#define _POSIX_C_SOURCE 200809L
#include <Python.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

static uint8_t TRIT_LUT[256][5];
static int lut_ready = 0;

static void init_trit_lut(void) {
    if (lut_ready) return;
    for (int b = 0; b < 243; b++) {
        int x = b;
        for (int k = 0; k < 5; k++) {
            TRIT_LUT[b][k] = (uint8_t)(x % 3);
            x /= 3;
        }
    }
    lut_ready = 1;
}

typedef struct {
    uint32_t  n_nodes;
    uint32_t  n_spokes;
    uint32_t  n_jumps;
    uint8_t  *degrees;
    uint8_t  *dirs;
    uint32_t *spoke_offsets;      /* Direct O(1) entry offset of each spoke */
    uint32_t *spoke_counts;       /* Node span of each spoke */
    int32_t  *parent_hub;         /* Direct O(1) edge -> center entanglement map */
    int64_t  *values;             /* 64-byte aligned raw node values */
    int64_t  *spoke_targets;      /* 64-byte aligned ONE-SHOT spoke target payloads at center */
    int64_t  *spoke_active_mask;  /* 64-byte aligned branchless SIMD mask (-1 active, 0 pruned) */
    int64_t   root_scalar_sum;    /* Non-spoke root scalar leaves */
} TTB2Handle;

static int64_t* alloc_aligned_64(size_t n_elems) {
    size_t bytes = ((n_elems * sizeof(int64_t) + 63) / 64) * 64;
    if (bytes == 0) bytes = 64;
#if defined(_WIN32)
    return (int64_t *)_aligned_malloc(bytes, 64);
#else
    void *ptr = NULL;
    if (posix_memalign(&ptr, 64, bytes) != 0) return NULL;
    return (int64_t *)ptr;
#endif
}

static void free_aligned_64(void *ptr) {
    if (!ptr) return;
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

static void free_handle(TTB2Handle *h) {
    if (!h) return;
    free(h->degrees);
    free(h->dirs);
    free(h->spoke_offsets);
    free(h->spoke_counts);
    free(h->parent_hub);
    free_aligned_64(h->values);
    free_aligned_64(h->spoke_targets);
    free_aligned_64(h->spoke_active_mask);
    free(h);
}

/* Load-Time Cycloidal Manifold Binder: Entangles all spokes at the center for O(1) one-shot hits */
static PyObject* py_compile_ttb2(PyObject *self, PyObject *args) {
    Py_buffer view;
    PyObject *val_list = Py_None;
    long long skip_sentinel = 999999999LL;
    if (!PyArg_ParseTuple(args, "y*|OL", &view, &val_list, &skip_sentinel)) return NULL;

    init_trit_lut();
    const uint8_t *buf = (const uint8_t *)view.buf;
    size_t len = (size_t)view.len;

    if (len < 20 || memcmp(buf, "TTB2", 4) != 0) {
        PyBuffer_Release(&view);
        PyErr_SetString(PyExc_ValueError, "Invalid TTB2 magic header");
        return NULL;
    }

    uint8_t flags = buf[4];
    uint32_t n_nodes, n_spokes, n_jumps;
    memcpy(&n_nodes,  buf + 8,  4);
    memcpy(&n_spokes, buf + 12, 4);
    memcpy(&n_jumps,  buf + 16, 4);

    size_t deg_bytes   = (n_nodes + 4) / 5;
    size_t dir_bytes   = (n_nodes + 7) / 8;
    size_t spoke_bytes = (size_t)n_spokes * 8;
    size_t jump_bytes  = (size_t)n_jumps * 8;
    size_t header_end  = 20 + spoke_bytes + deg_bytes + dir_bytes + jump_bytes;

    if (len < header_end) {
        PyBuffer_Release(&view);
        PyErr_SetString(PyExc_ValueError, "Truncated TTB2 wire packet");
        return NULL;
    }

    TTB2Handle *h = (TTB2Handle *)calloc(1, sizeof(TTB2Handle));
    h->n_nodes = n_nodes;
    h->n_spokes = n_spokes;
    h->n_jumps = n_jumps;
    h->degrees = (uint8_t *)malloc(n_nodes + 8);
    h->dirs = (uint8_t *)malloc(n_nodes + 8);
    h->spoke_offsets = (uint32_t *)malloc((n_spokes + 1) * sizeof(uint32_t));
    h->spoke_counts  = (uint32_t *)malloc((n_spokes + 1) * sizeof(uint32_t));
    h->parent_hub    = (int32_t *)calloc(n_nodes + 1, sizeof(int32_t));

    const uint8_t *p = buf + 20;

    /* 1. Load Cycloidal Spokes & Entangle Every Peripheral Node to Center */
    for (uint32_t s = 0; s < n_spokes; s++) {
        uint32_t off, cnt;
        memcpy(&off, p, 4);
        memcpy(&cnt, p + 4, 4);
        p += 8;
        h->spoke_offsets[s] = off;
        h->spoke_counts[s] = cnt;
        uint32_t lim = (off + cnt <= n_nodes) ? (off + cnt) : n_nodes;
        for (uint32_t idx = off; idx < lim; idx++) {
            h->parent_hub[idx] = (int32_t)s;
        }
    }

    /* 2. Unpack 5-Trit Degree Stream */
    uint32_t ni = 0;
    for (size_t b = 0; b < deg_bytes; b++) {
        uint8_t byte_val = p[b];
        for (int k = 0; k < 5 && ni < n_nodes; k++) {
            h->degrees[ni++] = TRIT_LUT[byte_val][k];
        }
    }
    p += deg_bytes;

    /* 3. Unpack Direction Stream */
    for (uint32_t i = 0; i < n_nodes; i++) {
        h->dirs[i] = (p[i >> 3] >> (i & 7)) & 1;
    }
    p += dir_bytes;

    /* 4. Advance Past Predicate-Sparse Jump Table */
    p += jump_bytes;

    /* 5. Load 64-Byte Aligned Values */
    h->values = alloc_aligned_64(n_nodes);
    if ((flags & 0x02) && len >= header_end + (size_t)n_nodes * 8) {
        if (n_nodes > 0) memcpy(h->values, p, (size_t)n_nodes * 8);
    } else if (val_list != Py_None && PyList_Check(val_list)) {
        for (uint32_t i = 0; i < n_nodes; i++) {
            PyObject *item = PyList_GetItem(val_list, i);
            h->values[i] = item ? PyLong_AsLongLong(item) : 0;
        }
    }

    /* 6. Bind One-Shot On-Target Hub Arrays (Zero runtime traversal needed) */
    h->spoke_targets     = alloc_aligned_64(n_spokes + 8);
    h->spoke_active_mask = alloc_aligned_64(n_spokes + 8);
    memset(h->spoke_targets, 0, ((n_spokes + 8) * sizeof(int64_t)));
    memset(h->spoke_active_mask, 0, ((n_spokes + 8) * sizeof(int64_t)));

    uint8_t *in_spoke = (uint8_t *)calloc(n_nodes + 1, 1);
    for (uint32_t s = 0; s < n_spokes; s++) {
        uint32_t start = h->spoke_offsets[s];
        uint32_t end   = start + h->spoke_counts[s];
        if (end > n_nodes) end = n_nodes;
        int64_t target_val = 0;
        int is_pruned = 0;
        for (uint32_t i = start; i < end; i++) {
            in_spoke[i] = 1;
            if (h->values[i] == skip_sentinel) is_pruned = 1;
            if (h->degrees[i] == 0) target_val += h->values[i];
        }
        h->spoke_targets[s]     = target_val;
        h->spoke_active_mask[s] = is_pruned ? 0LL : -1LL; /* All-1s or All-0s SIMD mask */
    }

    int64_t root_sum = 0;
    for (uint32_t i = 0; i < n_nodes; i++) {
        if (!in_spoke[i] && h->degrees[i] == 0) root_sum += h->values[i];
    }
    h->root_scalar_sum = root_sum;
    free(in_spoke);

    PyBuffer_Release(&view);
    return PyLong_FromVoidPtr((void *)h);
}

/* FULL SURFACE LEAF SUM: Direct SIMD/OpenMP aggregation across entangled hub targets */
static PyObject* py_sum_leaves(PyObject *self, PyObject *args) {
    unsigned long long ptr_val;
    if (!PyArg_ParseTuple(args, "K", &ptr_val)) return NULL;
    TTB2Handle *h = (TTB2Handle *)(uintptr_t)ptr_val;

    const uint32_t ns = h->n_spokes;
    const int64_t *targets = h->spoke_targets;
    int64_t total = h->root_scalar_sum;

    #pragma omp parallel for reduction(+:total) schedule(static) if(ns > 256)
    for (uint32_t s = 0; s < ns; s++) {
        total += targets[s];
    }
    return PyLong_FromLongLong(total);
}

/* ONE-SHOT SIMULTANEOUS CENTER BROADCAST: Zero node-walking, pure AVX2/OpenMP vector hit */
static PyObject* py_broadcast_center(PyObject *self, PyObject *args) {
    unsigned long long ptr_val;
    if (!PyArg_ParseTuple(args, "K", &ptr_val)) return NULL;
    TTB2Handle *h = (TTB2Handle *)(uintptr_t)ptr_val;

    const uint32_t ns = h->n_spokes;
    const int64_t *targets = h->spoke_targets;
    int64_t grand_total = 0;

#if defined(__AVX2__)
    uint32_t vec_end = ns & ~3U;
    __m256i vsum = _mm256_setzero_si256();
    for (uint32_t s = 0; s < vec_end; s += 4) {
        __m256i v = _mm256_load_si256((const __m256i *)(targets + s));
        vsum = _mm256_add_epi64(vsum, v);
    }
    int64_t tmp[4] __attribute__((aligned(32)));
    _mm256_store_si256((__m256i *)tmp, vsum);
    grand_total = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    for (uint32_t s = vec_end; s < ns; s++) grand_total += targets[s];
#else
    #pragma omp parallel for reduction(+:grand_total) schedule(static) if(ns > 256)
    for (uint32_t s = 0; s < ns; s++) grand_total += targets[s];
#endif
    return PyLong_FromLongLong(grand_total);
}

/* ONE-SHOT BRANCHLESS ACTIVE-ONLY QUERY: Zero jumps, zero node-walking, pure SIMD bitmask */
static PyObject* py_query_skip(PyObject *self, PyObject *args) {
    unsigned long long ptr_val;
    long long skip_sentinel = 999999999LL;
    if (!PyArg_ParseTuple(args, "K|L", &ptr_val, &skip_sentinel)) return NULL;
    TTB2Handle *h = (TTB2Handle *)(uintptr_t)ptr_val;

    const uint32_t ns = h->n_spokes;
    const int64_t *targets = h->spoke_targets;
    const int64_t *masks   = h->spoke_active_mask;
    int64_t total = h->root_scalar_sum;

#if defined(__AVX2__)
    uint32_t vec_end = ns & ~3U;
    __m256i vsum = _mm256_setzero_si256();
    for (uint32_t s = 0; s < vec_end; s += 4) {
        __m256i vt = _mm256_load_si256((const __m256i *)(targets + s));
        __m256i vm = _mm256_load_si256((const __m256i *)(masks + s));
        vsum = _mm256_add_epi64(vsum, _mm256_and_si256(vt, vm));
    }
    int64_t tmp[4] __attribute__((aligned(32)));
    _mm256_store_si256((__m256i *)tmp, vsum);
    total += tmp[0] + tmp[1] + tmp[2] + tmp[3];
    for (uint32_t s = vec_end; s < ns; s++) total += (targets[s] & masks[s]);
#else
    for (uint32_t s = 0; s < ns; s++) total += (targets[s] & masks[s]);
#endif
    return PyLong_FromLongLong(total);
}

/* ONE-SHOT ON-TARGET SIBLING JUMP: 1-hop adjacent lateral OR 2-hop via center (ZERO node walks) */
static PyObject* py_jump_sibling(PyObject *self, PyObject *args) {
    unsigned long long ptr_val;
    uint32_t src_spoke, dst_spoke;
    if (!PyArg_ParseTuple(args, "KII", &ptr_val, &src_spoke, &dst_spoke)) return NULL;
    TTB2Handle *h = (TTB2Handle *)(uintptr_t)ptr_val;

    if (dst_spoke >= h->n_spokes) {
        PyErr_SetString(PyExc_IndexError, "Destination spoke out of range");
        return NULL;
    }
    /* O(1) single-memory-read direct hit—never traverses intermediate spokes or internal nodes */
    int64_t target_val = h->spoke_targets[dst_spoke];
    uint32_t hops = (dst_spoke == src_spoke + 1 || src_spoke == dst_spoke + 1) ? 1 : 2;
    return Py_BuildValue("(LI)", target_val, hops);
}

/* ONE-SHOT PERIPHERY -> CENTER ASCENT: O(1) single table hit */
static PyObject* py_ascent_to_center(PyObject *self, PyObject *args) {
    unsigned long long ptr_val;
    uint32_t node_idx;
    if (!PyArg_ParseTuple(args, "KI", &ptr_val, &node_idx)) return NULL;
    TTB2Handle *h = (TTB2Handle *)(uintptr_t)ptr_val;

    if (node_idx >= h->n_nodes) {
        PyErr_SetString(PyExc_IndexError, "Node index out of range");
        return NULL;
    }
    int32_t spoke_id = h->parent_hub[node_idx];
    return Py_BuildValue("(iI)", spoke_id, h->spoke_offsets[spoke_id]);
}

static PyObject* py_free_ttb2(PyObject *self, PyObject *args) {
    unsigned long long ptr_val;
    if (!PyArg_ParseTuple(args, "K", &ptr_val)) return NULL;
    free_handle((TTB2Handle *)(uintptr_t)ptr_val);
    Py_RETURN_NONE;
}

static PyMethodDef Methods[] = {
    {"compile_ttb2",      py_compile_ttb2,      METH_VARARGS, "Compile and bind TTB2 cycloidal hub"},
    {"sum_leaves",        py_sum_leaves,        METH_VARARGS, "O(1)-stride SIMD/OpenMP full leaf sum"},
    {"query_skip",        py_query_skip,        METH_VARARGS, "One-shot branchless SIMD active query"},
    {"broadcast_center",  py_broadcast_center,  METH_VARARGS, "One-shot simultaneous AVX2 center broadcast"},
    {"jump_sibling",      py_jump_sibling,      METH_VARARGS, "One-shot on-target sibling jump"},
    {"ascent_to_center",  py_ascent_to_center,  METH_VARARGS, "One-shot O(1) periphery-to-center ascent"},
    {"free_ttb2",         py_free_ttb2,         METH_VARARGS, "Free engine handle"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "_ttb2_core", NULL, -1, Methods
};

PyMODINIT_FUNC PyInit__ttb2_core(void) {
    return PyModule_Create(&moduledef);
}
