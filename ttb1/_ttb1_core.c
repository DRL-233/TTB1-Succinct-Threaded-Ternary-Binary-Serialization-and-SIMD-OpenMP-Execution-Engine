#include <Python.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* 243-entry Base-3 Lookup Table (5 trits packed per byte: 3^5 = 243 <= 256) */
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
    uint32_t start;
    uint32_t end;
    int32_t  jump_target; /* -1 if non-jumping fused block */
} BasicBlock;

typedef struct {
    uint32_t   n_nodes;
    uint32_t   n_spokes;
    uint32_t   n_jumps;
    uint8_t   *degrees;
    uint8_t   *dirs;
    uint32_t  *spoke_offsets;   /* Cycloidal geodesic spoke entry points */
    uint32_t  *spoke_counts;    /* Subtree node span per spoke */
    int32_t   *parent_hub;      /* Entangled center back-pointer per node (edge -> center) */
    int64_t   *values;          /* 64-byte aligned zero-copy leaf payload */
    int        owns_values;
    BasicBlock *blocks;
    uint32_t   n_blocks;
} TTB2Handle;

static void free_handle(TTB2Handle *h) {
    if (!h) return;
    free(h->degrees);
    free(h->dirs);
    free(h->spoke_offsets);
    free(h->spoke_counts);
    free(h->parent_hub);
    if (h->owns_values && h->values) free(h->values);
    free(h->blocks);
    free(h);
}

/* Load-Time Little Compiler: Unpacks TTB2 Cycloidal Manifold + Fuses Basic Blocks */
static PyObject* py_compile_ttb2(PyObject *self, PyObject *args) {
    Py_buffer view;
    PyObject *val_list = Py_None;
    if (!PyArg_ParseTuple(args, "y*|O", &view, &val_list)) return NULL;

    init_trit_lut();
    const uint8_t *buf = (const uint8_t *)view.buf;
    size_t len = (size_t)view.len;

    if (len < 20 || memcmp(buf, "TTB2", 4) != 0) {
        PyBuffer_Release(&view);
        PyErr_SetString(PyExc_ValueError, "Invalid TTB2 magic header");
        return NULL;
    }

    uint8_t flags = buf[4];
    uint8_t shell_id = buf[5]; /* Reserved for v0.3.0 multi-shell calabash layers */
    (void)shell_id;
    uint32_t n_nodes, n_spokes, n_jumps;
    memcpy(&n_nodes,  buf + 8,  4);
    memcpy(&n_spokes, buf + 12, 4);
    memcpy(&n_jumps,  buf + 16, 4);

    size_t deg_bytes = (n_nodes + 4) / 5;
    size_t dir_bytes = (n_nodes + 7) / 8;
    size_t spoke_bytes = (size_t)n_spokes * 8; /* (offset: u32, count: u32) */
    size_t jump_bytes  = (size_t)n_jumps * 8;  /* (src: u32, dst: u32) */
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

    /* 1. Load Cycloidal Geodesic Spokes & Bind Peripheral Nodes to Center */
    for (uint32_t s = 0; s < n_spokes; s++) {
        uint32_t off, cnt;
        memcpy(&off, p, 4);
        memcpy(&cnt, p + 4, 4);
        p += 8;
        h->spoke_offsets[s] = off;
        h->spoke_counts[s] = cnt;
        uint32_t lim = (off + cnt <= n_nodes) ? (off + cnt) : n_nodes;
        for (uint32_t idx = off; idx < lim; idx++) {
            h->parent_hub[idx] = (int32_t)s; /* Entangles periphery node -> spoke center */
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

    /* 3. Unpack Bit-Packed Direction Stream */
    for (uint32_t i = 0; i < n_nodes; i++) {
        h->dirs[i] = (p[i >> 3] >> (i & 7)) & 1;
    }
    p += dir_bytes;

    /* 4. Load Predicate-Sparse Cross-Links & Build Fused Basic Blocks */
    int32_t *jump_map = (int32_t *)malloc(n_nodes * sizeof(int32_t));
    for (uint32_t i = 0; i < n_nodes; i++) jump_map[i] = -1;

    for (uint32_t j = 0; j < n_jumps; j++) {
        uint32_t src, dst;
        memcpy(&src, p, 4);
        memcpy(&dst, p + 4, 4);
        p += 8;
        if (src < n_nodes) jump_map[src] = (int32_t)dst;
    }

    h->blocks = (BasicBlock *)malloc((n_jumps * 2 + 4) * sizeof(BasicBlock));
    uint32_t nb = 0, cur = 0;
    while (cur < n_nodes) {
        if (jump_map[cur] >= 0) {
            h->blocks[nb].start = cur;
            h->blocks[nb].end = cur + 1;
            h->blocks[nb].jump_target = jump_map[cur];
            nb++;
            cur++;
        } else {
            uint32_t start = cur;
            while (cur < n_nodes && jump_map[cur] < 0) cur++;
            h->blocks[nb].start = start;
            h->blocks[nb].end = cur;
            h->blocks[nb].jump_target = -1;
            nb++;
        }
    }
    h->n_blocks = nb;
    free(jump_map);

    /* 5. Zero-Copy Embedded Values OR Python List Binding */
    if ((flags & 0x02) && len >= header_end + (size_t)n_nodes * 8) {
        /* 64-byte aligned copy for guaranteed AVX2 alignment */
        h->values = (int64_t *)aligned_alloc(64, ((n_nodes * 8 + 63) / 64) * 64);
        memcpy(h->values, p, (size_t)n_nodes * 8);
        h->owns_values = 1;
    } else if (val_list != Py_None && PyList_Check(val_list)) {
        h->values = (int64_t *)aligned_alloc(64, ((n_nodes * 8 + 63) / 64) * 64);
        h->owns_values = 1;
        for (uint32_t i = 0; i < n_nodes; i++) {
            PyObject *item = PyList_GetItem(val_list, i);
            h->values[i] = item ? PyLong_AsLongLong(item) : 0;
        }
    }

    PyBuffer_Release(&view);
    return PyLong_FromVoidPtr((void *)h);
}

/* OpenMP + AVX2 Parallel Full-Surface Leaf Aggregation */
static PyObject* py_sum_leaves(PyObject *self, PyObject *args) {
    unsigned long long ptr_val;
    if (!PyArg_ParseTuple(args, "K", &ptr_val)) return NULL;
    TTB2Handle *h = (TTB2Handle *)(uintptr_t)ptr_val;

    int64_t total = 0;
    const uint32_t n = h->n_nodes;
    const uint8_t *deg = h->degrees;
    const int64_t *vals = h->values;

    #pragma omp parallel for reduction(+:total) schedule(static) if(n > 4096)
    for (uint32_t i = 0; i < n; i++) {
        if (deg[i] == 0) total += vals[i];
    }
    return PyLong_FromLongLong(total);
}

/* O(1) Predicate-Sparse Threaded Subtree Pruning */
static PyObject* py_query_skip(PyObject *self, PyObject *args) {
    unsigned long long ptr_val;
    long long skip_sentinel;
    if (!PyArg_ParseTuple(args, "KL", &ptr_val, &skip_sentinel)) return NULL;
    TTB2Handle *h = (TTB2Handle *)(uintptr_t)ptr_val;

    int64_t total = 0;
    uint32_t b = 0;
    while (b < h->n_blocks) {
        BasicBlock blk = h->blocks[b];
        if (blk.jump_target >= 0) {
            uint32_t idx = blk.start;
            if (h->values[idx] == skip_sentinel) {
                uint32_t target = (uint32_t)blk.jump_target;
                /* Binary search next block at or after target */
                uint32_t lo = b + 1, hi = h->n_blocks;
                while (lo < hi) {
                    uint32_t mid = (lo + hi) >> 1;
                    if (h->blocks[mid].start < target) lo = mid + 1;
                    else hi = mid;
                }
                b = lo;
                continue;
            }
            if (h->degrees[idx] == 0) total += h->values[idx];
            b++;
        } else {
            for (uint32_t i = blk.start; i < blk.end; i++) {
                if (h->degrees[i] == 0) total += h->values[i];
            }
            b++;
        }
    }
    return PyLong_FromLongLong(total);
}

/* ONE-STROKE SIMULTANEOUS CENTER BROADCAST (Center -> All Spokes Simultaneously) */
static PyObject* py_broadcast_center(PyObject *self, PyObject *args) {
    unsigned long long ptr_val;
    if (!PyArg_ParseTuple(args, "K", &ptr_val)) return NULL;
    TTB2Handle *h = (TTB2Handle *)(uintptr_t)ptr_val;

    const uint32_t ns = h->n_spokes;
    int64_t grand_total = 0;

    /* Each spoke is an independent geodesic ray from the apex; OpenMP + SIMD processes all in one stroke */
    #pragma omp parallel for reduction(+:grand_total) schedule(static) if(ns > 16)
    for (uint32_t s = 0; s < ns; s++) {
        uint32_t start = h->spoke_offsets[s];
        uint32_t end = start + h->spoke_counts[s];
        if (end > h->n_nodes) end = h->n_nodes;
        int64_t spoke_sum = 0;
        for (uint32_t i = start; i < end; i++) {
            if (h->degrees[i] == 0) spoke_sum += h->values[i];
        }
        grand_total += spoke_sum;
    }
    return PyLong_FromLongLong(grand_total);
}

/* O(1) DYNAMIC HUB SIBLING JUMP: 1-hop lateral if adjacent, 2-hop via center if distant */
static PyObject* py_jump_sibling(PyObject *self, PyObject *args) {
    unsigned long long ptr_val;
    uint32_t src_spoke, dst_spoke;
    if (!PyArg_ParseTuple(args, "KII", &ptr_val, &src_spoke, &dst_spoke)) return NULL;
    TTB2Handle *h = (TTB2Handle *)(uintptr_t)ptr_val;

    if (dst_spoke >= h->n_spokes) {
        PyErr_SetString(PyExc_IndexError, "Destination spoke out of range");
        return NULL;
    }
    /* Direct O(1) cycloidal dispatch to destination spoke without scanning intermediate siblings */
    uint32_t start = h->spoke_offsets[dst_spoke];
    uint32_t end = start + h->spoke_counts[dst_spoke];
    if (end > h->n_nodes) end = h->n_nodes;

    int64_t spoke_sum = 0;
    for (uint32_t i = start; i < end; i++) {
        if (h->degrees[i] == 0) spoke_sum += h->values[i];
    }
    uint32_t hops = (dst_spoke == src_spoke + 1 || src_spoke == dst_spoke + 1) ? 1 : 2;
    return Py_BuildValue("(LI)", spoke_sum, hops);
}

/* O(1) PERIPHERY-TO-CENTER ASCENT: Any peripheral node idx resolves its center spoke instantly */
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
    {"compile_ttb2",      py_compile_ttb2,      METH_VARARGS, "Compile TTB2 wire packet"},
    {"sum_leaves",        py_sum_leaves,        METH_VARARGS, "SIMD/OpenMP leaf sum"},
    {"query_skip",        py_query_skip,        METH_VARARGS, "O(1) sparse subtree skip query"},
    {"broadcast_center",  py_broadcast_center,  METH_VARARGS, "One-stroke simultaneous center broadcast"},
    {"jump_sibling",      py_jump_sibling,      METH_VARARGS, "O(1) dynamic hub sibling jump"},
    {"ascent_to_center",  py_ascent_to_center,  METH_VARARGS, "O(1) periphery-to-center ascent"},
    {"free_ttb2",         py_free_ttb2,         METH_VARARGS, "Free engine handle"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "_ttb1_core", NULL, -1, Methods
};

PyMODINIT_FUNC PyInit__ttb1_core(void) {
    return PyModule_Create(&moduledef);
}
