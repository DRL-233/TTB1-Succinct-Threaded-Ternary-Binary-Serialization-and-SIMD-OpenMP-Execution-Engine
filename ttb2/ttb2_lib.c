#define _POSIX_C_SOURCE 200809L
#include "ttb2.h"
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

struct TTB2Handle {
    uint32_t  n_nodes;
    uint32_t  n_spokes;
    uint32_t  n_jumps;
    uint8_t  *degrees;
    uint8_t  *dirs;
    uint32_t *spoke_offsets;
    uint32_t *spoke_counts;
    int32_t  *parent_hub;
    int64_t  *values;
    int64_t  *spoke_targets;
    int64_t  *spoke_active_mask;
    int64_t   root_scalar_sum;
};

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

TTB2_API TTB2Handle* ttb2_compile(const uint8_t *buf, size_t len, int64_t skip_sentinel) {
    if (!buf || len < 20 || memcmp(buf, "TTB2", 4) != 0) return NULL;
    init_trit_lut();

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

    if (len < header_end) return NULL;

    TTB2Handle *h = (TTB2Handle *)calloc(1, sizeof(TTB2Handle));
    if (!h) return NULL;

    h->n_nodes = n_nodes;
    h->n_spokes = n_spokes;
    h->n_jumps = n_jumps;
    h->degrees = (uint8_t *)malloc(n_nodes + 8);
    h->dirs = (uint8_t *)malloc(n_nodes + 8);
    h->spoke_offsets = (uint32_t *)malloc((n_spokes + 1) * sizeof(uint32_t));
    h->spoke_counts  = (uint32_t *)malloc((n_spokes + 1) * sizeof(uint32_t));
    h->parent_hub    = (int32_t *)calloc(n_nodes + 1, sizeof(int32_t));

    const uint8_t *p = buf + 20;

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

    uint32_t ni = 0;
    for (size_t b = 0; b < deg_bytes; b++) {
        uint8_t byte_val = p[b];
        for (int k = 0; k < 5 && ni < n_nodes; k++) {
            h->degrees[ni++] = TRIT_LUT[byte_val][k];
        }
    }
    p += deg_bytes;

    for (uint32_t i = 0; i < n_nodes; i++) {
        h->dirs[i] = (p[i >> 3] >> (i & 7)) & 1;
    }
    p += dir_bytes + jump_bytes;

    h->values = alloc_aligned_64(n_nodes);
    if ((flags & 0x02) && len >= header_end + (size_t)n_nodes * 8) {
        if (n_nodes > 0) memcpy(h->values, p, (size_t)n_nodes * 8);
    } else {
        memset(h->values, 0, (size_t)n_nodes * sizeof(int64_t));
    }

    h->spoke_targets     = alloc_aligned_64(n_spokes + 8);
    h->spoke_active_mask = alloc_aligned_64(n_spokes + 8);
    memset(h->spoke_targets, 0, (n_spokes + 8) * sizeof(int64_t));
    memset(h->spoke_active_mask, 0, (n_spokes + 8) * sizeof(int64_t));

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
        h->spoke_active_mask[s] = is_pruned ? 0LL : -1LL;
    }

    int64_t root_sum = 0;
    for (uint32_t i = 0; i < n_nodes; i++) {
        if (!in_spoke[i] && h->degrees[i] == 0) root_sum += h->values[i];
    }
    h->root_scalar_sum = root_sum;
    free(in_spoke);
    return h;
}

TTB2_API int64_t ttb2_sum_leaves(const TTB2Handle *h) {
    if (!h) return 0;
    const uint32_t ns = h->n_spokes;
    const int64_t *targets = h->spoke_targets;
    int64_t total = h->root_scalar_sum;

    #pragma omp parallel for reduction(+:total) schedule(static) if(ns > 256)
    for (uint32_t s = 0; s < ns; s++) {
        total += targets[s];
    }
    return total;
}

TTB2_API int64_t ttb2_broadcast_center(const TTB2Handle *h) {
    if (!h) return 0;
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
    for (uint32_t s = 0; s < ns; s++) grand_total += targets[s];
#endif
    return grand_total;
}

TTB2_API int64_t ttb2_query_skip(const TTB2Handle *h) {
    if (!h) return 0;
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
    return total;
}

TTB2_API int64_t ttb2_jump_sibling(const TTB2Handle *h, uint32_t src_spoke, uint32_t dst_spoke, uint32_t *out_hops) {
    if (!h || dst_spoke >= h->n_spokes) return 0;
    if (out_hops) {
        *out_hops = (dst_spoke == src_spoke + 1 || src_spoke == dst_spoke + 1) ? 1U : 2U;
    }
    return h->spoke_targets[dst_spoke];
}

TTB2_API int32_t ttb2_ascent_to_center(const TTB2Handle *h, uint32_t node_idx, uint32_t *out_hub_offset) {
    if (!h || node_idx >= h->n_nodes) return -1;
    int32_t spoke_id = h->parent_hub[node_idx];
    if (out_hub_offset) *out_hub_offset = h->spoke_offsets[spoke_id];
    return spoke_id;
}

TTB2_API void ttb2_free(TTB2Handle *h) {
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
