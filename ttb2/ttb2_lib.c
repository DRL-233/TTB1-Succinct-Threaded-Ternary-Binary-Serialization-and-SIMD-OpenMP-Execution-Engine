/* ttb2_lib.c - TTB v0.3.0 Standalone C/AVX2/OpenMP Engine */
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

static void inflate_compact_values_avx2(
    const uint8_t *src, uint8_t width_code, uint32_t count, int64_t *dst
) {
    uint32_t i = 0;
    if (width_code == 1) {
#if defined(__AVX2__)
        for (; i + 4 <= count; i += 4) {
            int32_t raw4;
            memcpy(&raw4, src + i, 4);
            __m128i v8 = _mm_cvtsi32_si128(raw4);
            __m256i v64 = _mm256_cvtepi8_epi64(v8);
            _mm256_storeu_si256((__m256i *)(dst + i), v64);
        }
#endif
        for (; i < count; i++) dst[i] = (int64_t)((int8_t)src[i]);
    } else if (width_code == 2) {
#if defined(__AVX2__)
        for (; i + 4 <= count; i += 4) {
            __m128i v16 = _mm_loadl_epi64((const __m128i *)(src + i * 2));
            __m256i v64 = _mm256_cvtepi16_epi64(v16);
            _mm256_storeu_si256((__m256i *)(dst + i), v64);
        }
#endif
        for (; i < count; i++) {
            int16_t v;
            memcpy(&v, src + i * 2, 2);
            dst[i] = (int64_t)v;
        }
    } else if (width_code == 4) {
#if defined(__AVX2__)
        for (; i + 4 <= count; i += 4) {
            __m128i v32 = _mm_loadu_si128((const __m128i *)(src + i * 4));
            __m256i v64 = _mm256_cvtepi32_epi64(v32);
            _mm256_storeu_si256((__m256i *)(dst + i), v64);
        }
#endif
        for (; i < count; i++) {
            int32_t v;
            memcpy(&v, src + i * 4, 4);
            dst[i] = (int64_t)v;
        }
    } else {
        memcpy(dst, src, (size_t)count * 8);
    }
}

TTB2Handle* ttb2_compile_with_values(
    const uint8_t *buf, size_t len, const int64_t *ext_values, int64_t skip_sentinel
) {
    if (!buf || len < 20 || memcmp(buf, "TTB2", 4) != 0) return NULL;
    init_trit_lut();

    uint8_t flags = buf[4];
    uint8_t width_code = buf[5];
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
    h->parent_hub    = (int32_t *)malloc((n_nodes + 1) * sizeof(int32_t));
    for (uint32_t i = 0; i <= n_nodes; i++) h->parent_hub[i] = -1;

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

    h->values = alloc_aligned_64(n_nodes + 8);
    memset(h->values, 0, (n_nodes + 8) * sizeof(int64_t));

    if ((flags & 0x04) && (width_code == 1 || width_code == 2 || width_code == 4 || width_code == 8)) {
        uint32_t n_leaves = 0;
        for (uint32_t i = 0; i < n_nodes; i++) {
            if (h->degrees[i] == 0) n_leaves++;
        }
        size_t req = (size_t)n_leaves * width_code;
        if (len >= header_end + req && n_leaves > 0) {
            int64_t *leaf_buf = alloc_aligned_64(n_leaves + 8);
            inflate_compact_values_avx2(p, width_code, n_leaves, leaf_buf);
            uint32_t li = 0;
            for (uint32_t i = 0; i < n_nodes; i++) {
                if (h->degrees[i] == 0) h->values[i] = leaf_buf[li++];
            }
            free_aligned_64(leaf_buf);
        }
    } else if ((flags & 0x02) && len >= header_end + (size_t)n_nodes * 8) {
        if (n_nodes > 0) memcpy(h->values, p, (size_t)n_nodes * 8);
    } else if (ext_values != NULL && n_nodes > 0) {
        memcpy(h->values, ext_values, (size_t)n_nodes * 8);
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

TTB2Handle* ttb2_compile(const uint8_t *buf, size_t len, int64_t skip_sentinel) {
    return ttb2_compile_with_values(buf, len, NULL, skip_sentinel);
}

int ttb2_update_leaf(TTB2Handle *h, uint32_t node_idx, int64_t new_val, int64_t skip_sentinel) {
    if (!h || node_idx >= h->n_nodes || h->degrees[node_idx] != 0) return 0;
    int64_t old_val = h->values[node_idx];
    if (old_val == new_val) return 1;
    h->values[node_idx] = new_val;

    int32_t s = h->parent_hub[node_idx];
    if (s >= 0 && (uint32_t)s < h->n_spokes) {
        h->spoke_targets[s] += (new_val - old_val);
        if (old_val == skip_sentinel || new_val == skip_sentinel) {
            uint32_t start = h->spoke_offsets[s];
            uint32_t end = start + h->spoke_counts[s];
            if (end > h->n_nodes) end = h->n_nodes;
            int pruned = 0;
            for (uint32_t i = start; i < end; i++) {
                if (h->values[i] == skip_sentinel) { pruned = 1; break; }
            }
            h->spoke_active_mask[s] = pruned ? 0LL : -1LL;
        }
    } else {
        h->root_scalar_sum += (new_val - old_val);
    }
    return 1;
}

int ttb2_set_spoke_status(TTB2Handle *h, uint32_t spoke_idx, int active) {
    if (!h || spoke_idx >= h->n_spokes) return 0;
    h->spoke_active_mask[spoke_idx] = active ? -1LL : 0LL;
    return 1;
}

int64_t ttb2_sum_leaves(const TTB2Handle *h) {
    if (!h) return 0;
    const uint32_t ns = h->n_spokes;
    const int64_t *targets = h->spoke_targets;
    int64_t total = h->root_scalar_sum;

    #pragma omp parallel for reduction(+:total) schedule(static) if(ns > 256)
    for (uint32_t s = 0; s < ns; s++) total += targets[s];
    return total;
}

int64_t ttb2_broadcast_center(const TTB2Handle *h) {
    if (!h) return 0;
    const uint32_t ns = h->n_spokes;
    const int64_t *targets = h->spoke_targets;
    int64_t grand_total = 0;

#if defined(__AVX2__)
    uint32_t vec_end = ns & ~3U;
    __m256i vsum = _mm256_setzero_si256();
    for (uint32_t s = 0; s < vec_end; s += 4) {
        vsum = _mm256_add_epi64(vsum, _mm256_load_si256((const __m256i *)(targets + s)));
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

int64_t ttb2_query_skip(const TTB2Handle *h) {
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

TTB2JumpResult ttb2_jump_sibling(const TTB2Handle *h, uint32_t src_spoke, uint32_t dst_spoke) {
    TTB2JumpResult res = {0, 0};
    if (!h || dst_spoke >= h->n_spokes) return res;
    res.value = h->spoke_targets[dst_spoke];
    res.hops  = (dst_spoke == src_spoke + 1 || src_spoke == dst_spoke + 1) ? 1U : 2U;
    return res;
}

TTB2AscentResult ttb2_ascent_to_center(const TTB2Handle *h, uint32_t node_idx) {
    TTB2AscentResult res = {-1, 0};
    if (!h || node_idx >= h->n_nodes) return res;
    int32_t sid = h->parent_hub[node_idx];
    res.spoke_id = sid;
    res.hub_offset = (sid >= 0) ? h->spoke_offsets[sid] : 0U;
    return res;
}

int32_t ttb2_ozip_verify_block_avx2(
    const TTB2OZIPBlock994 *blk,
    uint64_t mask_A[6],
    uint64_t mask_B[6],
    uint64_t mask_C[6]
) {
    if (!blk) return 0;
    uint64_t local_A[6] = {0}, local_B[6] = {0}, local_C[6] = {0};

    for (int i = 0; i < 81; ++i) {
        int v = blk->A_idx[i];
        if (v < 0 || v >= 324) return 0;
        uint64_t bit = 1ULL << (v & 63);
        if (local_A[v >> 6] & bit) return 0;
        local_A[v >> 6] |= bit;
    }
    for (int i = 0; i < 36; ++i) {
        int vb = blk->B_idx[i], vc = blk->C_idx[i];
        if (vb < 0 || vb >= 324 || vc < 0 || vc >= 324) return 0;
        uint64_t bit_b = 1ULL << (vb & 63);
        uint64_t bit_c = 1ULL << (vc & 63);
        if ((local_B[vb >> 6] & bit_b) || (local_C[vc >> 6] & bit_c)) return 0;
        local_B[vb >> 6] |= bit_b;
        local_C[vc >> 6] |= bit_c;
    }
#if defined(__AVX2__)
    if (!_mm256_testz_si256(_mm256_loadu_si256((const __m256i *)mask_A), _mm256_loadu_si256((const __m256i *)local_A))) return 0;
    if (!_mm256_testz_si256(_mm256_loadu_si256((const __m256i *)mask_B), _mm256_loadu_si256((const __m256i *)local_B))) return 0;
    if (!_mm256_testz_si256(_mm256_loadu_si256((const __m256i *)mask_C), _mm256_loadu_si256((const __m256i *)local_C))) return 0;
#endif
    for (int w = 0; w < 6; ++w) {
        if ((mask_A[w] & local_A[w]) || (mask_B[w] & local_B[w]) || (mask_C[w] & local_C[w])) return 0;
    }
    for (int w = 0; w < 6; ++w) {
        mask_A[w] |= local_A[w];
        mask_B[w] |= local_B[w];
        mask_C[w] |= local_C[w];
    }
    return 1;
}

void ttb2_ozip_contract_3x3x2(const int64_t A[9], const int64_t B[6], int64_t C[6]) {
    const int64_t b00 = B[0], b01 = B[1];
    const int64_t b10 = B[2], b11 = B[3];
    const int64_t b20 = B[4], b21 = B[5];
    for (int i = 0; i < 3; ++i) {
        const int64_t a0 = A[i * 3 + 0];
        const int64_t a1 = A[i * 3 + 1];
        const int64_t a2 = A[i * 3 + 2];
        C[i * 2 + 0] = a0 * b00 + a1 * b10 + a2 * b20;
        C[i * 2 + 1] = a0 * b01 + a1 * b11 + a2 * b21;
    }
}

void ttb2_ozip_contract_batch_3x3x2(
    const int64_t *A_batch, const int64_t *B_batch, int64_t *C_batch, uint32_t n_blocks
) {
    #pragma omp parallel for schedule(static) if(n_blocks > 256)
    for (uint32_t m = 0; m < n_blocks; ++m) {
        ttb2_ozip_contract_3x3x2(A_batch + m * 9, B_batch + m * 6, C_batch + m * 6);
    }
}

void ttb2_free(TTB2Handle *h) {
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
