/* ttb2.h - TTB v0.3.0 (TTB3-Bridge) Pure-C ABI
 * Unifies:
 *   1. Cycloidal Hub-Entangled Succinct Ternary Execution (DOI: 10.5281/zenodo.24379185)
 *   2. Lean-4-Verified OZIP-Algebra v2.0.0 Mixed-Radix Z3 Triality Contractions (DOI: 10.5281/zenodo.24528210)
 * Author: Matthias Asare-Darko (Darkon Research Laboratory, Ghana)
 * License: MIT
 */
#ifndef TTB2_H
#define TTB2_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TTB2Handle TTB2Handle;

typedef struct {
    int64_t  value;
    uint32_t hops;
} TTB2JumpResult;

typedef struct {
    int32_t  spoke_id;
    uint32_t hub_offset;
} TTB2AscentResult;

typedef struct {
    int32_t A_idx[81];
    int32_t B_idx[36];
    int32_t C_idx[36];
} TTB2OZIPBlock994;

TTB2Handle* ttb2_compile(const uint8_t *buf, size_t len, int64_t skip_sentinel);
TTB2Handle* ttb2_compile_with_values(const uint8_t *buf, size_t len, const int64_t *ext_values, int64_t skip_sentinel);

int64_t          ttb2_sum_leaves(const TTB2Handle *h);
int64_t          ttb2_query_skip(const TTB2Handle *h);
int64_t          ttb2_broadcast_center(const TTB2Handle *h);
TTB2JumpResult   ttb2_jump_sibling(const TTB2Handle *h, uint32_t src_spoke, uint32_t dst_spoke);
TTB2AscentResult ttb2_ascent_to_center(const TTB2Handle *h, uint32_t node_idx);

int ttb2_update_leaf(TTB2Handle *h, uint32_t node_idx, int64_t new_val, int64_t skip_sentinel);
int ttb2_set_spoke_status(TTB2Handle *h, uint32_t spoke_idx, int active);

static inline uint32_t ttb2_ozip_psi1(uint32_t s, uint32_t g, uint32_t i) {
    return 9U * s + 3U * g + i;
}

int32_t ttb2_ozip_verify_block_avx2(
    const TTB2OZIPBlock994 *blk,
    uint64_t mask_A[6],
    uint64_t mask_B[6],
    uint64_t mask_C[6]
);

void ttb2_ozip_contract_3x3x2(
    const int64_t A[9],
    const int64_t B[6],
    int64_t       C[6]
);

void ttb2_ozip_contract_batch_3x3x2(
    const int64_t *A_batch,
    const int64_t *B_batch,
    int64_t       *C_batch,
    uint32_t       n_blocks
);

void ttb2_free(TTB2Handle *h);

#ifdef __cplusplus
}
#endif

#endif
