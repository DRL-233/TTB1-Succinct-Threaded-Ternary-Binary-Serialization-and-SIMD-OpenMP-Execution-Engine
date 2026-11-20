#ifndef TTB2_H
#define TTB2_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
  #define TTB2_API __declspec(dllexport)
#else
  #define TTB2_API __attribute__((visibility("default")))
#endif

typedef struct TTB2Handle TTB2Handle;

TTB2_API TTB2Handle* ttb2_compile(const uint8_t *buf, size_t len, int64_t skip_sentinel);
TTB2_API int64_t     ttb2_sum_leaves(const TTB2Handle *h);
TTB2_API int64_t     ttb2_query_skip(const TTB2Handle *h);
TTB2_API int64_t     ttb2_broadcast_center(const TTB2Handle *h);
TTB2_API int64_t     ttb2_jump_sibling(const TTB2Handle *h, uint32_t src_spoke, uint32_t dst_spoke, uint32_t *out_hops);
TTB2_API int32_t     ttb2_ascent_to_center(const TTB2Handle *h, uint32_t node_idx, uint32_t *out_hub_offset);
TTB2_API void        ttb2_free(TTB2Handle *h);

#ifdef __cplusplus
}
#endif

#endif /* TTB2_H */
