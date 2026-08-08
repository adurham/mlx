// Copyright © 2024-25 Apple Inc.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"

#include "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h"

#define instantiate_attn(tname, dtype, bq, bk, bd, wm, wn, mname, mtype) \
  instantiate_kernel(                                                    \
      "steel_attention_" #tname "_bq" #bq "_bk" #bk "_bd" #bd            \
      "_wm" #wm "_wn" #wn "_mask" #mname,                                \
  attention, dtype, bq, bk, bd, wm, wn, mtype, float)

// 2026-08-08, real production incident: the bq=16/wm=1/bd=512
// instantiation below (added by commit 21008ab1a, "SDPA D=512 bq=16
// spike", an abandoned same-day A/B experiment gated behind
// MLX_SDPA_D512_BQ16 -- unset by default, never in start_cluster.sh's
// env allow-list, never actually used in production) is a compile-time
// dead end: TQ = BQ / (WM*WN*kFragSize) = 16 / (1*1*8) = 2, but
// attention()'s own static_assert(TQ == 1, "Check TQ") requires TQ==1
// unconditionally (see steel_attention.h). This was pushed to
// adurham/mlx main already broken and only surfaced tonight because
// every cluster deploy since 2026-07-16 happened to reuse an
// already-built cached wheel -- this was the first genuinely fresh
// Metal compile since the spike commit landed, on ANY toolchain (this
// is deterministic integer arithmetic, not a toolchain-version issue).
// Removing the dead instantiation only affects the never-exercised
// bq=16 spike path -- the bq=8 D=512 path (the actual production
// kernel for BD=512, "wm=1, (8+8+8)*512*2=24KB" per that same
// concurrent code path's own comment) is untouched.
#define instantiate_attn_shapes_helper(iname, itype, mname, mtype)  \
    instantiate_attn(iname, itype, 32, 16, 256, 4, 1, mname, mtype) \
    instantiate_attn(iname, itype, 32, 16, 128, 4, 1, mname, mtype) \
    instantiate_attn(iname, itype, 32, 32,  80, 4, 1, mname, mtype) \
    instantiate_attn(iname, itype, 32, 32,  64, 4, 1, mname, mtype) \
    instantiate_attn(iname, itype,  8,  8, 512, 1, 1, mname, mtype)

#define instantiate_attn_mask_helper(iname, itype) \
    instantiate_attn_shapes_helper(iname, itype, iname, itype) \
    instantiate_attn_shapes_helper(iname, itype, bool_, bool)

instantiate_attn_mask_helper(float16, half);
instantiate_attn_mask_helper(bfloat16, bfloat16_t);

instantiate_attn_mask_helper(float32, float);
// clang-format on