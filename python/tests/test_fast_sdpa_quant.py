"""Unit tests for mx.fast.scaled_dot_product_attention_quant.

Validates that the quantized-KV SDPA kernel's output matches a reference
computed via mx.dequantize + mx.fast.scaled_dot_product_attention, across
the full v1 supported grid:

  bits      ∈ {4, 5, 8}
  head_dim  ∈ {64, 128}
  N (T_kv)  ∈ {256, 8192, 66560}

The 66K case is pulled straight from the MiniMax-M2.7 66K-context decode
profile — if the kernel drifts there, it breaks the actual production
workload. 8192 covers the mid-range block-count heuristic; 256 covers the
small-N edge case where the block heuristic falls back to smaller
shards.

Tolerances reflect the fact that the reference path goes dequantize →
bf16-SDPA (two kernels, with an intermediate materialized write) while
the quant kernel fuses the dequantize into the inner loop, so the
intermediate roundings differ by exactly the bf16 precision they would
in the reference case.
"""

import math
import unittest

import mlx.core as mx
import mlx_tests


def _make_quant_kv(shape, bits, group_size, dtype):
    """Random K or V tensor, quantized and then returned as both the
    dequantized-reference and the (packed, scales, biases) triple.
    """
    x = mx.random.normal(shape=shape, dtype=mx.float32).astype(dtype)
    packed, scales, biases = mx.quantize(
        x, group_size=group_size, bits=bits
    )
    x_deq = mx.dequantize(
        packed, scales, biases, group_size=group_size, bits=bits
    ).astype(dtype)
    return x_deq, packed, scales, biases


class TestFastSDPAQuant(mlx_tests.MLXTestCase):
    """v1 kernel correctness vs dequantize-then-SDPA reference."""

    def _run_case(self, *, bits, head_dim, N, dtype, do_causal=False):
        mx.random.seed(1234)

        B = 1
        n_q_heads = 8
        n_kv_heads = 2  # GQA factor 4, matches common Qwen/MiniMax layouts
        group_size = 64
        scale = 1.0 / math.sqrt(head_dim)

        q = mx.random.normal(
            shape=(B, n_q_heads, 1, head_dim), dtype=mx.float32
        ).astype(dtype)
        k_deq, k_packed, k_scales, k_biases = _make_quant_kv(
            (B, n_kv_heads, N, head_dim), bits, group_size, dtype
        )
        v_deq, v_packed, v_scales, v_biases = _make_quant_kv(
            (B, n_kv_heads, N, head_dim), bits, group_size, dtype
        )

        ref = mx.fast.scaled_dot_product_attention(
            q,
            k_deq,
            v_deq,
            scale=scale,
            mask="causal" if do_causal else None,
        )
        out = mx.fast.scaled_dot_product_attention_quant(
            q,
            k_packed,
            k_scales,
            k_biases,
            v_packed,
            v_scales,
            v_biases,
            scale=scale,
            group_size=group_size,
            bits=bits,
            do_causal=do_causal,
        )
        mx.eval(ref, out)

        # The quant kernel fuses dequantize into the dot product; the
        # reference kernel writes dequantized K/V through bf16/fp16
        # before the dot product. That's an extra rounding step on the
        # reference side, so tolerances need to cover one ULP of
        # accumulated error at N positions.
        if dtype == mx.bfloat16:
            atol, rtol = 2e-2, 2e-2
        elif dtype == mx.float16:
            atol, rtol = 1e-2, 1e-2
        else:
            atol, rtol = 2e-3, 2e-3

        self.assertEqual(out.shape, ref.shape)
        self.assertEqual(out.dtype, ref.dtype)
        self.assertTrue(
            mx.allclose(out, ref, atol=atol, rtol=rtol).item(),
            f"bits={bits} head_dim={head_dim} N={N} dtype={dtype} "
            f"max abs diff: {mx.max(mx.abs(out - ref)).item():g}",
        )

    # ---------------------------------------------------------------
    # Supported grid: bits × head_dim × N × dtype
    # ---------------------------------------------------------------
    def test_bits4_hd64_small(self):
        self._run_case(bits=4, head_dim=64, N=256, dtype=mx.bfloat16)

    def test_bits4_hd128_small(self):
        self._run_case(bits=4, head_dim=128, N=256, dtype=mx.bfloat16)

    def test_bits5_hd64_small(self):
        self._run_case(bits=5, head_dim=64, N=256, dtype=mx.bfloat16)

    def test_bits5_hd128_small(self):
        self._run_case(bits=5, head_dim=128, N=256, dtype=mx.bfloat16)

    def test_bits8_hd64_small(self):
        self._run_case(bits=8, head_dim=64, N=256, dtype=mx.bfloat16)

    def test_bits8_hd128_small(self):
        self._run_case(bits=8, head_dim=128, N=256, dtype=mx.bfloat16)

    # Mid-range context (hits different block heuristic).
    def test_bits5_hd128_mid(self):
        self._run_case(bits=5, head_dim=128, N=8192, dtype=mx.bfloat16)

    def test_bits4_hd128_mid(self):
        self._run_case(bits=4, head_dim=128, N=8192, dtype=mx.bfloat16)

    # MiniMax production context shape (66K tokens). Guardrail for the
    # real workload that motivated this kernel.
    def test_bits5_hd128_production(self):
        self._run_case(bits=5, head_dim=128, N=66560, dtype=mx.bfloat16)

    # Dtype coverage.
    def test_bits5_hd128_fp16(self):
        self._run_case(bits=5, head_dim=128, N=256, dtype=mx.float16)

    def test_bits5_hd128_fp32(self):
        self._run_case(bits=5, head_dim=128, N=256, dtype=mx.float32)

    # Causal masking.
    def test_causal_bits5_hd128(self):
        self._run_case(
            bits=5, head_dim=128, N=8192, dtype=mx.bfloat16, do_causal=True
        )

    # ---------------------------------------------------------------
    # Fallback path: prefill (q_seq_len > 1) must still produce
    # matching output via dequantize + SDPA.
    # ---------------------------------------------------------------
    def test_fallback_prefill_matches(self):
        mx.random.seed(7)
        B, n_q, n_kv, head_dim, group_size, bits = 1, 4, 4, 64, 64, 5
        T_q, T_kv = 16, 256
        dtype = mx.bfloat16
        scale = 1.0 / math.sqrt(head_dim)

        q = mx.random.normal(shape=(B, n_q, T_q, head_dim), dtype=mx.float32).astype(
            dtype
        )
        k_deq, k_packed, k_scales, k_biases = _make_quant_kv(
            (B, n_kv, T_kv, head_dim), bits, group_size, dtype
        )
        v_deq, v_packed, v_scales, v_biases = _make_quant_kv(
            (B, n_kv, T_kv, head_dim), bits, group_size, dtype
        )
        ref = mx.fast.scaled_dot_product_attention(
            q, k_deq, v_deq, scale=scale
        )
        out = mx.fast.scaled_dot_product_attention_quant(
            q,
            k_packed,
            k_scales,
            k_biases,
            v_packed,
            v_scales,
            v_biases,
            scale=scale,
            group_size=group_size,
            bits=bits,
        )
        mx.eval(ref, out)
        self.assertTrue(mx.allclose(out, ref, atol=2e-2, rtol=2e-2).item())


if __name__ == "__main__":
    unittest.main()
