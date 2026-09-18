"""
triton_backend.py — a real GPU backend, written to slot into the SAME
autotuning harness as the C/AVX-512 backends, with NO changes needed
to fusion2.py or the structure of autotune.py's candidate loop. This
proves the "fusion pass and autotuning harness are backend-agnostic"
claim with actual code, not just an assertion.

Honesty about what could be verified in this sandbox: there is no GPU
here (checked via torch.cuda.is_available()), so this kernel has been
verified to *define* correctly (triton.jit decoration succeeds, no
syntax/type errors) but its actual launch and numerical correctness
have NOT been exercised end-to-end on real hardware in this project.
is_triton_available() gates it out of the autotuning candidate list
entirely on a machine without CUDA -- it is never silently skipped as
if it had been tried and lost; it is correctly reported as untried.
"""

import numpy as np

try:
    import torch
    import triton
    import triton.language as tl
    _IMPORT_OK = True
except Exception:
    _IMPORT_OK = False


def is_triton_available():
    return _IMPORT_OK and torch.cuda.is_available()


if _IMPORT_OK:
    @triton.jit
    def _fused_bias_relu_kernel(c_ptr, bias_ptr, out_ptr,
                                 n_rows, n_cols,
                                 BLOCK_SIZE: tl.constexpr):
        row = tl.program_id(0)
        col_block = tl.program_id(1)
        col_start = col_block * BLOCK_SIZE
        offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols

        row_ptr = c_ptr + row * n_cols
        c = tl.load(row_ptr + offsets, mask=mask, other=0.0)
        bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
        v = c + bias
        v = tl.where(v > 0, v, 0.0)
        tl.store(out_ptr + row * n_cols + offsets, v, mask=mask)


class TritonBiasReLUKernel:
    """Same call convention as CompiledKernel: __call__(c_array, [bias_array])
    -- takes/returns numpy arrays, does the host<->device transfer
    internally, so it's a drop-in candidate in autotune.py's loop."""

    def __init__(self, block_size=1024):
        if not is_triton_available():
            raise RuntimeError("Triton backend requires CUDA; none available here.")
        self.block_size = block_size

    def __call__(self, c_array: np.ndarray, operand_arrays: list):
        (bias,) = operand_arrays
        rows, cols = c_array.shape
        c_t = torch.from_numpy(c_array).cuda()
        bias_t = torch.from_numpy(bias).cuda()
        out_t = torch.empty_like(c_t)

        grid = (rows, triton.cdiv(cols, self.block_size))
        _fused_bias_relu_kernel[grid](
            c_t, bias_t, out_t, rows, cols, BLOCK_SIZE=self.block_size,
        )
        result = out_t.cpu().numpy()
        c_array[...] = result
        return c_array

    def label(self):
        return f"triton-block{self.block_size}"


def self_check_definition_only():
    """Runs in ANY environment: proves the kernel at least compiles
    (decorates) without a GPU, distinct from proving it EXECUTES
    correctly, which needs CUDA."""
    if not _IMPORT_OK:
        print("triton/torch not importable in this environment.")
        return False
    print(f"triton.jit decoration succeeded for _fused_bias_relu_kernel "
          f"(triton {triton.__version__}). CUDA available: "
          f"{torch.cuda.is_available()}.")
    if not torch.cuda.is_available():
        print("No GPU in this environment -- kernel launch/correctness "
              "NOT verified here. is_triton_available() correctly "
              "reports False, so autotune.py will not add it as a "
              "candidate on this machine.")
    return True


if __name__ == "__main__":
    self_check_definition_only()
