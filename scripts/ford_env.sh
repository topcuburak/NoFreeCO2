# Source this before running anything on the ford testbed:  source scripts/ford_env.sh
#
# Why each line exists (learned the hard way on the CUDA-13 / vLLM stack):
#  - vLLM V1 uses torch.compile, which needs nvcc. The torch wheel ships only the
#    CUDA runtime, so we point CUDA_HOME at the pip CUDA wheel that carries nvcc.
#  - flashinfer's JIT sampler refuses to build when nvcc's minor version differs
#    from the CUDA headers (13.3 nvcc vs 13.0 headers) and there's no matching
#    13.0 nvcc wheel. So we disable the flashinfer sampler and use prebuilt
#    FlashAttention-2 instead -- nothing else needs to JIT-compile.

# locate the pip CUDA wheel's nvcc inside the active conda env
_NVCC=$(find "$CONDA_PREFIX" -name nvcc -path '*nvidia*' 2>/dev/null | head -1)
if [ -n "$_NVCC" ]; then
  export CUDA_HOME="$(dirname "$(dirname "$_NVCC")")"
  export PATH="$CUDA_HOME/bin:$PATH"
  hash -r
fi

# avoid the flashinfer JIT sampler; keep attention on prebuilt FA2
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

echo "[ford_env] CUDA_HOME=${CUDA_HOME:-<not found>}"
echo "[ford_env] VLLM_USE_FLASHINFER_SAMPLER=0  VLLM_ATTENTION_BACKEND=FLASH_ATTN"
command -v nvcc >/dev/null && nvcc --version | tail -1 || echo "[ford_env] WARNING: nvcc not on PATH (ok if you run with --enforce-eager)"
