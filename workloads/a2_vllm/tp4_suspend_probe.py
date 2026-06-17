#!/usr/bin/env python3
"""A2 TP=4 feasibility probe: can real vLLM TP=4 survive destroy_process_group -> reinit?

Mirrors A1's FSDP suspend path (destroy NCCL -> [cuda-checkpoint] -> reinit fresh rendezvous
-> rebind the cached TP group), injected into vLLM's 4 worker processes via collective_rpc.

Stages (each guarded so we LEARN the API from the output even if a step fails):
  1. serve TP=4 + warmup generate
  2. collective_rpc(worker_info)    -> pid/rank per worker + the parallel_state API surface
  3. collective_rpc(worker_destroy) -> quiesce + destroy model-parallel + world group
  4. [--hold only] write /tmp/vllm_destroyed.{rank}, print PIDs, wait /tmp/vllm_resume
       (so an external root driver can cuda-checkpoint the 4 PIDs in between)
  5. collective_rpc(worker_reinit)  -> init fresh rendezvous + initialize_model_parallel
  6. generate again -> validates serving resumed (the rebind took)

Default (no --hold) does 3->5 back-to-back: isolates whether the vLLM PG teardown/rebuild
works, no cuda-checkpoint (= A1's first test). Add --hold for the full transparent suspend.

    python workloads/a2_vllm/tp4_suspend_probe.py --model meta-llama/Llama-3.1-8B [--hold]

Needs 4 free GPUs + HF_TOKEN. TP=4 uses multiprocessing workers (separate PIDs) by default --
do NOT set VLLM_ENABLE_V1_MULTIPROCESSING=0.
"""
from __future__ import annotations

import argparse
import os
import time

# Same vLLM workarounds the TP=1 sweep uses: the default FlashInfer sampler JIT-compiles a
# CUDA kernel at runtime (needs nvcc, absent here) -> disable it; pin the attention backend.
# Set before importing vLLM. (Do NOT disable V1 multiprocessing -- TP=4 needs the multiproc
# executor so the 4 workers are separate, checkpointable PIDs.)
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
# collective_rpc ships a Python callable to the workers; the default msgpack serializer rejects
# functions, so allow the pickle/cloudpickle fallback (single-node, our own code -> safe).
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")


# ---- functions shipped to each worker via collective_rpc (worker passed as 1st arg) ----
def worker_info(worker):
    import os as _os
    info = {"pid": _os.getpid()}
    for a in ("rank", "local_rank"):
        info[a] = getattr(worker, a, None)
    try:
        import torch.distributed as dist
        info["dist_inited"] = dist.is_initialized()
        if dist.is_initialized():
            info["world"] = dist.get_world_size()
    except Exception as e:
        info["dist_err"] = repr(e)
    try:
        from vllm.distributed import parallel_state as ps
        info["ps_api"] = sorted(f for f in dir(ps)
                                if any(k in f for k in ("destroy", "init", "get_tp", "get_tensor")))
    except Exception as e:
        info["ps_err"] = repr(e)
    return info


def worker_destroy(worker):
    import os as _os
    out = []
    try:
        import torch
        torch.cuda.synchronize()
    except Exception as e:
        out.append(f"sync: {e!r}")
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.barrier()
            out.append("barrier ok")
    except Exception as e:
        out.append(f"barrier: {e!r}")
    try:
        from vllm.distributed import parallel_state as ps
        ps.destroy_model_parallel()
        out.append("destroy_model_parallel ok")
    except Exception as e:
        out.append(f"destroy_model_parallel: {e!r}")
    try:
        from vllm.distributed import parallel_state as ps
        ps.destroy_distributed_environment()
        out.append("destroy_distributed_environment ok")
    except Exception as e:
        out.append(f"destroy_distributed_environment: {e!r}")
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()
                out.append("torch.destroy_process_group ok")
        except Exception as e2:
            out.append(f"torch.destroy: {e2!r}")
    try:
        import torch.distributed as dist
        out.append(f"dist_inited_after={dist.is_initialized()}")
    except Exception:
        pass
    return {"pid": _os.getpid(), "rank": getattr(worker, "rank", None), "steps": out}


def worker_reinit(worker, port, world):
    out = []
    rank = getattr(worker, "rank", 0)
    local = getattr(worker, "local_rank", rank)
    try:
        from vllm.distributed import parallel_state as ps
        ps.init_distributed_environment(
            world_size=world, rank=rank,
            distributed_init_method=f"tcp://127.0.0.1:{port}", local_rank=local)
        out.append("init_distributed_environment ok")
        ps.initialize_model_parallel(tensor_model_parallel_size=world)
        out.append("initialize_model_parallel ok")
    except Exception as e:
        out.append(f"reinit FAILED: {type(e).__name__}: {e}")
    return {"rank": rank, "steps": out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--hold", action="store_true",
                    help="wait for an external cuda-checkpoint between destroy and reinit")
    ap.add_argument("--reinit-port", type=int, default=29600)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, tensor_parallel_size=args.tp, enforce_eager=True)
    sp = SamplingParams(max_tokens=8, temperature=0.0)
    prompt = "The capital of France is"

    out = llm.generate([prompt], sp)
    print(f"[probe] WARMUP OK: {out[0].outputs[0].text!r}", flush=True)

    print(f"[probe] === worker info ===", flush=True)
    for r in llm.collective_rpc(worker_info):
        print(f"   {r}", flush=True)

    print(f"[probe] === destroy (quiesce + destroy NCCL) ===", flush=True)
    d = llm.collective_rpc(worker_destroy)
    for r in d:
        print(f"   {r}", flush=True)
    pids = [r.get("pid") for r in d]

    if args.hold:
        for r in range(args.tp):
            open(f"/tmp/vllm_destroyed.{r}", "w").close()
        try: os.remove("/tmp/vllm_resume")
        except OSError: pass
        print(f"[probe] HELD -- cuda-checkpoint these PIDs then `touch /tmp/vllm_resume`: {pids}",
              flush=True)
        while not os.path.exists("/tmp/vllm_resume"):
            time.sleep(1)
        print("[probe] resume flag seen", flush=True)

    print(f"[probe] === reinit + rebind (fresh rendezvous port {args.reinit_port}) ===", flush=True)
    r = llm.collective_rpc(worker_reinit, args=(args.reinit_port, args.tp))
    for x in r:
        print(f"   {x}", flush=True)

    print(f"[probe] === validate: generate after reinit ===", flush=True)
    try:
        out = llm.generate([prompt], sp)
        print(f"[probe] RESUME OK: {out[0].outputs[0].text!r}", flush=True)
        print("[probe] SUCCESS -- vLLM TP=4 survived destroy -> reinit -> generate", flush=True)
    except Exception as e:
        print(f"[probe] RESUME FAILED: {type(e).__name__}: {e}", flush=True)
        print("[probe] -> the PG rebind didn't take; need to also rebind cached TP refs "
              "(vLLM analogue of rebind_fsdp_pg)", flush=True)


if __name__ == "__main__":
    main()
