# socc-bench

Measurement harness for the SoCC 2026 short paper on **mechanism cost in
carbon-aware spatiotemporal scheduling**. Measures the energy + latency of
checkpoint/dump/restore/migrate, decomposed per hardware component across two
domains (**host** and **accelerator**).

Developed here; **measurements run on `ford`** (4× A100 SXM4 40 GB, AMD EPYC
75F3, PM1733 RAID-0 NVMe). The harness imports and smoke-tests off-testbed —
telemetry sources self-detect and degrade gracefully when absent.

## Layout

```
harness/                 reusable, operation-agnostic measurement core
  schema.py              RunRecord / SourceResult (one JSON line per operation)
  telemetry.py           CLOCK_MONOTONIC sampler thread + power-trace integration
  measure.py             measure_operation(): window-based differential protocol
  configload.py          yaml loader
  sources/               self-detecting telemetry sources
    nvml_source.py        GPU package power (HBM+SM+PCIe, lumped)   [power]
    perf_source.py        AMD pkg energy via amd_energy hwmon       [energy counter]
    ipmi_source.py        whole-node power via IPMI DCMI            [power]
    nvme_smart.py         bytes written (cross-check)               [byte counter]
config/
  ford.yaml              testbed + telemetry config
  coefficients.yaml      locked per-byte energy coefficients (pJ/bit)
microbench/isolation.py  single-path benches: hbm_copy / pcie_copy / nvme_write
workloads/a2_vllm/       A2 Llama-3-8B serving: serve.py + dump_restore.py
scripts/
  smoke.py               off-testbed wiring check (run this first, here)
  run_microbench.py      derive measured ε_HBM / ε_PCIe / ε_NVMe (run first on ford)
  run_a2_slice.py        the A2 vertical slice: steady-state -> dump -> restore
data/                    per-run JSONL records
```

## Two-domain decomposition

Energy is attributed top-down. **Host** workloads (A6, CPU/CRIU): cores, memory,
storage. **Accelerator** workloads (A1–A5, GPU): accel cores, accel memory, host
memory, storage (+NIC for migrate); dump path `HBM → PCIe → host DRAM → NVMe`.

Three things are **modeled, not measured** on this hardware (state in §4):
1. **Host DRAM** — no DRAM RAPL on EPYC → ~10 pJ/bit.
2. **SM vs HBM** — NVML package is monolithic → attribute HBM = `S·3.5 pJ/bit`.
3. **NVMe power** — back out from IPMI − CPU package; cross-check SMART bytes.

`run_microbench.py` produces the measured coefficients that anchor (2), and the
`Σ(components)` vs end-to-end-dump check validates the whole decomposition.

## Build order (vertical slice first)

1. **`scripts/smoke.py`** — here, off-testbed. Confirms wiring.
2. **`scripts/run_microbench.py`** — first on ford. Proves the full loop +
   gives measured ε coefficients. No model weights needed.
3. **`scripts/run_a2_slice.py`** — A2 steady-state → dump → restore. Go/no-go.
4. Fan out: remaining workloads (A1, A3, A4, A5, A6) + migrate, all via the same
   `measure_operation()`.

## Setup (on the server)

```bash
conda env create -f environment.yml      # or: pip install -r requirements.txt
conda activate socc-bench
export HF_TOKEN=...                       # gated Llama-3.x weights/tokenizers
# system packages for telemetry / A6: ipmitool, nvme-cli, criu
```

## Run

```bash
python scripts/smoke.py                   # wiring check (no GPU needed)

# A2 serving (the completed workload):
python workloads/a2_vllm/prep_longbench_v2.py \
    --tokenizer meta-llama/Llama-3.1-8B --min-tokens 70000 --max-tokens 90000 \
    --num-samples 32 --shuffle --output data/lbv2_70k_90k.jsonl
python workloads/a2_vllm/serve.py \
    --model meta-llama/Llama-3.1-8B --tensor-parallel-size 4 --max-model-len 98304 \
    --dataset data/lbv2_70k_90k.jsonl --prompt-field prompt --num-prompts 32 --max-tokens 256

# measurement loop (run microbench first):
python scripts/run_microbench.py --bytes 8e9
```

> Status: A2 serving + LongBench-v2 prep are complete. The dump/restore mechanism
> (`workloads/a2_vllm/dump_restore.py`) and the per-workload runners (A1, A3–A6)
> are scaffolds with `TODO(ford)` markers. The measurement loop will be driven by a
> generic `run_op.py` + per-workload adapter registry (planned).

## Measurement protocol (per operation)

Baseline 30 s under the same workload state → read counters at T_start → run op →
read counters at T_end. PowerSource: `∫(P − P_baseline) dt`. EnergyCounter /
ByteCounter: end − start. Pin GPU clock (`nvidia-smi -lgc`) and NUMA
(`numactl --membind`) before measuring; align all sources on CLOCK_MONOTONIC.
```
```
