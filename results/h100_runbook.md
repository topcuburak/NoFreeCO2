# H100 rental runbook — paste-and-run, zero on-box thinking

Every billed minute counts. All phases below are PASTE blocks. Decomposition and model
fitting happen LOCALLY afterwards (decompose_mech_energy.py / fit_n1_model.py with the
filled `config/h100.yaml`); the box only produces raw jsonl + facts + logs.

**SKU guidance:** NVLink is NOT needed (independent hold_gpu processes, no NCCL). Pick by
HBM + blockers: H100 NVL 94 GB (more footprint range) or H100 SXM 80 GB both fine. H200
(141 GB HBM3e) = optional third generation point for the trend if budget allows. Record
the exact SKU; its datasheet power limits go into h100.yaml.

**Budget model (2x GPUs billed per wall hour):** probe ~1 h, core ~10-12 h wall
(~20-25 GPU-h), +GDS ~4-6 h wall. Do NOT leave the box idle; release immediately after
the collection scp.

---

## Phase 0 — PROBE (separate cheap ~1 h rental; kill on FAIL)

```bash
git clone --depth 1 https://github.com/topcuburak/NoFreeCO2.git && cd NoFreeCO2
sudo bash scripts/h100_probe.sh /scratch     # <- replace /scratch with the NVMe mount
```
- Verdict `GO` -> commit to the real session (same provider/instance type).
- Any BLOCKER (no root, driver <550, -lgc denied, no RAPL, no NVMe) -> **release the box**,
  try another provider. Open-driver WARN only kills the GDS branch, not the campaign.

## Phase 1 — SETUP (~30-45 min; start downloads FIRST, they run while you set up)

```bash
git clone --depth 1 https://github.com/topcuburak/NoFreeCO2.git && cd NoFreeCO2
python3 -m venv ~/venv && source ~/venv/bin/activate
pip install torch nvidia-ml-py pyyaml numpy psutil &        # bg; vllm NOT needed for core

# cuda-checkpoint (prebuilt, driver R550+)
git clone --depth 1 https://github.com/NVIDIA/cuda-checkpoint.git ~/cc
sudo cp ~/cc/bin/x86_64_Linux/cuda-checkpoint /usr/local/bin/ && cuda-checkpoint --help | head -3

# pin clocks on ALL GPUs (comparability with ford; probe verified this works)
MAXC=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits | head -1)
sudo nvidia-smi -lgc "$MAXC"
mkdir -p ~/logs
wait   # pip done?
python -c "import torch; print(torch.cuda.device_count(), 'GPUs, torch OK')"
```

## Phase 2 — FACTS capture (~10 min; fills h100.yaml LOCALLY later)

```bash
sudo bash scripts/h100_fill_config.sh /scratch     # -> ~/h100_facts.txt
nvidia-smi -q > ~/h100_smi_q.txt                   # full snapshot (power limits, SKU)
```

## Phase 3 — CORE SWEEPS (nohup chain, ~3-4 h wall; NVMe scratch path!)

Sizes assume >=170 GB host RAM (probe told you). If RAM-capped, drop the 64s.

```bash
source ~/venv/bin/activate
PY=$(which python); sudo -v
nohup sudo -E bash -c "
  # 1-GPU footprint sweep (trend core)
  $PY scripts/sweep_multigpu_suspend.py --gpu-counts 1 --sizes 4,8,16,32,64 --cycles 4 \
    --store-out /scratch --tag h100_mg_nvme &&
  # 2-GPU (nGPU slope: verifies p_board transfers; 1-4 linearity already shown on A100)
  $PY scripts/sweep_multigpu_suspend.py --gpu-counts 2 --sizes 8,16,32,64 --cycles 4 \
    --store-out /scratch --tag h100_mg_nvme &&
  # hold power (park vs suspended, 60 s steady baselines)
  $PY scripts/sweep_multigpu_suspend.py --gpu-counts 1,2 --sizes 8,32 --cycles 2 \
    --baseline 60 --store-out /scratch --tag h100_hold_power
" > ~/logs/h100_core.log 2>&1 &
echo "PID: $!"; tail -f ~/logs/h100_core.log     # first FOOTPRINT line sanity-checks context overhead
```

## Phase 4 — GDS branch (ONLY if probe said gdscheck present; ~1-2 h)

```bash
/usr/local/cuda/gds/tools/gdscheck -p | head -30          # supported FS on the scratch?
nvcc -O2 -o ~/gds_bench scripts/gds_direct_bench.cpp -lcufile
for GB in 4 8 16 32; do
  sudo ~/gds_bench --gb $GB --file /scratch/gds.bin --mode both | tee -a ~/logs/gds_bench.log
done
# RESULT,mode,dir,GB,seconds,GBps,cpu_J  -> direct-vs-staged measured, incl. CPU energy
```

## Phase 4.5 — DRAM-model validation byproduct (Intel host only, ~10 min, FREE win)

ford (AMD EPYC) has no DRAM RAPL domain, so DRAM is MODELED at 0.03-0.08 W/GB. If this box
is Intel (h100_facts.txt lists a `dram` domain), MEASURE it: hold N GiB resident and read
the dram-domain counter. Validates the band -> one MODELED label becomes measured.

```bash
DRAM_DOM=$(grep -l dram /sys/class/powercap/intel-rapl:*:*/name 2>/dev/null | head -1 | xargs dirname)
if [ -n "$DRAM_DOM" ]; then
  for GB in 16 64 128; do
    python scripts/work_dram_mp.py --gb $GB --procs 1 --threads 1 --seconds 70 & W=$!
    sleep 5; E0=$(cat $DRAM_DOM/energy_uj); sleep 60; E1=$(cat $DRAM_DOM/energy_uj)
    echo "DRAM ${GB}GiB: $(( (E1-E0)/60 )) uW avg" | tee -a ~/logs/dram_validate.log
    wait $W
  done
else echo "no dram RAPL domain on this box"; fi
```

## Phase 5 — optional workload anchor (~1-2 h; skip if budget tight)

One real GPU workload validates the model on H100 beyond synthetics. Cheapest: ViT-style
single-GPU hold via the existing harness, or skip entirely — synthetic sweeps carry the trend.

## Phase 6 — COLLECT + RELEASE (never skip; box off right after scp)

```bash
grep -hE "h100_mg_nvme|h100_hold_power" data/timed_dump.jsonl > ~/h100_records.jsonl
wc -l ~/h100_records.jsonl
tar czf ~/h100_bundle.tar.gz -C ~ h100_records.jsonl h100_facts.txt h100_smi_q.txt logs
# from the LOCAL machine:
#   scp <box>:~/h100_bundle.tar.gz ~/Desktop/socc/socc-bench/data/incoming/
# verify the tar landed intact locally, THEN release the box.
sudo nvidia-smi -rgc     # unpin clocks (courtesy)
```

## Afterwards (local, no billing)

1. Fill `config/h100.yaml` from `h100_facts.txt` + SKU datasheet (power_model, clocks, storage).
2. `decompose_mech_energy.py --records data/incoming/h100_records.jsonl --config h100.yaml`.
3. Transfer validation: predict H100 legs from the A100-fit structure + H100 datasheet
   params; report error (the cross-generation claim). p_board slope from the 1->2 step.
4. GDS: RESULT lines vs the staged decomposition's ~53% projected dump saving.
