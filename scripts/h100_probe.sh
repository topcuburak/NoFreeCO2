#!/usr/bin/env bash
# h100_probe.sh -- 5-minute GO/NO-GO check for a rented GPU box.
# Run this FIRST, on a cheap ~1-hour rental. If a BLOCKER fails, RELEASE THE BOX
# (do not burn billed hours debugging). See results/h100_runbook.md.
#
#   sudo bash scripts/h100_probe.sh [scratch_dir]     # scratch_dir default /tmp
#
# BLOCKERs (comparability with ford; fail -> the cross-gen trend cannot be claimed):
#   root, driver >= 550 (cuda-checkpoint), clock pin allowed (-lgc), RAPL readable, NVMe scratch.
# WARNs: virtualization (NVML attribution), low host RAM (caps sweep sizes).
# OPTIONAL: GDS/cuFile presence (direct-leg bench) -- absence does NOT kill the rental.
set -u
SCRATCH="${1:-/tmp}"
PASS=0; FAIL=0; WARN=0
ok()   { echo "  [PASS] $1"; PASS=$((PASS+1)); }
bad()  { echo "  [FAIL] $1  <-- BLOCKER"; FAIL=$((FAIL+1)); }
warn() { echo "  [WARN] $1"; WARN=$((WARN+1)); }

echo "== h100_probe: $(hostname) $(date -u +%FT%TZ) =="

# 1. root
[ "$(id -u)" = "0" ] && ok "root" || bad "not root (RAPL + cuda-checkpoint need it)"

# 2. GPUs + driver version
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader | sed 's/^/       /'
  DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
  [ "${DRV:-0}" -ge 550 ] && ok "driver ${DRV} >= 550 (cuda-checkpoint ok)" || bad "driver ${DRV} < 550"
  NGPU=$(nvidia-smi -L | wc -l); [ "$NGPU" -ge 2 ] && ok "$NGPU GPUs" || warn "only $NGPU GPU (nGPU axis needs 2)"
else
  bad "nvidia-smi missing"
fi

# 3. driver flavor (GDS only: open kernel module does not export nvidia_p2p_* -> nvidia_fs fails, as on ford)
LIC=$(modinfo -F license nvidia 2>/dev/null || echo "?")
case "$LIC" in
  *MIT*|*GPL*) warn "OPEN kernel module ($LIC): GDS/nvidia_fs will likely FAIL (ford failure mode); staged legs unaffected";;
  NVIDIA*)     ok  "proprietary kernel module (GDS possible)";;
  *)           warn "driver flavor unknown ($LIC)";;
esac

# 4. clock pin allowed? (set max SM clock, then reset)
MAXC=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits 2>/dev/null | head -1)
if [ -n "${MAXC:-}" ] && nvidia-smi -i 0 -lgc "$MAXC" >/dev/null 2>&1; then
  nvidia-smi -i 0 -rgc >/dev/null 2>&1
  ok "clock pin allowed (-lgc $MAXC)"
else
  bad "clock pin denied (-lgc): numbers not comparable to ford"
fi

# 5. RAPL readable
R=$(cat /sys/class/powercap/intel-rapl:*/energy_uj 2>/dev/null | head -1)
[ -n "${R:-}" ] && ok "RAPL energy_uj readable" || bad "RAPL not readable (no CPU energy)"

# 6. CPU vendor + host RAM (cuda-checkpoint stages HBM in DRAM: need RAM >= 1.2x max total footprint)
VEND=$(lscpu | awk -F: '/Vendor ID/{gsub(/ /,"",$2);print $2}')
echo "       CPU vendor: ${VEND:-?} (Intel => DRAM RAPL domain exists; keep DRAM MODELED anyway for like-for-like)"
RAMG=$(awk '/MemTotal/{printf "%d", $2/1024/1024}' /proc/meminfo)
if [ "${RAMG:-0}" -ge 170 ]; then ok "host RAM ${RAMG} GB (2x64GiB sweep fits)"
elif [ "${RAMG:-0}" -ge 90 ]; then warn "host RAM ${RAMG} GB: cap sweep at ~1x64 + 2x32 GiB"
else warn "host RAM ${RAMG} GB LOW: cap per-GPU sizes accordingly"; fi

# 7. NVMe scratch + quick O_DIRECT write speed
lsblk -o NAME,TRAN,SIZE,MOUNTPOINT 2>/dev/null | grep -i nvme | sed 's/^/       /'
if lsblk -o TRAN 2>/dev/null | grep -qi nvme; then
  ok "NVMe present"
  AV=$(df -BG --output=avail "$SCRATCH" 2>/dev/null | tail -1 | tr -dc 0-9)
  [ "${AV:-0}" -ge 200 ] && ok "scratch $SCRATCH: ${AV} GB free" || warn "scratch $SCRATCH only ${AV:-?} GB free"
  if dd if=/dev/zero of="$SCRATCH/.probe_dd" bs=64M count=16 oflag=direct 2>/tmp/dd.err; then
    grep -o '[0-9.]* [GM]B/s' /tmp/dd.err | tail -1 | sed 's/^/       seq write: /'
    rm -f "$SCRATCH/.probe_dd"
  fi
else
  bad "no NVMe device (dump target)"
fi

# 8. bare metal?
V=$(systemd-detect-virt 2>/dev/null || echo "?")
[ "$V" = "none" ] && ok "bare metal" || warn "virtualization: $V (NVML power attribution may be distorted)"

# 9. GDS (optional)
GDSCHK=$(ls /usr/local/cuda*/gds/tools/gdscheck 2>/dev/null | head -1)
if [ -n "$GDSCHK" ]; then ok "gdscheck present: $GDSCHK (run '$GDSCHK -p' after setup)"
elif ldconfig -p 2>/dev/null | grep -q libcufile; then warn "libcufile present, gdscheck missing (GDS maybe installable)"
else warn "no GDS stack (direct-leg bench unavailable; staged campaign unaffected)"; fi
lsmod | grep -q nvidia_fs && ok "nvidia_fs loaded" || warn "nvidia_fs not loaded (needed for GDS only)"

# 10. cuda-checkpoint + python
command -v cuda-checkpoint >/dev/null && ok "cuda-checkpoint in PATH" \
  || warn "cuda-checkpoint missing (runbook installs it: github.com/NVIDIA/cuda-checkpoint prebuilt)"
command -v python3 >/dev/null && ok "python3 $(python3 -V 2>&1 | cut -d' ' -f2)" || bad "python3 missing"

echo
echo "== VERDICT: $PASS pass, $WARN warn, $FAIL BLOCKER(s) =="
if [ "$FAIL" -gt 0 ]; then echo ">> RELEASE THE BOX (blockers above make it non-comparable to ford)."; exit 1
else echo ">> GO. Proceed with results/h100_runbook.md Phase 1."; fi
