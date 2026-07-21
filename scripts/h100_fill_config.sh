#!/usr/bin/env bash
# h100_fill_config.sh -- capture EVERY fact needed to fill config/h100.yaml (power_model,
# cpu/gpu topology, storage) into ONE file. Do NOT edit yaml on the billed box: run this,
# include h100_facts.txt in the collection tar, fill the yaml locally afterwards.
#
#   sudo bash scripts/h100_fill_config.sh [scratch_dir]      # default /tmp
set -u
SCRATCH="${1:-/tmp}"
OUT=~/h100_facts.txt
exec > >(tee "$OUT") 2>&1

echo "== h100_facts: $(hostname) $(date -u +%FT%TZ) =="

echo; echo "## GPU (model, HBM, driver, max clocks -> gpus.* + telemetry.pin_gpu_clock_mhz)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version,clocks.max.sm,power.limit --format=csv
echo "-- link (PCIe gen/width per GPU) --"
nvidia-smi --query-gpu=index,pcie.link.gen.current,pcie.link.width.current --format=csv
echo "-- topology (gpu_numa_map) --"
nvidia-smi topo -m 2>/dev/null | head -15

echo; echo "## CPU (cpu.* + RAPL domains)"
lscpu | grep -E "Model name|Vendor|Socket|Core|Thread|NUMA"
echo "-- RAPL domains (dram domain present? -> telemetry.dram_rapl) --"
for d in /sys/class/powercap/intel-rapl:*; do
  [ -e "$d/name" ] && echo "  $d : $(cat $d/name) max=$(cat $d/max_energy_range_uj 2>/dev/null)uJ"
done

echo; echo "## DRAM (DIMM inventory -> power_model.dram.w_per_gb from datasheet)"
dmidecode -t memory 2>/dev/null | grep -E "Size:|Type: DDR|Speed:|Manufacturer:|Part Number:" \
  | grep -v "No Module" | sort | uniq -c
awk '/MemTotal/{printf "  MemTotal: %.0f GB\n", $2/1024/1024}' /proc/meminfo

echo; echo "## NVMe (power states -> power_model.drive; find the datasheet for active R/W)"
for dev in /dev/nvme?n1; do
  [ -e "$dev" ] || continue
  echo "-- $dev --"
  nvme id-ctrl "$dev" 2>/dev/null | grep -iE "^mn |^ps [0-9]" | head -8
  b=$(basename "$dev" | sed 's/n1//'); cat /sys/class/nvme/$b/device/current_link_speed 2>/dev/null
done
lsblk -o NAME,TRAN,SIZE,MODEL,MOUNTPOINT | grep -vi loop

echo; echo "## Storage seq bench (O_DIRECT, 4 GiB -> storage.seq_write/read_gbps)"
F="$SCRATCH/.fillcfg_dd"
dd if=/dev/zero of="$F" bs=64M count=64 oflag=direct 2>&1 | tail -1
sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null
dd if="$F" of=/dev/null bs=64M iflag=direct 2>&1 | tail -1
rm -f "$F"

echo; echo "## GDS stack"
ls /usr/local/cuda*/gds/tools/gdscheck 2>/dev/null || echo "  no gdscheck"
ldconfig -p 2>/dev/null | grep cufile || echo "  no libcufile"
modinfo -F license nvidia 2>/dev/null | sed 's/^/  nvidia module license: /'

echo; echo "== done -> $OUT (include this file in the collection tar) =="
