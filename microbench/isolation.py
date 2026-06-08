"""Single-path isolation microbenchmarks.

Each exercises ONE leg of the dump path so its energy can be attributed and the
lumped GPU package power (flag 2) decomposed. Wrap each with measure_operation();
divide measured energy by bytes moved to get a measured pJ/bit, then validate
Σ(components) against an end-to-end dump (the C1 consistency check).

  hbm_copy   : in-GPU D2D copy, no PCIe        -> ε_HBM
  pcie_copy  : GPU HBM -> pinned host, NVMe idle -> ε_PCIe + ε_DRAM_write
  nvme_write : host DRAM -> NVMe, GPU idle      -> ε_NVMe_write

All return bytes moved. Need torch + CUDA (ford). Import is lazy so the file
loads off-testbed.
"""
from __future__ import annotations


def hbm_copy(nbytes: int, gpu: int = 0, iters: int = 50) -> int:
    import torch
    dev = torch.device(f"cuda:{gpu}")
    n = nbytes // 2  # fp16 elements
    src = torch.empty(n, dtype=torch.float16, device=dev)
    dst = torch.empty(n, dtype=torch.float16, device=dev)
    torch.cuda.synchronize(dev)
    for _ in range(iters):
        dst.copy_(src)            # D2D within HBM
    torch.cuda.synchronize(dev)
    return nbytes * iters


def pcie_copy(nbytes: int, gpu: int = 0, iters: int = 50) -> int:
    """HBM -> pinned host (D2H): the dump/suspend EXTRACT leg. Returns bytes moved."""
    import torch
    dev = torch.device(f"cuda:{gpu}")
    n = nbytes // 2
    src = torch.empty(n, dtype=torch.float16, device=dev)
    host = torch.empty(n, dtype=torch.float16, pin_memory=True)  # pinned host DRAM
    torch.cuda.synchronize(dev)
    for _ in range(iters):
        host.copy_(src)           # D2H over PCIe -> DRAM write
    torch.cuda.synchronize(dev)
    return nbytes * iters


def pcie_copy_h2d(nbytes: int, gpu: int = 0, iters: int = 50) -> int:
    """pinned host -> HBM (H2D): the RESTORE leg (reverse of pcie_copy). Bytes moved."""
    import torch
    dev = torch.device(f"cuda:{gpu}")
    n = nbytes // 2
    dst = torch.empty(n, dtype=torch.float16, device=dev)
    host = torch.empty(n, dtype=torch.float16, pin_memory=True)
    torch.cuda.synchronize(dev)
    for _ in range(iters):
        dst.copy_(host)           # H2D over PCIe -> HBM write
    torch.cuda.synchronize(dev)
    return nbytes * iters


def pcie_copy_rate(nbytes: int, gpu: int = 0, direction: str = "d2h",
                   target_gbps: float | None = None, chunk_mb: int = 128) -> int:
    """Move nbytes over PCIe at a CONTROLLED effective bandwidth, by transferring in
    chunks and pacing (sleep) to hit target_gbps. target_gbps=None/0 -> unthrottled
    (the hardware ceiling). One chunk buffer is reused, so the alloc is tiny regardless
    of nbytes -- we measure the cost of moving nbytes of PCIe traffic, not of holding it.

    direction: 'd2h' = HBM->host (suspend/extract leg); 'h2d' = host->HBM (restore leg).
    Used by characterize_bw_sweep.py to fit E = e_byte*S + P_hold*t (slope vs latency)."""
    import time
    import torch
    dev = torch.device(f"cuda:{gpu}")
    chunk = (int(chunk_mb) << 20)
    chunk -= chunk % 2                                   # fp16-align
    n_elem = chunk // 2
    nchunks = max(1, nbytes // chunk)
    host = torch.empty(n_elem, dtype=torch.float16, pin_memory=True)
    if direction == "h2d":
        dst = torch.empty(n_elem, dtype=torch.float16, device=dev)
        do = lambda: dst.copy_(host)                     # H2D
    else:
        src = torch.empty(n_elem, dtype=torch.float16, device=dev)
        do = lambda: host.copy_(src)                     # D2H
    torch.cuda.synchronize(dev)
    bps = (target_gbps * 1e9) if target_gbps else None
    t0 = time.perf_counter()
    moved = 0
    for _ in range(nchunks):
        do()
        torch.cuda.synchronize(dev)                      # per-chunk so pacing is accurate
        moved += chunk
        if bps:                                          # pace to target: sleep if ahead
            ahead = moved / bps - (time.perf_counter() - t0)
            if ahead > 0:
                time.sleep(ahead)
    return moved


def dd_write(path: str, nbytes: int, bs: int = 1 << 26) -> int:
    """O_DIRECT durable write (host->NVMe). Returns bytes written. Device-rate proxy
    for the checkpoint STORE leg (writes zeros; same write-side I/O cost)."""
    import subprocess
    count = nbytes // bs
    subprocess.run(["dd", "if=/dev/zero", f"of={path}", f"bs={bs}", f"count={count}",
                    "oflag=direct", "conv=fdatasync"], check=True, capture_output=True)
    return count * bs


def dd_read(path: str, nbytes: int, bs: int = 1 << 26) -> int:
    """O_DIRECT read (NVMe->host). Returns bytes read. Device-rate proxy for the
    checkpoint LOAD leg (drop caches first for a cold/device read)."""
    import subprocess
    count = nbytes // bs
    subprocess.run(["dd", f"if={path}", "of=/dev/null", f"bs={bs}", f"count={count}",
                    "iflag=direct"], check=True, capture_output=True)
    return count * bs


def nvme_read(path: str, nbytes: int, iters: int = 1) -> int:
    """host <- NVMe: read nbytes from an existing file. For a true device read,
    drop page cache first (the characterize script does `echo 3 > drop_caches`).
    Returns bytes read."""
    import os
    bufsize = 1 << 26  # 64 MB
    total = 0
    fd = os.open(path, os.O_RDONLY)
    try:
        for _ in range(iters):
            os.lseek(fd, 0, os.SEEK_SET)
            remaining = nbytes
            while remaining > 0:
                chunk = os.read(fd, min(bufsize, remaining))
                if not chunk:
                    break
                total += len(chunk)
                remaining -= len(chunk)
    finally:
        os.close(fd)
    return total


def nvme_write(nbytes: int, path: str, iters: int = 10) -> int:
    """host DRAM -> NVMe. O_DIRECT-ish: flush + fsync to defeat page cache."""
    import os
    buf = bytearray(os.urandom(min(nbytes, 1 << 26)))  # <=64MB pattern, repeated
    mv = memoryview(buf)                                # avoid per-chunk slice copies
    written = 0
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        for _ in range(iters):
            remaining = nbytes
            os.lseek(fd, 0, os.SEEK_SET)
            while remaining > 0:
                n = min(len(buf), remaining)
                os.write(fd, mv[:n])
                remaining -= n
                written += n
            os.fsync(fd)
    finally:
        os.close(fd)
    return written
