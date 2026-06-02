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
