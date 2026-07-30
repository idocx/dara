"""Portable CPU core-count detection for Ray + BGMN concurrency sizing.

Priority, most authoritative first:

1. An explicit SLURM allocation (``SLURM_CPUS_PER_TASK``, then
   ``SLURM_CPUS_ON_NODE``) -- trusts a scheduler-assigned budget over
   anything detected locally.
2. OS-level CPU affinity (``os.sched_getaffinity``, Linux only), narrowed by
   a cgroup CPU quota if one is set and implies fewer cores than affinity --
   this matters on shared workstations, containers, and HPC nodes.
3. ``os.cpu_count()`` -- universal fallback, also the only option on macOS,
   where ``sched_getaffinity`` does not exist.
"""

from __future__ import annotations

import math
import os
from pathlib import Path


def _slurm_cpu_count() -> int | None:
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        value = os.environ.get(var)
        if not value:
            continue
        try:
            n = int(value)
        except ValueError:
            continue
        if n > 0:
            return n
    return None


def _cgroup_quota_cpu_count() -> int | None:
    """Return the core count implied by a cgroup CPU quota, if one is set."""
    cgroup_v2_path = Path("/sys/fs/cgroup/cpu.max")
    if cgroup_v2_path.exists():
        try:
            quota_str, period_str = cgroup_v2_path.read_text().split()
        except (OSError, ValueError):
            quota_str = period_str = None
        if quota_str is not None and quota_str != "max":
            try:
                quota, period = int(quota_str), int(period_str)
                if quota > 0 and period > 0:
                    return max(1, math.floor(quota / period))
            except ValueError:
                pass

    quota_path = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    period_path = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if quota_path.exists() and period_path.exists():
        try:
            quota = int(quota_path.read_text().strip())
            period = int(period_path.read_text().strip())
            if quota > 0 and period > 0:
                return max(1, math.floor(quota / period))
        except (OSError, ValueError):
            pass

    return None


def _affinity_cpu_count() -> int | None:
    sched_getaffinity = getattr(os, "sched_getaffinity", None)
    if sched_getaffinity is None:
        return None
    try:
        affinity = len(sched_getaffinity(0))
    except OSError:
        return None
    return affinity if affinity > 0 else None


def detect_available_cores() -> int:
    """
    Detect the number of CPU cores actually available to this process.

    Returns
    -------
        the detected core count, always >= 1. Never returns None or 0.
    """
    slurm = _slurm_cpu_count()
    if slurm is not None:
        return slurm

    affinity = _affinity_cpu_count()
    if affinity is not None:
        cgroup = _cgroup_quota_cpu_count()
        if cgroup is not None:
            return max(1, min(affinity, cgroup))
        return affinity

    return max(1, os.cpu_count() or 1)
