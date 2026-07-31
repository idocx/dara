"""Runtime CPU-core detection, used to size worker/thread counts portably."""

from __future__ import annotations

import os


def detect_cpu_count() -> int:
    """Detect the number of CPU cores usable by this process.

    Checked in priority order:
        1. SLURM_CPUS_PER_TASK, then SLURM_CPUS_ON_NODE (SLURM-managed jobs).
        2. ``os.sched_getaffinity(0)`` (Linux only; reflects taskset/affinity
           restrictions that ``os.cpu_count()`` ignores).
        3. ``os.cpu_count()``.

    Returns
    -------
        A positive integer. Falls back to 1 if nothing above yields a usable count.
    """
    for env_var in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        value = os.environ.get(env_var)
        if value:
            try:
                count = int(value)
            except ValueError:
                count = 0
            if count > 0:
                return count

    if hasattr(os, "sched_getaffinity"):
        count = len(os.sched_getaffinity(0))
        if count > 0:
            return count

    count = os.cpu_count()
    if count:
        return count

    return 1
