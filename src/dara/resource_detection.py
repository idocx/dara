"""Portable CPU/thread-budget detection for Ray + BGMN.

BGMN's own internal thread scaling saturates early (measured: a 2-phase
refinement sped up 1.36x at 2 threads vs. 1.71x at 10 threads, on a 12-core
machine), while Ray's default `num_cpus=1` per refinement task lets as many
concurrent tasks run as there are detected CPUs -- decoupled from the actual
`n_threads` a task's BGMN subprocess uses. Left at defaults (e.g.
`n_threads=10` with 12 concurrent tasks), this oversubscribes the machine by
up to ~10x (120 OS threads on 12 physical cores), which can starve other
processes -- including Ray's own GCS -- of scheduling time under sustained
load.

This module derives a `(n_threads, concurrency)` pair such that
`concurrency * n_threads` stays close to the actually-available core count,
on whatever machine dara is running on -- a laptop, a shared workstation, or
an HPC compute node -- rather than assuming a fixed core count.
"""

import os


def detect_available_cores() -> int:
    """
    Detect the number of CPUs actually available to this process.

    Priority, most authoritative first:
    1. An explicit SLURM allocation (`SLURM_CPUS_PER_TASK`), if present --
       trusts a scheduler-assigned budget over anything detected locally.
    2. OS-level CPU affinity (`os.sched_getaffinity`, Linux only) -- respects
       cgroup/container/taskset restrictions that `os.cpu_count()` ignores,
       which matters on shared workstations and HPC nodes.
    3. `os.cpu_count()` -- universal fallback (also the only option on
       macOS, where `sched_getaffinity` doesn't exist).

    Returns
    -------
        the detected core count, at least 1.
    """
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        try:
            return max(1, int(slurm_cpus))
        except ValueError:
            pass

    sched_getaffinity = getattr(os, "sched_getaffinity", None)
    if sched_getaffinity is not None:
        try:
            return max(1, len(sched_getaffinity(0)))
        except OSError:
            pass

    return max(1, os.cpu_count() or 1)


def default_bgmn_n_threads(available_cores: int) -> int:
    """
    Default BGMN thread count per refinement task, given `available_cores`.

    Fixed at a small constant (2) regardless of machine size: BGMN's own
    thread scaling saturates well before 10 threads (see module docstring),
    so additional cores are better spent on more *concurrent* refinement
    tasks -- exploring more of the tree search at once -- than on making any
    single task more heavily threaded.
    """
    return min(2, available_cores)


def default_ray_concurrency(available_cores: int, n_threads: int) -> int:
    """
    How many refinement tasks can run concurrently without exceeding
    `available_cores`, given each one uses `n_threads` OS threads.
    """
    return max(1, available_cores // max(1, n_threads))
