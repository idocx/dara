import os
import unittest
from unittest.mock import patch

from dara.resource_detection import (
    default_bgmn_n_threads,
    default_ray_concurrency,
    detect_available_cores,
)


class TestDetectAvailableCores(unittest.TestCase):
    def test_slurm_env_var_takes_priority(self):
        with patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "64"}):
            self.assertEqual(detect_available_cores(), 64)

    def test_slurm_env_var_invalid_falls_through(self):
        with patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "not-a-number"}):
            # should fall through to affinity/cpu_count, not raise
            result = detect_available_cores()
            self.assertGreaterEqual(result, 1)

    def test_falls_back_to_cpu_count_when_no_slurm_or_affinity(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("os.cpu_count", return_value=16):
                # simulate a platform with no sched_getaffinity (e.g. macOS)
                with patch("dara.resource_detection.os") as mock_os:
                    mock_os.environ.get.return_value = None
                    mock_os.cpu_count.return_value = 16
                    del mock_os.sched_getaffinity  # simulate attribute absent
                    self.assertEqual(detect_available_cores(), 16)

    def test_uses_affinity_when_available(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("os.sched_getaffinity", return_value=set(range(4)), create=True):
                self.assertEqual(detect_available_cores(), 4)

    def test_never_returns_less_than_one(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("os.cpu_count", return_value=None):
                with patch("dara.resource_detection.os") as mock_os:
                    mock_os.environ.get.return_value = None
                    mock_os.cpu_count.return_value = None
                    del mock_os.sched_getaffinity
                    self.assertEqual(detect_available_cores(), 1)


class TestDefaultBgmnNThreads(unittest.TestCase):
    def test_small_machine_gets_all_cores(self):
        self.assertEqual(default_bgmn_n_threads(1), 1)

    def test_capped_at_two_regardless_of_scale(self):
        self.assertEqual(default_bgmn_n_threads(12), 2)
        self.assertEqual(default_bgmn_n_threads(64), 2)
        self.assertEqual(default_bgmn_n_threads(256), 2)


class TestDefaultRayConcurrency(unittest.TestCase):
    def test_matches_measured_12_core_case(self):
        # This machine, measured: 12 cores, n_threads=2 -> concurrency=6
        self.assertEqual(default_ray_concurrency(12, 2), 6)

    def test_64_core_example(self):
        self.assertEqual(default_ray_concurrency(64, 2), 32)

    def test_never_returns_less_than_one(self):
        self.assertEqual(default_ray_concurrency(1, 2), 1)  # 1 // 2 = 0, floored to 1

    def test_product_stays_close_to_available_cores(self):
        for cores in [1, 2, 3, 4, 12, 17, 64, 256]:
            n_threads = default_bgmn_n_threads(cores)
            concurrency = default_ray_concurrency(cores, n_threads)
            product = n_threads * concurrency
            # product should never exceed available cores, and shouldn't
            # waste more than one "n_threads-sized" chunk of slack
            self.assertLessEqual(product, cores)
            self.assertGreater(product, cores - n_threads)


if __name__ == "__main__":
    unittest.main()
