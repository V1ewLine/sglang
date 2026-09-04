"""CPU-only tests for Unified Memory log rank selection."""

import argparse
import logging
import unittest

from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.common import _UnifiedMemoryLogRankFilter
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _record(message: str) -> logging.LogRecord:
    return logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
    )


class TestUnifiedMemoryLogRankScope(CustomTestCase):
    def test_server_arg_defaults_to_tp0_and_accepts_all(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)

        default_args = parser.parse_args(["--model-path", "dummy"])
        all_rank_args = parser.parse_args(
            [
                "--model-path",
                "dummy",
                "--unified-memory-log-rank-scope",
                "all",
            ]
        )

        self.assertEqual(default_args.unified_memory_log_rank_scope, "tp0")
        self.assertEqual(all_rank_args.unified_memory_log_rank_scope, "all")

    def test_tp0_scope_filters_only_unified_memory_logs(self):
        rank_zero_filter = _UnifiedMemoryLogRankFilter(
            tp_rank=0,
            rank_scope="tp0",
        )
        rank_one_filter = _UnifiedMemoryLogRankFilter(
            tp_rank=1,
            rank_scope="tp0",
        )

        unified_messages = (
            "[unified-memory] initialized",
            "[unified-memory-pool] allocated",
            "[unified-memory-reclaim] Mamba needs 1 slot",
            "[unified-memory-flush] pool=full",
            "[Prefill] requests: new=1",
            "[Decode] requests: running=1",
        )
        for message in unified_messages:
            with self.subTest(message=message):
                self.assertTrue(rank_zero_filter.filter(_record(message)))
                self.assertFalse(rank_one_filter.filter(_record(message)))

        self.assertTrue(
            rank_one_filter.filter(_record("CUDA out of memory on TP rank 1"))
        )

    def test_all_scope_keeps_unified_memory_logs_on_nonzero_rank(self):
        rank_filter = _UnifiedMemoryLogRankFilter(tp_rank=3, rank_scope="all")

        self.assertTrue(
            rank_filter.filter(_record("[unified-memory-reclaim] allocation failed"))
        )


if __name__ == "__main__":
    unittest.main()
