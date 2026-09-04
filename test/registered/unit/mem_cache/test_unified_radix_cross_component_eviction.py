"""CPU-only tests for Kimi FULL/MAMBA adaptive memory management."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import torch

from sglang.srt.managers.schedule_policy import PrefillAdder
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.multi_ended_allocator import (
    UnifiedMambaTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestUnifiedRadixCrossComponentEviction(CustomTestCase):
    @staticmethod
    def _build_cache(
        *,
        full_available: int = 0,
        mamba_available: int = 0,
        full_virtual_available: int = 1024,
        mamba_virtual_available: int = 16,
        full_evictable: int = 0,
        mamba_evictable: int = 0,
        full_tokens_per_mamba: int = 64,
        defer_full_eviction: bool = False,
    ):
        cache = object.__new__(UnifiedRadixCache)
        cache.disable = False
        cache.tree_components = (ComponentType.FULL, ComponentType.MAMBA)
        cache.is_swa_enabled = False
        cache.cache_controller = None
        cache.metrics_collector = None

        capacity = {
            ComponentType.FULL: full_available,
            ComponentType.MAMBA: mamba_available,
        }
        deferred_capacity = {
            ComponentType.FULL: 0,
            ComponentType.MAMBA: 0,
        }
        evictable = {
            ComponentType.FULL: full_evictable,
            ComponentType.MAMBA: mamba_evictable,
        }

        allocator = MagicMock()
        allocator.supports_full_mamba_cross_reclaim = True
        allocator.available_size.side_effect = lambda: capacity[ComponentType.FULL]
        allocator.full_virtual_available_size.return_value = full_virtual_available
        allocator.mamba_virtual_available_size.return_value = mamba_virtual_available
        allocator.get_memory_snapshot.side_effect = lambda: {
            "full_available_schedulable_tokens": capacity[ComponentType.FULL],
            "mamba_available_schedulable_slots": capacity[ComponentType.MAMBA],
            "shared_gap_bytes": capacity[ComponentType.FULL] * 4,
        }
        if defer_full_eviction:

            def drain_group():
                for component_type in deferred_capacity:
                    capacity[component_type] += deferred_capacity[component_type]
                    deferred_capacity[component_type] = 0

            allocator.drain_free_group.side_effect = drain_group
        cache.token_to_kv_pool_allocator = allocator

        mamba_allocator = MagicMock()
        mamba_allocator.schedulable_available_size.side_effect = (
            lambda: capacity[ComponentType.MAMBA]
        )
        cache.req_to_token_pool = MagicMock(mamba_allocator=mamba_allocator)

        cache.tree_core = MagicMock()
        cache.tree_core.component_evictable_size.side_effect = (
            lambda component_type: evictable[component_type]
        )

        active_victim = {"component": None}

        def next_node(component_type, tracker):
            active_victim["component"] = component_type
            if tracker[component_type] >= evictable[component_type]:
                return None, False
            return 1, True

        def evict_leaf(_node_id, tracker):
            victim = active_victim["component"]
            if victim == ComponentType.FULL:
                tracker[ComponentType.FULL] += 32
                target = deferred_capacity if defer_full_eviction else capacity
                target[ComponentType.FULL] += 32
                target[ComponentType.MAMBA] += 1
            else:
                tracker[ComponentType.MAMBA] += 1
                capacity[ComponentType.FULL] += full_tokens_per_mamba
                capacity[ComponentType.MAMBA] += 1
            return None

        cache._evict_device_next_node = MagicMock(side_effect=next_node)
        cache._evict_device_leaf = MagicMock(side_effect=evict_leaf)
        return cache, capacity

    def test_full_shortfall_evicts_mamba_peer(self):
        cache, capacity = self._build_cache(mamba_evictable=3)

        with self.assertLogs(
            "sglang.srt.mem_cache.unified_radix_cache", level="INFO"
        ) as logs:
            result = cache.evict_for_alloc(EvictParams(num_tokens=64))

        self.assertEqual(capacity[ComponentType.FULL], 64)
        self.assertEqual(result.num_tokens_evicted, 0)
        self.assertEqual(result.mamba_num_evicted, 1)
        self.assertEqual(
            cache.tree_core.evict_device_start.call_args_list,
            [call(ComponentType.FULL, 64), call(ComponentType.MAMBA, 3)],
        )
        output = "\n".join(logs.output)
        self.assertEqual(output.count("[unified-memory-reclaim]"), 1)
        self.assertIn("KV needs 64 tokens", output)
        self.assertIn("evicted 1 Mamba slot", output)
        self.assertIn(
            "after reclaim: KV free=64 tokens, Mamba free=1 slot, "
            "shared gap=0.0 MiB",
            output,
        )
        self.assertIn("allocation possible=yes", output)

    def test_kimi_prefill_regression_reclaims_mamba_for_full_shortfall(self):
        cache, capacity = self._build_cache(
            full_available=14_592,
            full_virtual_available=20_000,
            mamba_evictable=3,
            full_tokens_per_mamba=2_032,
        )

        result = cache.evict_for_alloc(EvictParams(num_tokens=1_755))

        self.assertGreaterEqual(capacity[ComponentType.FULL], 16_347)
        self.assertEqual(result.mamba_num_evicted, 1)

    def test_mamba_shortfall_evicts_full_peer(self):
        cache, capacity = self._build_cache(full_evictable=64)

        result = cache.evict_for_alloc(EvictParams(mamba_num=1))

        self.assertEqual(capacity[ComponentType.MAMBA], 1)
        self.assertEqual(result.num_tokens_evicted, 32)
        self.assertEqual(result.mamba_num_evicted, 0)
        self.assertEqual(
            cache.tree_core.evict_device_start.call_args_list,
            [call(ComponentType.MAMBA, 1), call(ComponentType.FULL, 64)],
        )

    def test_grouped_full_free_is_visible_before_mamba_eviction(self):
        """Regression for allocation recovery inside prefill's free group."""
        cache, capacity = self._build_cache(full_evictable=64)

        def drain_group():
            # A previously released FULL page becomes shared capacity for one
            # Mamba slot only when the surrounding free group is drained.
            capacity[ComponentType.FULL] += 32
            capacity[ComponentType.MAMBA] += 1
            cache.token_to_kv_pool_allocator.drain_free_group.side_effect = None

        cache.token_to_kv_pool_allocator.drain_free_group.side_effect = drain_group

        result = cache.evict_for_alloc(EvictParams(mamba_num=1))

        self.assertEqual(result.num_tokens_evicted, 0)
        self.assertEqual(result.mamba_num_evicted, 0)
        self.assertEqual(capacity[ComponentType.MAMBA], 1)
        cache.tree_core.evict_device_start.assert_not_called()
        cache.token_to_kv_pool_allocator.drain_free_group.assert_called()

    def test_cross_evicted_full_free_is_visible_to_mamba_retry(self):
        cache, capacity = self._build_cache(
            full_evictable=64,
            defer_full_eviction=True,
        )

        result = cache.evict_for_alloc(EvictParams(mamba_num=1))

        self.assertEqual(result.num_tokens_evicted, 32)
        self.assertEqual(capacity[ComponentType.MAMBA], 1)
        self.assertEqual(cache._evict_device_leaf.call_count, 1)

    def test_virtual_id_shortage_does_not_evict_peer(self):
        cache, _ = self._build_cache(
            mamba_virtual_available=0,
            full_evictable=64,
        )

        with self.assertLogs(
            "sglang.srt.mem_cache.unified_radix_cache", level="WARNING"
        ) as logs:
            result = cache.evict_for_alloc(EvictParams(mamba_num=1))

        self.assertEqual(result.num_tokens_evicted, 0)
        cache.tree_core.evict_device_start.assert_called_once_with(
            ComponentType.MAMBA, 1
        )
        self.assertIn("reason=not enough unused virtual IDs", "\n".join(logs.output))

    def test_no_peer_victim_exits_without_looping(self):
        cache, _ = self._build_cache()

        with self.assertLogs(
            "sglang.srt.mem_cache.unified_radix_cache", level="WARNING"
        ) as logs:
            result = cache.evict_for_alloc(EvictParams(num_tokens=64))

        self.assertEqual(result.num_tokens_evicted, 0)
        self.assertEqual(result.mamba_num_evicted, 0)
        cache.tree_core.evict_device_start.assert_called_once_with(
            ComponentType.FULL, 64
        )
        self.assertIn("reason=no unlocked cache available", "\n".join(logs.output))

    def test_virtual_capacity_uses_allocator_units(self):
        allocator = object.__new__(UnifiedMambaTokenToKVPoolAllocator)
        allocator.full_attn_allocator = SimpleNamespace(
            free_virtual_ids=[1, 2, 3], page_size=8
        )
        allocator.mamba_allocator = SimpleNamespace(
            free_virtual_ids=[1, 2, 3, 4], page_size=1
        )

        self.assertEqual(allocator.full_virtual_available_size(), 24)
        self.assertEqual(allocator.mamba_virtual_available_size(), 4)

    def test_drain_free_group_keeps_outer_group_open(self):
        allocator = object.__new__(UnifiedMambaTokenToKVPoolAllocator)
        allocator.free_group = None
        allocator.free_page_reps_group = None
        allocator.full_attn_allocator = MagicMock()
        allocator.mamba_allocator = MagicMock()

        allocator.free_group_begin()
        allocator.free(torch.tensor([7], dtype=torch.int64))
        allocator.drain_free_group()

        allocator.full_attn_allocator.free.assert_called_once()
        self.assertEqual(allocator.free_group, [])
        self.assertEqual(allocator.free_page_reps_group, [])

        allocator.free(torch.tensor([8], dtype=torch.int64))
        allocator.full_attn_allocator.free.assert_called_once()
        allocator.free_group_end()
        self.assertEqual(allocator.full_attn_allocator.free.call_count, 2)
        self.assertIsNone(allocator.free_group)
        self.assertIsNone(allocator.free_page_reps_group)


class TestUnifiedFullMambaAdmission(CustomTestCase):
    @staticmethod
    def _build_allocator(*, mamba_available: int = 10):
        allocator = object.__new__(UnifiedMambaTokenToKVPoolAllocator)
        allocator.mamba_slot_full_token_cost = lambda: 64
        allocator.mamba_allocator = SimpleNamespace(
            schedulable_available_size=lambda: mamba_available
        )
        return allocator

    @staticmethod
    def _build_tree_cache(*, mamba_evictable: int = 2):
        req_pool = SimpleNamespace(
            enable_mamba_extra_buffer_lazy=False,
            mamba_ping_pong_track_buffer_size=2,
            mamba_ckpt_pool=None,
        )
        return SimpleNamespace(
            supports_mamba=lambda: True,
            enable_mamba_extra_buffer=True,
            req_to_token_pool=req_pool,
            mamba_evictable_size=lambda: mamba_evictable,
        )

    @staticmethod
    def _build_adder(allocator, tree_cache):
        with (
            patch(
                "sglang.srt.managers.schedule_policy.is_dsa_prefill_cp_in_seq_split",
                return_value=False,
            ),
            patch(
                "sglang.srt.managers.schedule_policy.is_prefill_context_parallel_enabled",
                return_value=False,
            ),
        ):
            return PrefillAdder(
                page_size=1,
                tree_cache=tree_cache,
                token_to_kv_pool_allocator=allocator,
                running_batch=None,
                new_token_ratio=1.0,
                rem_input_tokens=10_000,
                rem_chunk_tokens=None,
            )

    def test_prefill_reserves_real_slots_and_runtime_headroom(self):
        allocator = self._build_allocator()
        tree_cache = self._build_tree_cache()
        adder = self._build_adder(allocator, tree_cache)

        # One global transient slot remains available for cache_unfinished_req.
        self.assertEqual(adder.rem_total_token_offset, 64)
        self.assertEqual(adder.cur_rem_token_offset, 64)
        self.assertEqual(adder.rem_mamba_slots, 11)  # 10 free + 2 evictable - 1

        new_req = SimpleNamespace(
            kv=SimpleNamespace(
                holds_mamba=False,
                mamba_ping_pong_track_buffer=None,
                mamba_cow_src_index=None,
            )
        )
        reserve = adder._mamba_gap_budget_for_req(new_req)
        self.assertEqual(reserve, 4 * 64)

        adder._update_prefill_budget(
            prefix_len=0,
            extend_input_len=1,
            max_new_tokens=0,
            retracted_stain=False,
            mamba_gap_reserve=reserve,
        )
        self.assertEqual(adder.rem_mamba_slots, 7)

    def test_prefill_only_reserves_slots_missing_from_existing_req(self):
        allocator = self._build_allocator()
        tree_cache = self._build_tree_cache()
        adder = self._build_adder(allocator, tree_cache)
        existing_req = SimpleNamespace(
            kv=SimpleNamespace(
                holds_mamba=True,
                mamba_ping_pong_track_buffer=object(),
                mamba_cow_src_index=None,
            )
        )
        self.assertEqual(adder._mamba_gap_budget_for_req(existing_req), 0)

    def test_prefill_charges_main_and_locked_checkpoint_after_new_cow(self):
        allocator = self._build_allocator()
        tree_cache = self._build_tree_cache()
        adder = self._build_adder(allocator, tree_cache)
        matched_req = SimpleNamespace(
            kv=SimpleNamespace(
                holds_mamba=True,
                mamba_ping_pong_track_buffer=None,
                mamba_cow_src_index=object(),
            )
        )

        self.assertEqual(adder._mamba_gap_budget_for_req(matched_req), 4 * 64)

    def test_prefill_rejects_req_larger_than_remaining_slot_budget(self):
        allocator = self._build_allocator(mamba_available=4)
        tree_cache = self._build_tree_cache(mamba_evictable=0)
        adder = self._build_adder(allocator, tree_cache)  # 3 after headroom
        new_req = SimpleNamespace(
            kv=SimpleNamespace(
                holds_mamba=False,
                mamba_ping_pong_track_buffer=None,
                mamba_cow_src_index=None,
            )
        )

        self.assertFalse(adder._mamba_req_fits(new_req))

    def test_decode_preserves_one_mamba_runtime_slot(self):
        allocator = self._build_allocator(mamba_available=0)
        allocator.evict_to_free_tokens = MagicMock()
        allocator.available_size = MagicMock(return_value=96)

        slot_state = {"available": 0}
        allocator.mamba_allocator.schedulable_available_size = (
            lambda: slot_state["available"]
        )
        tree_cache = MagicMock()
        tree_cache.supports_mamba.return_value = True
        tree_cache.evict_for_alloc.side_effect = lambda _params: slot_state.update(
            available=1
        )

        self.assertTrue(
            allocator.check_decode_capacity(num_tokens=32, tree_cache=tree_cache)
        )
        allocator.evict_to_free_tokens.assert_called_once_with(tree_cache, 96)
        params = tree_cache.evict_for_alloc.call_args.args[0]
        self.assertEqual(params, EvictParams(mamba_num=1))

    def test_decode_retracts_when_runtime_slot_cannot_be_reclaimed(self):
        allocator = self._build_allocator(mamba_available=0)
        allocator.evict_to_free_tokens = MagicMock()
        allocator.available_size = MagicMock(return_value=96)
        tree_cache = MagicMock()
        tree_cache.supports_mamba.return_value = True

        self.assertFalse(
            allocator.check_decode_capacity(num_tokens=32, tree_cache=tree_cache)
        )


if __name__ == "__main__":
    unittest.main()
