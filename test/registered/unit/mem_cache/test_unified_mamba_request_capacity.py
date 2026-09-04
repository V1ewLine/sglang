# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import unittest

from sglang.srt.mem_cache.kv_cache_configurator import (
    _compute_unified_mamba_request_capacity,
    _limit_unified_max_running_requests,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestUnifiedMambaRequestCapacity(unittest.TestCase):
    def test_capacity_accounts_for_fixed_and_per_request_memory(self):
        capacity = _compute_unified_mamba_request_capacity(
            total_bytes=3_700,
            full_token_bytes=10,
            draft_token_bytes=0,
            fixed_shared_bytes=100,
            shared_bytes_per_request=200,
            fixed_external_bytes=50,
            external_bytes_per_request=30,
        )

        self.assertEqual(capacity, 15)

    def test_capacity_accounts_for_draft_kv_pool(self):
        capacity = _compute_unified_mamba_request_capacity(
            total_bytes=3_700,
            full_token_bytes=10,
            draft_token_bytes=10,
            fixed_shared_bytes=100,
            shared_bytes_per_request=200,
            fixed_external_bytes=50,
            external_bytes_per_request=30,
        )

        self.assertEqual(capacity, 8)

    def test_user_limit_is_capped_by_unified_capacity(self):
        self.assertEqual(
            _limit_unified_max_running_requests(
                requested=100, capacity=50, attn_dp_size=1
            ),
            50,
        )
        self.assertEqual(
            _limit_unified_max_running_requests(
                requested=32, capacity=50, attn_dp_size=1
            ),
            32,
        )

    def test_auto_limit_uses_unified_capacity(self):
        self.assertEqual(
            _limit_unified_max_running_requests(
                requested=None, capacity=50, attn_dp_size=1
            ),
            50,
        )

    def test_user_limit_is_converted_to_one_dp_worker(self):
        self.assertEqual(
            _limit_unified_max_running_requests(
                requested=100, capacity=80, attn_dp_size=2
            ),
            50,
        )


if __name__ == "__main__":
    unittest.main()
