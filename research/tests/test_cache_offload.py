"""同 dtype 的缓存搬运与消费旧缓存不得改变实际模型运算结果。"""

from __future__ import annotations

import unittest

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from kvstudy.groups import fit_per_channel
from phase1.runner import cache_lists, decode_step, prefill
from phase1.generation import _config, _decode_step, _prefill_cache
from experiments.generation_error_sources.code.trajectory_replay import (
    ComparisonObserver, ReferenceObserver, _rebuild_generated_history, reference_run,
)


class ObserverMetricsTest(unittest.TestCase):
    """关闭未用层诊断时仍保留完全相同的预测、KL 和间隔。"""

    def test_reset_keeps_prediction_metrics_exact(self) -> None:
        reference = ReferenceObserver()
        reference.logits = torch.tensor([3.0, 2.0, -1.0])
        reference.layers = [{"hidden": torch.tensor([2.0, 1.0])}]
        full = ComparisonObserver(reference)
        reset = ComparisonObserver(reference, collect_layers=False)
        for observer in (full, reset):
            observer({"layer": 0, "hidden": torch.tensor([1.0, 2.0])})
            observer({"logits": torch.tensor([2.0, 3.0, -1.0])})
        expected = {key: value for key, value in full.output.items() if key != "layers"}
        self.assertEqual(reset.output, expected)
        self.assertEqual(full.output["layers"], [{"hidden": 0.4}])
        self.assertEqual(reset.output["bf16_argmax"], 0)
        self.assertEqual(reset.output["candidate_argmax"], 1)
        self.assertLess(reset.output["candidate_reference_margin"], 0)


@unittest.skipUnless(torch.cuda.is_available(), "缓存搬运逐值检查需要 CUDA")
class CacheOffloadTest(unittest.TestCase):
    """小型真实 Llama 覆盖 128-token 环形窗口跨界及历史重建。"""

    def setUp(self) -> None:
        torch.manual_seed(123)
        config = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128,
                             num_hidden_layers=2, num_attention_heads=2,
                             num_key_value_heads=1, max_position_embeddings=1024)
        config._attn_implementation = "sdpa"
        self.model = LlamaForCausalLM(config).to(device="cuda", dtype=torch.bfloat16).eval()
        self.ids = (torch.arange(129, device="cuda") % 128).view(1, -1)

    @torch.inference_mode()
    def test_consumed_reference_cache_is_exact(self) -> None:
        outputs = prefill(self.model, self.ids)
        keys, values = cache_lists(outputs.past_key_values)
        original = decode_step(self.model, keys, values, 5, 129)
        consumed_k, consumed_v = list(keys), list(values)
        consumed = decode_step(self.model, consumed_k, consumed_v, 5, 129,
                               consume_cache=True)
        self.assertTrue(all(x is None for x in consumed_k + consumed_v))
        self.assertTrue(torch.equal(original[0], consumed[0]))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(
            original[1] + original[2], consumed[1] + consumed[2])))

    def _same(self, left: tuple, right: tuple, length: int) -> None:
        for a_group, b_group in zip(left[1:5], right[1:5]):
            for a, b in zip(a_group, b_group):
                end = min(length, a.shape[2])
                self.assertTrue(torch.equal(a[:, :, :end].cpu(), b[:, :, :end].cpu()))

    @torch.inference_mode()
    def test_all_candidates_across_ring_boundary_and_reset(self) -> None:
        tokens, _, references = reference_run(self.model, self.ids, set(), 132)
        for name, h in (("k8v4-per_token-pre", 0), ("k8v4-per_token-post", 0),
                        ("k8v4-per_channel-pre", 8)):
            with self.subTest(name=name, h=h):
                config = _config(name)
                params = fit_per_channel(torch.randn(1, 16, 32, device="cuda"), 8)
                calibrated = {"key": [params, params]} if h else {}
                left = _prefill_cache(self.model, self.ids, config, calibrated, 132, h)
                right = _prefill_cache(self.model, self.ids, config, calibrated, 132, h,
                                       offload_layers=1)
                self.assertEqual(left[0], right[0])
                self._same(left, right, 129)
                for step in range(131):
                    original = _decode_step(self.model, *left[1:5], tokens[step],
                                            129 + step, config, calibrated, h)
                    offloaded = _decode_step(self.model, *right[1:5], tokens[step],
                                             129 + step, config, calibrated, h)
                    self.assertEqual(original, offloaded)
                    self._same(left, right, 130 + step)
                del left, right
                left = _prefill_cache(self.model, self.ids, config, calibrated, 132, h)
                right = _prefill_cache(self.model, self.ids, config, calibrated, 132, h,
                                       offload_layers=1)
                for buffers in (left, right):
                    _rebuild_generated_history(self.model, buffers, references, config,
                                               calibrated, h, 129, 130)
                self._same(left, right, 259)


if __name__ == "__main__":
    unittest.main()
