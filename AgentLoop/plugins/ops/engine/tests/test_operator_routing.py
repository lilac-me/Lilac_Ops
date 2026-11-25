import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from operator_routing import classify, route_plan


class RouteTests(unittest.TestCase):
    def test_native_not_triton(self):
        self.assertEqual(classify("aclnnMatMul")["target"], "ascendc")
        self.assertEqual(classify("hot_kernel")["backend"], "unknown")
        self.assertEqual(classify("HcclAllReduce")["backend"], "excluded")
        self.assertEqual(classify("triton_kernel")["backend"], "triton_hint")
        self.assertEqual(classify("hot_kernel", {}, "@triton.jit\ndef hot_kernel(): pass")["backend"], "triton")

    def test_same_language_and_mixed_fusion(self):
        mapping = {"my_kernel": {"backend": "triton", "evidence": "actual dispatch"}}
        opportunities = {"matmul_mfu": [{"name": "aclnnMatMul", "mean_op_ratio": .2}],
                         "vector_fusion": [
                             {"names": ["aclnnSilu", "aclnnMul"], "mean_op_ratio": .1},
                             {"names": ["my_kernel", "aclnnMul"], "mean_op_ratio": .1},
                             {"names": ["my_kernel", "my_kernel"], "mean_op_ratio": .1}]}
        r = route_plan(opportunities, [], mapping, {}, ".")
        self.assertEqual(r[0]["plugin"], "tilelang2ascendc-ops-generator")
        self.assertEqual(r[0]["objective"], "matmul_mfu")
        self.assertEqual(r[1]["action"], "generate_fused")
        self.assertEqual(r[2]["action"], "split_or_review")
        self.assertEqual(r[3]["plugin"], "triton-op-generator")

    def test_conflicting_dispatch(self):
        self.assertEqual(classify("aclnnMul", {"backend": "triton", "evidence": "bad"})["backend"], "unknown")
