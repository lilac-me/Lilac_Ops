"""Synthetic data tests for routing and CSV contract."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))
sys.path.insert(0, str(ENGINE.parents[2] / "skills/ops-skills/triton/profile-bottleneck-analyzer/scripts"))
from training_backend.profile_contract import check_profile
from opportunities import identify


class OpportunityTests(unittest.TestCase):
    def test_profile_and_routes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "kernel_details.csv"
            keys = ["Step Id", "Name", "Type", "Accelerator Core", "Start Time(us)",
                    "Duration(us)", "Stream ID", "Input Shapes", "Input Data Types",
                    "Output Shapes", "Output Data Types", "aic_mac_ratio"]
            def row(name, kind, core, start, duration, stream=0):
                return [1, name, kind, core, start, duration, stream, "4,8", "BF16", "4,8", "BF16", 0.4]
            with path.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(keys)
                # Out of order in CSV: actual timeline must decide adjacency.
                w.writerows([row("mul_kernel", "Mul", "AI_VECTOR_CORE", 5, 4),
                             row("aclnnMatMul", "MatMulV2", "AI_CORE", 10, 20),
                             row("silu_kernel", "Silu", "AI_VECTOR_CORE", 0, 4),
                             row("add_kernel", "Add", "AI_VECTOR_CORE", 31, 4, 1)])
            (root / "step_trace_time.csv").write_text("Step\\n1\\n".replace("\\\\n", "\\n"))
            (root / "trace_view.json").write_text("[]")
            self.assertGreater(check_profile(root)["shape_rows"], 0)
            result = identify(path, ["1"])
            self.assertEqual(result["matmul_mfu"][0]["name"], "aclnnMatMul")
            self.assertEqual(result["vector_fusion"][0]["names"], ["silu_kernel", "mul_kernel"])
            self.assertEqual(len(result["vector_fusion"]), 1)
            (root / "trace_view.json").unlink()
            with self.assertRaises(ValueError):
                check_profile(root)

    def test_missing_timing_suppresses_fusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "kernel_details.csv"
            path.write_text("Step Id,Name,Duration(us),Accelerator Core,Stream ID,Input Shapes,Input Data Types\\n1,silu_kernel,10,AI_VECTOR_CORE,0,4,BF16\\n".replace("\\\\n", "\\n"))
            result = identify(path, ["1"])
            self.assertFalse(result["fusion_scan_complete"])
            self.assertEqual(result["vector_fusion"], [])
