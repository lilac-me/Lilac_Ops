"""CPU/HTTP regressions for cluster coverage and multi-node producer placement."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
import tempfile
import unittest
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DIRECTORY / "common"))
sys.path.insert(0, str(DIRECTORY.parent))
from producer_topology import load_inference_device_map, resolve_ipc_topology, validate_producer_records
from weight_update_validation import WeightUpdateValidation
from common.dual_api import ENDPOINTS


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, DIRECTORY / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RELOAD = load_script("test_reload_weights")
IPC = load_script("test_npu_ipc_weight_update")
HCCL = load_script("test_hccl_weight_update")


def cluster_arguments(**overrides):
    return SimpleNamespace(**{
        "expected_world_size": 4, "data_parallel_size": 2,
        "inference_nodes": "10.0.0.1,10.0.0.2", **overrides,
    })


def producer_records():
    return [{"rank": rank, "node_rank": rank // 2, "node_size": 2,
             "local_rank": rank % 2, "npu_uuid": f"10.0.0.{rank // 2 + 1}-{rank % 2}",
             "packed_buffer_bytes": 1024,
             "checkpoint": {"manifest_sha256": "same", "tensor_count": 8,
                            "total_tensor_bytes": 2048, "max_tensor_bytes": 1024}}
            for rank in range(4)]


class ProducerTopologyTests(unittest.TestCase):
    def arguments(self, **overrides):
        return SimpleNamespace(**{"devices": "1,0", "node_size": 2, "node_rank": 1,
                                  "master_addr": None, **overrides})

    def environment(self, **overrides):
        return {"WORLD_SIZE": "4", "LOCAL_WORLD_SIZE": "2", "RANK": "3", "LOCAL_RANK": "1",
                "GROUP_RANK": "1", "MASTER_ADDR": "10.0.0.1", **overrides}

    def test_two_nodes_use_local_device_not_global_rank(self):
        topology = resolve_ipc_topology(self.arguments(), self.environment())
        self.assertEqual(topology["node_rank"], 1)
        self.assertEqual(topology["device"], 0)
        self.assertEqual(topology["rank"], 3)

    def test_invalid_torchrun_placement_and_rendezvous_rejected(self):
        for overrides in ({"RANK": "4"}, {"LOCAL_RANK": "0"}, {"WORLD_SIZE": "3"},
                          {"GROUP_RANK": "0"}, {"MASTER_ADDR": "127.0.0.1"}, {"LOCAL_WORLD_SIZE": "0"}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                resolve_ipc_topology(self.arguments(), self.environment(**overrides))

    def test_explicit_node_flags_must_match_torchrun(self):
        for flags in ({"node_size": 3}, {"node_rank": 0}, {"devices": "0,0"}):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                resolve_ipc_topology(self.arguments(**flags), self.environment())

    def test_single_process_retains_device_selection(self):
        topology = resolve_ipc_topology(self.arguments(devices="2", node_size=None, node_rank=None), {})
        self.assertEqual((topology["rank"], topology["node_size"], topology["device"]), (0, 1, 2))

    def test_all_node_records_and_buffers_agree(self):
        validate_producer_records(producer_records(), 4)

    def test_physical_mapping_required_for_multinode_and_checked(self):
        with self.assertRaisesRegex(ValueError, "inference-devices-json"):
            load_inference_device_map(None, 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "devices.json"
            path.write_text(json.dumps([{"host": "10.0.0.1", "devices": [0, 1]},
                                        {"host": "10.0.0.2", "devices": [0, 1]}]), encoding="utf-8")
            expected = load_inference_device_map(str(path), 2)
            validate_producer_records(producer_records(), 4, expected)
            with self.assertRaisesRegex(ValueError, "推理设备映射不一致"):
                validate_producer_records(producer_records(), 4, {"wrong-host-0"})

    def test_duplicate_device_manifest_buffer_and_node_errors_rejected(self):
        mutations = [
            lambda records: records[3].update(npu_uuid=records[2]["npu_uuid"]),
            lambda records: records[3]["checkpoint"].update(manifest_sha256="different"),
            lambda records: records[3].update(packed_buffer_bytes=2048),
            lambda records: records[3]["checkpoint"].update(max_tensor_bytes=2048),
            lambda records: records[3].update(local_rank=0),
            lambda records: records[3].update(node_size=1),
            lambda records: records[3].update(npu_uuid="another-host-1"),
        ]
        for mutate in mutations:
            records = producer_records()
            mutate(records)
            with self.subTest(records=records), self.assertRaises(ValueError):
                validate_producer_records(records, 4)

    def test_different_node_ranks_cannot_claim_same_host(self):
        records = producer_records()
        records[2]["npu_uuid"] = "10.0.0.1-2"
        records[3]["npu_uuid"] = "10.0.0.1-3"
        with self.assertRaisesRegex(ValueError, "同一物理主机"):
            validate_producer_records(records, 4)

    def test_remote_rank_error_reaches_coordinator(self):
        class FakeDistributed:
            def is_initialized(self): return True
            def get_world_size(self): return 4
            def all_gather_object(self, results, local_error, group):
                results[:] = [local_error, None, "rank 2: remote checkpoint missing", None]
        with self.assertRaisesRegex(RuntimeError, "rank 2: remote checkpoint missing"):
            IPC.run_distributed_step(SimpleNamespace(control_group=None), FakeDistributed(), 0, "preflight", lambda: None)

    def test_ipc_factory_receives_global_rank_on_second_node(self):
        captured = {}
        class FakeFactory:
            @staticmethod
            def register_engine(*args): pass
            @staticmethod
            def trainer_init(info, **kwargs):
                captured["rank"] = info.rank
                return "producer"
        factory_module = SimpleNamespace(WeightTransferTrainerFactory=FakeFactory,
                                         HTTPVLLMWeightSyncClient=lambda *args, **kwargs: None)
        backend_module = SimpleNamespace(NPUIPCTrainerInitInfo=SimpleNamespace)
        modules = {"vllm.distributed.weight_transfer": factory_module,
                   "vllm_ascend.distributed.weight_transfer.npu_ipc_engine": backend_module}
        args = SimpleNamespace(packed_buffer_mb=1, server_url="http://head:8000", timeout=5)
        source = SimpleNamespace(summary=lambda: {"max_tensor_bytes": 1024})
        with patch.dict(sys.modules, modules), patch.object(IPC, "_uses_unsupported_vllm_version", return_value=False):
            self.assertEqual(IPC.create_checkpoint_producer(args, 3, source), "producer")
        self.assertEqual(captured["rank"], 3)


@contextmanager
def inference_cluster(*, changed_replica=None, world_size=4, group_size=2, fail_reload=False):
    state = {"sleeping": False, "updated": False, "requests": [], "weights_awake": False}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_): pass
        def respond(self, body, status=200):
            encoded = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
        def do_GET(self):
            if self.path.startswith("/get_world_size"):
                self.respond({"world_size": group_size if "false" in self.path else world_size})
            elif self.path == "/v1/models": self.respond({"data": [{"id": "policy"}]})
            elif self.path == "/is_sleeping": self.respond({"is_sleeping": state["sleeping"]})
            else: self.respond({})
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length)) if length else None
            state["requests"].append((self.path, self.headers.get("X-data-parallel-rank"), payload))
            if self.path.startswith("/sleep"):
                state["sleeping"] = True
            elif self.path == "/wake_up?tags=weights":
                state["weights_awake"] = True
            elif self.path == "/wake_up?tags=kv_cache":
                state["sleeping"] = False
            elif self.path == "/collective_rpc":
                if fail_reload:
                    return self.respond({"error": "worker on node 1 failed to reload"}, 500)
                state["updated"] = True
                return self.respond({"results": [None] * group_size})
            elif self.path in ENDPOINTS:
                rank = int(self.headers.get("X-data-parallel-rank", "0"))
                text = f"replica-{rank}"
                if state["updated"] and rank == changed_replica: text = "changed"
                if self.path == "/v1/completions":
                    choice = {"text": text, "logprobs": {"tokens": [text], "token_logprobs": [-.5]}}
                else:
                    choice = {"message": {"content": text}, "logprobs": {"content": [{"token": text, "logprob": -.5}]}}
                return self.respond({"choices": [choice]})
            self.respond({})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class ClusterValidationTests(unittest.TestCase):
    def validation(self, url, report):
        return WeightUpdateValidation(url, "auto", ["first", "second"], 4, 5, report)

    def test_every_dp_replica_and_both_endpoints_are_compared(self):
        with inference_cluster() as (url, state):
            report = {}
            validation = self.validation(url, report)
            self.assertEqual(validation.configure_cluster(cluster_arguments()), 4)
            validation.record_baseline()
            validation.verify_updated_outputs()
            self.assertEqual(report["outputs_identical_by_dp_rank"], {"0": True, "1": True})
            for endpoint in ENDPOINTS:
                self.assertEqual(len(report["baseline_by_endpoint"][endpoint]), 4)
                self.assertEqual({rank for path, rank, _ in state["requests"] if path == endpoint}, {"0", "1"})

    def test_changed_remote_replica_fails_even_when_replica_zero_matches(self):
        with inference_cluster(changed_replica=1) as (url, state):
            report = {}
            validation = self.validation(url, report)
            validation.configure_cluster(cluster_arguments())
            validation.record_baseline()
            state["updated"] = True
            with self.assertRaisesRegex(RuntimeError, "不一致"):
                validation.verify_updated_outputs()
            self.assertEqual(report["outputs_identical_by_dp_rank"], {"0": True, "1": False})

    def test_missing_or_duplicate_replica_output_fails_acceptance(self):
        with inference_cluster() as (url, _):
            validation = self.validation(url, {})
            validation.configure_cluster(cluster_arguments())
            validation.record_baseline()
            validation.verify_updated_outputs()
            for endpoint in ENDPOINTS:
                validation.report["post_update_by_endpoint"][endpoint][-1]["dp_rank"] = 0
            with self.assertRaisesRegex(RuntimeError, "完整覆盖"):
                validation.compare_outputs()

    def test_bad_expected_world_size_fails_before_sleep(self):
        with inference_cluster() as (url, state):
            with self.assertRaises(ValueError):
                self.validation(url, {}).configure_cluster(cluster_arguments(expected_world_size=8))
            self.assertEqual(state["requests"], [])

    def test_bad_dp_size_or_node_list_fails_before_sleep(self):
        with inference_cluster() as (url, state):
            for flags in ({"data_parallel_size": 4}, {"inference_nodes": "a,a"}, {"inference_nodes": "a,"}):
                with self.subTest(flags=flags), self.assertRaises(ValueError):
                    self.validation(url, {}).configure_cluster(cluster_arguments(**flags))
            self.assertEqual(state["requests"], [])

    def test_single_dp_multinode_pipeline_needs_no_routing_header(self):
        with inference_cluster(world_size=4, group_size=4) as (url, state):
            validation = self.validation(url, {})
            validation.configure_cluster(cluster_arguments(data_parallel_size=1))
            validation.record_baseline()
            validation.verify_updated_outputs()
            self.assertEqual(validation.report["outputs_identical_by_dp_rank"], {"0": True})
            self.assertTrue(all(rank is None for _, rank, _ in state["requests"]))

    def test_checkpoint_lifecycle_against_two_replica_http_cluster(self):
        with inference_cluster() as (url, state):
            args = cluster_arguments(server_url=url, model=None, served_model_name="auto", updated_model="/shared/model",
                                     max_tokens=4, timeout=5, prompts_json="unused")
            report = {"checks": {}}
            with patch.object(RELOAD, "load_prompts", return_value=["first", "second"]):
                RELOAD.run_validation(args, report)
            self.assertFalse(state["sleeping"])
            self.assertTrue(report["checks"]["checkpoint_reload"])
            self.assertTrue(report["checks"]["outputs_identical"])
            self.assertEqual(report["outputs_identical_by_dp_rank"], {"0": True, "1": True})

    def test_remote_reload_failure_leaves_kv_cache_asleep(self):
        with inference_cluster(fail_reload=True) as (url, state):
            args = cluster_arguments(server_url=url, model=None, served_model_name="auto", updated_model="/shared/model",
                                     max_tokens=4, timeout=5, prompts_json="unused")
            report = {"checks": {}}
            with patch.object(RELOAD, "load_prompts", return_value=["first"]), self.assertRaises(RuntimeError):
                RELOAD.run_validation(args, report)
            self.assertTrue(state["sleeping"])
            self.assertTrue(report["server_left_sleeping"])
            self.assertNotIn("/wake_up?tags=kv_cache", [path for path, _, _ in state["requests"]])

    def test_all_three_entrypoints_expose_cluster_arguments(self):
        common = ["--server-url", "http://head:8000", "--prompts-json", "prompts.json", "--report", "out.json",
                  "--inference-nodes", "node0,node1", "--expected-world-size", "4", "--data-parallel-size", "2"]
        for module, backend_args in [(RELOAD, ["--updated-model", "/model"]),
                                     (HCCL, ["--checkpoint", "/model", "--master-address", "10.0.0.3", "--master-port", "29501"]),
                                     (IPC, ["--checkpoint", "/model", "--devices", "0,1", "--node-size", "2", "--node-rank", "1"])]:
            with self.subTest(module=module.__name__):
                args = module.parse_args(common + backend_args)
                self.assertEqual(args.expected_world_size, 4)

    def test_hccl_initializes_all_global_workers_plus_one_producer(self):
        observed = {}
        class FakeClient:
            def init_weight_transfer_engine(self, info): observed["workers"] = info
        class FakeBackend:
            @staticmethod
            def trainer_init(info):
                observed["trainer"] = info
                return SimpleNamespace(available=True, disabled=False)
        args = SimpleNamespace(master_address="10.0.0.3", master_port=29501, timeout=5)
        with patch.object(HCCL.time, "monotonic", side_effect=[0, 6]):
            HCCL.initialize_hccl_transfer(args, FakeClient(), FakeBackend(), 4)
        self.assertEqual(observed["workers"], {"master_address": "10.0.0.3", "master_port": 29501,
                                                "rank_offset": 1, "world_size": 5})
        self.assertEqual(observed["trainer"]["world_size"], 5)
        self.assertNotIn("rank_offset", observed["trainer"])


if __name__ == "__main__":
    unittest.main()
