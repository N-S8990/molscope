import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from molscope.benchmarking import build_benchmark, inspect_mdxplain_checkout, load_benchmark_spec, write_benchmark


class BenchmarkingTests(unittest.TestCase):
    def _manifest(self, path: Path, *, replica: str = "replica_1") -> None:
        pd.DataFrame([{
            "system_id": "system", "ensemble_id": "ensemble", "replica_id": replica, "trajectory_id": replica,
        }]).to_csv(path, index=False)

    def _spec(self, directory: Path, *, method_manifest: str | None = None, mdxplain_path: str | None = None) -> Path:
        shared = directory / "shared.csv"
        self._manifest(shared)
        results = directory / "result.csv"
        pd.DataFrame([{"metric_id": "runtime_seconds", "value": 12.5, "notes": "CPU"}]).to_csv(results, index=False)
        payload = {
            "benchmark_id": "synthetic", "input_manifest": str(shared),
            "metrics": [{"metric_id": "runtime_seconds", "unit": "seconds", "higher_is_better": False}],
            "methods": [
                {"method_id": "platform", "label": "Platform", "kind": "molscope", "input_manifest": method_manifest or str(shared), "results_csv": str(results)},
                {"method_id": "mdxplain", "label": "mdxplain", "kind": "mdxplain", "checkout_path": mdxplain_path, "input_manifest": str(shared)},
            ],
        }
        spec = directory / "benchmark.json"
        spec.write_text(json.dumps(payload), encoding="utf-8")
        return spec

    def test_collates_declared_metrics_and_records_pending_external_method(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            outputs = build_benchmark(load_benchmark_spec(self._spec(directory)))
            self.assertEqual(outputs["method_status"].set_index("method_id").loc["platform", "status"], "ready")
            self.assertEqual(outputs["method_status"].set_index("method_id").loc["mdxplain", "status"], "pending_results")
            self.assertEqual(outputs["metrics"].iloc[0]["metric_id"], "runtime_seconds")
            output_dir = write_benchmark(outputs, directory / "out")
            self.assertTrue((output_dir / "benchmark_report.md").is_file())
            self.assertTrue((output_dir / "metrics.csv").is_file())

    def test_rejects_method_with_nonmatching_input_identities(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            mismatch = directory / "mismatch.csv"
            self._manifest(mismatch, replica="replica_2")
            outputs = build_benchmark(load_benchmark_spec(self._spec(directory, method_manifest=str(mismatch))))
            status = outputs["method_status"].set_index("method_id").loc["platform", "status"]
            self.assertEqual(status, "input_mismatch")
            self.assertTrue(outputs["metrics"].empty)

    def test_mdxplain_adapter_reports_recognizable_checkout_without_importing(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary)
            (checkout / "mdxplain").mkdir()
            (checkout / "pyproject.toml").write_text("[project]\nname='mdxplain'\n")
            (checkout / "mdxplain" / "__init__.py").write_text("")
            result = inspect_mdxplain_checkout(checkout)
        self.assertEqual(result["status"], "available_not_executed")

    def test_unknown_result_metric_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            spec = self._spec(directory)
            pd.DataFrame([{"metric_id": "not_declared", "value": 1.0}]).to_csv(directory / "result.csv", index=False)
            with self.assertRaisesRegex(ValueError, "undeclared metrics"):
                build_benchmark(load_benchmark_spec(spec))

    def test_preserves_declared_matched_protocol_without_executing_external_tool(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            spec_path = self._spec(directory)
            payload = json.loads(spec_path.read_text(encoding="utf-8"))
            payload["shared_protocol"] = {"frame_stride": 5, "imaging_policy": "declared"}
            payload["methods"][0]["protocol"] = {"runner": "molscope"}
            spec_path.write_text(json.dumps(payload), encoding="utf-8")
            outputs = build_benchmark(load_benchmark_spec(spec_path))
            self.assertEqual(outputs["summary"]["shared_protocol"]["frame_stride"], 5)
            status = outputs["method_status"].set_index("method_id")
            self.assertTrue(status.loc["platform", "protocol_declared"])
            self.assertIn("Declared shared protocol", outputs["markdown"])


if __name__ == "__main__":
    unittest.main()
