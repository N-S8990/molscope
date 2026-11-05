import json
import tempfile
import unittest
from pathlib import Path

from molscope.project_spec import load_project_spec, project_summary
from molscope.openmm_plan import build_openmm_plan, write_openmm_plan
from molscope.run_project import _target_column


class ProjectSpecTests(unittest.TestCase):
    def _write_project(self, directory: Path) -> Path:
        (directory / "topology.pdb").write_text("HEADER test\nEND\n", encoding="utf-8")
        (directory / "trajectory.dcd").write_bytes(b"test")
        payload = {
            "project_id": "generic-study",
            "systems": [{
                "system_id": "arbitrary-system",
                "topology": "topology.pdb",
                "ensembles": [{
                    "ensemble_id": "reference-condition",
                    "metadata": {"condition": "reference"},
                    "replicas": [{"replica_id": "run-01", "trajectory": "trajectory.dcd"}],
                }],
            }],
            "prediction_tasks": [{
                "task_id": "optional-task",
                "task_type": "classification",
                "target_metadata_key": "condition",
            }],
            "analysis": {
                "generic_feature_parameters": {"max_frames": 5, "contact_cutoff_angstrom": 5.0},
                "plugins": ["protein.salt_bridges"],
            },
        }
        path = directory / "project.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_relative_paths_and_generic_metadata_are_loaded(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = load_project_spec(self._write_project(Path(temporary)))
            self.assertEqual(project.project_id, "generic-study")
            item = project.trajectory_inputs()[0]
            self.assertEqual(item.system_id, "arbitrary-system")
            self.assertEqual(item.ensemble_id, "arbitrary-system:reference-condition")
            self.assertEqual(project_summary(project)["missing_paths"], [])
            self.assertEqual(project.analysis.plugins, ("protein.salt_bridges",))
            self.assertEqual(project.analysis.generic_feature_parameters["max_frames"], 5)

    def test_duplicate_system_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self._write_project(Path(temporary))
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["systems"].append(payload["systems"][0])
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate system_id"):
                load_project_spec(path)

    def test_declared_trajectory_labels_are_preserved_separately_from_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = self._write_project(directory)
            payload = json.loads(path.read_text(encoding="utf-8"))
            trajectory_id = "arbitrary-system:reference-condition:run-01"
            payload["labels"] = [{
                "label_id": "experimental_condition",
                "label_type": "categorical",
                "level": "trajectory",
                "source": "external_annotation",
                "description": "Experimentally assigned condition.",
                "assignments": {trajectory_id: "reference"},
            }]
            payload["prediction_tasks"] = [{
                "task_id": "condition_from_label",
                "task_type": "classification",
                "target_label_id": "experimental_condition",
            }]
            path.write_text(json.dumps(payload), encoding="utf-8")
            project = load_project_spec(path)
            item = project.trajectory_inputs()[0]
            self.assertEqual(item.metadata["label_experimental_condition"], "reference")
            self.assertIsNone(project.prediction_tasks[0].target_metadata_key)
            self.assertEqual(project.prediction_tasks[0].target_label_id, "experimental_condition")
            self.assertEqual(project_summary(project)["labels"][0]["source"], "external_annotation")
            self.assertEqual(_target_column(project, "condition_from_label"), "metadata_label_experimental_condition")

    def test_frame_or_derived_state_labels_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = self._write_project(directory)
            payload = json.loads(path.read_text(encoding="utf-8"))
            trajectory_id = "arbitrary-system:reference-condition:run-01"
            payload["labels"] = [{
                "label_id": "invalid_state",
                "label_type": "categorical",
                "level": "frame",
                "source": "derived_state",
                "assignments": {trajectory_id: "state_0"},
            }]
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "frame-level labels"):
                load_project_spec(path)
            payload["labels"][0]["level"] = "trajectory"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "derived_state"):
                load_project_spec(path)

    def test_label_target_requires_complete_trajectory_assignments(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = self._write_project(directory)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["labels"] = [{
                "label_id": "condition",
                "label_type": "categorical",
                "level": "trajectory",
                "source": "researcher_provided",
                "assignments": {"unknown:trajectory": "control"},
            }]
            payload["prediction_tasks"] = [{
                "task_id": "condition_task",
                "task_type": "classification",
                "target_label_id": "condition",
            }]
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown trajectory IDs"):
                load_project_spec(path)

    def test_openmm_protocol_generates_deterministic_reviewable_jobs(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = self._write_project(directory)
            payload = json.loads(path.read_text(encoding="utf-8"))
            protocol = {
                "engine": "openmm",
                "forcefield_files": ["amber14-all.xml", "implicit/gbn2.xml"],
                "temperature_kelvin": 310,
                "production_steps": 5000,
                "report_interval_steps": 100,
                "seed": 101,
            }
            payload["systems"][0]["simulation_protocol"] = protocol
            path.write_text(json.dumps(payload), encoding="utf-8")
            project = load_project_spec(path)
            plan = build_openmm_plan(project, directory / "generated")
            self.assertEqual(plan["job_count"], 1)
            job = plan["jobs"][0]
            self.assertEqual(job["protocol"]["temperature_kelvin"], 310.0)
            self.assertEqual(job["protocol"]["seed"], 101)
            self.assertTrue(job["trajectory"].endswith("simulations/arbitrary-system/reference-condition/run-01/trajectory.dcd"))
            plan_path, launcher = write_openmm_plan(project, directory / "generated")
            self.assertTrue(plan_path.is_file())
            self.assertTrue(launcher.is_file())
            self.assertIn("openmm_plan.json", launcher.read_text(encoding="utf-8"))

    def test_openmm_protocol_rejects_invalid_execution_settings(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = self._write_project(directory)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["systems"][0]["simulation_protocol"] = {"engine": "openmm", "production_steps": 0}
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "production_steps"):
                load_project_spec(path)


if __name__ == "__main__":
    unittest.main()
