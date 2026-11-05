import contextlib
import io
import json
import hashlib
from dataclasses import asdict
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace
from pathlib import Path

from molscope.openmm_plan import PLAN_VERSION, execute_openmm_plan, inspect_openmm_plan, main
from molscope.project_spec import _openmm_protocol


class OpenMMExecutionTests(unittest.TestCase):
    def _recovery(self, directory):
        plan = self._plan(directory)
        job = plan['jobs'][0]
        prepared = Path(job['prepared_topology'])
        prepared.parent.mkdir(parents=True)
        prepared.write_text('END\n')
        checkpoint = prepared.parent / 'checkpoint.chk'
        checkpoint.write_bytes(b'checkpoint')
        (prepared.parent / 'system.xml').write_text('system')
        digest = hashlib.sha256(b'END\n').hexdigest()
        provenance = {'job_id': job['job_id'], 'protocol': asdict(_openmm_protocol({}, 'test')),
                      'input_sha256': digest, 'prepared_sha256': digest,
                      'outputs': {'prepared_topology': str(prepared)},
                      'runtime': {'openmm_version': 'test', 'platform': 'CPU',
                                  'system_sha256': hashlib.sha256(b'system').hexdigest()}}
        (prepared.parent / 'execution_provenance.json').write_text(json.dumps(provenance))
        return plan, checkpoint

    def test_preview_estimates_report_frames_on_absolute_step_schedule(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = self._plan(Path(temporary))
            plan['jobs'][0]['protocol'] = {'equilibration_steps': 9, 'production_steps': 11, 'report_interval_steps': 10}
            estimate = inspect_openmm_plan(plan)['jobs'][0]['estimate']
            self.assertEqual(estimate['expected_frames'], 2)
            self.assertIsNone(estimate['approximate_dcd_bytes'])

    def test_resume_rejects_modified_input_and_protocol_before_runtime(self):
        for mutation in ('input', 'protocol', 'system'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                plan, checkpoint = self._recovery(Path(temporary))
                if mutation == 'input':
                    Path(plan['jobs'][0]['input_structure']).write_text('changed')
                elif mutation == 'protocol':
                    plan['jobs'][0]['protocol']['temperature_kelvin'] = 400
                else:
                    (checkpoint.parent / 'system.xml').write_text('changed')
                with patch('molscope.openmm_plan._runtime_dependencies') as runtime:
                    with self.assertRaisesRegex(ValueError, 'differs'):
                        execute_openmm_plan(plan, resume_checkpoint=checkpoint)
                    runtime.assert_not_called()

    def test_resume_uses_checkpoint_and_new_segment_without_reinitialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan, checkpoint = self._recovery(Path(temporary))
            original = checkpoint.read_bytes()
            mm, app = MagicMock(), MagicMock()
            mm.__version__ = 'test'
            mm.XmlSerializer.serialize.return_value = 'system'
            simulation = app.Simulation.return_value
            simulation.context.getPlatform.return_value.getName.return_value = 'CPU'
            simulation.currentStep = 2000
            simulation.reporters = []
            unit = SimpleNamespace(kelvin=1, picosecond=1, femtoseconds=1)
            with patch('molscope.openmm_plan._runtime_dependencies', return_value=(mm, app, unit, MagicMock())), patch('molscope.openmm_plan._prepare') as prepare, patch('molscope.openmm_plan._validate_outputs_and_write_project') as validate:
                execute_openmm_plan(plan, resume_checkpoint=checkpoint)
            prepare.assert_not_called()
            simulation.minimizeEnergy.assert_not_called()
            simulation.context.setVelocitiesToTemperature.assert_not_called()
            simulation.loadCheckpoint.assert_called_once_with(str(checkpoint.resolve()))
            simulation.step.assert_called_once_with(199000)
            resumed_job = validate.call_args.args[0]
            self.assertIn('/segments/', resumed_job['trajectory'])
            self.assertEqual(checkpoint.read_bytes(), original)
            self.assertNotEqual(resumed_job['trajectory'], plan['jobs'][0]['trajectory'])

    def test_resume_rejects_incompatible_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan, checkpoint = self._recovery(Path(temporary))
            mm, app = MagicMock(), MagicMock()
            mm.__version__ = 'different'
            unit = SimpleNamespace(kelvin=1, picosecond=1, femtoseconds=1)
            with patch('molscope.openmm_plan._runtime_dependencies', return_value=(mm, app, unit, MagicMock())):
                with self.assertRaisesRegex(ValueError, 'same recorded OpenMM'):
                    execute_openmm_plan(plan, resume_checkpoint=checkpoint)
            app.Simulation.return_value.loadCheckpoint.assert_not_called()

    def test_resume_dry_run_validates_without_creating_segments_or_importing_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            plan, checkpoint = self._recovery(directory)
            path = directory / 'plan.json'
            path.write_text(json.dumps(plan))
            with patch('molscope.openmm_plan._runtime_dependencies') as runtime, contextlib.redirect_stdout(io.StringIO()) as output:
                main(['--plan', str(path), '--resume-checkpoint', str(checkpoint), '--dry-run'])
            runtime.assert_not_called()
            self.assertEqual(json.loads(output.getvalue())['jobs'][0]['status'], 'ready_to_run')
            self.assertFalse((checkpoint.parent / 'segments').exists())
            with self.assertRaisesRegex(ValueError, 'without --run'):
                main(['--plan', str(path), '--resume-checkpoint', str(checkpoint)])

    def test_resume_rejects_completed_checkpoint_without_stepping(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan, checkpoint = self._recovery(Path(temporary))
            mm, app = MagicMock(), MagicMock()
            mm.__version__ = 'test'
            mm.XmlSerializer.serialize.return_value = 'system'
            simulation = app.Simulation.return_value
            simulation.context.getPlatform.return_value.getName.return_value = 'CPU'
            simulation.currentStep = 201000
            unit = SimpleNamespace(kelvin=1, picosecond=1, femtoseconds=1)
            with patch('molscope.openmm_plan._runtime_dependencies', return_value=(mm, app, unit, MagicMock())):
                with self.assertRaisesRegex(ValueError, 'completed runs cannot be resumed'):
                    execute_openmm_plan(plan, resume_checkpoint=checkpoint)
            simulation.step.assert_not_called()

    def _plan(self, directory: Path) -> dict:
        (directory / 'input.pdb').write_text('END\n', encoding='utf-8')
        root = directory / "generated"
        job_dir = root / "simulations" / "system" / "ensemble" / "replica"
        return {
            "plan_version": PLAN_VERSION,
            "project_id": "test-project",
            "generated_output_root": str(root),
            "jobs": [{
                "job_id": "system:ensemble:replica",
                "input_structure": str(directory / "input.pdb"),
                "prepared_topology": str(job_dir / "prepared.pdb"),
                "trajectory": str(job_dir / "trajectory.dcd"),
                "state_log": str(job_dir / "state.tsv"),
                "protocol": {},
            }],
        }

    def test_dry_run_preview_is_no_import_and_no_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            plan = self._plan(directory)
            plan_path = directory / "openmm_plan.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                main(["--plan", str(plan_path), "--dry-run"])
            payload = json.loads(stream.getvalue())
            self.assertEqual(payload["jobs"][0]["status"], "ready_to_run")
            self.assertFalse(Path(payload["jobs"][0]["execution_provenance"]).exists())

    def test_execution_requires_explicit_run_authorization(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            plan_path = directory / "openmm_plan.json"
            plan_path.write_text(json.dumps(self._plan(directory)), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "without --run"):
                main(["--plan", str(plan_path)])

    def test_existing_output_is_reported_before_optional_runtime_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            plan = self._plan(directory)
            trajectory = Path(plan["jobs"][0]["trajectory"])
            trajectory.parent.mkdir(parents=True)
            trajectory.write_bytes(b"existing")
            preview = inspect_openmm_plan(plan)
            self.assertEqual(preview["jobs"][0]["status"], "blocked_existing_output")
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                execute_openmm_plan(plan)

    def test_plan_cannot_redirect_generated_outputs_outside_its_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = self._plan(Path(temporary))
            plan["jobs"][0]["trajectory"] = str(Path(temporary) / "outside.dcd")
            with self.assertRaisesRegex(ValueError, "outside generated_output_root"):
                inspect_openmm_plan(plan)

    def test_missing_input_is_blocked_before_runtime_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = self._plan(Path(temporary))
            plan['jobs'][0]['input_structure'] = str(Path(temporary) / 'missing.pdb')
            self.assertEqual(inspect_openmm_plan(plan)['jobs'][0]['status'], 'blocked_missing_input')
            with self.assertRaises(FileNotFoundError):
                execute_openmm_plan(plan)

    def test_colliding_job_outputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = self._plan(Path(temporary))
            plan['jobs'].append({**plan['jobs'][0], 'job_id': 'another'})
            with self.assertRaisesRegex(ValueError, 'collision'):
                inspect_openmm_plan(plan)

    def test_invalid_production_interval_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = self._plan(Path(temporary))
            plan['jobs'][0]['protocol'] = {'production_steps': 5, 'report_interval_steps': 10}
            with self.assertRaisesRegex(ValueError, 'report_interval_steps'):
                inspect_openmm_plan(plan)

    def test_execution_seeds_integrator_and_validates_before_completed_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = self._plan(Path(temporary))
            plan['jobs'][0]['protocol'] = {'seed': 123}
            mm, app = MagicMock(), MagicMock()
            unit = SimpleNamespace(kelvin=1, picosecond=1, femtoseconds=1)
            simulation = app.Simulation.return_value
            simulation.reporters = []
            with patch('molscope.openmm_plan._runtime_dependencies', return_value=(mm, app, unit, MagicMock())), patch('molscope.openmm_plan._prepare', return_value=(MagicMock(), MagicMock())), patch('molscope.openmm_plan._validate_outputs_and_write_project') as validate:
                completed = execute_openmm_plan(plan)
            self.assertEqual(completed, ['system:ensemble:replica'])
            mm.LangevinMiddleIntegrator.return_value.setRandomNumberSeed.assert_called_once_with(123)
            simulation.saveCheckpoint.assert_called_once()
            validate.assert_called_once()
            provenance = Path(plan['jobs'][0]['trajectory']).parent / 'execution_provenance.json'
            self.assertEqual(json.loads(provenance.read_text())['status'], 'completed')

    def test_output_integrity_failure_is_recorded_as_failed(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = self._plan(Path(temporary))
            mm, app = MagicMock(), MagicMock()
            unit = SimpleNamespace(kelvin=1, picosecond=1, femtoseconds=1)
            app.Simulation.return_value.reporters = []
            with patch('molscope.openmm_plan._runtime_dependencies', return_value=(mm, app, unit, MagicMock())), patch('molscope.openmm_plan._prepare', return_value=(MagicMock(), MagicMock())), patch('molscope.openmm_plan._validate_outputs_and_write_project', side_effect=RuntimeError('invalid trajectory')):
                with self.assertRaisesRegex(RuntimeError, 'invalid trajectory'):
                    execute_openmm_plan(plan)
            provenance = Path(plan['jobs'][0]['trajectory']).parent / 'execution_provenance.json'
            self.assertEqual(json.loads(provenance.read_text())['status'], 'failed')


if __name__ == "__main__":
    unittest.main()
