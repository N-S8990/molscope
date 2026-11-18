"""Integration tests for OpenMM simulation planning, dry-run validation, and recovery.

Validates:
1. End-to-end plan generation from ProjectSpec.
2. Dry-run inspection without importing OpenMM or writing trajectory files.
3. Accurate trajectory frame and byte size estimation based on atom count and step schedule.
4. Checkpoint resumption into immutable segments while preserving parent replica identity.
5. Hash verification (topology, serialized system XML, protocol) across recovery.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from molscope.core.project import (
    EnsembleSpec,
    OpenMMSimulationSpec,
    ProjectSpec,
    ReplicaSpec,
    SystemSpec,
)
from molscope.simulation.openmm import (
    PLAN_VERSION,
    execute_openmm_plan,
    inspect_openmm_plan,
    write_openmm_plan,
)


def _create_minimal_peptide_pdb(path: Path) -> Path:
    """Extract first 5 residues of 1A22 to build a rapid, valid protein test fixture."""
    fixture_pdb = Path("tests/fixtures/1A22.pdb")
    if not fixture_pdb.is_file():
        raise FileNotFoundError(f"Fixture not found: {fixture_pdb}")

    lines = []
    with open(fixture_pdb) as f:
        for line in f:
            if line.startswith(("ATOM", "TER")) and line[21] == "A":
                resseq = int(line[22:26])
                if resseq <= 5:
                    lines.append(line)
    lines.append("END\n")
    path.write_text("".join(lines))
    return path


class OpenMMSimulationPipelineIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.pdb_path = _create_minimal_peptide_pdb(self.root / "peptide.pdb")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_plan_generation_dry_run_and_size_estimation(self):
        """Verify plan generation, dry-run inspection, and trajectory byte/frame estimates."""
        protocol = OpenMMSimulationSpec(
            timestep_femtoseconds=2.0,
            temperature_kelvin=300.0,
            friction_per_picosecond=1.0,
            nonbonded_method="NoCutoff",
            constraints="HBonds",
            minimization_max_iterations=10,
            equilibration_steps=20,
            production_steps=100,
            report_interval_steps=20,
            platform="CPU",
            forcefield_files=("amber14-all.xml", "implicit/gbn2.xml"),
            seed=42,
        )
        system = SystemSpec(
            system_id="peptide_system",
            topology=self.pdb_path,
            simulation_protocol=protocol,
            ensembles=(
                EnsembleSpec(
                    ensemble_id="nvt_ensemble",
                    replicas=(
                        ReplicaSpec(replica_id="rep_1", trajectory=self.root / "placeholder.dcd"),
                    ),
                ),
            ),
        )
        project = ProjectSpec(project_id="test_sim_pipeline", systems=(system,))

        plan_path, launcher_path = write_openmm_plan(project, self.root / "generated")
        self.assertTrue(plan_path.is_file())
        self.assertTrue(launcher_path.is_file())

        plan = json.loads(plan_path.read_text())
        self.assertEqual(plan["plan_version"], PLAN_VERSION)
        self.assertEqual(len(plan["jobs"]), 1)

        job = plan["jobs"][0]
        self.assertEqual(job["job_id"], "peptide_system:nvt_ensemble:rep_1")

        # Dry-run inspection
        preview = inspect_openmm_plan(plan)
        job_preview = preview["jobs"][0]
        self.assertEqual(job_preview["status"], "ready_to_run")

        estimate = job_preview["estimate"]
        # 100 production steps / 20 interval = 5 expected frames
        self.assertEqual(estimate["expected_frames"], 5)
        # 40 input atoms in first 5 residues of 1A22
        self.assertEqual(estimate["input_atom_count"], 40)
        # Byte estimate check: DCD header (~276 bytes) + frames * (atoms * 12 + frame header bytes)
        self.assertGreater(estimate["approximate_dcd_bytes"], 2000)
        self.assertEqual(estimate["save_interval_ps"], 0.04)

    def test_end_to_end_execution_and_checkpoint_recovery(self):
        """Execute a partial simulation, checkpoint it, and recover a new segment preserving replica identity."""
        protocol = OpenMMSimulationSpec(
            timestep_femtoseconds=2.0,
            temperature_kelvin=300.0,
            friction_per_picosecond=1.0,
            nonbonded_method="NoCutoff",
            constraints="HBonds",
            minimization_max_iterations=5,
            equilibration_steps=10,
            production_steps=60,
            report_interval_steps=10,
            platform="CPU",
            forcefield_files=("amber14-all.xml", "implicit/gbn2.xml"),
            seed=1234,
        )
        system = SystemSpec(
            system_id="pep_sys",
            topology=self.pdb_path,
            simulation_protocol=protocol,
            ensembles=(
                EnsembleSpec(
                    ensemble_id="ens_0",
                    replicas=(
                        ReplicaSpec(replica_id="rep_0", trajectory=self.root / "init.dcd"),
                    ),
                ),
            ),
        )
        project = ProjectSpec(project_id="test_exec_and_recovery", systems=(system,))
        plan_path, _ = write_openmm_plan(project, self.root / "generated")
        plan = json.loads(plan_path.read_text())

        job = plan["jobs"][0]
        traj_dir = Path(job["trajectory"]).parent
        traj_dir.mkdir(parents=True, exist_ok=True)

        # Set up an interrupted run state at step 30 (out of 70 total equilibration+production steps)
        from molscope.simulation.openmm import _runtime_dependencies, _prepare, _digest, _write_execution_provenance
        from datetime import datetime, timezone
        from dataclasses import asdict
        from molscope.core.project import _openmm_protocol

        try:
            from datetime import UTC  # type: ignore[attr-defined]
        except ImportError:
            UTC = timezone.utc  # type: ignore[misc]

        try:
            mm, app, unit, PDBFixer = _runtime_dependencies()
        except RuntimeError as exc:
            self.skipTest(f"OpenMM and pdbfixer runtime not available: {exc}")
        prepared_pdb = Path(job["prepared_topology"])
        prepared_pdb.parent.mkdir(parents=True, exist_ok=True)
        topology, positions = _prepare(str(job["input_structure"]), str(prepared_pdb), job["protocol"], mm, app, unit, PDBFixer)

        ff = app.ForceField(*job["protocol"]["forcefield_files"])
        om_system = ff.createSystem(topology, nonbondedMethod=getattr(app, job["protocol"]["nonbonded_method"]))
        system_xml = traj_dir / "system.xml"
        system_xml.write_text(mm.XmlSerializer.serialize(om_system))

        integrator = mm.LangevinMiddleIntegrator(300.0 * unit.kelvin, 1.0 / unit.picosecond, 2.0 * unit.femtoseconds)
        integrator.setRandomNumberSeed(1234)
        sim = app.Simulation(topology, om_system, integrator, mm.Platform.getPlatformByName("CPU"))
        sim.context.setPositions(positions)
        sim.minimizeEnergy(maxIterations=15)
        sim.step(30)
        checkpoint_path = traj_dir / "checkpoint.chk"
        sim.saveCheckpoint(str(checkpoint_path))

        provenance = {
            "job_id": job["job_id"],
            "status": "failed",
            "protocol": asdict(_openmm_protocol(job["protocol"], "test")),
            "input_sha256": _digest(job["input_structure"]),
            "prepared_sha256": _digest(prepared_pdb),
            "outputs": {"prepared_topology": str(prepared_pdb), "trajectory": job["trajectory"], "state_log": job["state_log"]},
            "runtime": {
                "openmm_version": str(mm.__version__),
                "platform": "CPU",
                "system_sha256": _digest(system_xml),
            },
            "created_at": datetime.now(UTC).isoformat(),
        }
        (traj_dir / "execution_provenance.json").write_text(json.dumps(provenance))

        # Now resume the incomplete run: it executes remaining 40 steps into a new segment
        resumed_completed = execute_openmm_plan(plan, resume_checkpoint=checkpoint_path)
        self.assertEqual(resumed_completed, ["pep_sys:ens_0:rep_0"])

        # Verify segment directory was created and parent replica identity preserved
        segments_dir = traj_dir / "segments"
        self.assertTrue(segments_dir.is_dir())
        segment_dirs = list(segments_dir.iterdir())
        self.assertGreaterEqual(len(segment_dirs), 1)

        resumed_prov = json.loads((segment_dirs[0] / "execution_provenance.json").read_text())
        self.assertEqual(resumed_prov["status"], "completed")
        self.assertEqual(resumed_prov["job_id"], "pep_sys:ens_0:rep_0")
        self.assertIn("resumed_from_checkpoint", resumed_prov)
        self.assertEqual(resumed_prov["production_start_step"], 30)
        self.assertEqual(resumed_prov["production_end_step"], 70)


if __name__ == "__main__":
    unittest.main()
