"""Generate and explicitly execute reproducible OpenMM simulation plans.

Generation never runs molecular dynamics.  A written plan can be reviewed,
archived, and executed later with this module when OpenMM and PDBFixer are
installed.  The runner deliberately imports those optional runtime packages
only when execution is requested.
"""

from __future__ import annotations

import argparse
import json
import hashlib
import shutil
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from molscope.core.project import ProjectSpec, load_project_spec, _openmm_protocol


PLAN_VERSION = "1"


def _digest(path: str | Path) -> str | None:
    source = Path(path)
    if not source.is_file():
        return None
    digest = hashlib.sha256()
    with source.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _resume_job(job: dict[str, Any], checkpoint: str | Path, root: Path) -> dict[str, Any]:
    """Authenticate a recovery source and allocate a fresh immutable segment."""
    checkpoint = Path(checkpoint).resolve()
    if root not in checkpoint.parents or checkpoint.name != 'checkpoint.chk':
        raise ValueError('Resume checkpoint must be checkpoint.chk inside generated_output_root.')
    provenance_path = checkpoint.parent / 'execution_provenance.json'
    if not checkpoint.is_file() or not provenance_path.is_file():
        raise ValueError('Resume requires an existing checkpoint and execution provenance.')
    previous = json.loads(provenance_path.read_text())
    protocol = asdict(_openmm_protocol(job['protocol'], 'resume protocol'))
    if previous.get('job_id') != job['job_id'] or previous.get('protocol') != json.loads(json.dumps(protocol)):
        raise ValueError('Resume job or protocol differs from checkpoint provenance.')
    if previous.get('input_sha256') != _digest(job['input_structure']) or not previous.get('input_sha256'):
        raise ValueError('Resume input structure differs from checkpoint provenance.')
    prepared = Path(previous.get('outputs', {}).get('prepared_topology', '')).resolve()
    if root not in prepared.parents or not previous.get('prepared_sha256') or _digest(prepared) != previous['prepared_sha256']:
        raise ValueError('Resume prepared topology differs from checkpoint provenance.')
    system_path = checkpoint.parent / 'system.xml'
    if not previous.get('runtime', {}).get('system_sha256') or _digest(system_path) != previous['runtime']['system_sha256']:
        raise ValueError('Resume serialized system differs from checkpoint provenance.')
    segment = Path(job['trajectory']).parent / 'segments' / uuid.uuid4().hex
    return {**job, 'prepared_topology': str(segment / 'prepared.pdb'),
            'trajectory': str(segment / 'trajectory.dcd'), 'state_log': str(segment / 'state.tsv'),
            '_resume_checkpoint': str(checkpoint), '_resume_prepared': str(prepared),
            '_resume_system': str(system_path),
            '_resume_runtime': previous.get('runtime', {})}


def _output_estimate(job: dict[str, Any]) -> dict[str, Any]:
    protocol = _openmm_protocol(job['protocol'], 'estimate protocol')
    source = Path(job['input_structure'])
    atoms = 0
    if source.is_file():
        with source.open(errors='replace') as handle:
            atoms = sum(line.startswith(('ATOM  ', 'HETATM')) for line in handle)
    start = protocol.equilibration_steps
    frames = (start + protocol.production_steps) // protocol.report_interval_steps - start // protocol.report_interval_steps
    return {'production_duration_ns': protocol.production_steps * protocol.timestep_femtoseconds / 1e6,
            'save_interval_ps': protocol.report_interval_steps * protocol.timestep_femtoseconds / 1000,
            'expected_frames': frames, 'input_atom_count': atoms,
            'approximate_dcd_bytes': frames * (12 * atoms + 80) + 1024 if atoms else None,
            'estimate_note': 'Uncompressed coordinate estimate using input atoms; preparation changes atom count. Logs/checkpoints excluded.'}


def build_openmm_plan(project: ProjectSpec, output_dir: str | Path) -> dict[str, Any]:
    """Build a deterministic, reviewable job plan for configured systems.

    Output paths are rooted under ``output_dir`` and never reuse declared
    analysis-trajectory paths, preventing plan generation from overwriting
    input data.
    """
    destination = Path(output_dir).resolve()
    jobs: list[dict[str, Any]] = []
    for system in project.systems:
        protocol = system.simulation_protocol
        if protocol is None:
            continue
        for ensemble in system.ensembles:
            for replica in ensemble.replicas:
                job_index = len(jobs)
                job_dir = destination / "simulations" / system.system_id / ensemble.ensemble_id / replica.replica_id
                seed = protocol.seed + job_index if protocol.seed is not None else None
                jobs.append({
                    "job_id": f"{system.system_id}:{ensemble.ensemble_id}:{replica.replica_id}",
                    "input_structure": str(system.topology),
                    "prepared_topology": str(job_dir / "prepared.pdb"),
                    "trajectory": str(job_dir / "trajectory.dcd"),
                    "state_log": str(job_dir / "state.tsv"),
                    "protocol": {**asdict(protocol), "seed": seed},
                    "metadata": {**system.metadata, **ensemble.metadata, **replica.metadata},
                })
    if not jobs:
        raise ValueError("No system declares simulation_protocol; no OpenMM plan can be generated.")
    return {
        "plan_version": PLAN_VERSION,
        "project_id": project.project_id,
        "project_source": str(project.source_path) if project.source_path else None,
        "generated_output_root": str(destination),
        "job_count": len(jobs),
        "jobs": jobs,
    }


def write_openmm_plan(project: ProjectSpec, output_dir: str | Path) -> tuple[Path, Path]:
    """Write a plan and a small, reviewable launcher without executing jobs."""
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    plan_path = destination / "openmm_plan.json"
    plan_path.write_text(json.dumps(build_openmm_plan(project, destination), indent=2) + "\n", encoding="utf-8")
    launcher_path = destination / "run_openmm_plan.py"
    launcher_path.write_text(
        "#!/usr/bin/env python3\n"
        "\"\"\"Explicitly execute the reviewed OpenMM plan next to this file.\"\"\"\n"
        "from pathlib import Path\n"
        "from molscope.openmm_plan import main\n\n"
        "if __name__ == '__main__':\n"
        "    main(['--plan', str(Path(__file__).with_name('openmm_plan.json')), '--run'])\n",
        encoding="utf-8",
    )
    launcher_path.chmod(launcher_path.stat().st_mode | 0o111)
    return plan_path, launcher_path


def _runtime_dependencies():
    try:
        import openmm as mm
        import openmm.app as app
        import openmm.unit as unit
        from pdbfixer import PDBFixer
    except ImportError as error:  # pragma: no cover - depends on optional local runtime
        raise RuntimeError("OpenMM execution requires openmm and pdbfixer; generate/review plans without them.") from error
    return mm, app, unit, PDBFixer


def _selected_jobs(plan: dict[str, Any], selected_job_ids: set[str] | None) -> list[dict[str, Any]]:
    """Validate the portable plan surface before optional runtime imports.

    This validation intentionally does not evaluate force-field suitability;
    that requires a researcher-reviewed system and an installed OpenMM stack.
    It does ensure a reviewed plan cannot redirect generated outputs outside
    its declared output root or overwrite prior simulation artifacts.
    """
    if plan.get("plan_version") != PLAN_VERSION:
        raise ValueError(f"Unsupported OpenMM plan version {plan.get('plan_version')!r}.")
    jobs = plan.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("OpenMM plan must contain at least one job.")
    root_value = plan.get("generated_output_root")
    if not isinstance(root_value, str) or not root_value:
        raise ValueError("OpenMM plan is missing generated_output_root.")
    root = Path(root_value).resolve()
    identifiers = [job.get("job_id") for job in jobs if isinstance(job, dict)]
    if len(identifiers) != len(jobs) or any(not isinstance(item, str) or not item for item in identifiers):
        raise ValueError("Every OpenMM plan job requires a non-empty job_id.")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("OpenMM plan job IDs must be unique.")
    unknown = selected_job_ids - set(identifiers) if selected_job_ids else set()
    if unknown:
        raise ValueError(f"Unknown job IDs: {sorted(unknown)}")
    selected = [job for job in jobs if selected_job_ids is None or job["job_id"] in selected_job_ids]
    if not selected:
        raise ValueError("No jobs selected from the plan.")
    required = {"job_id", "input_structure", "prepared_topology", "trajectory", "state_log", "protocol"}
    output_paths: set[Path] = set()
    inputs = {Path(job['input_structure']).resolve() for job in jobs if isinstance(job.get('input_structure'), str)}
    for job in selected:
        if not isinstance(job, dict) or required - set(job):
            raise ValueError("Each selected OpenMM job must declare input_structure, generated paths, and protocol.")
        for output_key, output in _job_outputs(job).items():
            output = output.resolve()
            if output == root or root not in output.parents:
                raise ValueError(f"Job {job['job_id']!r} writes {output_key} outside generated_output_root.")
            if output in inputs or output in output_paths:
                raise ValueError(f"Job {job['job_id']!r} has an input/output collision: {output}")
            output_paths.add(output)
        if not isinstance(job["protocol"], dict):
            raise ValueError(f"Job {job['job_id']!r} protocol must be an object.")
        protocol = _openmm_protocol(job['protocol'], f"job {job['job_id']} protocol")
        if protocol.production_steps < protocol.report_interval_steps:
            raise ValueError("production_steps must be at least report_interval_steps to produce a trajectory.")
        if protocol.seed is not None and not 1 <= protocol.seed <= 2147483647:
            raise ValueError("OpenMM seed must be in [1, 2147483647].")
    return selected


def _job_outputs(job: dict[str, Any]) -> dict[str, Path]:
    directory = Path(job['trajectory']).parent
    return {
        **{key: Path(job[key]) for key in ('prepared_topology', 'trajectory', 'state_log')},
        'checkpoint': directory / 'checkpoint.chk',
        'system': directory / 'system.xml',
        'provenance': directory / 'execution_provenance.json',
        'integrity': directory / 'output_integrity.json',
        'analysis_project': directory / 'analysis_project.json',
    }


def inspect_openmm_plan(plan: dict[str, Any], *, selected_job_ids: set[str] | None = None) -> dict[str, Any]:
    """Return a no-import, no-write execution preview for a reviewed plan."""
    jobs = _selected_jobs(plan, selected_job_ids)
    preview = []
    for job in jobs:
        outputs = list(_job_outputs(job).values())
        provenance = Path(job["trajectory"]).parent / "execution_provenance.json"
        existing = [str(path) for path in [*outputs, provenance] if path.exists()]
        preview.append({
            "job_id": job["job_id"],
            "input_structure": job["input_structure"],
            "outputs": {key: job[key] for key in ("prepared_topology", "trajectory", "state_log")},
            "execution_provenance": str(provenance),
            "status": "blocked_existing_output" if existing else "blocked_missing_input" if not Path(job['input_structure']).is_file() else "ready_to_run",
            "existing_outputs": existing,
            "estimate": _output_estimate(job),
        })
    return {"plan_version": PLAN_VERSION, "project_id": plan.get("project_id"), "jobs": preview}


def _write_execution_provenance(job: dict[str, Any], *, status: str, started_at: str, error: Exception | None = None) -> None:
    path = Path(job["trajectory"]).parent / "execution_provenance.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "plan_version": PLAN_VERSION,
        "job_id": job["job_id"],
        "status": status,
        "started_at_utc": started_at,
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "input_structure": job["input_structure"],
        "outputs": {key: job[key] for key in ("prepared_topology", "trajectory", "state_log")},
        "protocol": job["protocol"],
        "input_sha256": _digest(job['input_structure']),
        "prepared_sha256": _digest(job['prepared_topology']),
        "runtime": job.get('_runtime', {}),
        "resumed_from_checkpoint": job.get('_resume_checkpoint'),
        "production_start_step": job.get('_production_start_step'),
        "production_end_step": job.get('_production_end_step'),
        "preparation_policy": {"missing_residues": "not_rebuilt", "nonstandard_residues": "replace_with_standard", "missing_atoms": "add", "hydrogens": "add_at_declared_ph", "waters": "removed", "heterogens": "retained" if job['protocol'].get('keep_heterogens', False) else "removed"},
    }
    if error is not None:
        payload["error"] = {"type": type(error).__name__, "message": str(error)}
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _prepare(input_structure: str, prepared_topology: str, protocol: dict[str, Any], mm, app, unit, PDBFixer) -> tuple[Any, Any]:
    fixer = PDBFixer(filename=input_structure)
    fixer.findMissingResidues()
    fixer.missingResidues = {}
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    if not protocol["keep_heterogens"]:
        fixer.removeHeterogens(False)
    else:
        waters = [residue for residue in fixer.topology.residues() if residue.name in {"HOH", "WAT", "TIP3", "H2O"}]
        if waters:
            modeller = app.Modeller(fixer.topology, fixer.positions)
            modeller.delete(waters)
            fixer.topology, fixer.positions = modeller.topology, modeller.positions
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(protocol["ph"])
    prepared = Path(prepared_topology)
    prepared.parent.mkdir(parents=True, exist_ok=True)
    with prepared.open("w", encoding="utf-8") as handle:
        app.PDBFile.writeFile(fixer.topology, fixer.positions, handle)
    return fixer.topology, fixer.positions


def _validate_outputs_and_write_project(job: dict[str, Any]) -> None:
    from molscope.io.validation import inspect_md_inputs

    paths = _job_outputs(job)
    for key in ('prepared_topology', 'trajectory', 'state_log', 'checkpoint'):
        if not paths[key].is_file() or paths[key].stat().st_size == 0:
            raise RuntimeError(f"Simulation output is missing or empty: {key}")
    report = inspect_md_inputs(job['prepared_topology'], job['trajectory'])
    paths['integrity'].write_text(json.dumps(report.to_dict(), indent=2) + '\n', encoding='utf-8')
    if report.errors:
        raise RuntimeError(f"Simulation trajectory integrity failed: {report.errors}")
    project = {
        'schema_version': '1', 'project_id': 'simulation-output',
        'systems': [{'system_id': 'simulated-system', 'topology': str(paths['prepared_topology'].resolve()),
                     'metadata': {'source_job_id': job['job_id']},
                     'ensembles': [{'ensemble_id': 'production', 'replicas': [
                         {'replica_id': 'replica-1', 'trajectory': str(paths['trajectory'].resolve())}
                     ]}]}],
    }
    paths['analysis_project'].write_text(json.dumps(project, indent=2) + '\n', encoding='utf-8')


def execute_openmm_plan(plan: dict[str, Any], *, selected_job_ids: set[str] | None = None, resume_checkpoint: str | Path | None = None) -> list[str]:
    """Run explicitly selected jobs from a reviewed plan.

    This function has no implicit retry, overwrite, or scheduling behaviour.
    Existing outputs are rejected so execution remains deliberate.
    """
    jobs = _selected_jobs(plan, selected_job_ids)
    if resume_checkpoint is not None:
        if len(jobs) != 1:
            raise ValueError('Resume requires exactly one selected job.')
        jobs = [_resume_job(jobs[0], resume_checkpoint, Path(plan['generated_output_root']).resolve())]
        plan = {**plan, 'jobs': jobs}
    preview = inspect_openmm_plan(plan, selected_job_ids=selected_job_ids)
    blocked = [job for job in preview["jobs"] if job["status"] == "blocked_existing_output"]
    if blocked:
        raise FileExistsError(f"Refusing to overwrite existing simulation output for {blocked[0]['job_id']!r}.")
    if any(job['status'] == 'blocked_missing_input' for job in preview['jobs']):
        raise FileNotFoundError("Selected OpenMM job input structure is missing.")
    mm, app, unit, PDBFixer = _runtime_dependencies()
    completed = []
    nonbonded = {name: getattr(app, name) for name in ("NoCutoff", "CutoffNonPeriodic", "CutoffPeriodic", "PME", "Ewald", "LJPME")}
    constraints = {"None": None, "HBonds": app.HBonds, "AllBonds": app.AllBonds, "HAngles": app.HAngles}
    for job in jobs:
        started_at = datetime.now(UTC).isoformat()
        try:
            protocol = asdict(_openmm_protocol(job["protocol"], "execution protocol"))
            job = {**job, 'protocol': protocol}
            if resume_checkpoint is not None:
                prepared = app.PDBFile(job['_resume_prepared'])
                topology, positions = prepared.topology, prepared.positions
            else:
                topology, positions = _prepare(job["input_structure"], job["prepared_topology"], protocol, mm, app, unit, PDBFixer)
            if resume_checkpoint is not None:
                serialized_system = Path(job['_resume_system']).read_text()
                system = mm.XmlSerializer.deserialize(serialized_system)
            else:
                forcefield = app.ForceField(*protocol["forcefield_files"])
                system = forcefield.createSystem(topology, nonbondedMethod=nonbonded[protocol["nonbonded_method"]], constraints=constraints[protocol["constraints"]])
                serialized_system = mm.XmlSerializer.serialize(system)
            integrator = mm.LangevinMiddleIntegrator(protocol["temperature_kelvin"] * unit.kelvin, protocol["friction_per_picosecond"] / unit.picosecond, protocol["timestep_femtoseconds"] * unit.femtoseconds)
            if protocol['seed'] is not None:
                integrator.setRandomNumberSeed(protocol['seed'])
            platform = mm.Platform.getPlatformByName(protocol["platform"]) if protocol["platform"] else None
            simulation = app.Simulation(topology, system, integrator, platform)
            runtime_version = getattr(mm, '__version__', None)
            runtime_platform = simulation.context.getPlatform().getName()
            job['_runtime'] = {'openmm_version': runtime_version if isinstance(runtime_version, str) else None,
                               'platform': runtime_platform if isinstance(runtime_platform, str) else None,
                               'system_sha256': hashlib.sha256(serialized_system.encode()).hexdigest() if isinstance(serialized_system, str) else None}
            if resume_checkpoint is not None:
                if not all(job['_runtime'].values()) or job['_runtime'] != job['_resume_runtime']:
                    raise ValueError('Resume requires the same recorded OpenMM version and platform; hardware must also be checkpoint-compatible.')
                simulation.loadCheckpoint(job['_resume_checkpoint'])
                remaining_steps = protocol['equilibration_steps'] + protocol['production_steps'] - simulation.currentStep
                if simulation.currentStep < protocol['equilibration_steps'] or remaining_steps <= 0:
                    raise ValueError('Checkpoint is outside the unfinished production interval; completed runs cannot be resumed.')
                interval = protocol['report_interval_steps']
                if (simulation.currentStep + remaining_steps) // interval == simulation.currentStep // interval:
                    raise ValueError('Remaining production has no reportable frames.')
                Path(job['prepared_topology']).parent.mkdir(parents=True, exist_ok=False)
                shutil.copyfile(job['_resume_prepared'], job['prepared_topology'])
                shutil.copyfile(job['_resume_system'], _job_outputs(job)['system'])
                job['_production_start_step'] = simulation.currentStep
            else:
                simulation.context.setPositions(positions)
                simulation.minimizeEnergy(maxIterations=protocol["minimization_max_iterations"])
                if protocol["seed"] is None:
                    simulation.context.setVelocitiesToTemperature(protocol["temperature_kelvin"] * unit.kelvin)
                else:
                    simulation.context.setVelocitiesToTemperature(protocol["temperature_kelvin"] * unit.kelvin, protocol["seed"])
                if protocol["equilibration_steps"]:
                    simulation.step(protocol["equilibration_steps"])
                remaining_steps = protocol['production_steps']
                job['_production_start_step'] = protocol['equilibration_steps']
                if isinstance(serialized_system, str):
                    _job_outputs(job)['system'].write_text(serialized_system)
            job['_production_end_step'] = protocol['equilibration_steps'] + protocol['production_steps']
            _write_execution_provenance(job, status='running', started_at=started_at)
            simulation.reporters.append(app.DCDReporter(job["trajectory"], protocol["report_interval_steps"]))
            simulation.reporters.append(app.StateDataReporter(job["state_log"], protocol["report_interval_steps"], step=True, potentialEnergy=True, temperature=True, separator="\t"))
            simulation.reporters.append(app.CheckpointReporter(str(_job_outputs(job)['checkpoint']), protocol['report_interval_steps']))
            simulation.step(remaining_steps)
            simulation.saveCheckpoint(str(_job_outputs(job)['checkpoint']))
            # Ensure reporter buffers are visible to the integrity reader.
            for reporter in simulation.reporters:
                stream = getattr(reporter, '_out', None)
                if hasattr(stream, 'flush'):
                    stream.flush()
            _validate_outputs_and_write_project(job)
        except Exception as error:
            _write_execution_provenance(job, status="failed", started_at=started_at, error=error)
            raise
        _write_execution_provenance(job, status="completed", started_at=started_at)
        completed.append(job["job_id"])
    return completed


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Execute an explicitly generated MolScope OpenMM plan.")
    parser.add_argument("--plan", required=True, help="Path to reviewed openmm_plan.json.")
    parser.add_argument("--job", action="append", help="Optional job ID; repeat to run selected jobs only.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print selected jobs without importing OpenMM or running them.")
    parser.add_argument("--run", action="store_true", help="Explicitly authorize execution after reviewing the plan or a dry run.")
    parser.add_argument('--resume-checkpoint', help='Recover one unfinished job into a new segment using checkpoint.chk and its provenance.')
    args = parser.parse_args(argv)
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    selected = set(args.job) if args.job else None
    if args.dry_run:
        preview_plan = plan
        if args.resume_checkpoint:
            jobs = _selected_jobs(plan, selected)
            if len(jobs) != 1:
                raise ValueError('Resume requires exactly one selected job.')
            preview_plan = {**plan, 'jobs': [_resume_job(jobs[0], args.resume_checkpoint, Path(plan['generated_output_root']).resolve())]}
        preview = inspect_openmm_plan(preview_plan, selected_job_ids=selected)
        if args.resume_checkpoint:
            preview['resume_note'] = 'Identity validated. Runtime compatibility and remaining steps are checked when loading the checkpoint; estimates describe the full original protocol.'
        print(json.dumps({"plan": str(Path(args.plan).resolve()), **preview}, indent=2))
        return
    if not args.run:
        raise ValueError("Refusing to execute without --run. Review with --dry-run first, then explicitly authorize execution.")
    print(json.dumps({"completed_jobs": execute_openmm_plan(plan, selected_job_ids=selected, resume_checkpoint=args.resume_checkpoint)}, indent=2))


if __name__ == "__main__":
    main()
