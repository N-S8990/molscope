"""Study directories and reproducible run records, separate from source code."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import platform

from molscope.core.project import load_project_spec
from molscope.io.storage import file_fingerprint


def write_json(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, default=str) + '\n')
    temporary.replace(path)


def initialize_study(destination, project_id='my_study') -> Path:
    """Create a fresh study skeleton; never overwrite an existing directory."""
    if not isinstance(project_id, str) or not project_id.strip():
        raise ValueError('project_id must be a nonempty string')
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=False)
    for name in ('inputs', 'preparation', 'simulations', 'runs', 'cache'):
        (root / name).mkdir()
    write_json(root / 'project.json', {
        'schema_version': '1', 'project_id': project_id,
        'systems': [{'system_id': 'system_01', 'topology': 'inputs/topology.pdb',
                     'ensembles': [{'ensemble_id': 'ensemble_01', 'replicas': [
                         {'replica_id': 'replica_01', 'trajectory': 'inputs/trajectory.dcd'}]}]}],
        'analysis': {'plugins': []},
    })
    (root / 'README.md').write_text(
        '# Study workspace\n\nEdit project.json and supply the declared inputs before running.\n'
        'Input paths may reference external files; no copying is required.\n'
        'Keep original inputs unchanged. Record preprocessing in preparation/.\n'
        'Use simulations/ for generated trajectories and checkpoints.\n'
        'Each run under runs/ preserves its configuration and provenance.\n'
        'cache/ contains disposable cached results.\n\n'
        'Run: python -m molscope run --study . --run-id first_run --extract-only\n'
    )
    return root


def begin_run(study, run_id):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', run_id or ''):
        raise ValueError('run_id must contain only letters, digits, dots, underscores and hyphens')
    study = Path(study).resolve()
    project = load_project_spec(study / 'project.json')
    missing = project.validate_paths()
    if missing:
        raise ValueError(f'Missing project inputs: {missing}')
    root = study / 'runs' / run_id
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / 'resolved_project.json', asdict(project))
    inputs = sorted({str(item.topology) for item in project.trajectory_inputs()} |
                    {str(item.trajectory) for item in project.trajectory_inputs()})
    provenance = {'project_id': project.project_id, 'run_id': run_id, 'status': 'running',
                  'started_at': datetime.now(timezone.utc).isoformat(),
                  'python': platform.python_version(),
                  'inputs': [file_fingerprint(path) for path in inputs]}
    write_json(root / 'provenance.json', provenance)
    return root, provenance


def organize_outputs(root: Path) -> dict[str, str]:
    """Group fresh native output files, retaining native links for report/session readers."""
    native = root / '.native'
    index = {}
    for source in sorted(native.iterdir()):
        name = source.name
        if source.suffix in {'.png', '.html', '.md'} or name.startswith('research_report'):
            stage = 'report'
        elif name.startswith(('supervised', 'predictive')):
            stage = 'predictive'
        elif name.startswith(('sam_mfp', 'pareto', 'rank_stability')):
            stage = 'prioritization'
        elif name.startswith(('feature_quality', 'quality')):
            stage = 'quality'
        elif name.startswith(('state', 'representation', 'pca', 'cluster')):
            stage = 'discovery'
        elif name.startswith(('extraction_evidence', 'input_integrity', 'system_profile', 'label_')):
            stage = 'validation'
        else:
            stage = 'features'
        destination = root / stage / name
        destination.parent.mkdir(exist_ok=True)
        source.rename(destination)
        source.symlink_to(Path('..') / stage / name, target_is_directory=destination.is_dir())
        index[name] = f'{stage}/{name}'
    # Native HTML assets/tables remain resolvable without rewriting report content.
    if (root / 'report').exists():
        for name, relative in index.items():
            link = root / 'report' / name
            if not link.exists():
                link.symlink_to(Path('..') / relative,
                                target_is_directory=(root / relative).is_dir())
    write_json(root / 'artifacts.json', index)
    return index
