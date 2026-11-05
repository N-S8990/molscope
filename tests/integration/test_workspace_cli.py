"""Public command compatibility and managed study output contracts."""
import importlib
import json
from pathlib import Path

import pytest

from molscope.cli.main import main
from molscope.core.workspace import initialize_study
from molscope.core.project import load_project_spec


def prepare_study(tmp_path):
    study = initialize_study(tmp_path / 'study', 'test_study')
    fixture = Path(__file__).parents[1] / 'fixtures' / '1A22.pdb'
    project_path = study / 'project.json'
    project = json.loads(project_path.read_text())
    system = project['systems'][0]
    system['topology'] = str(fixture)
    system['ensembles'][0]['replicas'][0]['trajectory'] = str(fixture)
    project_path.write_text(json.dumps(project))
    return study


def test_initializer_is_portable_and_never_overwrites(tmp_path):
    study = initialize_study(tmp_path / 'study')
    spec = load_project_spec(study / 'project.json')
    assert spec.systems[0].topology == study / 'inputs/topology.pdb'
    assert len(spec.validate_paths()) == 2
    with pytest.raises(FileExistsError):
        initialize_study(study)


def test_managed_run_records_resolved_inputs_and_groups_outputs(tmp_path, monkeypatch):
    from molscope.cli import main as cli
    study = prepare_study(tmp_path)

    def workflow(module, args):
        project = load_project_spec(args[args.index('--project') + 1])
        assert project.validate_paths() == []
        output = Path(args[args.index('--out-dir') + 1])
        output.mkdir()
        (output / 'feature_quality_metrics.csv').write_text('feature,quality\na,1\n')
        (output / 'research_report.md').write_text('![plot](state_populations.png)')
        (output / 'state_populations.png').write_bytes(b'fixture')
        (output / 'extraction_evidence.json').write_text('{}')
        (output / 'label_definitions.json').write_text('[]')
        print('workflow executed')

    monkeypatch.setattr(cli, 'dispatch', workflow)
    main(['run', '--study', str(study), '--run-id', 'run1'])
    run = study / 'runs/run1'
    provenance = json.loads((run / 'provenance.json').read_text())
    assert provenance['status'] == 'completed'
    assert len(provenance['inputs']) == 1
    assert (run / 'quality/feature_quality_metrics.csv').exists()
    assert (run / 'validation/extraction_evidence.json').exists()
    assert (run / 'validation/label_definitions.json').exists()
    assert not (run / 'predictive').exists()
    assert (run / 'report/state_populations.png').exists()
    assert (run / '.native/feature_quality_metrics.csv').read_text().startswith('feature')
    assert 'workflow executed' in (run / 'logs/run.log').read_text()
    with pytest.raises(FileExistsError):
        main(['run', '--study', str(study), '--run-id', 'run1'])


def test_failed_run_is_recorded_and_run_id_cannot_escape(tmp_path, monkeypatch):
    from molscope.cli import main as cli
    study = prepare_study(tmp_path)
    def fail(*args):
        raise RuntimeError('controlled failure')
    monkeypatch.setattr(cli, 'dispatch', fail)
    with pytest.raises(ValueError):
        main(['run', '--study', str(study), '--run-id', '../escape'])
    with pytest.raises(RuntimeError, match='controlled failure'):
        main(['run', '--study', str(study), '--run-id', 'failed'])
    record = json.loads((study / 'runs/failed/provenance.json').read_text())
    assert record['status'] == 'failed'
    assert 'controlled failure' in record['error']


@pytest.mark.parametrize('old,new', [
    ('pipeline', 'execution.pipeline'), ('openmm_plan', 'simulation.openmm'),
    ('plugins', 'features.plugins'), ('registries', 'features.registry'),
    ('project_spec', 'core.project'), ('streaming', 'execution.streaming'),
])
def test_legacy_modules_alias_canonical_implementations(old, new):
    assert importlib.import_module('molscope.' + old) is importlib.import_module('molscope.' + new)
