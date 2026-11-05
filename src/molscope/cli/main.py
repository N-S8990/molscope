"""Unified CLI; scientific work remains in domain packages."""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import importlib
from pathlib import Path
import sys

COMMANDS = {
    'run': ('molscope.cli.run_project', 'Run the configured full project workflow'),
    'validate': ('molscope.cli.validate_inputs', 'Validate topology and trajectory inputs'),
    'profile': ('molscope.cli.inspect_system', 'Inspect molecular capabilities and feature applicability'),
    'plan': ('molscope.cli.generate_openmm_plan', 'Generate an OpenMM execution plan'),
    'simulate': ('molscope.simulation.openmm', 'Preview, execute or recover an OpenMM plan'),
    'extract': ('molscope.execution.streaming', 'Extract supported features in resumable chunks'),
    'analyze': ('molscope.execution.stream_analysis', 'Analyze completed chunks with bounded batches'),
    'stream-project': ('molscope.execution.stream_project', 'Run the limited streaming project workflow'),
    'report': ('molscope.cli.build_report', 'Build a research report from analysis tables'),
    'benchmark': ('molscope.cli.run_benchmark', 'Run a configured matched benchmark'),
    'check-environment': ('molscope.cli.check_environment', 'Inspect installed runtime capabilities'),
}


def dispatch(module, arguments):
    previous = sys.argv
    try:
        sys.argv = [f'molscope {module.rsplit(".", 1)[-1]}', *arguments]
        return importlib.import_module(module).main()
    finally:
        sys.argv = previous


def study_run(arguments):
    from molscope.core.workspace import begin_run, organize_outputs, write_json
    parser = argparse.ArgumentParser(description='Run a study into a fresh, recorded run directory')
    parser.add_argument('--study', required=True)
    parser.add_argument('--run-id', required=True)
    args, rest = parser.parse_known_args(arguments)
    if any(item.split('=')[0] in {'--project', '--out-dir'} for item in rest):
        parser.error('--study determines --project and --out-dir; do not combine them')
    root, provenance = begin_run(args.study, args.run_id)
    provenance['arguments'] = arguments
    (root / 'logs').mkdir()
    try:
        with (root / 'logs' / 'run.log').open('w') as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                dispatch(COMMANDS['run'][0], ['--project', str(root / 'resolved_project.json'),
                         '--out-dir', str(root / '.native'), *rest])
        organize_outputs(root)
        provenance['status'] = 'completed'
    except BaseException as error:
        provenance['status'] = 'failed'
        provenance['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        provenance['finished_at'] = datetime.now(timezone.utc).isoformat()
        write_json(root / 'provenance.json', provenance)
    print(f'Completed run: {root}')


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {'-h', '--help'}:
        parser = argparse.ArgumentParser(description='MolScope molecular-dynamics workflows')
        parser.add_argument('command', nargs='?', choices=['init', *COMMANDS])
        parser.epilog = '\n'.join(['init: Create a separate study workspace'] +
                                  [f'{key}: {value[1]}' for key, value in COMMANDS.items()])
        parser.formatter_class = argparse.RawDescriptionHelpFormatter
        parser.print_help()
        return
    command, *rest = arguments
    if command == 'init':
        from molscope.core.workspace import initialize_study
        parser = argparse.ArgumentParser(description='Create a fresh study workspace')
        parser.add_argument('directory')
        parser.add_argument('--project-id', default='my_study')
        args = parser.parse_args(rest)
        print(initialize_study(args.directory, args.project_id))
    elif command == 'run' and any(item.split('=')[0] == '--study' for item in rest):
        study_run(rest)
    elif command in COMMANDS:
        if command == 'run' and any(item in {'-h', '--help'} for item in rest):
            print('Managed study mode: molscope run --study PATH --run-id ID [workflow options]')
            print('Standalone mode (options below): molscope run --project FILE --out-dir PATH\n')
        dispatch(COMMANDS[command][0], rest)
    else:
        raise SystemExit(f'Unknown command {command!r}; use molscope --help')
