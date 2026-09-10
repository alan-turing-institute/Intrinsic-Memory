"""Workers a spawned child can import.

A spawned child imports the module its target came from. Anything reachable from
`tasks.envs` pulls in every simulator, so these live apart from the rest of the
fakes and import nothing of this repository.
"""

import os
import pathlib
import signal
import time


def kill_this_worker_on_seed_2(experiment_config: dict) -> dict:
    """Dies the way an OOM kill or a fatal signal in a native extension kills one."""
    if experiment_config['seed'] == 2:
        os.kill(os.getpid(), signal.SIGKILL)
    return {
        'status': 'success',
        'ran': experiment_config['seed'],
        'seed': experiment_config['seed'],
    }


def report_this_process(experiment_config: dict) -> dict:
    """Reports which process ran it, so a caller can tell in-process from spawned."""
    return {
        'status': 'success',
        'pid': os.getpid(),
        'seed': experiment_config['seed'],
    }


def record_concurrency(experiment_config: dict) -> dict:
    """Counts how many workers were live alongside it, through the filesystem.

    A spawned child shares nothing with the parent, so the count goes through
    `db_dir` rather than a counter in memory.
    """
    directory = pathlib.Path(experiment_config['db_dir'])
    marker = directory / f'live-{os.getpid()}'
    marker.write_text('')
    time.sleep(0.2)
    live = len(list(directory.glob('live-*')))
    marker.unlink()
    return {'status': 'success', 'seed': experiment_config['seed'], 'live': live}
