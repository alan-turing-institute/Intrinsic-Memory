"""The window and the cache hold up under the processes a sweep actually runs.

Both are files rather than objects because no two workers of a sweep share
anything else: a job's pool workers are `spawn`ed, and an experiment set is
submitted as one job per dataset. So nothing here uses a `multiprocessing`
primitive, and every test below runs real processes - a lock or a semaphore
handed out by one interpreter is invisible to the other 39.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tasks.envs.wiki_cache import PageCache, RequestWindow

WORKER = Path(__file__).parent / 'wiki_cache_worker.py'

# Allowed for every worker to be up before any of them starts, so that they meet
# the window together rather than in launch order.
STARTUP = 0.8


def worker(*arguments) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, str(WORKER), *(str(argument) for argument in arguments)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def reaped(workers: list) -> list:
    """What each worker reported, once every one of them has finished."""
    reported = []
    for process in workers:
        written, failed = process.communicate(timeout=120)
        assert process.returncode == 0, failed
        reported.append(json.loads(written))
    return reported


def test_no_two_processes_are_granted_the_same_slot(tmp_path):
    """Workers meeting one window queue behind each other rather than all firing.

    Without a lock over the read-revise-write, each process reads the same
    moment out of the file and is handed it, which is the burst the window
    exists to prevent.
    """
    interval = 0.03
    per_worker = 5
    start_at = time.time() + STARTUP

    granted = reaped([
        worker('reserve', tmp_path / 'window', interval, per_worker, start_at)
        for _ in range(4)
    ])

    slots = sorted(moment for report in granted for moment in report)
    assert len(slots) == 4 * per_worker
    apart = [later - earlier for earlier, later in zip(slots, slots[1:])]
    assert min(apart) >= interval - 0.005, (
        f'two slots {min(apart):.4f}s apart, closer than the {interval}s interval'
    )


def test_a_refusal_one_process_met_holds_the_other_processes_off(tmp_path):
    """The 429 that broke the sweep: one worker waited, the other 39 kept asking."""
    deferral = 2.0

    reaped([worker('defer', tmp_path / 'window', 0.03, deferral)])

    owed = RequestWindow(tmp_path / 'window', min_interval=0.03).reserve()
    assert owed == pytest.approx(deferral, abs=0.5), (
        f'a process that met no refusal was held off {owed:.2f}s'
    )


def test_an_entry_another_process_is_rewriting_is_never_read_half_written(tmp_path):
    """A reader arriving mid-write sees the last whole entry, never part of one.

    This is why the write needs no lock: it is staged elsewhere and renamed
    over the entry, and a rename is not something a reader can catch part way.
    """
    size = 400_000
    start_at = time.time() + STARTUP
    writers = [
        worker('rewrite', tmp_path / 'pages', 'Telemundo', size, 20, start_at)
        for _ in range(3)
    ]

    cache = PageCache(tmp_path / 'pages')
    reads = whole = torn = 0
    while any(process.poll() is None for process in writers):
        entry = cache.get('Telemundo')
        reads += 1
        if entry is not None and len(entry.get('content', '')) == size:
            whole += 1
        elif whole:
            torn += 1
    reaped(writers)

    assert whole, f'{reads} reads never saw the entry at all'
    assert torn == 0, f'{torn} of {reads} reads saw a partially written entry'


def test_processes_writing_at_once_do_not_lose_each_others_entries(tmp_path):
    """Every worker writes the same directory, tens of thousands of times a sweep."""
    prefixes = ['worker-0', 'worker-1', 'worker-2', 'worker-3']
    titles = 10
    start_at = time.time() + STARTUP

    reaped([
        worker('write_each', tmp_path / 'pages', prefix, titles, 20_000, start_at)
        for prefix in prefixes
    ])

    cache = PageCache(tmp_path / 'pages')
    written = [f'{prefix}-{index}' for prefix in prefixes for index in range(titles)]
    assert [title for title in written if cache.get(title) is None] == []


def test_processes_writing_at_once_leave_nothing_but_entries_behind(tmp_path):
    """A staged file left in place is read as an entry by nobody and swept up by nobody."""
    start_at = time.time() + STARTUP

    reaped([
        worker('write_each', tmp_path / 'pages', prefix, 10, 20_000, start_at)
        for prefix in ('worker-0', 'worker-1', 'worker-2', 'worker-3')
    ])

    left = sorted({path.suffix for path in (tmp_path / 'pages').iterdir()})
    assert left == ['.json'], f'the cache directory also holds {left}'
