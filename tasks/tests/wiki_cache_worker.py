"""One worker of the concurrency tests, run as a process of its own.

A sweep's workers are separate process trees - the pool workers of a job are
`spawn`ed, and an experiment set is submitted as one job per dataset - so a test
of what they share between them has to be separate processes too.
"""

import importlib.util
import json
import sys
import time
from pathlib import Path


def wiki_cache():
    """The module under test, by path: importing `tasks.envs` needs every simulator."""
    source = Path(__file__).resolve().parents[1] / 'envs' / 'wiki_cache.py'
    spec = importlib.util.spec_from_file_location('wiki_cache', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _together(start_at: str) -> None:
    time.sleep(max(0.0, float(start_at) - time.time()))


def reserve(path, min_interval, count, start_at) -> list:
    """Take `count` slots, reporting the moment each was granted for."""
    window = wiki_cache().RequestWindow(path, min_interval=float(min_interval))
    _together(start_at)

    granted = []
    for _ in range(int(count)):
        # Read the clock before the call, not after: `reserve` reports a wait
        # from a moment inside itself, and adding it to a later reading moves
        # the slot forward by however long the call took.
        asked_at = time.time()
        owed = window.reserve()
        granted.append(asked_at + owed)
        time.sleep(owed)
    return granted


def defer(path, min_interval, seconds) -> list:
    window = wiki_cache().RequestWindow(path, min_interval=float(min_interval))
    window.defer(float(seconds))
    return []


def rewrite(directory, title, size, count, start_at) -> list:
    """Write one title `count` times over, as every arm re-fetching it would."""
    cache = wiki_cache().PageCache(directory)
    entry = {'content': 'x' * int(size), 'url': f'https://x/{title}'}
    _together(start_at)

    for _ in range(int(count)):
        cache.put(title, entry)
    return []


def write_each(directory, prefix, titles, size, start_at) -> list:
    cache = wiki_cache().PageCache(directory)
    _together(start_at)

    for index in range(int(titles)):
        title = f'{prefix}-{index}'
        cache.put(title, {'content': 'x' * int(size), 'url': f'https://x/{title}'})
    return []


MODES = {
    'reserve': reserve,
    'defer': defer,
    'rewrite': rewrite,
    'write_each': write_each,
}


if __name__ == '__main__':
    mode, *arguments = sys.argv[1:]
    json.dump(MODES[mode](*arguments), sys.stdout)
