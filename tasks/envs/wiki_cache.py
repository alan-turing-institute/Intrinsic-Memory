"""What a sweep has already fetched from Wikipedia, and when it may fetch again.

Both are files because the workers are separate processes started with `spawn`,
sharing nothing but the filesystem, and because an experiment set is submitted as
several Slurm jobs that share nothing but the filesystem either.

Neither is load-bearing: a directory that cannot be written costs the saving, not
the run, which is why every operation here swallows its own IO errors.
"""

from pathlib import Path
from typing import Optional, Union
import errno
import fcntl
import hashlib
import json
import logging
import os
import tempfile
import time

logger = logging.getLogger(__name__)

Entry = dict


class PageCache:
    """Search outcomes kept by title, in one JSON file each.

    An outcome is worth keeping only if asking again would give the same answer:
    a page, or the fact that there is no such page. A search that could not reach
    Wikipedia is not an outcome and is never stored - see `LangChainWiki.search`.
    """

    def __init__(self, directory: Optional[Union[str, Path]]) -> None:
        self.directory = Path(directory) if directory is not None else None

    def get(self, title: str) -> Optional[Entry]:
        if self.directory is None:
            return None
        try:
            with open(self._path(title), encoding='utf-8') as handle:
                return json.load(handle)
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as unreadable:
            logger.debug('cache entry for %r unreadable: %s', title, unreadable)
            return None

    def put(self, title: str, entry: Entry) -> None:
        if self.directory is None:
            return
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            self._write(self._path(title), entry)
        except OSError as unwritable:
            logger.debug('cache entry for %r not written: %s', title, unwritable)

    def _path(self, title: str) -> Path:
        """A filename for `title`, which may be any string the model produced."""
        digest = hashlib.sha256(title.encode('utf-8')).hexdigest()
        return self.directory / f'{digest}.json'

    @staticmethod
    def _write(path: Path, entry: Entry) -> None:
        """Whole or not at all: another worker may read this while it is written."""
        handle, staged = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
        try:
            with os.fdopen(handle, 'w', encoding='utf-8') as writer:
                json.dump(entry, writer)
            os.replace(staged, path)
        except BaseException:
            _discard(staged)
            raise


def _discard(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


class RequestWindow:
    """The earliest moment the next live request may go out, shared by every process.

    `reserve` takes a slot and reports the seconds to sleep before using it, so
    the caller waits outside any lock and callers spread rather than pile onto
    one moment. Wall clock rather than monotonic: the file is read by processes
    that do not share a monotonic origin.
    """

    def __init__(
        self, path: Optional[Union[str, Path]], min_interval: float
    ) -> None:
        self.path = Path(path) if path is not None else None
        self.min_interval = min_interval

    def reserve(self) -> float:
        """Seconds to wait before the request this call has just taken a slot for."""
        if self.path is None or self.min_interval <= 0:
            return 0.0
        now = time.time()
        opens_at = self._update(lambda stamp: max(stamp, now) + self.min_interval)
        return max(0.0, (opens_at - self.min_interval) - now)

    def defer(self, seconds: float) -> None:
        """Hold every process off for `seconds`, never shortening a longer wait."""
        if self.path is None or seconds <= 0:
            return
        until = time.time() + seconds
        self._update(lambda stamp: max(stamp, until))

    def _update(self, revise) -> float:
        """`revise` applied to the stored moment, under a lock, and stored back.

        Returns the moment now stored, or what it would have been had the file
        been usable.
        """
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, 'a+', encoding='utf-8') as handle:
                with _locked(handle, self.path):
                    handle.seek(0)
                    revised = revise(_moment(handle.read()))
                    handle.seek(0)
                    handle.truncate()
                    handle.write(f'{revised!r}')
                    handle.flush()
                    return revised
        except OSError as unusable:
            logger.debug('request window %s unusable: %s', self.path, unusable)
            return revise(0.0)


def _moment(written: str) -> float:
    """The moment `written` holds, or the epoch if it holds nothing usable."""
    try:
        return float(written)
    except ValueError:
        return 0.0


_UNLOCKABLE: set = set()


class _locked:
    """An exclusive lock on the open file, if the filesystem grants one.

    Lustre supports flock only when mounted with it. Without one the window still
    spaces requests, it just races over the last one of a burst, which is a
    better outcome than refusing to search.
    """

    def __init__(self, handle, path) -> None:
        self.handle = handle
        self.path = path
        self.locked = False

    def __enter__(self):
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)
            self.locked = True
        except OSError as error:
            if self.path not in _UNLOCKABLE:
                _UNLOCKABLE.add(self.path)
                logger.warning(
                    'cannot lock %s (%s); request spacing is approximate',
                    self.path, error,
                )
        return self

    def __exit__(self, *_) -> None:
        if self.locked:
            try:
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            except OSError as error:
                if error.errno != errno.EBADF:
                    raise
