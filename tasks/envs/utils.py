from pathlib import Path
from typing import Any, Optional, Union
import logging
import random
import string
import re
import time

from langchain_core.documents import Document
import wikipedia

from mas.utils import repo_path

from .wiki_cache import PageCache, RequestWindow

logger = logging.getLogger(__name__)

# Wikimedia refuses the `wikipedia` package's own default User-Agent, answering
# 403 with a text/plain body that the package then fails to parse as JSON. Its
# policy asks for a descriptive agent identifying the project:
# https://foundation.wikimedia.org/wiki/Policy:Wikimedia_Foundation_User-Agent_Policy
wikipedia.wikipedia.USER_AGENT = (
    "Intrinsic-Memory/0.1 "
    "(research; https://github.com/alan-turing-institute/Intrinsic-Memory)"
)
# The package still defaults to http, which Wikimedia redirects.
wikipedia.wikipedia.API_URL = "https://en.wikipedia.org/w/api.php"

WIKIPEDIA_ATTEMPTS = 6
WIKIPEDIA_RETRY_SECONDS = 2.0
WIKIPEDIA_MAX_WAIT = 60.0

# Wikimedia's User-Agent policy asks unauthenticated clients to make requests in
# series rather than in parallel. A sweep's workers are one client as far as the
# endpoint is concerned, so the interval is what they share, not what each keeps.
WIKIPEDIA_MIN_INTERVAL = 0.2
WIKIPEDIA_CACHE_DIR = repo_path('data', 'wikipedia-cache')
WINDOW_FILE = '.window'

# Every worker of a sweep meets the same burst and is handed the same wait, so
# the waits are jittered apart by a fraction of the wait itself, so that the
# spread scales with the burst. This generator is its own: the global one
# carries the experiment's seed.
_backoff = random.Random()


class WikipediaRefused(RuntimeError):
    """The API would not answer this request for now.

    `retry_after` is the wait before asking again: what the response asked for,
    bounded by `WIKIPEDIA_MAX_WAIT`, or the backoff default if it named none.
    """

    def __init__(self, status: int, asked: Optional[float] = None) -> None:
        named = f"asked for {asked}s" if asked is not None else "named no wait"
        super().__init__(f"Wikimedia answered {status} and {named}")
        self.status = status
        self.retry_after = min(
            WIKIPEDIA_MAX_WAIT, WIKIPEDIA_RETRY_SECONDS if asked is None else asked
        )


class StatusAwareRequests:
    """`wikipedia`'s `requests`, with a refusal raised rather than parsed as JSON.

    The package hands every response to `r.json()` whatever its status, so
    Wikimedia's 429 - an HTML error page carrying `Retry-After` - reached the
    caller as a bare JSONDecodeError with both the status and the wait it asked
    for discarded. There is nowhere else to read them: the response never leaves
    `_wiki_request`.
    """

    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)

    def get(self, *args: Any, **kwargs: Any) -> Any:
        response = self.delegate.get(*args, **kwargs)
        if response.status_code == 429 or response.status_code >= 500:
            raise WikipediaRefused(response.status_code, _retry_after(response))
        return response


def _retry_after(response: Any) -> Optional[float]:
    """The seconds `response` asked for, or None if it named none in seconds.

    The header may be an HTTP-date instead, which is not worth parsing to learn
    the same thing the backoff already assumes.
    """
    try:
        return float(response.headers.get("Retry-After"))
    except (TypeError, ValueError):
        return None


# `tasks/` is on the path as well as the repo root, so this module is imported
# once as `envs.utils` and once as `tasks.envs.utils`. `delegate` is the marker
# that the first copy already wrapped the package's fetcher.
if not hasattr(wikipedia.wikipedia.requests, "delegate"):
    wikipedia.wikipedia.requests = StatusAwareRequests(wikipedia.wikipedia.requests)


class WikipediaUnavailable(RuntimeError):
    """Wikipedia could not be reached, as distinct from a page not existing.

    Kept separate because a run that cannot reach Wikipedia at all still scores:
    every Search returns nothing found, every episode fails, and the results look
    like a weak agent rather than a broken one.
    """


class LangChainWiki:

    def __init__(
        self,
        attempts: int = WIKIPEDIA_ATTEMPTS,
        retry_seconds: float = WIKIPEDIA_RETRY_SECONDS,
        cache_dir: Optional[Union[str, Path]] = None,
        min_interval: float = WIKIPEDIA_MIN_INTERVAL,
    ) -> None:
        """Without a `cache_dir` nothing is kept and nothing is spaced.

        The default is off because the caller that wants a cache is a sweep,
        which has a directory to put one in; an explorer built for a single
        lookup should not leave a directory behind.
        """
        self.attempts = attempts
        self.retry_seconds = retry_seconds
        self.cache = PageCache(cache_dir)
        self.window = RequestWindow(
            Path(cache_dir) / WINDOW_FILE if cache_dir is not None else None,
            min_interval,
        )
        self.document: Optional[Document] = None
        self.lookup_str = ""
        self.lookup_index = 0

    def search(self, search: str) -> Union[str, Document]:
        """`search`'s first paragraph, or why there is no page to take one from.

        A search that could not reach Wikipedia raises instead of answering, and
        is the one outcome not kept: caching it would make one bad minute of the
        endpoint permanent for every later run sharing the directory.
        """
        result = self._remembered(search)
        if result is None:
            try:
                page = self._page(search)
                result = Document(
                    page_content=page.content, metadata={"page": page.url}
                )
                self.cache.put(search, {"content": page.content, "url": page.url})
            except (wikipedia.PageError, wikipedia.DisambiguationError):
                result = f"Could not find [{search}]. Similar: {self._similar(search)}"
                self.cache.put(search, {"absent": result})
            except Exception as unreachable:
                raise WikipediaUnavailable(
                    f"Wikipedia could not be reached while searching for {search!r}: "
                    f"{type(unreachable).__name__}: {unreachable}"
                ) from unreachable

        if isinstance(result, Document):
            self.document = result
            return self._sumary
        else:
            self.document = None
            return result

    def _remembered(self, search: str) -> Optional[Union[str, Document]]:
        """What a previous search of `search` found, or None if none has."""
        entry = self.cache.get(search)
        if entry is None:
            return None
        if "absent" in entry:
            return entry["absent"]
        if "content" not in entry:
            return None
        return Document(
            page_content=entry["content"], metadata={"page": entry.get("url", "")}
        )

    def _page(self, search: str) -> Any:
        """`search`'s page, retrying the transport faults that arrive in bursts.

        A page that does not exist is an answer and is not retried; anything else
        is the API refusing to answer, and it says for how long.

        `auto_suggest` is off because the package's default is to replace the
        title asked for with Wikipedia's spelling suggestion and fetch that,
        which reports existing pages as absent.
        """
        for attempt in range(1, self.attempts + 1):
            try:
                self._hold()
                return wikipedia.page(search, auto_suggest=False)
            except (wikipedia.PageError, wikipedia.DisambiguationError):
                raise
            except Exception as unreachable:
                logger.warning(
                    "Wikipedia lookup of %r failed (attempt %d/%d): %s: %s",
                    search, attempt, self.attempts,
                    type(unreachable).__name__, unreachable,
                )
                wait = self._wait(unreachable, attempt)
                # Before the process that met the refusal sleeps it off, every
                # other process is held off too: a refusal answers the client,
                # and the workers of a sweep are one client.
                self.window.defer(wait)
                if attempt == self.attempts:
                    raise
                time.sleep(wait)

    def _hold(self) -> None:
        """Wait for this request's slot in the window every process shares."""
        owed = self.window.reserve()
        if owed > 0:
            time.sleep(owed)

    def _wait(self, unreachable: Exception, attempt: int) -> float:
        """Seconds before the attempt after `attempt`, jittered by up to half."""
        asked = getattr(unreachable, "retry_after", 0)
        wait = asked + self.retry_seconds * 2 ** (attempt - 1)
        return wait + _backoff.uniform(0, wait / 2)

    @staticmethod
    def _similar(search: str) -> list:
        """Titles close to `search`, or an empty list if even that fails."""
        try:
            return wikipedia.search(search)
        except Exception:
            return []
    
    def lookup(self, term: str):

        if self.document is None:
            raise ValueError("Cannot lookup without a successful search first")
        if term.lower() != self.lookup_str:
            self.lookup_str = term.lower() 
            self.lookup_index = 0
        else:
            self.lookup_index += 1
        lookups = [p for p in self._paragraphs if self.lookup_str in p.lower()]
        if len(lookups) == 0:
            return "No Results"
        elif self.lookup_index >= len(lookups):
            return "No More Results"
        else:
            result_prefix = f"(Result {self.lookup_index + 1}/{len(lookups)})"
            return f"{result_prefix} {lookups[self.lookup_index]}"

    @property
    def _sumary(self) -> str:
        return self._paragraphs[0]
    
    @property
    def _paragraphs(self) -> list[str]:
        if self.document is None:
            raise ValueError("Cannot get paragraphs without a document")
        return self.document.page_content.split("\n\n")
    


def normalize_answer(s: str):

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    
    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def match_exactly(answer, key) -> bool:

    n_answer = normalize_answer(answer)
    n_key = normalize_answer(key)
    return n_answer == n_key
