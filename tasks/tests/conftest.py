"""
conftest.py — executed before test collection.

Stubs the heavy or unavailable dependencies that mas and tasks.envs import at
module scope, so the whole package can be imported offline.

Also tees all stdout/stderr output to a timestamped .txt file in tasks/tests/logs/
so that a full record of every run (including -s print output) is preserved.
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from types import ModuleType
from unittest.mock import MagicMock

import pytest

# An endpoint no test will reach, so a developer's .env cannot leak into a run.
os.environ["OPENAI_API_BASE"] = "http://localhost:9999"
os.environ["OPENAI_API_KEY"] = "test-key"


@pytest.fixture(autouse=True)
def installed_llm_settings():
    """The settings a test's GPTChat falls back on, as a run would install them."""
    from mas.settings import LLMSettings, use_llm_settings

    settings = LLMSettings(
        api_base=os.environ["OPENAI_API_BASE"],
        api_key=os.environ["OPENAI_API_KEY"],
        max_tokens=512,
        max_tokens_ceiling=8192,
        temperature=0.1,
        request_timeout=300.0,
        log_responses=False,
    )
    use_llm_settings(settings)
    return settings


def _stub_module(name: str):
    if name not in sys.modules:
        sys.modules[name] = MagicMock()


def _stub_module_exporting(name: str, *exports: str):
    """Stub a module exposing only `exports`, so a wrong import name still fails."""
    if name in sys.modules:
        return
    module = ModuleType(name)
    for export in exports:
        setattr(module, export, MagicMock(name=f"{name}.{export}"))
    module.__all__ = list(exports)
    sys.modules[name] = module


def _stub_langchain_documents():
    """A `Document` that is a real class, because the code type-checks against it.

    `LangChainWiki.search` decides whether a search succeeded with
    `isinstance(result, Document)`, and isinstance against a MagicMock raises
    TypeError - so the whole function is unreachable in a test without this.
    """
    if "langchain_core.documents" in sys.modules:
        return

    class Document:
        def __init__(self, page_content: str = "", metadata: dict = None):
            self.page_content = page_content
            self.metadata = metadata or {}

    core = sys.modules.get("langchain_core") or ModuleType("langchain_core")
    documents = ModuleType("langchain_core.documents")
    documents.Document = Document
    core.documents = documents

    sys.modules["langchain_core"] = core
    sys.modules["langchain_core.documents"] = documents


def _stub_wikipedia():
    """A `wikipedia` stub with real exception classes and a settable module.

    A MagicMock cannot stand in here: `LangChainWiki.search` names the package's
    exceptions in an `except` clause, which raises TypeError on anything that is
    not an exception class, and it sets `wikipedia.wikipedia.USER_AGENT`. So the
    error paths that decide whether a failed search is reported or masked are
    only reachable in a test against a stub shaped like this.
    """
    if "wikipedia" in sys.modules:
        return

    module = ModuleType("wikipedia")

    class PageError(Exception):
        pass

    class DisambiguationError(Exception):
        pass

    module.PageError = PageError
    module.DisambiguationError = DisambiguationError
    module.page = MagicMock(name="wikipedia.page")
    module.search = MagicMock(name="wikipedia.search")

    # the package keeps its settings on an inner module of the same name
    settings = ModuleType("wikipedia.wikipedia")
    settings.USER_AGENT = "wikipedia (https://github.com/goldsmith/Wikipedia/)"
    settings.API_URL = "http://en.wikipedia.org/w/api.php"
    # every request the package makes goes through this, and the status of what
    # comes back is readable nowhere else
    settings.requests = ModuleType("requests")
    settings.requests.get = MagicMock(name="requests.get")
    module.wikipedia = settings

    sys.modules["wikipedia"] = module
    sys.modules["wikipedia.wikipedia"] = settings


# The task environments import their simulators at module scope, so the env and
# recorder registries need these stubbed to be importable offline.
for _mod in (
    "gymnasium",
    "jericho",
    "langchain_chroma",
    "minigrid",
    "minigrid.core",
    "minigrid.core.actions",
    "nltk",
    "pddlgym",
    "pddlgym.structs",
    "scienceworld",
):
    _stub_module(_mod)

_stub_langchain_documents()
_stub_wikipedia()

# finch-clust's whole public surface is one function.
_stub_module_exporting("finch", "FINCH")

# ALFWorld reaches the code through one factory. The parent packages are stubbed
# too because `from alfworld.agents.environment import ...` walks the whole path.
_stub_module_exporting("alfworld", "agents")
_stub_module_exporting("alfworld.agents", "environment")
_stub_module_exporting("alfworld.agents.environment", "get_environment")


# ── output logging ────────────────────────────────────────────────────────────

class _TeeWriter:
    """Writes to multiple streams simultaneously."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, text):
        for s in self._streams:
            s.write(text)

    def flush(self):
        for s in self._streams:
            s.flush()

    def fileno(self):
        return self._streams[0].fileno()

    def isatty(self):
        return getattr(self._streams[0], "isatty", lambda: False)()

    def __getattr__(self, name):
        return getattr(self._streams[0], name)


def pytest_configure(config):
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{timestamp}.txt"

    log_file = log_path.open("w", encoding="utf-8")
    config._tee_log_file = log_file
    config._tee_log_path = log_path

    config._orig_stdout = sys.stdout
    config._orig_stderr = sys.stderr
    sys.stdout = _TeeWriter(sys.stdout, log_file)
    sys.stderr = _TeeWriter(sys.stderr, log_file)


def pytest_unconfigure(config):
    if hasattr(config, "_orig_stdout"):
        sys.stdout = config._orig_stdout
    if hasattr(config, "_orig_stderr"):
        sys.stderr = config._orig_stderr
    if hasattr(config, "_tee_log_file"):
        config._tee_log_file.close()
        print(f"\nTest output saved to: {config._tee_log_path}")
