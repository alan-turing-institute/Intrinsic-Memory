"""LLM settings, resolved on demand rather than at import.

Nothing here runs while `mas` is being imported. Importing the package therefore
needs no credentials and no particular working directory: the credentials are
read when a run installs its settings, and every value that sizes a request is
the run's own flag rather than a number chosen here.
"""

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

from .utils import repo_path

DEFAULT_ENV_PATH = repo_path(".env")


class MissingSettings(RuntimeError):
    """A required setting is absent, named so the reader knows what to set."""


@dataclass(frozen=True)
class LLMSettings:
    """Everything GPTChat needs to reach an endpoint and size a request.

    Attributes:
        api_base: OpenAI-compatible endpoint URL.
        api_key: Credential for that endpoint.
        max_tokens: Ceiling on the tokens generated per response, sent as
            `max_completion_tokens`. A truncated answer is still returned.
        max_tokens_ceiling: The largest `max_completion_tokens` a retry may climb
            to. A reasoning model that spends the whole budget reasoning answers
            nothing at all, and asking again for the same budget cannot change
            that, so the retry doubles it up to here. Where the endpoint reports
            a context window, the climb also stops at what the prompt leaves of
            it, so this being the wider of the two is harmless.
        thinking_token_budget: Tokens a reasoning model may spend thinking before
            the endpoint ends its reasoning block for it, sent per request. None
            leaves the model to stop on its own. vLLM implements this only
            alongside a `--reasoning-config` naming the reasoning delimiters; an
            endpoint that refuses the field is sent it once and then not again.
        temperature: Sampling temperature for any call that does not set its own.
        request_timeout: Seconds one request may take before it is abandoned. The
            openai default is 600, which multiplied by that client's retries and
            the retry loops above it lets one action block for over an hour.
        log_responses: Echo every LLM response and memory-update prompt to
            stderr. One Slurm job writes one .out file, from every worker at
            once, over the order of 100,000 requests.
    """

    api_base: str
    api_key: str
    max_tokens: int
    max_tokens_ceiling: int
    temperature: float
    request_timeout: float
    log_responses: bool
    thinking_token_budget: Optional[int] = None

    @classmethod
    def load(
        cls,
        max_tokens: int,
        max_tokens_ceiling: int,
        temperature: float,
        request_timeout: float,
        log_responses: bool,
        thinking_token_budget: Optional[int] = None,
    ) -> "LLMSettings":
        """Credentials from .env or the environment, the rest from the caller.

        An exported environment variable takes precedence over .env.
        """
        load_dotenv(dotenv_path=DEFAULT_ENV_PATH)

        api_base = os.getenv("OPENAI_API_BASE")
        api_key = os.getenv("OPENAI_API_KEY")
        missing = [
            name
            for name, value in (("OPENAI_API_BASE", api_base), ("OPENAI_API_KEY", api_key))
            if not value
        ]
        if missing:
            raise MissingSettings(
                f"{' and '.join(missing)} must be set, in the environment or in a .env "
                f"file at the repository root."
            )

        if not api_base.startswith(("http://", "https://")):
            raise MissingSettings(
                f"OPENAI_API_BASE must carry a scheme: 'http://{api_base}', not '{api_base}'. "
                f"A base URL without one parses as a relative path, with no host, so "
                f"requests never reach the server."
            )

        return cls(
            api_base=api_base,
            api_key=api_key,
            max_tokens=max_tokens,
            max_tokens_ceiling=max_tokens_ceiling,
            temperature=temperature,
            request_timeout=request_timeout,
            log_responses=log_responses,
            thinking_token_budget=thinking_token_budget,
        )


_INSTALLED: Optional[LLMSettings] = None


def use_llm_settings(settings: LLMSettings) -> None:
    """Make `settings` what every GPTChat in this process uses."""
    global _INSTALLED
    _INSTALLED = settings


def default_llm_settings() -> LLMSettings:
    """The settings installed for this process, for a GPTChat not given its own."""
    if _INSTALLED is None:
        raise MissingSettings(
            "No LLM settings are installed in this process. A run installs them "
            "from its flags through experiment.install_llm_settings; anything else "
            "must build LLMSettings.load(...) and either pass it to GPTChat or "
            "hand it to use_llm_settings."
        )
    return _INSTALLED


def reset_default_llm_settings() -> None:
    """Forget them. For tests that install different settings between cases."""
    global _INSTALLED
    _INSTALLED = None
