import logging
import sys

from typing import (
    Protocol, 
    Literal,  
    Optional, 
    List,
)
from openai import OpenAI
from dataclasses import dataclass, fields
from abc import ABC, abstractmethod

from .settings import LLMSettings, default_llm_settings
from datetime import datetime

logger = logging.getLogger(__name__)


def _refuses_temperature(error: BaseException) -> bool:
    """Whether an endpoint error is a complaint about the temperature parameter.

    Matched on the message, not a status code: OpenAI-compatible servers do not
    agree on one for an unsupported parameter.
    """
    return "temperature" in str(error).lower()


def _refuses_thinking_budget(error: BaseException) -> bool:
    """Whether an endpoint error is a complaint about `thinking_token_budget`.

    A vLLM without a `--reasoning-config` rejects the field by name; matched on
    the message for the same reason the temperature check is.
    """
    return "thinking_token_budget" in str(error).lower()


def _trace(message) -> Optional[str]:
    """A reasoning model's hidden reasoning, whichever field it arrived under.

    vLLM 0.15 sends `reasoning_content` and 0.28 sends `reasoning`. Reading only
    one of them turns a budget spent entirely on reasoning - which the retry can
    fix by growing it - into a reply with no explanation, which it cannot.
    """
    for field_name in ("reasoning_content", "reasoning"):
        trace = getattr(message, field_name, None)
        if trace:
            return trace
    return None


_WINDOW_UNASKED = object()


class LLMCallFailed(RuntimeError):
    """Every retry of an LLM call was exhausted without a usable answer.

    Distinct from a model that answered with an empty string, which is returned
    as the answer it is. Callers need to tell the two apart to decide whether
    retrying could help.
    """


@dataclass
class TokenTracker:
    """Per-instance token accounting, scoped to whatever owns it (typically one GPTChat)."""

    completion_tokens: int = 0
    prompt_tokens: int = 0
    intrinsic_completion_tokens: int = 0
    intrinsic_prompt_tokens: int = 0

    def record(self, prompt_tokens: int, completion_tokens: int, intrinsic: bool = False) -> None:
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        if intrinsic:
            self.intrinsic_prompt_tokens += prompt_tokens
            self.intrinsic_completion_tokens += completion_tokens

    def snapshot(self) -> "TokenTracker":
        """The counts as they stand, detached from further recording."""
        return TokenTracker(**{field.name: getattr(self, field.name) for field in fields(self)})

    def since(self, earlier: "TokenTracker") -> "TokenTracker":
        """What has been spent since `earlier` was snapshotted."""
        return TokenTracker(**{
            field.name: getattr(self, field.name) - getattr(earlier, field.name)
            for field in fields(self)
        })


@dataclass(frozen=True)
class Message:
    role: Literal["system", "user", "assistant"]
    content: str

# None on any of these means "whatever the settings say".
class LLMCallable(Protocol):

    def __call__(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_strs: Optional[List[str]] = None,
        intrinsic: bool = False # pass intrinsic flag to count tokens used by intrinsic memory
    ) -> str:
        pass

class LLM(ABC):
    
    def __init__(self, model_name: str):
        self.model_name: str = model_name

    @abstractmethod
    def __call__(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_strs: Optional[List[str]] = None,
        intrinsic: bool = False
    ) -> str:
        pass

class GPTChat(LLM):

    def __init__(
        self,
        model_name: str,
        tracker: Optional["TokenTracker"] = None,
        settings: Optional[LLMSettings] = None,
    ):
        super().__init__(model_name=model_name)
        self.settings: LLMSettings = settings if settings is not None else default_llm_settings()
        self.client = OpenAI(
            base_url=self.settings.api_base,
            api_key=self.settings.api_key,
            timeout=self.settings.request_timeout,
        )
        self.tracker: TokenTracker = tracker if tracker is not None else TokenTracker()
        self._sends_temperature: bool = True
        self._sends_stop: bool = True
        self._sends_thinking_budget: bool = self.settings.thinking_token_budget is not None
        self._context_window: object = _WINDOW_UNASKED

    def _endpoint_context_window(self) -> Optional[int]:
        """The endpoint's context length for this model, if it reports one.

        Asked for rather than configured: a number of our own would be free to
        disagree with the server that enforces it. vLLM puts `max_model_len` on
        the OpenAI model card; other servers do not, and then there is nothing
        to size a budget against and None says so.
        """
        if self._context_window is _WINDOW_UNASKED:
            self._context_window = None
            try:
                for card in self.client.models.list().data:
                    if card.id == self.model_name:
                        self._context_window = getattr(card, 'max_model_len', None)
                        break
            except Exception as error:
                print(
                    f'Could not read a context window for {self.model_name} from '
                    f'{self.settings.api_base} ({error}); a starved retry will climb '
                    f'to max_tokens_ceiling without checking the prompt leaves room',
                    file=sys.stderr,
                )
        return self._context_window

    def _create(
        self,
        messages: List[dict],
        temperature: float,
        max_tokens: int,
        stop_strs: Optional[List[str]],
    ):
        """One request, dropping the temperature if the endpoint refuses it.

        A refusal is remembered, and absorbed here rather than spending one of the
        caller's retries. A stop sequence that turns out to truncate a reasoning
        model's hidden reasoning before it answers is remembered the same way, in
        `__call__`, since only the response reveals that.
        """
        request = dict(
            model=self.model_name,
            messages=messages,
            max_completion_tokens=max_tokens,
            stop=stop_strs if self._sends_stop else None,
        )
        if self._sends_thinking_budget:
            request["extra_body"] = {
                "thinking_token_budget": self.settings.thinking_token_budget
            }

        if self._sends_temperature:
            try:
                return self.client.chat.completions.create(temperature=temperature, **request)
            except Exception as error:
                if _refuses_thinking_budget(error):
                    return self._without_thinking_budget(request, error, temperature=temperature)
                if not _refuses_temperature(error):
                    raise
                self._sends_temperature = False
                print(
                    f"{self.model_name} refused temperature={temperature} ({error}); "
                    f"sending subsequent calls without it",
                    file=sys.stderr,
                )

        try:
            return self.client.chat.completions.create(**request)
        except Exception as error:
            if not _refuses_thinking_budget(error):
                raise
            return self._without_thinking_budget(request, error)

    def _without_thinking_budget(self, request: dict, error: BaseException, **extra):
        """Retry a refused request without the thinking budget, and stop sending it.

        The endpoint cannot end the reasoning block itself, so the budget climb
        in `__call__` is all that is left to get an answer out of a model that
        thinks past its budget.
        """
        self._sends_thinking_budget = False
        print(
            f"{self.model_name} refused thinking_token_budget="
            f"{self.settings.thinking_token_budget} ({error}); sending subsequent calls "
            f"without it - serve it with --reasoning-config to have it honoured",
            file=sys.stderr,
        )
        request.pop("extra_body", None)
        return self.client.chat.completions.create(**request, **extra)

    def __call__(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_strs: Optional[List[str]] = None,
        intrinsic: bool = False,
    ) -> str:
        import time

        if max_tokens is None:
            max_tokens = self.settings.max_tokens
        if temperature is None:
            temperature = self.settings.temperature

        messages = [{"role": msg.role, "content": msg.content} for msg in messages]

        max_retries = 5
        wait_time = 1
        last_error: Optional[BaseException] = None
        # The ceiling bounds where a retry may climb to, not what the caller
        # asked for: the budget starts where it was asked to and only ever grows.
        budget = max_tokens
        ceiling = self.settings.max_tokens_ceiling
        attempts = 0
        # The budget of the most recent starved attempt, cleared by any other
        # outcome, so the failure names whatever actually ended the call.
        starved_at: Optional[int] = None
        # Set only where the endpoint's context window, rather than the run's
        # ceiling, is what the climb ran into: the two need different fixes.
        starved_window: Optional[tuple] = None

        for attempt in range(max_retries):
            try:
                attempts += 1
                response = self._create(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=budget,
                    stop_strs=stop_strs,
                )

                answer = response.choices[0].message.content
                self.tracker.record(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    intrinsic=intrinsic,
                )

                reasoning = _trace(response.choices[0].message)

                # Having asked for a stop sequence and been given nothing back,
                # the stop sequence is the first suspect: it is matched against
                # the raw stream, so it can fire inside reasoning the caller
                # never sees. Endpoints report that two ways - gpt-oss on vLLM
                # sends content=None with the text in a reasoning field, ollama
                # sends content='' and no reasoning field - so neither the shape
                # nor the reasoning field can be what this turns on.
                if not (answer or '').strip() and self._sends_stop and stop_strs:
                    self._sends_stop = False
                    print(
                        f'{self.model_name} answered nothing with stop={stop_strs}; '
                        f'sending subsequent calls without a stop sequence'
                        + (' (its reasoning was cut off mid-sentence)' if reasoning else ''),
                        file=sys.stderr,
                    )
                    continue

                if answer is None:
                    # Reasoning present with no content, once the stop sequence is
                    # out of the picture, is what too small a
                    # `max_completion_tokens` looks like: the budget went on
                    # reasoning and left nothing to answer with.
                    starved = bool(reasoning)
                    starved_at = budget if starved else None
                    cause = (
                        f'its reasoning did not reach an answer within '
                        f'max_completion_tokens={budget}'
                        if starved else 'it sent no content and no reasoning'
                    )
                    print(
                        f'Error: {self.model_name} returned no answer - {cause}. '
                        f'Full response:\n{response}',
                        file=sys.stderr,
                    )
                    if starved:
                        window = self._endpoint_context_window()
                        prompt_spent = response.usage.prompt_tokens
                        headroom = window - prompt_spent if window else None
                        capped = min(budget * 2, ceiling)
                        grown = capped if headroom is None else min(capped, headroom)
                        # Nothing but the budget can answer this, so where it
                        # cannot grow, another attempt is the same request and
                        # the same bill.
                        if grown <= budget:
                            if headroom is not None and headroom < capped:
                                starved_window = (window, prompt_spent)
                            break
                        budget = grown
                        print(
                            f'{self.model_name} ran out of budget before answering; '
                            f'retrying with max_completion_tokens={budget}',
                            file=sys.stderr,
                        )
                    continue
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                logger.debug('LLM RESPONSE at %s\n%s', current_time, answer)
                return answer  

            except Exception as e:
                last_error = e
                starved_at = None
                error_message = str(e)
                if "rate limit" in error_message.lower() or "429" in error_message:
                    print(f"Rate limited, waiting {wait_time}s before retry {attempt + 2}/{max_retries}", file=sys.stderr)
                    time.sleep(wait_time)
                    wait_time *= 2
                else:
                    print(f"Error during API call: {error_message}", file=sys.stderr)
                    break

        exhausted = ''
        if starved_at:
            exhausted = (
                f' within max_completion_tokens={starved_at}, the largest budget its '
                f'retries could climb to'
            )
            if starved_window:
                window, prompt_spent = starved_window
                exhausted += (
                    f" - the endpoint's {window}-token context window, less a "
                    f'{prompt_spent}-token prompt'
                )
        raise LLMCallFailed(
            f'{self.model_name} returned no answer after {attempts} attempts{exhausted}'
        ) from last_error
