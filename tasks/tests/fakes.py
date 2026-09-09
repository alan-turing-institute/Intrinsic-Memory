"""Fakes shared by the contract and smoke tests.

Each one stands in for something needing a network, a GPU or a simulator binary,
so the real workflow, recorder and memory code paths can run offline. These are
working minimal implementations of the interfaces the production code expects,
not mocks with recorded expectations.
"""

import hashlib
from types import SimpleNamespace

from mas.llm import GPTChat, Message, TokenTracker
from mas.reasoning import ReasoningConfig, ReasoningIO


class RunawayLoop(AssertionError):
    """A fake was called far more often than any bounded loop would need."""


class FakeLLM:
    """A GPTChat stand-in that returns scripted replies and counts tokens.

    `replies` is cycled, so a single-element list is an always-the-same LLM.

    `max_calls` is a runaway guard rather than a behaviour: it turns a test that
    would otherwise hang into one that fails immediately, naming the cause.
    """

    def __init__(self, replies=None, model_name="fake-model", tracker=None, max_calls=200):
        self.replies = list(replies) if replies else ["look at desk 1"]
        self.model_name = model_name
        self.tracker = tracker if tracker is not None else TokenTracker()
        self.max_calls = max_calls
        self.calls: list[list[Message]] = []

    def __call__(self, messages, temperature=None, max_tokens=None,
                 stop_strs=None, intrinsic=False):
        if len(self.calls) >= self.max_calls:
            raise RunawayLoop(
                f"FakeLLM was called {self.max_calls} times - the caller is not bounding its retries"
            )
        self.calls.append(list(messages))
        self.tracker.record(prompt_tokens=10, completion_tokens=5, intrinsic=intrinsic)
        return self.replies[(len(self.calls) - 1) % len(self.replies)]


def fake_reasoning(replies=None) -> ReasoningIO:
    """The real ReasoningIO over a FakeLLM - the workflows type-check for it."""
    return ReasoningIO(llm_model=FakeLLM(replies=replies))


class FakeEnv:
    """A deterministic environment satisfying the BaseEnv interface.

    Reports `done` after `steps_to_done` steps, so a test can choose between an
    episode that solves, one that exhausts the trial budget, and one that never
    terminates.
    """

    def __init__(self, max_trials: int = 3, steps_to_done: int = 1, reward: float = 1.0):
        self.max_trials = max_trials
        self.steps_to_done = steps_to_done
        self.reward = reward
        self.actions: list[str] = []
        self.reset()

    def reset(self) -> None:
        self.actions = []
        self.done = False

    def set_env(self, task_config: dict) -> tuple[str, str]:
        return "fake task main", "fake task description"

    def step(self, action: str) -> tuple[str, float, bool]:
        self.actions.append(action)
        self.done = len(self.actions) >= self.steps_to_done
        return f"observation after {action}", (self.reward if self.done else 0.0), self.done

    @classmethod
    def process_action(cls, action: str) -> str:
        return action.strip()

    @staticmethod
    def is_thought(action: str) -> bool:
        """Both spellings, so one fake serves every task's vocabulary."""
        return 'think' in action.lower() or 'thought' in action.lower()

    def feedback(self) -> tuple[float, bool, str]:
        return (
            (self.reward, True, "You successfully finished this task!")
            if self.done
            else (0.0, False, "You failed the task.")
        )


class FakeEmbeddingFunc:
    """A deterministic embedding, so cosine similarity is real arithmetic.

    Callers feed these vectors into np.dot and np.linalg.norm, so the values have
    to be floats. Derived from a hash of the text, and stable across processes.
    """

    dims = 16

    def __init__(self, model_type: str = "fake-embeddings"):
        self.model_type = model_type

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(str(text).encode("utf-8")).digest()
        return [digest[i] / 255.0 for i in range(self.dims)]

    def embed_query(self, query: str) -> list[float]:
        return self._vector(query)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]


class FakeCompletions:
    """Stands in for client.chat.completions, scripted per call.

    Each entry in `script` is either an answer string, None (the model returned
    no content), or an exception instance to raise. The last entry repeats once
    the script runs out.
    """

    def __init__(self, script, prompt_tokens: int = 11, completion_tokens: int = 7):
        self.script = list(script)
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.script[min(len(self.calls) - 1, len(self.script) - 1)]
        if isinstance(outcome, BaseException):
            raise outcome
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=outcome))],
            usage=SimpleNamespace(
                prompt_tokens=self.prompt_tokens, completion_tokens=self.completion_tokens
            ),
        )


class TemperatureRejectingCompletions(FakeCompletions):
    """An endpoint that refuses the `temperature` parameter.

    Every attempt is appended to `calls`, refused ones included.
    """

    def __init__(self, script, message=None, status_code: int = 400):
        super().__init__(script)
        self.message = message or (
            "Unsupported value: 'temperature' does not support 0.0 with this model"
        )
        self.status_code = status_code
        self.refusals = 0

    def create(self, **kwargs):
        if "temperature" in kwargs:
            self.calls.append(kwargs)
            self.refusals += 1
            error = ValueError(self.message)
            error.status_code = self.status_code
            raise error
        return super().create(**kwargs)


class StarvedReasoningCompletions(FakeCompletions):
    """A reasoning model that answers only once its budget reaches `needs`.

    Below that the whole budget goes on reasoning and the answer never starts,
    which vLLM reports as content=None with the text in a reasoning field and
    finish_reason='length'. The usage it reports is the whole budget, because
    that is what was generated.

    `reasoning_field` is the name that field arrives under, which differs by
    server version: `reasoning_content` on the vLLM 0.15 serving gpt-oss,
    `reasoning` on the 0.28 serving Qwen3.6.
    """

    def __init__(self, script, needs: int, reasoning="thinking it over",
                 reasoning_field: str = "reasoning_content"):
        super().__init__(script)
        self.needs = needs
        self.reasoning = reasoning
        self.reasoning_field = reasoning_field

    def create(self, **kwargs):
        budget = kwargs.get("max_completion_tokens") or 0
        if budget >= self.needs:
            return super().create(**kwargs)

        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content=None, **{self.reasoning_field: self.reasoning},
            ))],
            usage=SimpleNamespace(prompt_tokens=self.prompt_tokens, completion_tokens=budget),
        )


class StopTruncatesReasoningCompletions(FakeCompletions):
    """An endpoint whose reasoning model is cut off by a stop sequence.

    Answers None with populated `reasoning_content` for as long as a call sends
    `stop`; answers from `script` once the caller stops sending it.
    """

    def __init__(self, script, reasoning="thinking it over\n"):
        super().__init__(script)
        self.reasoning = reasoning
        self.truncations = 0

    def create(self, **kwargs):
        if kwargs.get("stop"):
            self.calls.append(kwargs)
            self.truncations += 1
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(
                    content=None, reasoning_content=self.reasoning
                ))],
                usage=SimpleNamespace(
                    prompt_tokens=self.prompt_tokens, completion_tokens=self.completion_tokens
                ),
            )
        return super().create(**kwargs)


class FakeModels:
    """Stands in for client.models, reporting one card per served model.

    `max_model_len` is the field vLLM adds to the OpenAI model card; `None`
    stands for an endpoint that does not report a context window at all.
    """

    def __init__(self, model_name: str = "fake-model", max_model_len=None):
        self.model_name = model_name
        self.max_model_len = max_model_len
        self.listings = 0

    def list(self):
        self.listings += 1
        card = SimpleNamespace(id=self.model_name)
        if self.max_model_len is not None:
            card.max_model_len = self.max_model_len
        return SimpleNamespace(data=[card])


def chat_over_fake_completions(
    script, tracker=None, model_name="fake-model", settings=None, completions=None,
    models=None,
):
    """A real GPTChat with a scripted client, for testing its request behaviour.

    A client without a `models` attribute stands for an endpoint whose context
    window cannot be discovered, which is the majority of these tests.
    """
    chat = GPTChat(model_name=model_name, tracker=tracker, settings=settings)
    completions = completions if completions is not None else FakeCompletions(script)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    if models is not None:
        client.models = models
    chat.client = client
    return chat, completions


class RecordingObserver:
    """Stands in for a recorder where a test only cares that messages arrive."""

    def __init__(self):
        self.messages: list[str] = []

    def log(self, message: str) -> None:
        self.messages.append(str(message))


REASONING_CONFIG = ReasoningConfig(temperature=0, stop_strs=["\n"])
