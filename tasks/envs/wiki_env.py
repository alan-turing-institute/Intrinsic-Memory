"""ReAct over live Wikipedia: the loop FEVER and HotpotQA both run.

An episode interleaves Thought with one of three actions - Search an entity,
Lookup a keyword in the page Search last found, Finish with an answer - and is
scored by whether the finished answer matches the dataset's. A subclass supplies
`task_prefix`, the word the question is presented to the agent under, and
nothing else: the datasets differ in what an answer looks like, which is the
system prompt's business rather than the environment's.
"""

from typing import Any, Literal
import re
from dataclasses import dataclass

from mas.mas import EpisodeResult

from .base_env import BaseEnv, BaseRecorder, clean_action_line
from .utils import (
    WIKIPEDIA_ATTEMPTS,
    WIKIPEDIA_CACHE_DIR,
    WIKIPEDIA_MIN_INTERVAL,
    WIKIPEDIA_RETRY_SECONDS,
    LangChainWiki,
    WikipediaUnavailable,
    match_exactly,
)

DEFAULT_UNREACHABLE_SEARCH_LIMIT = 3

# What a Search that was still unreachable after every retry `LangChainWiki`
# makes becomes, for the first `unreachable_search_limit` - 1 such Searches of a
# task. The one after those raises instead.
UNREACHABLE_OBSERVATION = (
    'Wikipedia could not be reached, so this Search returned nothing. This is a '
    'temporary fault rather than a missing page: Search for the same entity again.'
)


class WikiReActEnv(BaseEnv):

    task_prefix: str = None

    def __init__(self, env_config: dict[str, Any], max_trials: int) -> None:
        super().__init__(env_config, max_trials)
        config: dict = env_config or {}
        self.explorer = LangChainWiki(
            attempts=config.get('wikipedia_attempts', WIKIPEDIA_ATTEMPTS),
            retry_seconds=config.get('wikipedia_retry_seconds', WIKIPEDIA_RETRY_SECONDS),
            cache_dir=config.get('wikipedia_cache_dir', WIKIPEDIA_CACHE_DIR),
            min_interval=config.get('wikipedia_min_interval', WIKIPEDIA_MIN_INTERVAL),
        )
        self.unreachable_search_limit: int = config.get(
            'unreachable_search_limit', DEFAULT_UNREACHABLE_SEARCH_LIMIT
        )

        self.reset()

    def set_env(self, configs: dict) -> tuple[str, str]:
        if configs.get('answer') is None:
            raise ValueError('Please provide the answer for the question.')
        if configs.get('task') is None:
            raise ValueError('The configs dict should have the `task` attribute.')
        self.config = configs

        task: str = f"{self.task_prefix}: {self.config.get('task')}"
        return task, task

    def reset(self) -> None:
        self.current_task: str = None
        self.reward: float = 0
        self.unreachable_searches: int = 0

    def step(self, action: str) -> tuple[str, float, bool]:

        action: str = self.process_action(action)

        if self._parse_action_type(action) == 'thought':
            return 'OK.', 0, False

        action_type, argument = self._parse_action(action)

        if action_type == 'Finish':
            if self.success_fn(argument):
                observation = 'Answer is CORRECT'
                self.reward = 1
                return observation, 1, True

            else:
                observation = 'Answer is INCORRECT'
                return observation, 0, True

        elif action_type == 'Search':
            # A page that does not exist comes back as text naming similar titles;
            # only being unable to reach Wikipedia raises. Those faults arrive in
            # bursts seconds wide, so the agent is told and spends a step retrying.
            # At `unreachable_search_limit` faults in one task they stop looking
            # transient and the exception propagates: run_task records the task as
            # failed, where answering every Search with a fault would look like a
            # weak agent.
            try:
                observation = self.explorer.search(argument).strip('\n').strip()
            except WikipediaUnavailable:
                self.unreachable_searches += 1
                if self.unreachable_searches >= self.unreachable_search_limit:
                    raise
                observation = UNREACHABLE_OBSERVATION
        elif action_type == 'Lookup':
            try:
                observation = self.explorer.lookup(argument).strip('\n').strip()
            except ValueError:
                observation = 'The last page Searched was not found, so you cannot Lookup a keyword in it. Please try one of the similar pages given.'
        else:
            observation = 'Invalid Action. Valid Actions are Lookup[<topic>] Search[<topic>] and Finish[<answer>].'

        if 'Invalid Action' in observation:
            processed_reward = -1
        else:
            processed_reward = 0
        return observation, processed_reward, False

    @classmethod
    def _parse_action_type(cls, action: str) -> Literal['action', 'thought']:
        return 'thought' if cls.is_thought(action) else 'action'

    @staticmethod
    def looks_like_an_action(line: str) -> bool:
        """Whether a line has the shape of a ReAct action: `Verb[argument]`.

        The three verbs are a closed set, so this is what lets an action be
        preferred over a thought it was written alongside.
        """
        return WikiReActEnv._parse_action(line.strip())[0] is not None

    @staticmethod
    def process_action(action: str) -> str:
        """The `Verb[argument]` from the model's line.

        The `Action N:` prefix comes off through the shared label rule rather
        than by splitting on the first colon, which destroyed any argument
        carrying one of its own - `Search[Star Wars: A New Hope]` became
        `Search[Star Wars`, which parses as nothing and scores -1.
        """
        return clean_action_line(action, recognises=WikiReActEnv.looks_like_an_action)

    @staticmethod
    def _parse_action(string: str) -> tuple[str, str]:

        pattern = r'^(\w+)\[(.+)\]$'
        match = re.match(pattern, string)

        if match:
            action_type = match.group(1)
            argument = match.group(2)
            return action_type, argument
        else:
            return None, None

    def success_fn(self, agent_ans: str) -> bool:
        return match_exactly(agent_ans, self.config.get('answer'))

    def feedback(self) -> tuple[float, bool, str]:

        feedback: str = 'You successfully finished this task.' if self.reward == 1 else 'You failed the task.'
        done = self.reward == 1

        return self.reward, done, feedback


@dataclass
class WikiReActRecorder(BaseRecorder):

    def task_begin(self, task_id: int, task_config: dict):
        super().task_begin(task_id, task_config)

        message: str = f'---------- Task: {task_id} ----------'
        self.log(message)

    def task_end(self, episode: EpisodeResult):
        super().task_end(episode)

        averages = self.average_results()
        self.log(
            f'reward: {episode.reward}, ave reward: {averages.mean_reward}.\n'
            f'done: {episode.done}, ave done: {averages.mean_done}'
        )
