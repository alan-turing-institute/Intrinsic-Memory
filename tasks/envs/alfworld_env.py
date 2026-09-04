"""ALFWorld: one ALFRED household task, played as text through TextWorld.

The dataset is `data/alfworld/alfworld_tasks_suffix.json`, and every row names
the `.tw-pddl` game file to play, so `set_env` assigns `game_files` directly and
the split `AlfredTWEnv` collects for itself is never played. That is what keeps
the games the only download this needs: `collect_game_files` accepts a game only
when ALFRED's `traj_data.json` sits beside it, and reports none otherwise.

Installing the simulator on aarch64 takes two extra steps; see data/data.md.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Union, Any
import re

from mas.mas import EpisodeResult
from mas.utils import repo_path

from .base_env import BaseEnv, BaseRecorder, aggregate

# tasks/envs imports every environment at module scope, and alfworld is the one
# simulator uv sync cannot install - so importing it eagerly here would take every
# other dataset down with it on any machine that has not installed it by hand.
try:
    from alfworld.agents.environment import get_environment
except ModuleNotFoundError:
    get_environment = None

prefixes = {  # tasks: task_type
    'pick_and_place': 'put',
    'pick_clean_then_place': 'clean',
    'pick_heat_then_place': 'heat',
    'pick_cool_then_place': 'cool',
    'look_at_obj': 'examine',
    'pick_two_obj': 'puttwo'
}

REJECTED = 'Nothing happens.'


def get_env_name_from_gamefile(gamefile: str) -> Union[str, None]:

    for k in prefixes.keys():
        if k in gamefile:
            return k
    return None


def limit_episode_steps(env_config: dict, max_trials: int) -> None:
    """Make TextWorld's own episode limit the trial budget.

    `AlfredTWEnv.init_env` takes `max_episode_steps` from whichever config
    section its `training_method` names, never from `max_trials`, so a budget
    larger than the number in the YAML is cut short by the simulator instead.
    """
    for section in ('rl', 'dagger'):
        training = env_config.setdefault(section, {}).setdefault('training', {})
        training['max_nb_steps_per_episode'] = max_trials


def process_ob(ob: str) -> str:
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ') + 2:]
    return ob


class AlfworldEnv(BaseEnv):
    """One ALFRED task, driven through TextWorld.

    The game is per-task, so nothing is loaded until `set_env`.
    """

    def __init__(self, env_config: dict[str, Any], max_trials: int):
        super().__init__(env_config, max_trials)

        if get_environment is None:
            raise ModuleNotFoundError(
                'alfworld is installed separately from the rest, and on aarch64 needs '
                'two extra steps; see the ALFWorld section of data/data.md.'
            )

        limit_episode_steps(self.env_config, max_trials)

        self.main_env = get_environment(self.env_config['env']['type'])(
            self.env_config, train_eval=self.env_config['split']
        )
        self.env = None
        self.gamefile: str = None
        self.env_name: str = None
        self.done: bool = False
        self.won: bool = False

    def set_env(self, configs: dict) -> tuple[str, str]:
        gamefile: str = (configs.get('env_kwargs') or {}).get('gamefile')
        env_name: str = configs.get('env_name')

        if gamefile is None or env_name is None:
            raise ValueError(
                'An alfworld task config needs an `env_kwargs.gamefile` and an `env_name`.'
            )

        path = Path(gamefile)
        if not path.is_absolute():
            path = repo_path(gamefile)
        if not path.exists():
            raise FileNotFoundError(
                f'No game for {env_name} at {path}. The tw-pddl games are downloaded '
                f'separately; see data/data.md.'
            )

        self.gamefile = str(path)
        self.env_name = env_name
        self.main_env.game_files = [self.gamefile]

        task = configs['task']

        self.reset()
        return self._parse_task_main(task), self._parse_task_description(task)

    def reset(self) -> None:
        """Start this task's game again, however many times it is asked for.

        `register_games` builds a batch out of `game_files`, and for an empty one
        the batch's `reset` never returns - so with no game set this refuses
        rather than hanging the experiment.
        """
        if self.gamefile is None:
            raise RuntimeError('set_env chooses the game; reset cannot run before it.')

        if self.env is not None:
            self.env.close()  # one interpreter per game, and 134 games per experiment
        self.env = self.main_env.init_env(batch_size=1)
        self.env.reset()

        self.done = False
        self.won = False

    def step(self, action: str) -> tuple[str, float, bool]:

        action = self.process_action(action)

        if self.is_thought(action):
            return 'OK.', -1, False

        observation, _, done, info = self.env.step([action])
        observation = process_ob(observation[0])

        self.won = bool(info['won'][0])
        self.done = bool(done[0])

        if observation == REJECTED:
            reward = -1
        else:
            reward = 1 if self.won else 0

        return observation, reward, self.done

    def feedback(self) -> tuple[float, bool, str]:
        """What the episode scored, which `done` alone cannot say.

        TextWorld reports `done` for an episode that ran out of steps as well as
        for one that was won, so the win is read off `info['won']` instead.
        """
        message = "You successfully finished this task!" if self.won else "You failed the task."

        return 1.0 if self.won else 0.0, self.won, message

    def _parse_task_main(self, task: str):
        return self.env_name + '-' + re.search(r'Your task is to:\s*(.+)', task, re.DOTALL).group(1).strip()

    @staticmethod
    def _parse_task_description(task: str) -> str:
        return task.split('___')[0]


@dataclass
class AlfworldRecorder(BaseRecorder):

    def __post_init__(self):

        super().__post_init__()
        self.task = 'alfworld'
        # Episodes grouped by ALFWorld task type, for the per-type breakdown in
        # the log. The overall aggregate comes from BaseRecorder.
        self.episodes_by_task_type: dict[str, list[EpisodeResult]] = {
            name: [] for name in prefixes
        }

    def task_begin(self, task_id, task_config):
        super().task_begin(task_id, task_config)

        message: str = f'---------- Task: {task_id} ----------'
        self.log(message)

    def task_end(self, episode: EpisodeResult):
        super().task_end(episode)

        gamefile: str = self.current_task_config['env_kwargs']['gamefile']
        env_name = get_env_name_from_gamefile(gamefile)
        if env_name is None:
            raise ValueError('Format of the task config is wrong.')

        self.episodes_by_task_type[env_name].append(episode)

        self.log(f'done: {episode.done}, ave done: {self.average_results().mean_done}')
        for name, episodes in self.episodes_by_task_type.items():
            if episodes:
                self.log(f'  {prefixes[name]}: {aggregate(episodes)}')
