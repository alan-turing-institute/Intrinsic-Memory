import json
import jsonlines

from mas.utils import repo_path

from .base_env import BaseEnv, BaseRecorder
from .alfworld_env import AlfworldEnv, AlfworldRecorder, get_env_name_from_gamefile, prefixes
from .babyai_env import BabyAIEnv, BabyAIRecorder
from .sciworld_env import SciworldEnv, SciworldRecorder, build_simplification_str
from .swebench_env import SWEBenchEnv, SWEBenchRecorder
from .fever_env import FeverEnv, FeverRecorder
from .hotpotqa_env import HotpotQAEnv, HotpotQARecorder
from .jericho_env import JerichoEnv, JerichoRecorder
from .pddl_env.pddl_env import PDDLEnv, PDDLRecorder, get_all_environment_configs

TASKS_PATH = {
    'alfworld': repo_path('data/alfworld/alfworld_tasks_suffix.json'),
    'babyai': repo_path('data/babyai/babyai_levels.jsonl'),
    'fever': repo_path('data/fever/fever_dev.jsonl'),
    'hotpotqa': repo_path('data/hotpotqa/hotpotqa_dev.jsonl'),
    'jericho': repo_path('data/jericho/jericho_games.jsonl'),
    'pddl': repo_path('data/pddl/test.jsonl'),
    'sciworld': repo_path('data/sciworld/test.jsonl'),
    'swebench': repo_path('data/swebench/verified.jsonl'),
}

PDDL_DOMAINS = ["barman", "blockworld", "gripper", "tyreworld"]


def load_alfworld_tasks() -> list[dict]:
    with open(TASKS_PATH['alfworld'], 'r') as reader:
        rows = json.load(reader)

    return [
        {
            'task': f'{row["goal"]}',
            'env_kwargs': {
                'config': 'alfworld',
                "gamefile": row["gamefile"],
            },
            'task_type': prefixes[get_env_name_from_gamefile(row["gamefile"])],
            'env_name': get_env_name_from_gamefile(row["gamefile"])
        } for row in rows
    ]


def load_sciworld_tasks() -> list[dict]:
    with open(TASKS_PATH['sciworld'], 'r', encoding='utf-8') as reader:
        return [
            {
                "id": item['id'],
                "task_name": item["additional_info"]["env_name"],
                "var": item["additional_info"]["var"],
                "modified_goal": item["goal"],
                "subgoals": item['subgoals'],
                "difficulty": item["difficulty"],
                "simplification_str": build_simplification_str()
            }
            for item in jsonlines.Reader(reader)
        ]


def load_fever_tasks() -> list[dict]:
    with open(TASKS_PATH['fever'], 'r') as reader:
        return [
            {
                'task': row['claim'],
                'answer': row['label'],
                'env_name': 'fever',
            }
            for row in (json.loads(line) for line in reader)
        ]


def load_hotpotqa_tasks() -> list[dict]:
    with open(TASKS_PATH['hotpotqa'], 'r') as reader:
        return [
            {
                'task': row['question'],
                'answer': row['answer'],
                'env_name': 'hotpotqa',
            }
            for row in (json.loads(line) for line in reader)
        ]


def load_babyai_tasks() -> list[dict]:
    """One task per (level, seed): the seed is what fixes the gridworld."""
    with open(TASKS_PATH['babyai'], 'r') as reader:
        return [
            {
                'id': row['id'],
                'level': row['level'],
                'seed': row['seed'],
                'env_name': 'babyai',
            }
            for row in (json.loads(line) for line in reader)
        ]


def load_jericho_tasks() -> list[dict]:
    """One task per game. The rom itself is not in the repo; see data/data.md."""
    with open(TASKS_PATH['jericho'], 'r') as reader:
        return [
            {
                'id': row['id'],
                'rom': row['rom'],
                'env_name': 'jericho',
            }
            for row in (json.loads(line) for line in reader)
        ]


def load_swebench_tasks() -> list[dict]:
    """One task per instance. The instance's image is not in the repo, and nor is
    the task repository the issue and the tests are read from; see
    containers/isambard.md."""
    with open(TASKS_PATH['swebench'], 'r') as reader:
        return [
            {
                'id': row['id'],
                'repo': row['repo'],
                'env_name': 'swebench',
            }
            for row in (json.loads(line) for line in reader)
        ]


def load_pddl_tasks() -> list[dict]:
    return get_all_environment_configs(PDDL_DOMAINS, TASKS_PATH['pddl'])


TASK_LOADERS = {
    'alfworld': load_alfworld_tasks,
    'babyai': load_babyai_tasks,
    'sciworld': load_sciworld_tasks,
    'fever': load_fever_tasks,
    'hotpotqa': load_hotpotqa_tasks,
    'jericho': load_jericho_tasks,
    'pddl': load_pddl_tasks,
    'swebench': load_swebench_tasks,
}

ENVS = {
    'alfworld': AlfworldEnv,
    'babyai': BabyAIEnv,
    'sciworld': SciworldEnv,
    'fever': FeverEnv,
    'hotpotqa': HotpotQAEnv,
    'jericho': JerichoEnv,
    'pddl': PDDLEnv,
    'swebench': SWEBenchEnv,
}

RECORDERS = {
    'alfworld': AlfworldRecorder,
    'babyai': BabyAIRecorder,
    'sciworld': SciworldRecorder,
    'fever': FeverRecorder,
    'hotpotqa': HotpotQARecorder,
    'jericho': JerichoRecorder,
    'pddl': PDDLRecorder,
    'swebench': SWEBenchRecorder,
}

_datasets: dict[str, list[dict]] = {}


def get_env(task: str, env_config: dict, max_trials: int) -> BaseEnv:
    
    if ENVS.get(task) is None:
        raise ValueError(f'Unsupported task type: {task}')
    
    return ENVS.get(task)(env_config, max_trials)

def get_recorder(task: str, working_dir: str, namespace: str) -> BaseRecorder:
    
    if RECORDERS.get(task) is None:
        raise ValueError(f'Unsupported task type: {task}')
    
    return RECORDERS.get(task)(working_dir=working_dir, namespace=namespace)

def get_task(task: str, max_tasks: int = None) -> list[dict]:
    """The dataset for `task`, parsed on first use and kept for the next call.

    `max_tasks` takes the first n tasks. It is per-task configuration, in
    tasks/configs.yaml - not a sample, so two runs of the same task cover the
    same tasks.
    """
    if TASK_LOADERS.get(task) is None:
        raise ValueError(f'Unsupported task type: {task}')

    if task not in _datasets:
        _datasets[task] = TASK_LOADERS[task]()

    dataset = _datasets[task]
    return dataset[:max_tasks] if max_tasks is not None else dataset
