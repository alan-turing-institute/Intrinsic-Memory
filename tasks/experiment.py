"""One experiment: build the system it asks for, then run it over a dataset.

An experiment is one point in a sweep - a task, a workflow, a memory module, a
model and a seed. `sweep.py` decides which ones run; this decides what running
one means.
"""

import os
import random
import sys
import traceback
import yaml
from dataclasses import dataclass, field
from tqdm import tqdm

from mas.logging_utils import log_mas_to_console
from mas.module_map import module_map
from mas.reasoning import ReasoningBase
from mas.memory import MASMemoryBase
from mas.llm import LLMCallable, GPTChat, TokenTracker
from mas.mas import EpisodeResult, MetaMAS
from mas.settings import LLMSettings, use_llm_settings
from mas.utils import EmbeddingFunc, repo_path

import results
from experiment_config import ExperimentConfig
from envs import BaseEnv, BaseRecorder, get_env, get_recorder, get_task
from mas_workflow import get_mas
from prompts import get_dataset_system_prompt, get_task_few_shots
from utils import model_dir_name

CONFIG_PATH = repo_path('tasks', 'configs.yaml')

with open(CONFIG_PATH) as reader:
    CONFIG: dict = yaml.safe_load(reader)


def install_llm_settings(experiment_config: ExperimentConfig) -> LLMSettings:
    """Make this experiment's LLM flags the settings its GPTChats will use.

    Workers are spawned rather than forked, so what the parent installed does
    not reach them: each one installs its own from the config it was handed.
    """
    settings = LLMSettings.load(
        max_tokens=experiment_config.max_tokens,
        max_tokens_ceiling=experiment_config.max_tokens_ceiling,
        thinking_token_budget=experiment_config.thinking_token_budget,
        temperature=experiment_config.temperature,
        request_timeout=experiment_config.request_timeout,
        log_responses=experiment_config.log_responses,
    )
    use_llm_settings(settings)

    if settings.log_responses:
        log_mas_to_console()

    return settings


@dataclass
class TaskManager:
    task_name: str              # task name
    mas_type: str               # type of mas
    memory_type: str            # memory type
    tasks: list[dict]           # all tasks
    env: BaseEnv                # interative datatset environment
    recorder: BaseRecorder      # record experiment results
    mas: MetaMAS                # multi-agent system
    seed: int = None            # this experiment's seed
    model: str = None           # the LLM this experiment queries
    mas_config: dict = field(default_factory=dict)   # mas configs
    mem_config: dict = field(default_factory=dict)   # memory configs
    token_tracker: TokenTracker = None   # token accounting for this experiment's LLM calls

    def identity(self) -> dict:
        """Which experiment a result row belongs to."""
        return results.identity(
            model=self.model,
            task=self.task_name,
            mas_type=self.mas_type,
            mas_memory=self.memory_type,
            use_validator=self.mas_config.get('use_validator', False),
            intrinsic_cross_task=self.mem_config.get('intrinsic_cross_task', False),
        )


def trial_budget(task: str, override: int = None) -> int:
    """Trials one episode of `task` gets: its `max_steps`, or `override`."""
    if override is not None:
        return override

    budget = CONFIG.get(task, {}).get('max_steps')
    if budget is None:
        raise KeyError(f"'{task}' has no max_steps in tasks/configs.yaml")
    return budget


def dataset_size(task: str, override: int = None) -> int:
    """How many of `task`'s tasks a run covers: its `max_tasks`, or `override`.

    None either way means the whole dataset.
    """
    if override is not None:
        return override

    return CONFIG.get(task, {}).get('max_tasks')


def build_task(
    task: str,
    mas_type: str,
    memory_type: str,
    seed: int,
    working_dir: str,
    model: str = None,
    max_trials: int = None,
    max_tasks: int = None,
    env_overrides: dict = None,
) -> TaskManager:

    with open(repo_path(CONFIG.get(task).get('env_config_path'))) as reader:
        config = yaml.safe_load(reader) or {}
    config.update(
        (name, value) for name, value in (env_overrides or {}).items() if value is not None
    )

    env: BaseEnv = get_env(task, config, trial_budget(task, max_trials))
    recorder: BaseRecorder = get_recorder(task, working_dir=working_dir, namespace=f'total_task-seed_{seed}')
    tasks: list[dict] = get_task(task, max_tasks=dataset_size(task, max_tasks))
    mas_workflow: MetaMAS = get_mas(mas_type)
    mas_config: dict = CONFIG.get(mas_type, {})

    return TaskManager(
        task_name=task,
        mas_type=mas_type,
        memory_type=memory_type,
        tasks=tasks,
        env=env,
        recorder=recorder,
        mas=mas_workflow,
        seed=seed,
        model=model,
        mas_config=mas_config
    )

def build_mas(
    task_manager: TaskManager,
    reasoning: str = None,
    mas_memory: str = None,
    llm_type: str = None,
) -> None:
    
    embed_func = EmbeddingFunc(
        CONFIG.get('embedding_model', "sentence-transformers/all-MiniLM-L6-v2"),
        device=CONFIG.get('embedding_device', 'cpu'),
    )
    reasoning_module_type, mas_memory_module_type = module_map(reasoning, mas_memory)

    llm_model: LLMCallable = GPTChat(model_name=llm_type)
    task_manager.token_tracker = llm_model.tracker
    reasoning_module: ReasoningBase = reasoning_module_type(llm_model=llm_model)
    mas_memory_module: MASMemoryBase = mas_memory_module_type(
        namespace=mas_memory,
        global_config=task_manager.mem_config,
        llm_model=llm_model,
        embedding_func=embed_func 
    )
    task_manager.mas.add_observer(task_manager.recorder)  
    task_manager.mas.build_system(reasoning_module, mas_memory_module, task_manager.env, task_manager.mas_config)

def run_task(
    task_manager: TaskManager,
    working_dir: str,
    failed_tasks_filename: str,
) -> None:
    task_manager.recorder.dataset_begin()
    task_results_path = results.task_results_path(
        working_dir, task_manager.task_name, task_manager.memory_type
    )
    progress_path = results.progress_path(
        working_dir, task_manager.task_name, task_manager.memory_type, task_manager.seed
    )

    failed_tasks: list[dict] = []

    for task_id, task_config in tqdm(enumerate(task_manager.tasks), total=len(task_manager.tasks), desc="Running Tasks"):
        before = task_manager.token_tracker.snapshot()
        try:
            task_manager.recorder.task_begin(task_id, task_config)

            task_main, task_description = task_manager.mas.env.set_env(task_config)
            few_shots: list[str] = get_task_few_shots(
                dataset=task_manager.task_name,
                task_config=task_config,
                few_shots_num=CONFIG.get(task_manager.task_name).get('few_shots_num', 0)
            )
            task_config.update(task_main=task_main, task_description=task_description, few_shots=few_shots)

            task_instruction: str = get_dataset_system_prompt(task_manager.task_name, task_config=task_config)
            for agent in task_manager.mas.agents_team.values():
                task_manager.recorder.log(f'------------ MAS Agent: {agent.name} ------------')
                task_manager.recorder.log(agent.add_task_instruction(task_instruction))

            episode: EpisodeResult = task_manager.mas.schedule(task_config) # Schedule method from the mas_workflow (e.g. autogen)
            task_manager.recorder.task_end(episode)
        except Exception as error:
            failed_tasks.append({'task_id': task_id, 'error': error})
            task_manager.recorder.log(
                f'TASK FAILED task_id={task_id}: {type(error).__name__}: {error}\n'
                f'{traceback.format_exc()}'
            )
            print(
                f'Task failed, continuing: task={task_manager.task_name} task_id={task_id} '
                f'{type(error).__name__}: {error}',
                file=sys.stderr,
            )
            continue

        tracker = task_manager.token_tracker
        completion_tokens, prompt_tokens = tracker.completion_tokens, tracker.prompt_tokens
        intrinsic_completion_tokens = tracker.intrinsic_completion_tokens
        intrinsic_prompt_tokens = tracker.intrinsic_prompt_tokens
        task_manager.recorder.log(f'completion_tokens:{completion_tokens}, prompt_tokens:{prompt_tokens}\n')
        task_manager.recorder.log(f'intrinsic completion tokens:{intrinsic_completion_tokens}, intrinsic_prompt_tokens:{intrinsic_prompt_tokens}\n')
        task_manager.recorder.log(f'seed: {task_manager.seed}\n')

        identity = task_manager.identity()
        max_trials = task_manager.env.max_trials

        # the raw numbers for this task, which no aggregate can be worked back to
        results.write_row(
            task_results_path,
            results.TASK_COLUMNS,
            results.task_row(
                identity_fields=identity,
                seed=task_manager.seed,
                max_trials=max_trials,
                task_id=task_id,
                episode=episode,
                spent=tracker.since(before),
            ),
        )

        # means so far, so a killed job leaves a result without processing
        print(results.write_row(
            progress_path,
            results.AGGREGATE_COLUMNS,
            results.aggregate_row(
                identity_fields=identity,
                seed=task_manager.seed,
                max_trials=max_trials,
                averages=task_manager.recorder.average_results(),
                tracker=tracker,
            ),
        ))

    if failed_tasks:
        summary = (
            f'{len(failed_tasks)}/{len(task_manager.tasks)} tasks failed and are excluded '
            f'from the averages above'
        )
        task_manager.recorder.log(summary)
        print(summary, file=sys.stderr)
        _write_failed_tasks(task_manager, failed_tasks, working_dir, failed_tasks_filename)

    task_manager.recorder.dataset_end()


def _write_failed_tasks(
    task_manager: TaskManager,
    failed_tasks: list[dict],
    working_dir: str,
    failed_tasks_filename: str,
) -> None:
    """One row per task that could not be run, alongside the experiment's results."""
    path = results.failed_tasks_path(working_dir, failed_tasks_filename)

    for failure in failed_tasks:
        results.write_row(
            path,
            results.FAILED_TASK_COLUMNS,
            {
                **task_manager.identity(),
                'task_id': failure['task_id'],
                'seed': task_manager.seed,
                **results.failure_fields(failure['error']),
            },
        )


# Flags that configure an environment rather than the sweep, merged into the
# env_config its task's YAML supplies.
ENV_FLAGS = (
    'wikipedia_attempts',
    'wikipedia_retry_seconds',
    'unreachable_search_limit',
)


def run_experiment(experiment_config: ExperimentConfig) -> dict:
    task_name = experiment_config.task
    mas_type = experiment_config.mas_type
    mas_memory_type = experiment_config.mas_memory
    reasoning_type = experiment_config.reasoning
    model_type = experiment_config.model
    max_trials = experiment_config.max_trials
    max_tasks = experiment_config.max_tasks
    seed = experiment_config.seed
    successful_topk = experiment_config.successful_topk
    failed_topk = experiment_config.failed_topk
    insights_topk = experiment_config.insights_topk
    threshold = experiment_config.threshold
    use_projector = experiment_config.use_projector
    use_validator = experiment_config.use_validator
    hop = experiment_config.hop
    intrinsic_cross_task = experiment_config.intrinsic_cross_task
    db_dir = experiment_config.db_dir
    overall_results_filename = experiment_config.overall_results_filename
    failed_tasks_filename = experiment_config.failed_tasks_filename
    failed_experiments_filename = experiment_config.failed_experiments_filename

    # set save dirs
    working_dir = os.path.join(db_dir, model_dir_name(model_type), task_name, mas_type, f'{mas_memory_type}')
    os.makedirs(working_dir, exist_ok=True)

    try:
        install_llm_settings(experiment_config)
        random.seed(seed)

        task_configs: TaskManager = build_task(
            task_name, mas_type, mas_memory_type, seed, working_dir,
            model=model_type, max_trials=max_trials, max_tasks=max_tasks,
            env_overrides={name: getattr(experiment_config, name) for name in ENV_FLAGS},
        )
        task_configs.mas_config['successful_topk'] = successful_topk
        task_configs.mas_config['failed_topk'] = failed_topk
        task_configs.mas_config['insights_topk'] = insights_topk
        task_configs.mas_config['threshold'] = threshold
        task_configs.mas_config['use_projector'] = use_projector
        task_configs.mas_config['use_validator'] = use_validator

        # each seed gets its own memory persistence dir so concurrent seeds of the same
        # experiment config never read/write the same graph/vector-store/insights files
        memory_dir = os.path.join(working_dir, f'seed_{seed}')
        task_configs.mem_config.update(
            working_dir=memory_dir,
            hop=hop,
            intrinsic_cross_task=intrinsic_cross_task,
        )

        build_mas(task_configs, reasoning_type, mas_memory_type, model_type)
        run_task(task_configs, working_dir, failed_tasks_filename)

        tracker = task_configs.token_tracker
        completion_tokens, prompt_tokens = tracker.completion_tokens, tracker.prompt_tokens
        intrinsic_completion_tokens = tracker.intrinsic_completion_tokens
        intrinsic_prompt_tokens = tracker.intrinsic_prompt_tokens
        task_configs.recorder.log(f'completion_tokens:{completion_tokens}, prompt_tokens:{prompt_tokens}')
        task_configs.recorder.log(f'intrinsic completion tokens:{intrinsic_completion_tokens}, intrinsic_prompt_tokens:{intrinsic_prompt_tokens}')

        result_line = results.write_row(
            results.overall_results_path(db_dir, overall_results_filename),
            results.AGGREGATE_COLUMNS,
            results.aggregate_row(
                identity_fields=task_configs.identity(),
                seed=seed,
                max_trials=task_configs.env.max_trials,
                averages=task_configs.recorder.average_results(),
                tracker=tracker,
            ),
        )
        print(result_line)

        # the experiment has its result, so its crash-recovery file has no reader
        results.remove_progress(
            results.progress_path(working_dir, task_name, mas_memory_type, seed)
        )

        return {
            'task': task_name,
            'mas_type': mas_type,
            'mas_memory': mas_memory_type,
            'seed': seed,
            'result_string': result_line,
            'status': 'success',
        }
    except Exception as e:
        print(f"Experiment failed: task={task_name} mas_memory={mas_memory_type} seed={seed}\n{traceback.format_exc()}", file=sys.stderr)
        failed_path = _write_failed_experiment(experiment_config, e, db_dir, failed_experiments_filename)
        return {
            'task': task_name,
            'mas_type': mas_type,
            'mas_memory': mas_memory_type,
            'seed': seed,
            'status': 'failed',
            'error': f'{type(e).__name__}: {e}',
            'failed_path': failed_path,
        }


def _write_failed_experiment(
    experiment_config: ExperimentConfig,
    error: Exception,
    db_dir: str,
    failed_experiments_filename: str,
) -> str:
    failed_path = results.failed_experiments_path(db_dir, failed_experiments_filename)
    results.write_row(
        failed_path,
        results.FAILED_EXPERIMENT_COLUMNS,
        {
            **experiment_config.identity(),
            'seed': experiment_config.seed,
            **results.failure_fields(error),
        },
    )
    return failed_path
