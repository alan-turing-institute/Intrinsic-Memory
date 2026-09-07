from .alfworld_prompt import alfworld_solver_system_prompt, alfworld_few_shots
from .babyai_prompt import babyai_solver_system_prompt, babyai_few_shots
from .sciworld_prompt import sciworld_solver_system_prompt, sciworld_few_shots
from .fever_prompt import fever_solver_system_prompt, fever_few_shots
from .hotpotqa_prompt import hotpotqa_solver_system_prompt, hotpotqa_few_shots
from .jericho_prompt import jericho_solver_system_prompt, jericho_few_shots
from .swebench_prompt import swebench_solver_system_prompt, swebench_few_shots
from .pddl_prompt import pddl_prompts

# Appended to every dataset's system prompt, so that what the agent is told about
# the shape of a reply does not depend on which memory module is being measured.
# Only `IntrinsicMASMemory` used to say it, in the task description it renders, so
# the intrinsic arms were the only ones asked for one line - and once a stop
# sequence is dropped that instruction is the only thing left asking.
ONE_ACTION_PER_REPLY: str = (
    "\nOutput exactly one line, and nothing else: your single next action, with no "
    "numbering, no quotes, no observation of your own, and no explanation around it.\n"
)


def get_dataset_system_prompt(task: str, task_config: dict) -> str:
    prompt_map: dict = {
        'alfworld': alfworld_solver_system_prompt,
        'babyai': babyai_solver_system_prompt,
        'sciworld': sciworld_solver_system_prompt,
        'fever': fever_solver_system_prompt,
        'hotpotqa': hotpotqa_solver_system_prompt,
        'jericho': jericho_solver_system_prompt,
        'swebench': swebench_solver_system_prompt,
        'pddl': pddl_prompts
    }

    if prompt_map.get(task) is None:
        raise ValueError(f'Unsupported task type: {task}')
    
    if task != 'pddl':
        return prompt_map.get(task) + ONE_ACTION_PER_REPLY
    else:
        task_type: str = task_config.get('game_name')
        return pddl_prompts[task_type]['instruction'] + ONE_ACTION_PER_REPLY
        


def get_task_few_shots(dataset: str, task_config: dict, few_shots_num: int) -> list[str]:
    
    if dataset == 'alfworld':
        task_type = task_config.get('task_type')
        if task_type is None:
            raise ValueError('The task config must have the `task_type` attribute.')
        return [alfworld_few_shots[f'react_{task_type}_2'], alfworld_few_shots[f'react_{task_type}_0']][:few_shots_num]
    
    elif dataset == 'babyai':
        return babyai_few_shots[:few_shots_num]

    elif dataset == 'jericho':
        return jericho_few_shots[:few_shots_num]

    elif dataset == 'swebench':
        return swebench_few_shots[:few_shots_num]

    elif dataset == 'sciworld':
        return sciworld_few_shots[:few_shots_num]
    
    elif dataset == 'fever':
        return fever_few_shots[:few_shots_num]
    
    elif dataset == 'hotpotqa':
        return hotpotqa_few_shots[:few_shots_num]

    elif dataset == 'pddl':
        task_type = task_config.get('game_name')
        if task_type is None:
            raise ValueError('The task config must have the `game_name` attribute.')
        return pddl_prompts[task_type]['examples'][:few_shots_num]
    
    else:
        raise ValueError(f'Unsupported dataset type: {dataset}')

