from dataclasses import dataclass

from mas.agents import Agent
from mas.memory.common import MASMessage, AgentMessage
from mas.mas import AgentCallFailed, EpisodeResult, MetaMAS
from mas.reasoning import ReasoningBase, ReasoningConfig
from mas.memory import MASMemoryBase
from mas.agents import Env

from .autogen_prompt import AUTOGEN_PROMPT 
from ..format import format_task_prompt_with_insights, format_task_context

import sys


@dataclass
class AutoGen(MetaMAS):
    """A solver agent acting in the environment, with a ground-truth agent as backstop.

    With `use_validator`, a third agent reviews each proposed action for format
    only - it answers `VALID`, or `INVALID: <reason>` - and a rejection re-prompts
    the solver with that feedback, spending one try from the episode's shared
    budget. If the budget runs out while the validator is still refusing, the last
    rejected action is taken anyway: a disputed but well-formed action can reach
    the environment where an empty one cannot. The validator keeps its own memory
    instance, so the two agents' updates cannot overwrite each other.
    """

    def __post_init__(self):

        self.solver_name: str = 'solver'
        self.validator_name: str = 'validator'
        self.ground_truth_name: str = 'ground_truth'
        self.observers = []   

        self.reasoning_config = ReasoningConfig(temperature=0, stop_strs=['\n'])

    def build_system(self, reasoning: ReasoningBase, mas_memory: MASMemoryBase, env: Env, config: dict):  
        """
        build_system is called once per benchmark in run.py to set up the multi-agent system. Then the schedule method is called once per task
        """
        self._successful_topk: int = config.get('successful_topk', 1)
        self._failed_topk: int = config.get('failed_topk', 1)
        self._insights_topk: int = config.get('insights_topk', 3)
        self._threshold: float = config.get('threshold', 0)
        self._use_projector: bool = config.get('use_projector', False)
        self._use_validator: bool = config.get('use_validator', False)
        self.notify_observers(f"Successful Topk   : {self._successful_topk}")
        self.notify_observers(f"Failed Topk       : {self._failed_topk}")
        self.notify_observers(f"Insights Topk     : {self._insights_topk}")
        self.notify_observers(f"Retrieve Threshold: {self._threshold}")
        self.notify_observers(f"Use Role Projector: {self._use_projector}")
        self.notify_observers(f"Use Validator     : {self._use_validator}")

        if not isinstance(reasoning, ReasoningBase):
            raise TypeError("reasoning module must be an instance of ReasoningBase")
        if not isinstance(mas_memory, MASMemoryBase):
            raise TypeError("mas_memory module must be an instance of MASMemoryBase")
        
        solver_agent: Agent = Agent(
            name=self.solver_name, 
            role='solver', 
            system_instruction=AUTOGEN_PROMPT.solver_system_prompt,
            reasoning_module=reasoning,
            memory_module=None
        )

        ground_truth_agent: Agent = Agent(
            name=self.ground_truth_name,
            role="ground truth agent",
            system_instruction=AUTOGEN_PROMPT.ground_truth_system_prompt,
            reasoning_module=reasoning,
            memory_module=None           
        )

        team: list[Agent] = [solver_agent, ground_truth_agent]
        self.meta_memory_validator: MASMemoryBase = None

        if self._use_validator:
            team.append(Agent(
                name=self.validator_name,
                role='validator',
                system_instruction=AUTOGEN_PROMPT.validator_system_prompt,
                reasoning_module=reasoning,
                memory_module=None
            ))
            # Its own instance, so solver and validator updates cannot overwrite
            # each other.
            self.meta_memory_validator = mas_memory.for_namespace("_validator")

        self.hire(team)
        self.set_env(env)
        self.meta_memory = mas_memory
        
    
    def schedule(self, task_config: dict) -> EpisodeResult:
        """
        Schedules and executes a task according to the given task configuration.
        This function initializes the task context based on the configuration, retrieves relevant memories and insights,
        and then executes the task by interacting with the environment and agents. It also handles memory updates and feedback.
        
        Parameters:
        - task_config (dict): A dictionary containing the task configuration, including the main task and description.
        
        Returns:
        - EpisodeResult: the final reward, whether the task was completed, and the trial it ended on.
        """
        if task_config.get('task_main') is None:
            raise ValueError("Missing required keys `task_main` in task_config")
        if task_config.get('task_description') is None:
            raise ValueError("Missing required keys `task_description` in task_config")
        
        task_main: str = task_config.get('task_main')
        task_description: str = task_config.get('task_description')
        few_shots: list[str] =  task_config.get("few_shots", [])
        
        # Initialize environment and agents
        env: Env = self.env
        solver: Agent = self.get_agent(self.solver_name)
        validator: Agent = self.get_agent(self.validator_name)
        ground_truth: Agent = self.get_agent(self.ground_truth_name)
        env.reset()
        
        self.meta_memory.init_task_context(task_main, task_description) 
        if self._use_validator:
            self.meta_memory_validator.init_task_context(task_main, task_description)
        
        # Retrieve successful trajectories and insights from memory
        successful_trajectories: list[MASMessage]
        insights: list[dict]
        
        successful_trajectories, _, insights = self.meta_memory.retrieve_memory(
            query_task=task_main,
            successful_topk=self._successful_topk,
            failed_topk=self._failed_topk,
            insight_topk=self._insights_topk,
            threshold=self._threshold
        )
        successful_shots: list[str] = [format_task_context(
            traj.task_description, traj.task_trajectory, traj.get_extra_field('key_steps')
        ) for traj in successful_trajectories]
        raw_rules: list[str] = [insight for insight in insights]
        roles_rules: dict[str, list[str]] = self._project_insights(raw_rules)
        
        # Generate initial user prompt with insights
        user_prompt: str = format_task_prompt_with_insights(
            few_shots=few_shots, 
            memory_few_shots=successful_shots,
            insights=raw_rules,
            task_description=self.meta_memory.summarize(solver_message=solver.total_system_instruction)
        )
        self.notify_observers(user_prompt)

        # Main loop for task execution
        action_history: list = [] 
        trials = 0
        
        for i in range(env.max_trials):    
            
            user_prompt: str = format_task_prompt_with_insights(
                few_shots=few_shots, 
                memory_few_shots=successful_shots,
                insights=roles_rules.get(solver.profile, raw_rules),
                task_description=self.meta_memory.summarize(solver_message=solver.total_system_instruction)
            )
            print(f"\n==== SOLVER AGENT PROMPT ====\n{user_prompt}\n==== END SOLVER AGENT PROMPT ====\n", file=sys.stderr)

            def solve() -> str:
                return env.process_action(solver.response(user_prompt, self.reasoning_config))

            attempt = solve

            if self._use_validator:

                def propose(rejected: tuple[str, str]) -> str:
                    if rejected is None:
                        return solver.response(user_prompt, self.reasoning_config)

                    refused, verdict = rejected
                    revision: str = AUTOGEN_PROMPT.solver_revision_prompt.format(
                        action=refused, evaluation=verdict
                    )
                    print(f'==== SOLVER INSTRUCTION FOR REVISION ====\n{revision}\n==== END SOLVER INSTRUCTION FOR REVISION ====\n', file=sys.stderr)
                    return solver.response(f"{revision}{user_prompt}", self.reasoning_config)

                def review(action: str) -> str:
                    validator_prompt: str = AUTOGEN_PROMPT.validator_user_prompt.format(
                        action=action,
                        task_description=task_config.get('task_description'),
                        few_shots="\n".join(few_shots),
                    )
                    print(f'==== VALIDATOR PROMPT ====\n{validator_prompt}\n==== END VALIDATOR PROMPT ====\n', file=sys.stderr)

                    evaluation: str = validator.response(validator_prompt, self.reasoning_config)
                    print(f'==== VALIDATOR EVALUATION ====\n{evaluation}\n==== END VALIDATOR EVALUATION ====\n', file=sys.stderr)

                    self.meta_memory_validator.summarize(
                        solver_message=f"## Your latest evaluation: \n {evaluation}"
                    )
                    return evaluation

                attempt = self._reviewed_attempt(propose, review, env.process_action)

            try:
                action: str = self._call_agent_with_retries(
                    attempt,
                    description='solver agent',
                )
            except AgentCallFailed as failure:
                # Ends this episode only; feedback() below still scores it. The
                # trial count goes unreported: the episode was cut short, so how
                # many turns the task needed was never established.
                self.notify_observers(f'Ending episode at trial {i + 1}: {failure}')
                trials = None
                break

            name: str = solver.name
            system_instruction = solver.system_instruction
            
            if self._solver_stuck(action, action_history):
                user_prompt: str = format_task_prompt_with_insights(
                    few_shots=few_shots, 
                    memory_few_shots=successful_shots,
                    insights=roles_rules.get(ground_truth.profile, raw_rules),
                    task_description=self.meta_memory.summarize(solver_message=solver.total_system_instruction)
                )

                print(f'==== GROUND TRUTH AGENT PROMPT ==== \n{user_prompt}\n====END GROUNDTRUTH AGENT PROMPT ====\n', file=sys.stderr)

                try:
                    action: str = self._call_agent_with_retries(
                        lambda: env.process_action(ground_truth.response(user_prompt, self.reasoning_config)),
                        description='ground truth agent',
                    )
                except AgentCallFailed as failure:
                    self.notify_observers(f'Ending episode at trial {i + 1}: {failure}')
                    trials = None
                    break
                name: str = ground_truth.name
                system_instruction = ground_truth.system_instruction
            
            agent_message: AgentMessage = AgentMessage(
                agent_name=name,
                system_instruction=system_instruction,
                user_instruction=user_prompt,
                message=action,
            )
            self.meta_memory.add_agent_node(agent_message, upstream_agent_ids=[])

            observation, reward, done = env.step(action)
            action_history.append(action)
            
            step_message: str = f'Act {i + 1}: {action}\nObs {i + 1}: {observation}'
            self.notify_observers(step_message)

            self.meta_memory.move_memory_state(action, observation, reward=reward)   
            trials = i + 1
            if done:  
                break

        # Final feedback and memory update
        final_reward, final_done, final_feedback = self.env.feedback()
        self.notify_observers(final_feedback)
        self.meta_memory.save_task_context(label=final_done, feedback=final_feedback)  
        if self._use_validator:
            self.meta_memory_validator.save_task_context(label=final_done, feedback=final_feedback)
        self.meta_memory.backward(final_done)    

        return EpisodeResult(reward=final_reward, done=final_done, trials=trials)
    
        
            
