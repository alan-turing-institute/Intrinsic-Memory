from dataclasses import dataclass
import logging

from .memory_base import MASMemoryBase
from .prompt import (
    INTRINSICMEMORY_ALFWORLD,
    INTRINSICMEMORY_BABYAI,
    INTRINSICMEMORY_DEFAULT,
    INTRINSICMEMORY_FEVER,
    INTRINSICMEMORY_HOTPOTQA,
    INTRINSICMEMORY_JERICHO,
    INTRINSICMEMORY_NOTEMPLATE,
    INTRINSICMEMORY_PDDL,
    INTRINSICMEMORY_SCIWORLD,
    INTRINSICMEMORY_SWEBENCH,
)
from ..common import MASMessage # a MASMessage, which is a specific type of message used in MAS
from mas.llm import Message, GPTChat # a "normal" message, not a MASMessage?

logger = logging.getLogger(__name__)

@dataclass
class IntrinsicMASMemory(MASMemoryBase):
    """
    IntrinsicMASMemory keeps agent-specific memory. 
    
    Task context is stored persistently in mas_message, which includes the task 
    description (mas_message.task_description) and trajectory (mas_message.task_trajectory) (i.e., the sequence of actions taken to complete the task).)

    A separate memory is used to store the agent's memory, which is updated with the latest information from the agent's messages.
    
    """
    # Class attributes, deliberately unannotated so they stay off the
    # constructor: build_system rebuilds a memory as `memory.__class__(...)`,
    # which drops anything passed in.
    system_prompt = ""
    memory_update_prompt = INTRINSICMEMORY_DEFAULT.memory_update_prompt

    def __post_init__(self):
        super().__post_init__()
        self.counter: int = 0
        self.agent_intrinsic_memory: str = ""

    def summarize(self, *, solver_message: str = "", template_instructions: str = "") -> str:

        """UPDATE AGENT MEMORY STEP"""

        # Update agent's memory using existing agent memory, latest output from the agent, and the memory update prompt
        mas_message: MASMessage = self.current_task_context
        if self.current_task_context is None:
            raise RuntimeError('The current task memory is empty.')

        # Construct the user prompt for memory update with memory update prompt, task description, and latest agent output
        memory_update_prompt = self.memory_update_prompt.format(
                custom_message=solver_message, # I have modified this so that we can pass custom instructions and information to each agent. It now appears in the intrinsicmemory prompt for baseline and notemplate (not implemented for the other intrinsicmemory prompt variants)
                template_instructions=template_instructions, # We never pass this to summarize, so it's also empty??
                task_description=mas_message.task_description,
                task_trajectory=mas_message.task_trajectory,
                current_memory=self.agent_intrinsic_memory,
        )

        messages = [Message("system", self.system_prompt), Message("user", memory_update_prompt)]

        logger.debug('MEMORY UPDATE PROMPT\n%s', memory_update_prompt)

        # only summarise after some history has been built up
        if len(mas_message.task_trajectory) > 5:
            self.agent_intrinsic_memory = self.llm_model(messages, intrinsic=True)

        summary_message = f"""{mas_message.task_description} \n\n### Agent Memory\n 
        {self.agent_intrinsic_memory} \n\n {mas_message.task_trajectory}"""

        return summary_message


    def save_task_context(self, label: bool, feedback: str = None) -> None:
        """End of a task. Returns nothing: this family stores no trajectory.

        The memory itself is wiped unless the experiment asked for it to be
        carried into the next task.
        """
        self.counter = 0
        if not self.global_config.get('intrinsic_cross_task', False):
            self.agent_intrinsic_memory = ""

        # reset self.llm_model, but keep accounting on the same tracker so
        # per-experiment token totals survive the swap
        llm_model_name = self.llm_model.model_name
        self.llm_model = GPTChat(model_name=llm_model_name, tracker=self.llm_model.tracker)

@dataclass
class IntrinsicMASMemoryPDDL(IntrinsicMASMemory):
    system_prompt = INTRINSICMEMORY_PDDL.system_prompt


@dataclass
class IntrinsicMASMemoryFEVER(IntrinsicMASMemory):
    system_prompt = INTRINSICMEMORY_FEVER.system_prompt


@dataclass
class IntrinsicMASMemoryHOTPOTQA(IntrinsicMASMemory):
    system_prompt = INTRINSICMEMORY_HOTPOTQA.system_prompt


@dataclass
class IntrinsicMASMemoryALFWORLD(IntrinsicMASMemory):
    system_prompt = INTRINSICMEMORY_ALFWORLD.system_prompt


@dataclass
class IntrinsicMASMemoryBABYAI(IntrinsicMASMemory):
    system_prompt = INTRINSICMEMORY_BABYAI.system_prompt


@dataclass
class IntrinsicMASMemoryJERICHO(IntrinsicMASMemory):
    system_prompt = INTRINSICMEMORY_JERICHO.system_prompt


@dataclass
class IntrinsicMASMemorySCIWORLD(IntrinsicMASMemory):
    system_prompt = INTRINSICMEMORY_SCIWORLD.system_prompt


@dataclass
class IntrinsicMASMemorySWEBENCH(IntrinsicMASMemory):
    system_prompt = INTRINSICMEMORY_SWEBENCH.system_prompt


@dataclass
class IntrinsicMASMemoryNoTemplate(IntrinsicMASMemory):
    """Keeps agent memory without a memory template."""

    system_prompt = INTRINSICMEMORY_NOTEMPLATE.system_prompt

