from dataclasses import dataclass

from .intrinsicmemory import agent_view
from .memory_base import MASMemoryBase
from .prompt import (
    INTRINSICMEMORY_ALFWORLD,
    INTRINSICMEMORY_BABYAI,
    INTRINSICMEMORY_FEVER,
    INTRINSICMEMORY_HOTPOTQA,
    INTRINSICMEMORY_JERICHO,
    INTRINSICMEMORY_PDDL,
    INTRINSICMEMORY_SCIWORLD,
    template_of,
)


@dataclass
class FrozenTemplateMemory(MASMemoryBase):
    """A hand-written memory template, shown to the agent as written and never updated."""

    # Unannotated, so it stays off the constructor, as on IntrinsicMASMemory.
    template = ""

    def summarize(self, *, solver_message: str = "", template_instructions: str = "") -> str:
        context = self.current_task_context
        return agent_view(context.task_description, self.template, context.task_trajectory)


@dataclass
class FrozenTemplatePDDL(FrozenTemplateMemory):
    template = template_of(INTRINSICMEMORY_PDDL.system_prompt)


@dataclass
class FrozenTemplateFEVER(FrozenTemplateMemory):
    template = template_of(INTRINSICMEMORY_FEVER.system_prompt)


@dataclass
class FrozenTemplateHOTPOTQA(FrozenTemplateMemory):
    template = template_of(INTRINSICMEMORY_HOTPOTQA.system_prompt)


@dataclass
class FrozenTemplateALFWORLD(FrozenTemplateMemory):
    template = template_of(INTRINSICMEMORY_ALFWORLD.system_prompt)


@dataclass
class FrozenTemplateBABYAI(FrozenTemplateMemory):
    template = template_of(INTRINSICMEMORY_BABYAI.system_prompt)


@dataclass
class FrozenTemplateJERICHO(FrozenTemplateMemory):
    template = template_of(INTRINSICMEMORY_JERICHO.system_prompt)


@dataclass
class FrozenTemplateSCIWORLD(FrozenTemplateMemory):
    template = template_of(INTRINSICMEMORY_SCIWORLD.system_prompt)
