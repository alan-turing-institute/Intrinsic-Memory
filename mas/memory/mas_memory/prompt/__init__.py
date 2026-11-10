"""Prompt constants, one module per memory module.

Every prompt here belongs to the memory module its file is named for, so a
constant edited for one memory system cannot silently be the one another reads.
The intrinsic variants share `intrinsic.py`: they differ only in the memory
template their system prompt asks for.

One name per bundle: the instance, not the dataclass it is an instance of. The
classes exist to group and default the fields and are importable from their own
module if a caller ever needs the type.
"""

from .chatdev import CHATDEV
from .generative import GENERATIVE
from .gmemory import GMemoryPrompts
from .intrinsic import (
    INTRINSICMEMORY_ALFWORLD,
    INTRINSICMEMORY_ALFWORLD_SCHEMA,
    INTRINSICMEMORY_BABYAI,
    INTRINSICMEMORY_BABYAI_SCHEMA,
    INTRINSICMEMORY_DEFAULT,
    INTRINSICMEMORY_FEVER,
    INTRINSICMEMORY_FEVER_SCHEMA,
    INTRINSICMEMORY_HOTPOTQA,
    INTRINSICMEMORY_HOTPOTQA_SCHEMA,
    INTRINSICMEMORY_JERICHO,
    INTRINSICMEMORY_JERICHO_SCHEMA,
    INTRINSICMEMORY_LLM_TEMPLATE,
    INTRINSICMEMORY_NOTEMPLATE,
    INTRINSICMEMORY_PDDL,
    INTRINSICMEMORY_PDDL_SCHEMA,
    INTRINSICMEMORY_SCIWORLD,
    INTRINSICMEMORY_SCIWORLD_SCHEMA,
    template_of,
)
from .macnet import MACNET
from .memorybank import MEMORYBANK
from .voyager import VOYAGER

__all__ = [
    'CHATDEV',
    'GENERATIVE',
    'GMemoryPrompts',
    'INTRINSICMEMORY_ALFWORLD',
    'INTRINSICMEMORY_ALFWORLD_SCHEMA',
    'INTRINSICMEMORY_BABYAI',
    'INTRINSICMEMORY_BABYAI_SCHEMA',
    'INTRINSICMEMORY_DEFAULT',
    'INTRINSICMEMORY_FEVER',
    'INTRINSICMEMORY_FEVER_SCHEMA',
    'INTRINSICMEMORY_HOTPOTQA',
    'INTRINSICMEMORY_HOTPOTQA_SCHEMA',
    'INTRINSICMEMORY_JERICHO',
    'INTRINSICMEMORY_JERICHO_SCHEMA',
    'INTRINSICMEMORY_LLM_TEMPLATE',
    'INTRINSICMEMORY_NOTEMPLATE',
    'INTRINSICMEMORY_PDDL',
    'INTRINSICMEMORY_PDDL_SCHEMA',
    'INTRINSICMEMORY_SCIWORLD',
    'INTRINSICMEMORY_SCIWORLD_SCHEMA',
    'template_of',
    'MACNET',
    'MEMORYBANK',
    'VOYAGER',
]
