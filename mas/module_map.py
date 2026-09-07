from typing import Type

from .reasoning import ReasoningBase, ReasoningIO
from .memory import *

REASONING_MODULES: dict[str, Type[ReasoningBase]] = {
    'io': ReasoningIO,
}

MAS_MEMORY_MODULES: dict[str, Type[MASMemoryBase]] = {
    'empty': MASMemoryBase,
    'voyager': VoyagerMASMemory,
    'memorybank': MemoryBankMASMemory,
    'chatdev': ChatDevMASMemory,
    'generative': GenerativeMASMemory,
    'metagpt': MetaGPTMASMemory,
    'g-memory': GMemory,
    'intrinsicmemory-pddl': IntrinsicMASMemoryPDDL,
    'intrinsicmemory-fever': IntrinsicMASMemoryFEVER,
    'intrinsicmemory-hotpotqa': IntrinsicMASMemoryHOTPOTQA,
    'intrinsicmemory-alfworld': IntrinsicMASMemoryALFWORLD,
    'intrinsicmemory-babyai': IntrinsicMASMemoryBABYAI,
    'intrinsicmemory-jericho': IntrinsicMASMemoryJERICHO,
    'intrinsicmemory-sciworld': IntrinsicMASMemorySCIWORLD,
    'intrinsicmemory-swebench': IntrinsicMASMemorySWEBENCH,
    'intrinsicmemory-llm-structured-template': IntrinsicMASMemoryLLMTemplate,
    'intrinsicmemory-notemplate': IntrinsicMASMemoryNoTemplate,
}


def module_map(
    reasoning: str, mas_memory: str = None
) -> tuple[Type[ReasoningBase], Type[MASMemoryBase]]:

    if reasoning not in REASONING_MODULES:
        raise ValueError(f"Invalid reasoning type '{reasoning}'. Allowed values: {list(REASONING_MODULES.keys())}")

    if mas_memory is not None and mas_memory not in MAS_MEMORY_MODULES:
        raise ValueError(f"Invalid MAS memory type '{mas_memory}'. Allowed values: {list(MAS_MEMORY_MODULES.keys())}")

    return (
        REASONING_MODULES[reasoning],
        MAS_MEMORY_MODULES.get(mas_memory, None)
    )
