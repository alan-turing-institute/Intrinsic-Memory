"""Sibling memories keep their configuration and have independent runtime state."""

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import Mock

import pytest

from mas.memory import MASMemoryBase
from mas.memory.mas_memory.intrinsicmemory import IntrinsicMASMemoryPDDL
from mas.module_map import MAS_MEMORY_MODULES
from tasks.mas_workflow.autogen.autogen import AutoGen
from tasks.tests.fakes import FakeEmbeddingFunc, FakeEnv, FakeLLM, fake_reasoning


def build_memory(memory_cls, tmp_path, **kwargs):
    return memory_cls(
        namespace="solver",
        global_config={"working_dir": str(tmp_path), "hop": 1},
        llm_model=FakeLLM(),
        embedding_func=FakeEmbeddingFunc(),
        **kwargs,
    )


@pytest.mark.parametrize("key", sorted(MAS_MEMORY_MODULES))
def test_sibling_preserves_type_dependencies_and_namespace(key, tmp_path):
    memory = build_memory(MAS_MEMORY_MODULES[key], tmp_path)

    sibling = memory.for_namespace("_validator")

    assert sibling is not memory
    assert type(sibling) is type(memory)
    assert memory.namespace == "solver"
    assert sibling.namespace == "solver_validator"
    assert Path(sibling.persist_dir) == tmp_path / "solver_validator"
    assert Path(sibling.persist_dir).is_dir()
    assert sibling.global_config is memory.global_config
    assert sibling.llm_model is memory.llm_model
    assert sibling.embedding_func is memory.embedding_func


def test_sibling_does_not_copy_solver_runtime_state(tmp_path):
    memory = build_memory(IntrinsicMASMemoryPDDL, tmp_path)
    original_context = memory.init_task_context("solver task", "solver description")
    memory.counter = 7
    memory.agent_intrinsic_memory = "solver-only history"

    sibling = memory.for_namespace("_validator")
    sibling_context = sibling.init_task_context("validator task", "validator description")

    assert sibling.counter == 0
    assert sibling.agent_intrinsic_memory == ""
    assert sibling_context is not original_context
    assert memory.current_task_context is original_context
    assert memory.agent_intrinsic_memory == "solver-only history"
    assert memory.counter == 7


@dataclass
class ConfiguredMemory(MASMemoryBase):
    option: str

    def for_namespace(self, suffix):
        return type(self)(
            namespace=self.namespace + suffix,
            global_config=self.global_config,
            llm_model=self.llm_model,
            embedding_func=self.embedding_func,
            option=self.option,
        )


@pytest.mark.parametrize("use_validator", [False, True])
def test_workflow_leaves_construction_to_the_memory(tmp_path, use_validator):
    memory = build_memory(ConfiguredMemory, tmp_path, option="custom setting")
    factory = Mock(wraps=memory.for_namespace)
    memory.for_namespace = factory
    workflow = AutoGen()

    workflow.build_system(
        fake_reasoning(), memory, FakeEnv(), {"use_validator": use_validator}
    )

    assert workflow.meta_memory is memory
    if use_validator:
        factory.assert_called_once_with("_validator")
        assert isinstance(workflow.meta_memory_validator, ConfiguredMemory)
        assert workflow.meta_memory_validator.option == "custom setting"
        assert workflow.meta_memory_validator is not memory
    else:
        factory.assert_not_called()
        assert workflow.meta_memory_validator is None
