"""`--use_projector` reaches the memory, and each role gets its own insights.

Parametrised over every registered workflow: `_project_insights` lives on
`MetaMAS`, so all of them resolve to one implementation.
"""

import inspect
import tempfile

import pytest

from mas.memory import MASMemoryBase
from mas.memory.mas_memory.memory_base import SupportsProjection

from tasks.mas_workflow import MAS
from tasks.tests.fakes import FakeEnv
from tasks.tests.test_contracts import build_workflow

WORKFLOWS = sorted(MAS)
INSIGHTS = ["keep the goal in view", "check the room first", "put it down gently"]


class ProjectingMemory(MASMemoryBase):
    """A memory that tailors insights to a role, as GMemory does."""

    def __post_init__(self):
        super().__post_init__()
        self.projected_for: list[str] = []

    def project_insights(self, raw_insights, role=None, task_traj=None):
        self.projected_for.append(role)
        return [f"{role} should {insight}" for insight in raw_insights]


def memory_of(workflow):
    """The memory the projector consults, whichever field the workflow keeps it in."""
    return workflow.meta_memory


# ── the protocol is the contract ──────────────────────────────────────────────

def test_a_projecting_memory_satisfies_the_protocol():
    memory = ProjectingMemory(
        namespace="protocol", global_config={"working_dir": tempfile.mkdtemp(), "hop": 1},
        llm_model=None, embedding_func=None,
    )

    assert isinstance(memory, SupportsProjection)


def test_a_memory_without_projection_does_not():
    memory = MASMemoryBase(
        namespace="protocol", global_config={"working_dir": tempfile.mkdtemp(), "hop": 1},
        llm_model=None, embedding_func=None,
    )

    assert not isinstance(memory, SupportsProjection)


@pytest.mark.parametrize("mas_type", WORKFLOWS)
def test_the_projector_does_not_test_for_a_concrete_memory_class(mas_type):
    source = inspect.getsource(MAS[mas_type]._project_insights)

    assert "GMemory" not in source, (
        f"{mas_type} still decides whether projection is available by naming one class"
    )


# ── B1 · the projector projects ───────────────────────────────────────────────

@pytest.mark.parametrize("mas_type", WORKFLOWS)
def test_the_flag_reaches_the_memory(mas_type):
    workflow = build_workflow(
        mas_type, FakeEnv(), memory_cls=ProjectingMemory, use_projector=True
    )

    workflow._project_insights(INSIGHTS)

    assert memory_of(workflow).projected_for, (
        f"{mas_type}: use_projector was set and project_insights was never called"
    )


@pytest.mark.parametrize("mas_type", WORKFLOWS)
def test_each_role_receives_insights_projected_for_it(mas_type):
    workflow = build_workflow(
        mas_type, FakeEnv(), memory_cls=ProjectingMemory, use_projector=True
    )

    projected = workflow._project_insights(INSIGHTS)

    assert projected, f"{mas_type}: no role received any insights"
    for role, rules in projected.items():
        assert all(rule.startswith(role) for rule in rules), (
            f"{mas_type}: {role} received insights projected for someone else"
        )


@pytest.mark.parametrize("mas_type", WORKFLOWS)
def test_roles_that_differ_receive_insights_that_differ(mas_type):
    """Upstream DyLAN's nine agents and MacNet's three all carry the profile
    `solver`, so there is one role to project onto."""
    workflow = build_workflow(
        mas_type, FakeEnv(), memory_cls=ProjectingMemory, use_projector=True
    )

    projected = workflow._project_insights(INSIGHTS)

    distinct = {tuple(rules) for rules in projected.values()}
    assert len(distinct) == len(projected), (
        f"{mas_type}: {len(projected)} roles received {len(distinct)} distinct insight lists"
    )


@pytest.mark.parametrize("mas_type", WORKFLOWS)
def test_the_raw_insights_are_shared_when_the_flag_is_unset(mas_type):
    workflow = build_workflow(
        mas_type, FakeEnv(), memory_cls=ProjectingMemory, use_projector=False
    )

    projected = workflow._project_insights(INSIGHTS)

    assert not memory_of(workflow).projected_for, (
        f"{mas_type} projected without being asked to"
    )
    assert {tuple(rules) for rules in projected.values()} == {tuple(INSIGHTS[:1])}


@pytest.mark.parametrize("mas_type", WORKFLOWS)
def test_a_memory_that_cannot_project_is_not_asked_to(mas_type):
    """Most memory modules have no projection; the flag must not break them."""
    workflow = build_workflow(
        mas_type, FakeEnv(), memory_cls=MASMemoryBase, use_projector=True
    )

    projected = workflow._project_insights(INSIGHTS)

    assert {tuple(rules) for rules in projected.values()} == {tuple(INSIGHTS[:1])}


@pytest.mark.parametrize("mas_type", WORKFLOWS)
def test_each_role_gets_at_most_insights_topk(mas_type):
    workflow = build_workflow(
        mas_type, FakeEnv(), memory_cls=ProjectingMemory, use_projector=True
    )
    workflow._insights_topk = 2

    projected = workflow._project_insights(INSIGHTS)

    for role, rules in projected.items():
        assert len(rules) <= 2, f"{mas_type}: {role} got {len(rules)} of at most 2"
