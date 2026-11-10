"""DyLAN with roles, and where each agent's memory comes from.

`dylan` is the upstream workflow: nine solvers over three rounds, every turn
reading one shared memory. The role variants seat a planner, a solver and a
verifier in each round and differ only in the memory those agents are shown:

- `dylan-shared`: one memory, updated once per step from every role's proposals.
- `dylan-private`: one memory per role, updated once per step from that role's
  prompt and its own proposals.
- `dylan-private-agnostic`: one memory per role, updated from its own proposals
  alone.
"""

import tempfile
from collections import Counter

import pytest

from mas.llm import TokenTracker
from mas.memory import MASMemoryBase
from mas.module_map import module_map

from tasks.mas_workflow import MAS
from tasks.mas_workflow.dylan.dylan_prompt import get_role_system_prompt
from tasks.tests.fakes import FakeEmbeddingFunc, FakeEnv, FakeLLM, RecordingObserver
from mas.reasoning import ReasoningIO

ROLES = ('planner', 'solver', 'verifier')
ROLE_VARIANTS = ['dylan-private', 'dylan-private-agnostic', 'dylan-shared']
PRIVATE_VARIANTS = ['dylan-private', 'dylan-private-agnostic']
STEPS = 3
TASK = {'task_main': 'put a mug in the cabinet', 'task_description': 'a description', 'few_shots': []}


class RoleLLM(FakeLLM):
    """Answers `<role> move` to an agent of that role, so a proposal names who made it."""

    def __call__(self, messages, **kwargs):
        super().__call__(messages, **kwargs)
        system = messages[0].content
        for role in ROLES:
            if system == get_role_system_prompt(role):
                return f'{role} move'
        if system == get_role_system_prompt('ranker'):
            return '1 > 2 > 3'
        return 'decision move'

    def agent_turns(self) -> list[tuple[str, str]]:
        """(role, user prompt) for every call an agent in the grid made."""
        turns = []
        for messages in self.calls:
            for role in ('solver', 'ground_truth', *ROLES):
                if messages[0].content == get_role_system_prompt(role):
                    turns.append((role, messages[1].content))
                    break
        return turns


def recording_memory():
    """A fresh memory class that keeps every instance built from it, the workflow's copies included."""

    class RecordingMemory(MASMemoryBase):
        instances: list = []

        def __post_init__(self):
            super().__post_init__()
            self.updates: list[str] = []
            self.saved: list[bool] = []
            type(self).instances.append(self)

        def summarize(self, *, solver_message: str = "", template_instructions: str = "") -> str:
            self.updates.append(solver_message)
            return f'VIEW[{self.namespace}]#{len(self.updates)}'

        def save_task_context(self, label: bool, feedback: str = None):
            self.saved.append(label)
            return super().save_task_context(label, feedback)

    return RecordingMemory


def build(mas_type: str, memory_cls, llm=None, env=None):
    llm = llm or RoleLLM(tracker=TokenTracker())
    workflow = MAS[mas_type]()
    workflow.add_observer(RecordingObserver())
    memory = memory_cls(
        namespace=mas_type,
        global_config={'working_dir': tempfile.mkdtemp(), 'hop': 1},
        llm_model=llm,
        embedding_func=FakeEmbeddingFunc(),
    )
    workflow.build_system(ReasoningIO(llm_model=llm), memory, env or FakeEnv(max_trials=STEPS, steps_to_done=STEPS), {})
    return workflow, memory, llm


def run(mas_type: str, memory_cls=None, llm=None):
    memory_cls = memory_cls or recording_memory()
    workflow, memory, llm = build(mas_type, memory_cls, llm)
    workflow.schedule(dict(TASK))
    return workflow, memory, llm


def role_memories(memory_cls, shared):
    return [instance for instance in memory_cls.instances if instance is not shared]


# ── the upstream workflow ─────────────────────────────────────────────────────

def test_upstream_dylan_is_nine_solvers():
    workflow, _, _ = build('dylan', recording_memory())

    profiles = [agent.profile for agent in workflow.agents_team.values()]
    assert profiles == ['solver'] * 9, f'upstream dylan seats {Counter(profiles)}'


def test_upstream_dylan_reads_its_one_memory_afresh_at_every_agent_turn():
    memory_cls = recording_memory()
    _, memory, llm = run('dylan', memory_cls)

    assert memory_cls.instances == [memory], 'upstream dylan built memories of its own'
    turns = len(llm.agent_turns())
    assert len(memory.updates) == turns + 1, (
        f'{len(memory.updates)} memory reads for {turns} agent turns and the opening prompt'
    )
    assert set(memory.updates) == {''}, 'upstream dylan passed the memory a message'


# ── the role variants ─────────────────────────────────────────────────────────

@pytest.mark.parametrize('mas_type', ROLE_VARIANTS)
def test_a_role_variant_seats_every_role_in_every_round(mas_type):
    workflow, _, _ = build(mas_type, recording_memory())

    profiles = Counter(agent.profile for agent in workflow.agents_team.values())
    assert profiles == {role: 2 for role in ROLES}, f'{mas_type} seats {profiles}'


def test_every_role_has_a_prompt_of_its_own_that_keeps_the_output_format():
    prompts = {role: get_role_system_prompt(role) for role in ROLES}

    assert len(set(prompts.values())) == len(ROLES), 'two roles share a prompt'
    for role, prompt in prompts.items():
        assert 'strictly follow the output format' in prompt, (
            f'the {role} prompt drops the instruction to answer in the environment\'s format'
        )


@pytest.mark.parametrize('mas_type', ROLE_VARIANTS)
def test_every_memory_follows_the_whole_episode(mas_type):
    memory_cls = recording_memory()
    workflow, _, _ = build(mas_type, memory_cls)
    env = workflow.env

    workflow.schedule(dict(TASK))

    for instance in memory_cls.instances:
        trajectory = instance.current_task_context.task_trajectory
        for action in env.actions:
            assert f'observation after {action}' in trajectory, (
                f'{mas_type}: {instance.namespace} missed a step of the episode'
            )
        assert instance.saved == [True], f'{mas_type}: {instance.namespace} saved {instance.saved}'


# ── one shared memory, updated once a step ────────────────────────────────────

def test_the_shared_variant_shows_every_agent_one_memory_updated_once_a_step():
    memory_cls = recording_memory()
    _, memory, llm = run('dylan-shared', memory_cls)

    assert memory_cls.instances == [memory], 'dylan-shared built memories of its own'
    assert len(memory.updates) == STEPS, f'{len(memory.updates)} updates over {STEPS} steps'
    for role, prompt in llm.agent_turns():
        assert 'VIEW[dylan-shared]' in prompt, f'a {role} was not shown the shared memory'


def test_the_shared_memory_is_updated_from_every_roles_proposals():
    _, memory, _ = run('dylan-shared')

    for role in ROLES:
        assert f'{role} move' in memory.updates[-1], (
            f'the shared memory was not told what the {role} proposed'
        )


# ── one memory per role ───────────────────────────────────────────────────────

@pytest.mark.parametrize('mas_type', PRIVATE_VARIANTS)
def test_each_role_is_shown_a_memory_of_its_own(mas_type):
    memory_cls = recording_memory()
    _, shared, llm = run(mas_type, memory_cls)
    private = {instance.namespace: instance for instance in role_memories(memory_cls, shared)}

    assert len(private) == len(ROLES), f'{mas_type} built {sorted(private)}'
    for role, prompt in llm.agent_turns():
        shown = [namespace for namespace in private if f'VIEW[{namespace}]' in prompt]
        assert len(shown) == 1 and role in shown[0], (
            f'{mas_type}: a {role} was shown {shown}'
        )
        assert f'VIEW[{shared.namespace}]' not in prompt, f'{mas_type}: a {role} was shown the team memory'


@pytest.mark.parametrize('mas_type', PRIVATE_VARIANTS)
def test_each_role_memory_is_updated_once_a_step(mas_type):
    memory_cls = recording_memory()
    _, shared, _ = run(mas_type, memory_cls)

    for instance in role_memories(memory_cls, shared):
        assert len(instance.updates) == STEPS, (
            f'{mas_type}: {instance.namespace} was updated {len(instance.updates)} times over {STEPS} steps'
        )
    assert not shared.updates, f'{mas_type} updated a memory no agent is shown'


@pytest.mark.parametrize('mas_type', PRIVATE_VARIANTS)
def test_a_role_memory_is_updated_from_its_own_proposals_only(mas_type):
    memory_cls = recording_memory()
    _, shared, _ = run(mas_type, memory_cls)

    for instance in role_memories(memory_cls, shared):
        owner = next(role for role in ROLES if role in instance.namespace)
        update = instance.updates[-1]
        assert f'{owner} move' in update, f'{mas_type}: the {owner} memory was not told its own proposals'
        others = [role for role in ROLES if role != owner and f'{role} move' in update]
        assert not others, f'{mas_type}: the {owner} memory was told what {others} proposed'


def test_a_role_conditioned_memory_is_told_its_role():
    memory_cls = recording_memory()
    _, shared, _ = run('dylan-private', memory_cls)

    for instance in role_memories(memory_cls, shared):
        owner = next(role for role in ROLES if role in instance.namespace)
        assert all(get_role_system_prompt(owner) in update for update in instance.updates), (
            f'the {owner} memory was updated without its role'
        )


def test_a_role_agnostic_memory_is_told_no_role():
    memory_cls = recording_memory()
    _, shared, _ = run('dylan-private-agnostic', memory_cls)

    for instance in role_memories(memory_cls, shared):
        for role in ROLES:
            assert not any(get_role_system_prompt(role) in update for update in instance.updates), (
                f'{instance.namespace} was told the {role} role'
            )


@pytest.mark.parametrize('mas_type', PRIVATE_VARIANTS)
def test_role_memories_are_the_arms_memory_module_writing_to_its_tracker(mas_type):
    _, arm_cls = module_map('io', 'intrinsicmemory-fever')

    class TrackedIntrinsic(arm_cls):
        instances: list = []

        def __post_init__(self):
            super().__post_init__()
            type(self).instances.append(self)

    tracker = TokenTracker()
    _, shared, _ = run(mas_type, TrackedIntrinsic, RoleLLM(tracker=tracker))
    private = role_memories(TrackedIntrinsic, shared)

    assert len(private) == len(ROLES)
    for instance in private:
        assert type(instance) is TrackedIntrinsic
        assert instance.system_prompt == arm_cls.system_prompt
    assert len({instance.persist_dir for instance in [shared, *private]}) == len(ROLES) + 1, (
        f'{mas_type}: two memories share a persistence directory'
    )
    assert tracker.intrinsic_prompt_tokens > 0, f'{mas_type}: no memory update reached the tracker'


def test_an_unknown_agent_memory_is_refused():
    class Misnamed(MAS['dylan-private']):
        agent_memory = 'privat'

    with pytest.raises(ValueError, match='privat'):
        Misnamed().build_system(
            ReasoningIO(llm_model=RoleLLM()),
            recording_memory()(namespace='m', global_config={'working_dir': tempfile.mkdtemp(), 'hop': 1},
                               llm_model=None, embedding_func=FakeEmbeddingFunc()),
            FakeEnv(), {},
        )
