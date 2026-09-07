"""FEVER and HotpotQA share a ReAct loop over Wikipedia but not an answer space.

Both present the dataset's task to the agent and score a Finish against the
dataset's answer, and the only thing separating them is which word the task is
presented under. Getting that wrong tells the agent a question is a claim, which
raises nothing - the episode runs to its trial budget answering the wrong kind
of thing.
"""

import json

import pytest
import wikipedia

from tasks.envs import ENVS, TASKS_PATH, get_task
from tasks.envs.utils import WikipediaUnavailable
from tasks.envs.wiki_env import WikiReActEnv

WIKI_TASKS = {
    'fever': ('Claim', 'Telemundo is a English-language television network.', 'REFUTES'),
    'hotpotqa': ('Question', 'Were Pavel Urysohn and Leonid Levin known for the same '
                             'type of work?', 'yes'),
}


def build(task: str, **env_config) -> WikiReActEnv:
    return ENVS[task](env_config=env_config or None, max_trials=30)


@pytest.mark.parametrize('task', sorted(WIKI_TASKS))
def test_the_task_is_presented_under_its_own_word(task):
    prefix, question, answer = WIKI_TASKS[task]
    env = build(task)

    task_main, task_description = env.set_env({'task': question, 'answer': answer})

    assert task_main == f'{prefix}: {question}'
    assert task_description == task_main


@pytest.mark.parametrize('task', sorted(WIKI_TASKS))
def test_a_finish_is_scored_against_the_datasets_answer(task):
    _, question, answer = WIKI_TASKS[task]
    env = build(task)
    env.set_env({'task': question, 'answer': answer})

    observation, reward, done = env.step(f'Finish[{answer}]')

    assert (reward, done) == (1, True), f'{task} scored its own answer {observation}'
    assert env.feedback()[:2] == (1, True)


@pytest.mark.parametrize('task', sorted(WIKI_TASKS))
def test_a_wrong_finish_ends_the_episode_unscored(task):
    _, question, answer = WIKI_TASKS[task]
    env = build(task)
    env.set_env({'task': question, 'answer': answer})

    _, reward, done = env.step('Finish[a different answer entirely]')

    assert (reward, done) == (0, True)
    assert env.feedback()[:2] == (0, False)


@pytest.mark.parametrize('task', sorted(WIKI_TASKS))
def test_a_reasoning_step_costs_a_trial_but_reaches_no_action(task):
    """One classifier per env: `step` and `_solver_stuck` must read the same one."""
    _, question, answer = WIKI_TASKS[task]
    env = build(task)
    env.set_env({'task': question, 'answer': answer})

    assert env.is_thought('Thought 1: I should search first.') is True
    assert env.step('Thought 1: I should search first.') == ('OK.', 0, False)


def test_a_hotpotqa_answer_matches_up_to_case_articles_and_punctuation():
    """FEVER's three labels never needed normalising; a free-text span does."""
    env = build('hotpotqa')
    env.set_env({'task': 'Which magazine was started first?', 'answer': "Arthur's Magazine"})

    assert env.step('Finish[the arthurs magazine]')[1] == 1


def test_the_hotpotqa_loader_asks_the_question_and_keeps_the_answer():
    """Reversing the two raises nothing: every task would hand the answer over."""
    with open(TASKS_PATH['hotpotqa'], encoding='utf-8') as reader:
        source = [json.loads(line) for line in reader][:5]

    tasks = get_task('hotpotqa', max_tasks=5)

    assert [task['task'] for task in tasks] == [row['question'] for row in source]
    assert [task['answer'] for task in tasks] == [row['answer'] for row in source]
    assert all(task['env_name'] == 'hotpotqa' for task in tasks)


# ── getting the action out of a ReAct line ────────────────────────────────────

@pytest.mark.parametrize(
    'raw, expected',
    [
        ('Search[Star Wars: A New Hope]', 'Search[Star Wars: A New Hope]'),
        ('Action 1: Search[Kill Bill: Volume 1]', 'Search[Kill Bill: Volume 1]'),
        ('Lookup[Chapter 3: the return]', 'Lookup[Chapter 3: the return]'),
    ],
)
def test_an_argument_carrying_a_colon_survives(raw, expected):
    """The `Action N:` prefix used to come off by splitting on the first colon,
    which took the argument with it.

    `Search[Star Wars: A New Hope]` became `Search[Star Wars`, which matches no
    action pattern, so the search never happened and the trial scored -1.
    Wikipedia titles carry colons constantly - films, albums, books with
    subtitles - so this was reachable by any agent searching for one.
    """
    assert ENVS['hotpotqa'].process_action(raw) == expected


@pytest.mark.parametrize('raw', ['Lookup[BOOK]', 'Search[OK Computer]', 'Finish[OK]'])
def test_an_argument_carrying_an_uppercase_ok_survives(raw):
    """`OK` was stripped as a substring, so `Lookup[BOOK]` became `Lookup[BO]`."""
    assert ENVS['hotpotqa'].process_action(raw) == raw


@pytest.mark.parametrize(
    'raw', ['Search[Thought experiment]', 'Lookup[thoughts]', 'Finish[thought]']
)
def test_an_action_that_merely_mentions_a_thought_is_still_an_action(raw):
    """`thought` was matched anywhere in the line, so searching for one was read
    as reasoning about one - and the search was discarded in silence.
    """
    env = ENVS['hotpotqa']
    processed = env.process_action(raw)

    assert env.is_thought(processed) is False, f'{raw!r} would never reach Wikipedia'
    assert processed == raw


def test_the_action_prefix_comes_off_but_the_verb_stays():
    assert ENVS['fever'].process_action('Action 2: Finish[SUPPORTS]') == 'Finish[SUPPORTS]'
    assert ENVS['fever'].process_action('Finish[SUPPORTS]') == 'Finish[SUPPORTS]'


def test_an_action_written_on_the_same_line_as_a_thought_is_taken():
    """The shape issue #31 described: one line carrying both, classified a
    thought, with the action in it discarded.
    """
    combined = 'Thought 1: I should search Telemundo. Action 1: Search[Telemundo]'

    assert ENVS['fever'].process_action(combined) == 'Search[Telemundo]'


def test_a_thought_alone_is_still_a_thought():
    env = ENVS['fever']

    for spelling in ('Thought 1: I need to search Telemundo.',
                     'thought: I need to search Telemundo.',
                     'THOUGHT 2: I need to search Telemundo.'):
        assert env.is_thought(env.process_action(spelling)) is True


def test_prose_after_a_thought_does_not_displace_it():
    """Only a line shaped like an action is preferred over a thought. A model that
    keeps talking has still only reasoned, and a thought costs nothing.
    """
    env = ENVS['fever']
    processed = env.process_action(
        'Thought 1: I should search Telemundo.\nThat seems like the right approach.'
    )

    assert env.is_thought(processed) is True


def unreachable_searches(env, failures: int):
    """Make the env's next `failures` Searches unreachable, then reachable."""
    attempts = []

    def search(argument):
        attempts.append(argument)
        if len(attempts) <= failures:
            raise WikipediaUnavailable('Wikipedia could not be reached')
        return 'Telemundo is a Spanish-language network.'

    env.explorer.search = search
    return attempts


def test_an_unreachable_search_is_reported_to_the_agent_rather_than_ending_the_task():
    """A transient fault costs a step, not the task.

    The bursts seen on Isambard were seconds wide and killed whole tasks that
    would have completed had the agent simply Searched again.
    """
    env = build('fever')
    env.set_env({'task': 'Telemundo is an English-language network.', 'answer': 'REFUTES'})
    unreachable_searches(env, failures=1)

    observation, reward, done = env.step('Search[Telemundo]')

    assert done is False, 'the episode ended on a transient fault'
    assert reward == 0, f'a transport fault scored {reward}'
    assert 'could not be reached' in observation, observation

    recovered, _, _ = env.step('Search[Telemundo]')
    assert 'Spanish-language' in recovered, recovered


def test_an_unreachable_search_does_not_replace_the_page_a_lookup_reads(monkeypatch):
    """Lookup must still address the page the last successful Search found.

    A transient fault says nothing about the page, so it leaves it alone - unlike
    a Search for a page that does not exist, which discards it.
    """
    class Page:
        content = 'Telemundo is a network mentioning zebras.'
        url = 'https://en.wikipedia.org/wiki/Telemundo'

    env = build('fever', wikipedia_attempts=1)
    env.set_env({'task': 'Telemundo is an English-language network.', 'answer': 'REFUTES'})
    monkeypatch.setattr(wikipedia, 'page', lambda title, **kwargs: Page())
    env.step('Search[Telemundo]')

    monkeypatch.setattr(wikipedia, 'page', lambda title, **kwargs: (_ for _ in ()).throw(
        ValueError('Expecting value: line 1 column 1 (char 0)')))
    unreachable, _, _ = env.step('Search[Univision]')
    assert 'could not be reached' in unreachable, unreachable

    observation, _, _ = env.step('Lookup[zebras]')

    assert 'zebras' in observation, observation


def test_the_search_retry_settings_come_from_the_env_config():
    """The knobs are the run's, so a sweep can loosen them without an edit."""
    env = build('fever', wikipedia_attempts=7, wikipedia_retry_seconds=0.5,
                unreachable_search_limit=9)

    assert env.explorer.attempts == 7
    assert env.explorer.retry_seconds == 0.5
    assert env.unreachable_search_limit == 9


def test_a_wikipedia_that_stays_unreachable_ends_the_task_rather_than_scoring_it_zero():
    """The environment must not answer every Search with a fault and score zero.

    A run against a blocked endpoint would otherwise read as a weak agent. Once
    the faults stop looking transient the exception propagates, and run_task
    records the task in failed_tasks.csv instead.
    """
    env = build('fever')
    env.set_env({'task': 'Telemundo is an English-language network.', 'answer': 'REFUTES'})
    unreachable_searches(env, failures=env.unreachable_search_limit + 1)

    with pytest.raises(WikipediaUnavailable):
        for _ in range(env.unreachable_search_limit):
            env.step('Search[Telemundo]')


def test_the_unreachable_allowance_is_per_episode():
    """Faults spent on one task must not end the next one early."""
    env = build('fever')
    env.set_env({'task': 'Telemundo is an English-language network.', 'answer': 'REFUTES'})
    unreachable_searches(env, failures=env.unreachable_search_limit - 1)
    for _ in range(env.unreachable_search_limit - 1):
        env.step('Search[Telemundo]')

    env.reset()
    unreachable_searches(env, failures=env.unreachable_search_limit - 1)
    for _ in range(env.unreachable_search_limit - 1):
        observation, _, done = env.step('Search[Telemundo]')
        assert done is False, observation
