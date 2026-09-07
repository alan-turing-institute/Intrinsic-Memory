"""The SWE-bench environment: the shell action space, and how an episode is graded.

`SWEBenchEnv` needs a container runtime and an instance image to take a step, so
the shell is driven through a fake session here and the container plumbing is
only ever verified by running it - see containers/isambard.md. Everything that
decides a reward or a command is exercised.
"""

import json
import textwrap

import pytest
import yaml

from tasks.envs import ENVS, TASKS_PATH, get_task
from tasks.envs import swebench_env as module
from tasks.envs.swebench_env import (
    ContainerSession,
    CommandResult,
    clean_shell_action,
    elide,
)
from tasks.envs.swebench_grading import (
    NO_CHANGE,
    PATCH_WRITTEN,
    RESOLVED,
    TESTS_RAN,
    grade,
    parse_pytest_log,
    parse_sympy_log,
    parse_unittest_log,
    statuses,
    bracketed_output,
)
from tasks.prompts import swebench_prompt

INSTANCE = 'django__django-16485'

TESTS_JSON = {
    'FAIL_TO_PASS': ['test_zero_values (m.FunctionTests.test_zero_values)'],
    'PASS_TO_PASS': ['test_inputs (m.FunctionTests.test_inputs)', '#15789'],
}

TASK_YAML = {
    'repo': 'django/django',
    'instance_id': INSTANCE,
    'base_commit': '39f83765e12b0e5d260b7939fc3fe281d879b279',
    'log_parser': 'parse_log_django',
}


class FakeSession:
    """A container session that answers scripted output and records commands."""

    def __init__(self, outputs=None):
        self.outputs = dict(outputs or {})
        self.commands: list[str] = []
        self.cwd = '/testbed'
        self.closed = False

    def start(self) -> None:
        self.cwd = '/testbed'

    def run(self, command: str, timeout: int) -> CommandResult:
        self.commands.append(command)
        for pattern, output in self.outputs.items():
            if pattern in command:
                return CommandResult(output, 0)
        return CommandResult('', 0)

    def close(self) -> None:
        # Deliberately still usable afterwards, so a test of the cached verdict
        # is evidence for the cache rather than for the closed container.
        self.closed = True


@pytest.fixture
def task_repo(tmp_path):
    """A task directory as the task repository lays one out."""
    task_dir = tmp_path / 'tasks' / INSTANCE
    task_dir.mkdir(parents=True)
    (task_dir / 'task.yaml').write_text(yaml.safe_dump(TASK_YAML))
    (task_dir / 'tests.json').write_text(json.dumps(TESTS_JSON))
    (task_dir / 'eval.sh').write_text('#!/bin/bash\n./tests/runtests.py\n')
    (task_dir / 'problem_statement.md').write_text('floatformat() crashes on "0.00".')
    return tmp_path


@pytest.fixture
def env(task_repo):
    return ENVS['swebench'](env_config={'task_repo': str(task_repo)}, max_trials=30)


@pytest.fixture
def started(env, monkeypatch):
    """An env whose next episode runs against a fake session."""
    sessions = []

    def build(**kwargs):
        session = FakeSession()
        sessions.append(session)
        return session

    monkeypatch.setattr(module, 'ContainerSession', build)
    env.set_env({'id': INSTANCE, 'repo': 'django/django'})
    env.reset()
    return env, sessions[-1]


# ── the shell action space ────────────────────────────────────────────────────

# Every one of these is a command the shared `clean_action_line` destroys,
# because `<`, `>`, `*`, `#`, backticks and a leading dot are decoration in
# every other dataset and syntax in this one.
@pytest.mark.parametrize('command', [
    'ls tests/*.py',
    'python -c "print(1)" > /tmp/out',
    './tests/runtests.py template_tests',
    'sed -i "s/#comment//" django/x.py',
    'grep -rn "def floatformat" django/ | head -5',
    'git diff HEAD~1 -- django/template/defaultfilters.py',
    'python -c "print(`echo hi`)"',
    'cat < /tmp/in',
    '"$HOME/bin/x" > "/tmp/out"',
])
def test_a_shell_command_reaches_the_shell_unchanged(command):
    assert clean_shell_action(command) == command


@pytest.mark.parametrize('reply, expected', [
    ('"ls tests"', 'ls tests'),
    ("'ls tests'", 'ls tests'),
    ('ls tests.', 'ls tests'),
    ('1. ls tests', 'ls tests'),
    ('- ls tests', 'ls tests'),
    ('**ls tests**', 'ls tests'),
    ('Action 1: ls tests', 'ls tests'),
    ('$ ls tests', 'ls tests'),
    ('```\nls tests\n```', 'ls tests'),
    ('```bash\nls tests\n```', 'ls tests'),
])
def test_the_way_a_model_dresses_up_a_command_comes_off(reply, expected):
    assert clean_shell_action(reply) == expected


@pytest.mark.parametrize('command', ['git add .', 'cp file ..', 'ls src/.'])
def test_a_full_stop_that_is_an_argument_is_kept(command):
    """A trailing full stop is a model writing a sentence, except when it is a path."""
    assert clean_shell_action(command) == command


def test_a_fenced_block_is_one_command_however_many_lines_it_spans():
    """A heredoc is the only way to write a file in one turn, and splitting it
    into lines leaves `cat EOF /tmp/p.py`."""
    reply = textwrap.dedent('''\
        ```
        cat > /tmp/p.py <<'EOF'
        print(1)
        EOF
        ```''')

    assert clean_shell_action(reply) == "cat > /tmp/p.py <<'EOF'\nprint(1)\nEOF"


def test_a_command_written_after_a_thought_is_the_one_taken():
    assert clean_shell_action('think: I should look first\nls tests') == 'ls tests'


def test_a_reply_that_is_only_a_thought_stays_a_thought():
    processed = clean_shell_action('think: I should look first')

    assert ENVS['swebench'].is_thought(processed)


def test_an_empty_reply_is_no_command():
    assert clean_shell_action('') == ''
    assert clean_shell_action('```\n```') == ''


# ── one command in the container ──────────────────────────────────────────────

def session(monkeypatch, output: str):
    """A real ContainerSession over a fake runtime, so the protocol is exercised."""
    calls = []

    class Completed:
        stdout, stderr = output, ''

    def fake_run(arguments, **kwargs):
        calls.append(arguments)
        return Completed()

    monkeypatch.setattr(module.subprocess, 'run', fake_run)

    return ContainerSession(image='img', name='container'), calls


def test_the_marker_carries_the_working_directory_across_commands(monkeypatch):
    """The next command runs where the last one left off, so `cd` works."""
    container, _ = session(monkeypatch, f'out\n{module._MARKER} 0 /testbed/tests\n')
    container.cwd = '/testbed'

    result = container.run('cd tests', timeout=5)

    assert (result.output, result.status) == ('out', 0)
    assert container.cwd == '/testbed/tests'


def test_output_that_lost_its_marker_leaves_the_working_directory_alone(monkeypatch):
    """A command that replaced the shell reports nothing about where it left it."""
    container, _ = session(monkeypatch, 'exec output with no marker')
    container.cwd = '/testbed'

    result = container.run('exec true', timeout=5)

    assert result.status is None
    assert container.cwd == '/testbed', 'the directory was guessed at'


def test_the_marker_is_set_before_the_command_runs(monkeypatch):
    """An unterminated heredoc swallows every line after it, so a marker written
    after the command would take the working directory with it."""
    container, calls = session(monkeypatch, f'{module._MARKER} 0 /testbed\n')
    container.cwd = '/testbed'

    container.run("cat <<EOF", timeout=5)

    script = calls[0][-1]
    assert script.index('trap') < script.index('cat <<EOF')


def test_the_runtimes_stderr_is_merged_into_its_output(monkeypatch):
    """`eval.sh` brackets the test output with xtrace lines on stderr while the
    tests print to whichever stream their runner uses. Captured apart and
    concatenated, the section between the markers holds no test output at all.
    """
    captured = {}

    def fake_run(arguments, **kwargs):
        captured.update(kwargs)

        class Completed:
            stdout, stderr = '', None
        return Completed()

    monkeypatch.setattr(module.subprocess, 'run', fake_run)
    container = ContainerSession(image='img', name='container')
    container.cwd = '/testbed'

    container.run('bash /task/eval.sh', timeout=5)

    assert captured['stderr'] is module.subprocess.STDOUT
    assert 'capture_output' not in captured, 'capture_output keeps the two streams apart'


def test_the_image_store_reaches_the_runtime(monkeypatch, task_repo):
    """podman-hpc looks for migrated images under SQUASH_DIR, and a job that does
    not name the directory they were built into sees none of them."""
    captured = {}

    def fake_run(arguments, **kwargs):
        captured.update(kwargs)

        class Completed:
            stdout, stderr, returncode = '', '', 0
        return Completed()

    monkeypatch.setattr(module.subprocess, 'run', fake_run)
    env = ENVS['swebench'](
        env_config={'task_repo': str(task_repo), 'squash_dir': '/store/images'}, max_trials=1
    )
    env.set_env({'id': INSTANCE, 'repo': 'django/django'})

    env.reset()

    assert captured['env']['SQUASH_DIR'] == '/store/images'


def test_a_command_that_outruns_its_timeout_is_reported_not_raised(monkeypatch):
    """One trial is lost rather than the episode: the edits are in the container,
    not in the shell that timed out."""
    def timeout(arguments, **kwargs):
        raise module.subprocess.TimeoutExpired(arguments, kwargs.get('timeout'))

    monkeypatch.setattr(module.subprocess, 'run', timeout)
    container = ContainerSession(image='img', name='container')
    container.cwd = '/testbed'

    result = container.run('sleep 500', timeout=120)

    assert result.status is None
    assert '120 seconds' in result.output


# ── what a step does ──────────────────────────────────────────────────────────

def test_a_command_is_run_and_its_output_observed(started):
    env, session = started
    session.outputs = {'ls': 'tests\nsetup.py'}

    observation, reward, done = env.step('ls')

    assert session.commands == ['ls']
    assert observation == 'tests\nsetup.py'
    assert (reward, done) == (0, False)


def test_a_thought_costs_a_trial_but_reaches_no_shell(started):
    env, session = started

    observation, reward, done = env.step('think: I should read the file first')

    assert session.commands == [], 'the thought was sent to the shell as a command'
    assert (observation, reward, done) == ('OK.', 0, False)


def test_a_reply_carrying_no_command_is_not_sent(started):
    env, session = started

    observation, reward, done = env.step('```\n```')

    assert session.commands == []
    assert reward == -1
    assert done is False


def test_submitting_ends_the_episode(started):
    env, session = started

    observation, reward, done = env.step('submit')

    assert done is True
    assert session.commands == [], 'submit is not a command to run'


def test_a_long_output_is_cut_in_the_middle(started):
    env, session = started
    env.max_output_chars = 100
    session.outputs = {'cat': 'x' * 5000}

    observation, _, _ = env.step('cat big.py')

    assert len(observation) < 5000
    assert 'omitted' in observation


def test_elide_keeps_both_ends():
    """A test run says what happened in its last lines, so the tail cannot go."""
    elided = elide('START' + 'x' * 1000 + 'END', 100)

    assert elided.startswith('START')
    assert elided.endswith('END')


def test_an_output_within_the_limit_is_untouched():
    assert elide('short', 100) == 'short'


# ── what it refuses to do ─────────────────────────────────────────────────────

def test_a_task_config_without_an_instance_id_is_rejected(env):
    with pytest.raises(ValueError, match='`id`'):
        env.set_env({'repo': 'django/django'})


def test_a_missing_task_directory_names_what_to_clone(env):
    """The task repository is not in this repo, so this is the first thing to go
    wrong on a fresh checkout and the error has to say what to fetch."""
    with pytest.raises(FileNotFoundError, match='isambard.md'):
        env.set_env({'id': 'django__django-99999'})


def test_the_task_description_carries_the_issue_and_not_the_answer(env):
    _, description = env.set_env({'id': INSTANCE, 'repo': 'django/django'})

    assert 'floatformat() crashes' in description
    assert '/testbed' in description
    assert 'gold' not in description.lower(), 'the description must not name the reference fix'


# ── reading a verdict out of a test log ───────────────────────────────────────

DJANGO_LOG = textwrap.dedent('''\
    + ./tests/runtests.py --verbosity 2
    test_inputs (m.FunctionTests.test_inputs) ... ok
    test_floatformat07 (m.FloatformatTests.test_floatformat07)
    #15789 ... ok
    test_zero_values (m.FunctionTests.test_zero_values) ... ERROR
    ======================================================================
    ERROR: test_zero_values (m.FunctionTests.test_zero_values)
    ValueError: valid range for prec is [1, MAX_PREC]
    Ran 10 tests in 0.03s
    FAILED (errors=1)
''')


def test_the_docstring_form_of_a_unittest_name_is_read():
    """A django test with a docstring is announced over two lines and the second
    carries the docstring, which is why `#15789` is a graded name."""
    parsed = parse_unittest_log(DJANGO_LOG)

    assert parsed['#15789'] == 'passed'
    assert parsed['test_inputs (m.FunctionTests.test_inputs)'] == 'passed'
    assert parsed['test_zero_values (m.FunctionTests.test_zero_values)'] == 'error'


def test_pytest_statuses_come_off_the_summary_lines():
    log = (
        'PASSED testing/test_mark.py::TestMark::test_mark_with_param\n'
        'FAILED testing/test_mark.py::TestFunctional::test_x - AssertionError: nope\n'
        'ERROR test_requests.py::TestRequests::test_binary_put\n'
    )

    parsed = parse_pytest_log(log)

    assert parsed['testing/test_mark.py::TestMark::test_mark_with_param'] == 'passed'
    assert parsed['testing/test_mark.py::TestFunctional::test_x'] == 'failed'
    assert parsed['test_requests.py::TestRequests::test_binary_put'] == 'error'


def test_sympy_statuses_come_off_its_status_letters():
    log = 'test_tensor_product_expand ok\ntest_tensor_product_simp F\ntest_issue_5923 E\n'

    parsed = parse_sympy_log(log)

    assert parsed == {
        'test_tensor_product_expand': 'passed',
        'test_tensor_product_simp': 'failed',
        'test_issue_5923': 'error',
    }


def test_only_the_bracketed_section_is_read():
    """eval.sh prints the whole environment setup first, and `pip install` output
    carries lines that read like test results."""
    log = 'PASSED setup\n>>>>> Start Test Output\nPASSED a::b\n>>>>> End Test Output\nPASSED after\n'

    assert bracketed_output(log).strip() == 'PASSED a::b'


def test_a_log_with_no_test_output_section_is_not_a_verdict():
    assert bracketed_output('pip install failed') is None


def test_an_unknown_log_parser_raises_rather_than_finding_no_tests():
    """No status found scores as every test failing, which is indistinguishable
    from an agent that achieved nothing."""
    with pytest.raises(ValueError, match='parse_log_matplotlib'):
        statuses('anything', 'parse_log_matplotlib')


# ── the milestone ladder ──────────────────────────────────────────────────────

def graded(f2p_status: str = 'ok', p2p_status: str = 'ok') -> str:
    return (
        f'>>>>> Start Test Output\n'
        f'{TESTS_JSON["FAIL_TO_PASS"][0]} ... {f2p_status}\n'
        f'{TESTS_JSON["PASS_TO_PASS"][0]} ... {p2p_status}\n'
        f'#15789 ... {p2p_status}\n'
        f'>>>>> End Test Output\n'
    )


def test_an_unchanged_repository_scores_nothing():
    assert grade('', None, TESTS_JSON, 'parse_log_django').reward == NO_CHANGE


def test_a_patch_whose_tests_never_ran_scores_the_patch():
    """A patch that breaks the environment is worth more than no patch and less
    than one the tests were run against."""
    result = grade('diff --git a/x b/x', 'pip install failed', TESTS_JSON, 'parse_log_django')

    assert result.reward == PATCH_WRITTEN
    assert result.resolved is False


def test_the_tests_running_scores_more_than_the_patch_alone():
    result = grade('diff', graded(f2p_status='ERROR'), TESTS_JSON, 'parse_log_django')

    assert result.reward == TESTS_RAN
    assert result.resolved is False


def test_all_target_tests_passing_with_no_regression_is_resolved():
    result = grade('diff', graded(), TESTS_JSON, 'parse_log_django')

    assert (result.reward, result.resolved) == (RESOLVED, True)


def test_a_regression_is_a_gate_rather_than_a_deduction():
    """A patch that makes the target test pass by breaking tests that passed
    before is not a fix, and SWE-bench scores it unresolved."""
    result = grade('diff', graded(p2p_status='FAIL'), TESTS_JSON, 'parse_log_django')

    assert result.reward == TESTS_RAN, 'the target test passing must not be credited'
    assert result.resolved is False


def test_some_of_the_target_tests_passing_scores_between_the_rungs():
    tests = {
        'FAIL_TO_PASS': ['test_a (m.T.test_a)', 'test_b (m.T.test_b)'],
        'PASS_TO_PASS': [],
    }
    log = (
        '>>>>> Start Test Output\n'
        'test_a (m.T.test_a) ... ok\ntest_b (m.T.test_b) ... FAIL\n'
        '>>>>> End Test Output\n'
    )

    result = grade('diff', log, tests, 'parse_log_django')

    assert TESTS_RAN < result.reward < RESOLVED
    assert result.resolved is False


# ── grading an episode ────────────────────────────────────────────────────────

def test_an_episode_that_changed_nothing_is_not_put_through_the_tests(started):
    """Running eval.sh costs minutes, and there is nothing to grade."""
    env, session = started

    reward, done, _ = env.feedback()

    assert (reward, done) == (NO_CHANGE, False)
    assert not any('runtests' in command for command in session.commands)


def test_a_resolved_episode_reports_the_reward_and_done(started):
    env, session = started
    session.outputs = {'git': 'diff --git a/x b/x', './tests/runtests.py': graded()}

    reward, done, message = env.feedback()

    assert (reward, done) == (RESOLVED, True)
    assert 'fixed' in message


def test_the_tests_are_run_once_however_often_the_verdict_is_asked_for(started):
    env, session = started
    session.outputs = {'git': 'diff --git a/x b/x', './tests/runtests.py': graded()}

    env.feedback()
    runs = [command for command in session.commands if 'runtests' in command]
    env.feedback()

    assert [command for command in session.commands if 'runtests' in command] == runs


def test_grading_removes_the_container(started):
    env, session = started

    env.feedback()

    assert session.closed is True, 'a container per task would be left running'


def test_a_new_episode_replaces_the_container_of_the_last(env, monkeypatch):
    sessions = []
    monkeypatch.setattr(module, 'ContainerSession', lambda **kwargs: sessions.append(FakeSession()) or sessions[-1])
    env.set_env({'id': INSTANCE, 'repo': 'django/django'})

    env.reset()
    env.reset()

    assert sessions[0].closed is True
    assert sessions[1].closed is False
    assert len(sessions) == 2


# ── the prompt and the dataset ────────────────────────────────────────────────

def test_the_prompt_asks_for_the_marker_the_env_recognises():
    assert 'think:' in swebench_prompt.swebench_solver_system_prompt
    assert swebench_prompt.swebench_few_shots, 'the few shot is what shows the format'


def test_every_reasoning_line_of_the_few_shot_is_recognised_as_one():
    for shot in swebench_prompt.swebench_few_shots:
        for line in shot.splitlines():
            if line.startswith('think'):
                assert ENVS['swebench'].is_thought(line), f'{line!r} would be run as a command'


def test_the_manifest_is_one_task_per_instance():
    with open(TASKS_PATH['swebench'], encoding='utf-8') as reader:
        rows = [json.loads(line) for line in reader]

    tasks = get_task('swebench')

    assert len(tasks) == len(rows)
    assert all(task['env_name'] == 'swebench' for task in tasks)
    assert all(task['id'] and task['repo'] for task in tasks)


def test_the_manifest_names_each_instance_once():
    tasks = get_task('swebench')

    assert len({task['id'] for task in tasks}) == len(tasks)


def test_every_repository_is_spread_over_the_whole_manifest():
    """A prefix of the dataset is then a proportional sample of the repositories
    rather than one repository's instances, and a memory carried across tasks
    cannot accumulate knowledge of one repository by accident of the ordering.
    """
    repos = [task['repo'] for task in get_task('swebench')]
    middle = len(repos) // 2

    for repo in set(repos):
        first, second = repos[:middle].count(repo), repos[middle:].count(repo)
        assert abs(first - second) <= 1, (
            f'{repo} appears {first} times in the first half and {second} in the second'
        )
