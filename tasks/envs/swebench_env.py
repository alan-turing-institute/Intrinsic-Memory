"""SWE-bench: fix a real issue in a real repository, graded by its own tests.

One task is one instance - a repository at the commit before its fix, the issue
text as it was reported, and the tests the fix has to make pass. The agent works
in a container built from that instance's image, one shell command per turn, and
the episode is graded by running the instance's own `eval.sh`. Building those
images for aarch64 is `containers/isambard.md`; nothing here builds one.

Two things about how the shell is driven:

- Each command is its own `exec` in a long-running container rather than a line
  fed to one persistent shell. A command that hangs then costs one trial instead
  of the episode's work, because the edits live in the container's filesystem
  rather than in the shell. The working directory is carried across turns
  explicitly; anything else shell-local - exported variables, shell functions -
  is not.
- Nothing from the task directory is mounted. `eval.sh` carries the test patch
  inline, so grading needs no mount, and mounting the directory would put
  `gold.patch` inside the container with the agent.
"""

import json
import os
import re
import shlex
import subprocess
import yaml
from dataclasses import dataclass
from pathlib import Path

from mas.mas import EpisodeResult

from .base_env import BaseEnv, BaseRecorder, is_thought_line
from .swebench_grading import Grade, grade

DEFAULT_TASK_REPO = '~/swebench-arm64/swe-bench-tasks'
DEFAULT_IMAGE_TEMPLATE = 'localhost/sweb.eval.arm64.{instance}:latest'
DEFAULT_RUNTIME = 'podman-hpc'
DEFAULT_SQUASH_DIR = '~/GMemory/swebench-images'
DEFAULT_SETUP_COMMAND = 'source /opt/miniconda3/bin/activate testbed'
DEFAULT_WORKDIR = '/testbed'
DEFAULT_COMMAND_TIMEOUT = 120
DEFAULT_EVAL_TIMEOUT = 1800
DEFAULT_START_TIMEOUT = 300
DEFAULT_MAX_OUTPUT_CHARS = 3000

SUBMIT = 'submit'
DIFF_COMMAND = 'git -c core.fileMode=false diff'

NO_COMMAND = (
    'That reply carried no command. Reply with one shell command, a line '
    f'beginning `think:`, or `{SUBMIT}` when your change is complete.'
)
SUBMITTED = 'Submitted. Your change is now being graded against the repository\'s tests.'
NO_OUTPUT = '(the command printed nothing)'

_FENCE = re.compile(r'^\s*```')
_LABEL = re.compile(r'^(?:action|command|step)\s*\d*\s*:\s*', re.IGNORECASE)
_LIST_MARKER = re.compile(r'^(?:[-+•]|\(?\d+[.)])\s+')
_SHELL_PROMPT = re.compile(r'^[$#]\s+')
_QUOTE_PAIRS = (('"', '"'), ("'", "'"), ('“', '”'), ('‘', '’'))

_MARKER = '__SWEBENCH_TRIAL__'
_MARKER_LINE = re.compile(rf'^{_MARKER} (?P<status>-?\d+) (?P<cwd>.*)$', re.MULTILINE)


def _fenced_command(reply: str) -> str | None:
    """The contents of the first fenced block, which is one command however
    many lines it spans.

    A heredoc is the only way to write a file in one turn, and it cannot survive
    being split into lines - so a fenced block is taken whole.
    """
    lines = reply.splitlines()
    opening = next((index for index, line in enumerate(lines) if _FENCE.match(line)), None)
    if opening is None:
        return None

    body: list[str] = []
    for line in lines[opening + 1:]:
        if _FENCE.match(line):
            break
        body.append(line)

    command = '\n'.join(body).strip('\n')

    return command or None


def _strip_wrapping(line: str) -> str:
    """Wrapping quotes and bold markers off a whole line, and nothing inside it.

    `clean_action_line` cannot be used for a shell: it deletes every `<`, `>`,
    `*`, `#` and backtick in the line, which are redirection, globs, comments
    and command substitution here. Measured against it, `ls tests/*.py` becomes
    `ls tests/.py` and `python -c "print(1)" > /tmp/out` loses its redirection.
    """
    previous = None
    while previous != line:
        previous = line
        line = line.strip()

        if len(line) > 4 and line.startswith('**') and line.endswith('**'):
            line = line[2:-2]
            continue

        for opening, closing in _QUOTE_PAIRS:
            # Only a quote wrapping the whole line is decoration; one inside it
            # is the shell's.
            if len(line) > 1 and line.startswith(opening) and line.endswith(closing):
                if opening not in line[1:-1] and closing not in line[1:-1]:
                    line = line[1:-1]
                    break

    return line


def _strip_trailing_stop(line: str) -> str:
    """A full stop a model wrote at the end of a sentence, but not a path.

    `git add .` and `cp file ..` end in one that is an argument.
    """
    if not line.endswith('.'):
        return line

    last = line.split()[-1] if line.split() else ''
    if last in ('.', '..') or last.endswith('/.') or last.endswith('/..'):
        return line

    return line[:-1].rstrip()


def _undecorate(line: str) -> str:
    """One line of model output with the formatting taken off, shell intact.

    Returns empty for a line carrying nothing to run, which is a stray code
    fence.
    """
    if _FENCE.match(line):
        return ''

    line = line.strip()
    previous = None
    while previous != line:
        previous = line
        line = _LIST_MARKER.sub('', line.strip())
        line = _LABEL.sub('', line.strip())
        line = _SHELL_PROMPT.sub('', line.strip())

    return _strip_trailing_stop(_strip_wrapping(line))


def clean_shell_action(reply: str) -> str:
    """The command out of what a model wrote, however it dressed it up.

    A fenced block is the command; otherwise it is the first line that is not a
    reasoning step, since a reply carrying a thought and then a command can only
    be done one way round and taking the thought spends the trial achieving
    nothing.
    """
    fenced = _fenced_command(reply)
    if fenced is not None:
        return fenced

    lines = [line for line in (_undecorate(raw) for raw in reply.splitlines()) if line]
    if not lines:
        return ''

    if not is_thought_line(lines[0]):
        return lines[0]

    for line in lines[1:]:
        if not is_thought_line(line):
            return line

    return lines[0]


def elide(output: str, limit: int) -> str:
    """`output` cut to `limit` characters, keeping both ends.

    Every turn re-sends the whole trajectory, so an untruncated `cat` of a
    source file is paid for on every later turn as well as this one. The tail is
    kept because a test run says what happened in its last lines.
    """
    if len(output) <= limit:
        return output

    head, tail = limit // 2, limit - limit // 2
    cut = len(output) - head - tail

    return f'{output[:head]}\n... [{cut} characters omitted] ...\n{output[-tail:]}'


@dataclass
class CommandResult:
    output: str
    status: int | None


@dataclass
class ContainerSession:
    """A container holding one instance's repository, one command at a time."""

    image: str
    name: str
    runtime: str = DEFAULT_RUNTIME
    setup_command: str = DEFAULT_SETUP_COMMAND
    workdir: str = DEFAULT_WORKDIR
    start_timeout: int = DEFAULT_START_TIMEOUT
    squash_dir: str = None
    cwd: str = None

    def start(self) -> None:
        self.cwd = self.workdir
        self._run_runtime(
            ['run', '--detach', '--rm', '--name', self.name, '--workdir', self.workdir,
             self.image, 'sleep', 'infinity'],
            timeout=self.start_timeout,
        )

    def run(self, command: str, timeout: int) -> CommandResult:
        """`command` in the container, in the directory the last one left.

        The marker carries the exit status and the working directory back out,
        so a `cd` is not lost between turns.

        Note stderr is merged into stdout rather than captured beside it.
        `eval.sh` brackets the test output with `>>>>> Start Test Output`, and
        those markers are xtrace lines on stderr while the tests print to
        whichever stream their runner uses - unittest's is stderr, pytest's is
        stdout. Captured apart and concatenated, the section between the markers
        holds none of the test output.
        """
        try:
            completed = subprocess.run(
                [self.runtime, 'exec', self.name, 'bash', '-c', self._script(command)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout,
                stdin=subprocess.DEVNULL, env=self._environment(),
            )
        except subprocess.TimeoutExpired:
            return CommandResult(
                f'The command was still running after {timeout} seconds and was abandoned.',
                None,
            )

        return self._read(completed.stdout)

    def _script(self, command: str) -> str:
        """`command` with the setup, the working directory and the marker around it.

        The marker is printed from an EXIT trap set before the command rather
        than from a line after it: a command whose heredoc is left unterminated
        would swallow that line, and the working directory would be lost for the
        rest of the episode.
        """
        marker = f'printf "\\n{_MARKER} %s %s\\n" "$?" "$PWD"'

        return (
            f'{self.setup_command}\n'
            f'cd {shlex.quote(self.cwd)} 2>/dev/null || cd {shlex.quote(self.workdir)}\n'
            f"trap '{marker}' EXIT\n"
            f'{command}\n'
        )

    def close(self) -> None:
        """Remove the container. Safe to call on a session that never started."""
        if self.cwd is None:
            return

        self.cwd = None
        try:
            self._run_runtime(['rm', '--force', self.name], timeout=self.start_timeout)
        except (OSError, subprocess.SubprocessError, RuntimeError):
            pass

    def _read(self, output: str) -> CommandResult:
        match = _MARKER_LINE.search(output)
        if match is None:
            # The command replaced or killed the shell, so nothing came back to
            # say where it left the working directory.
            return CommandResult(output.strip(), None)

        self.cwd = match.group('cwd').strip()

        return CommandResult(output[:match.start()].strip(), int(match.group('status')))

    def _environment(self) -> dict:
        """The runtime's environment, with the image store this session reads.

        `podman-hpc` looks for migrated images under `SQUASH_DIR`, so a job that
        does not name the same directory the images were built into sees none of
        them and tries to pull from a registry instead.
        """
        if self.squash_dir is None:
            return None

        return {**os.environ, 'SQUASH_DIR': self.squash_dir}

    def _run_runtime(self, arguments: list[str], timeout: int) -> str:
        completed = subprocess.run(
            [self.runtime, *arguments], capture_output=True, text=True, timeout=timeout,
            stdin=subprocess.DEVNULL, env=self._environment(),
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f'`{self.runtime} {" ".join(arguments)}` failed: '
                f'{completed.stderr.strip() or completed.stdout.strip()}'
            )

        return completed.stdout


class SWEBenchEnv(BaseEnv):
    """One SWE-bench instance, worked on through a shell in its own container."""

    def __init__(self, env_config: dict, max_trials: int):
        super().__init__(env_config, max_trials)
        config: dict = env_config or {}
        self.task_repo = Path(os.path.expanduser(config.get('task_repo', DEFAULT_TASK_REPO)))
        self.image_template: str = config.get('image_template', DEFAULT_IMAGE_TEMPLATE)
        self.runtime: str = config.get('runtime', DEFAULT_RUNTIME)
        self.setup_command: str = config.get('setup_command', DEFAULT_SETUP_COMMAND)
        self.workdir: str = config.get('workdir', DEFAULT_WORKDIR)
        self.command_timeout: int = config.get('command_timeout', DEFAULT_COMMAND_TIMEOUT)
        self.eval_timeout: int = config.get('eval_timeout', DEFAULT_EVAL_TIMEOUT)
        self.start_timeout: int = config.get('start_timeout', DEFAULT_START_TIMEOUT)
        self.max_output_chars: int = config.get('max_output_chars', DEFAULT_MAX_OUTPUT_CHARS)
        self.squash_dir: str = os.path.expanduser(config.get('squash_dir', DEFAULT_SQUASH_DIR))

        self.instance: str = None
        self.session: ContainerSession = None
        self.submitted: bool = False
        self.verdict: Grade = None

    def set_env(self, configs: dict) -> tuple[str, str]:
        instance: str = configs.get('id')
        if instance is None:
            raise ValueError('A swebench task config needs an `id`, the instance id.')

        task_dir = self.task_repo / 'tasks' / instance
        if not task_dir.is_dir():
            raise FileNotFoundError(
                f'No task directory for {instance} at {task_dir}. The task repository is '
                f'cloned separately; see containers/isambard.md.'
            )

        self.instance = instance
        self.task_dir = task_dir
        self.task = yaml.safe_load((task_dir / 'task.yaml').read_text())
        self.graded_tests = json.loads((task_dir / 'tests.json').read_text())
        self.eval_script = (task_dir / 'eval.sh').read_text()
        self.image = self.image_template.format(instance=instance.lower())

        return instance, self._task_description((task_dir / 'problem_statement.md').read_text())

    def reset(self) -> None:
        """A container of its own for this episode, replacing any still running."""
        if self.session is not None:
            self.session.close()

        self.submitted = False
        self.verdict = None
        # The workers of a sweep run concurrently on one node, so the name has
        # to be unique per process as well as per instance.
        self.session = ContainerSession(
            image=self.image,
            name=f'swebench-{self.instance.lower()}-{os.getpid()}',
            runtime=self.runtime,
            setup_command=self.setup_command,
            workdir=self.workdir,
            start_timeout=self.start_timeout,
            squash_dir=self.squash_dir,
        )
        self.session.start()

    def step(self, action: str) -> tuple[str, float, bool]:
        action = self.process_action(action)

        if not action:
            return NO_COMMAND, -1, False

        if self.is_thought(action):
            return 'OK.', 0, False

        if action.strip().lower() == SUBMIT:
            self.submitted = True
            return SUBMITTED, 0, True

        result = self.session.run(action, timeout=self.command_timeout)

        return elide(result.output, self.max_output_chars) or NO_OUTPUT, 0, False

    @staticmethod
    def process_action(action: str) -> str:
        return clean_shell_action(action)

    def feedback(self) -> tuple[float, bool, str]:
        """The episode's verdict, from running the instance's own eval.sh once.

        The tests are the same whether the agent submitted or ran out of trials:
        an episode that never submitted is graded on the change it had made by
        then, which is what SWE-bench would score.
        """
        if self.verdict is None:
            self.verdict = self._score()

        return self.verdict.reward, self.verdict.resolved, self.verdict.message

    def _score(self) -> Grade:
        if self.session is None or self.session.cwd is None:
            return grade('', None, self.graded_tests, self.task['log_parser'])

        try:
            patch = self.session.run(DIFF_COMMAND, timeout=self.command_timeout).output
            log = None
            if patch.strip():
                log = self.session.run(self.eval_script, timeout=self.eval_timeout).output

            return grade(patch, log, self.graded_tests, self.task['log_parser'])
        finally:
            self.session.close()

    def _task_description(self, problem_statement: str) -> str:
        tests = len(self.graded_tests.get('FAIL_TO_PASS', []))

        return (
            f'You are working in a checkout of {self.task["repo"]} at '
            f'{self.workdir}, at the commit before this issue was fixed.\n\n'
            f'## The issue\n\n{problem_statement.strip()}\n\n'
            f'Change the repository so that the issue is fixed. '
            f'{tests} test{"" if tests == 1 else "s"} that fail today must pass, and '
            f'every test that passes today must still pass. Do not change the tests '
            f'themselves: they are restored from the repository before they are run.'
        )


@dataclass
class SWEBenchRecorder(BaseRecorder):

    def __post_init__(self):
        super().__post_init__()
        self.task = 'swebench'

    def task_begin(self, task_id: int, task_config: dict):
        super().task_begin(task_id, task_config)
        self.log(
            f'---------- Task: {task_id} ({task_config.get("id")}, '
            f'{task_config.get("repo")}) ----------'
        )

    def task_end(self, episode: EpisodeResult):
        super().task_end(episode)

        averages = self.average_results()
        self.log(
            f'reward: {episode.reward}, resolved: {episode.done}.\n'
            f'ave reward: {averages.mean_reward}, resolve rate: {averages.mean_done}'
        )
