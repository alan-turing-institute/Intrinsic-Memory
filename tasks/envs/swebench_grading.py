"""Reading a SWE-bench verdict out of a test log, and the milestone ladder.

An instance's `eval.sh` runs whole test files and brackets their output with
`>>>>> Start Test Output`. Only the tests named in `tests.json` are graded, so
the verdict is read per test name and never off the summary line: on
`psf__requests-2931` both the patched and the unpatched run report 81 errors,
all of them tests wanting a live httpbin and none of them graded.

Each repository's test runner prints its own status lines, and `task.yaml` names
the parser SWE-bench itself uses. Three formats cover the repositories whose
images build on aarch64: unittest's verbose output (django), pytest's `-rA`
summary (pytest, requests), and sympy's own runner.

The reward is a ladder rather than the fraction of tests passing. The median
Verified instance has one FAIL_TO_PASS test, so a fraction would be binary in
practice and would score nothing for the work of getting as far as a patch that
runs.
"""

import re
from dataclasses import dataclass

PASSED = 'passed'
FAILED = 'failed'
ERROR = 'error'
SKIPPED = 'skipped'

TEST_OUTPUT_START = '>>>>> Start Test Output'
TEST_OUTPUT_END = '>>>>> End Test Output'

# The rungs. `resolved` is the top, and is what SWE-bench itself scores.
NO_CHANGE = 0.0
PATCH_WRITTEN = 0.25
TESTS_RAN = 0.5
RESOLVED = 1.0

_UNITTEST_LINE = re.compile(
    r'^(?P<name>.+?) \.\.\. (?P<status>ok|OK|FAIL|ERROR|skipped.*|'
    r'expected failure|unexpected success)$'
)
_UNITTEST_HEADERS = {'FAIL: ': FAILED, 'ERROR: ': ERROR}

_PYTEST_STATUSES = {
    'PASSED': PASSED,
    'FAILED': FAILED,
    'ERROR': ERROR,
    'SKIPPED': SKIPPED,
    'XFAIL': SKIPPED,
    'XPASS': SKIPPED,
}

# sympy's runner reports a status letter per test, and only `ok` is a pass.
_SYMPY_STATUSES = {'ok': PASSED, 'F': FAILED, 'E': ERROR, 'f': SKIPPED, 's': SKIPPED, 'w': PASSED}


def _unittest_status(status: str) -> str:
    if status in ('ok', 'OK'):
        return PASSED
    if status == 'FAIL' or status == 'unexpected success':
        return FAILED
    if status == 'ERROR':
        return ERROR
    return SKIPPED


def parse_unittest_log(log: str) -> dict[str, str]:
    """Statuses out of unittest's verbose output, as django's runtests prints it.

    A test with a docstring is announced over two lines, the second carrying the
    docstring rather than the name - which is why `#15789` is a graded name in
    `django__django-16485`. Taking whatever precedes ` ... ok` on one line gets
    both forms, and is the name `tests.json` uses.
    """
    statuses: dict[str, str] = {}

    for raw in log.splitlines():
        line = raw.strip()
        match = _UNITTEST_LINE.match(line)
        if match is not None:
            statuses[match.group('name').strip()] = _unittest_status(match.group('status'))
            continue

        for header, status in _UNITTEST_HEADERS.items():
            if line.startswith(header):
                statuses[line[len(header):].strip()] = status

    return statuses


def parse_pytest_log(log: str) -> dict[str, str]:
    """Statuses out of pytest's `-rA` summary: `PASSED tests/test_x.py::test_y`."""
    statuses: dict[str, str] = {}

    for raw in log.splitlines():
        status, _, rest = raw.strip().partition(' ')
        if status not in _PYTEST_STATUSES or not rest:
            continue

        # A failure line carries its reason after the node id.
        name = rest.split(' - ', 1)[0].strip()
        statuses[name] = _PYTEST_STATUSES[status]

    return statuses


def parse_sympy_log(log: str) -> dict[str, str]:
    """Statuses out of sympy's own runner: a padded name and a status letter."""
    statuses: dict[str, str] = {}

    for raw in log.splitlines():
        line = raw.strip()
        if not line.startswith('test_'):
            continue

        name, _, status = line.rpartition(' ')
        if name and status in _SYMPY_STATUSES:
            statuses[name.strip()] = _SYMPY_STATUSES[status]

    return statuses


# The parser names are `task.yaml`'s, so an instance is graded by the parser
# SWE-bench grades it with.
LOG_PARSERS = {
    'parse_log_django': parse_unittest_log,
    'parse_log_pytest': parse_pytest_log,
    'parse_log_pytest_options': parse_pytest_log,
    'parse_log_pytest_v2': parse_pytest_log,
    'parse_log_requests': parse_pytest_log,
    'parse_log_sympy': parse_sympy_log,
}


def bracketed_output(log: str) -> str | None:
    """The bracketed test output out of an eval.sh run, or None if it never ran.

    Everything before the marker is the environment being set up - a failed
    `pip install -e .` leaves a log with no test output at all, which is a
    different outcome from tests that ran and failed.
    """
    if TEST_OUTPUT_START not in log:
        return None

    section = log.split(TEST_OUTPUT_START, 1)[1]

    return section.split(TEST_OUTPUT_END, 1)[0]


def statuses(log: str, log_parser: str) -> dict[str, str]:
    """Every test status in a log, by the parser `task.yaml` names.

    An unknown parser raises rather than returning nothing: no status found
    scores as every test failing, which is indistinguishable from an agent that
    achieved nothing.
    """
    if log_parser not in LOG_PARSERS:
        raise ValueError(
            f'No log parser for {log_parser!r}. Known: {sorted(LOG_PARSERS)}. '
            f'A repository whose runner is not one of these cannot be graded.'
        )

    return LOG_PARSERS[log_parser](log)


@dataclass(frozen=True)
class Grade:
    """What an episode reached, and what to tell the agent about it."""

    reward: float
    resolved: bool
    message: str


def grade(patch: str, log: str | None, graded_tests: dict, log_parser: str) -> Grade:
    """How far up the ladder an episode got.

    `patch` is the diff the agent left in the repository and `log` the output of
    the instance's own eval.sh, or None where it was never run.

    PASS_TO_PASS is a gate rather than a score: a patch that makes the target
    test pass by breaking tests that passed before is not a fix, and SWE-bench
    scores it unresolved.
    """
    if not patch.strip():
        return Grade(NO_CHANGE, False, 'You left the repository unchanged, so nothing was graded.')

    section = None if log is None else bracketed_output(log)
    if section is None:
        return Grade(
            PATCH_WRITTEN,
            False,
            'You changed the repository, but its tests could not be run against your '
            'change, so no test verdict was reached.',
        )

    found = statuses(section, log_parser)
    fail_to_pass = list(graded_tests.get('FAIL_TO_PASS', []))
    pass_to_pass = list(graded_tests.get('PASS_TO_PASS', []))

    fixed = [name for name in fail_to_pass if found.get(name) == PASSED]
    regressed = [name for name in pass_to_pass if found.get(name) != PASSED]

    if regressed:
        return Grade(
            TESTS_RAN,
            False,
            f'Your change broke {len(regressed)} of the {len(pass_to_pass)} tests that '
            f'passed before it, so it is not a fix however many of the target tests pass.',
        )

    if fail_to_pass and len(fixed) == len(fail_to_pass):
        return Grade(
            RESOLVED,
            True,
            f'You fixed the issue: all {len(fail_to_pass)} target tests pass and nothing '
            f'that passed before regressed.',
        )

    share = len(fixed) / len(fail_to_pass) if fail_to_pass else 0.0

    return Grade(
        TESTS_RAN + (RESOLVED - TESTS_RAN) * share / 2,
        False,
        f'The tests ran against your change and {len(fixed)} of the {len(fail_to_pass)} '
        f'target tests pass. The issue is not fixed.',
    )
