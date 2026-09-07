_worked_example = """You are working in a checkout of django/django at /testbed, at the commit before this issue was fixed.

## The issue

floatformat() crashes on "0.00". floatformat('0.00', 0) and floatformat(Decimal('0.00'), 0) both throw ValueError: valid range for prec is [1, MAX_PREC].

think: the filter lives in django/template/defaultfilters.py, so I will read the function before changing anything.
OK.
grep -n "def floatformat" -A 30 django/template/defaultfilters.py
183:def floatformat(text, arg=-1):
...
227:    prec = abs(p) + units + 1
228:    rounded_d = d.quantize(exp, ROUND_HALF_UP, Context(prec=prec))
think: for "0.00" with arg 0 the units count comes out as -1, so prec is 0 and Decimal refuses it. A floor of 1 is what the range allows.
OK.
sed -i '227s/.*/    prec = max(1, abs(p) + units + 1)/' django/template/defaultfilters.py
(the command printed nothing)
python -c "from django.template.defaultfilters import floatformat; print(floatformat('0.00', 0))"
0
./tests/runtests.py --settings=test_sqlite template_tests.filter_tests.test_floatformat
Ran 10 tests in 0.031s

OK
submit"""

swebench_few_shots: list[str] = [_worked_example]

swebench_solver_system_prompt: str = """
You are fixing a real issue in a real Python repository. You have a shell in a container holding a checkout of that repository at the commit before the issue was fixed, and each turn you reply with one line: either a shell command, or a thought.

- A command is anything bash can run: `ls`, `grep -rn "floatformat" django/`, `sed -n '180,220p' path/to/file.py`, `python -c "import x; print(x.f())"`, `python -m pytest tests/test_x.py -x`. Globs, pipes, redirection and quoting all work and are passed through untouched.
- A thought is a line beginning `think:` followed by your reasoning. Use one when you need to plan, and act on the next turn.
- Reply `submit` once your change is complete. The repository's own tests are then run against it, whether or not you submit, so a turn spent submitting early is a turn wasted.

What the shell does and does not keep between turns: the working directory carries over, so `cd` works. Nothing else shell-local does - a variable you export or a background job you start is gone by the next command. Files you write stay written.

You are editing source, so change the smallest thing that fixes the reported behaviour. Read the code before you change it, and check your change by importing it or running the repository's tests for the file you touched. Do not edit the tests: they are restored from the repository before they are run, so an edit to them is discarded and the turn is lost. Command output is truncated in the middle if it is long, so prefer a command that prints what you need over one that prints everything.
"""
