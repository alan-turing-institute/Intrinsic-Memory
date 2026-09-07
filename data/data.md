# Dataset
## ALFWorld

For the tasks suffix json
```
curl -o alfworld_tasks_suffix.json https://raw.githubusercontent.com/LeapLabTHU/ExpeL/e41ec9a24823e7b560c561ab191441b56d9bcefc/data/alfworld/alfworld_tasks_suffix.json
```

For PPDL and game files

```
curl -L -o alfworld.zip https://github.com/alfworld/alfworld/releases/download/0.4.2/json_2.1.3_tw-pddl.zip
```

## PDDL

```
curl -L -o data.tar.gz https://huggingface.co/datasets/hkust-nlp/agentboard/resolve/main/data.tar.gz

tar -zxvf data.tar.gz
```

Get the test.jsonl from data/pddl/test.jsonl

## FEVER
curl -L -o train.jsonl https://fever.ai/download/fever/train.jsonl

## HotpotQA

The original `hotpot_dev_distractor_v1.json` host (`curtis.ml.cmu.edu`) no longer
answers, so the dev split comes from the HuggingFace mirror. Only the question and
the answer are used - the agent searches live Wikipedia rather than a supplied
context - so the manifest keeps the five scalar fields and drops `context` and
`supporting_facts`, which are 45 MB of the 46.

```
for off in $(seq 0 100 7400); do
  curl -s "https://datasets-server.huggingface.co/rows?dataset=hotpotqa%2Fhotpot_qa&config=distractor&split=validation&offset=$off&length=100" \
    | jq -c '.rows[].row | {id, question, answer, level, type}'
done > hotpotqa/hotpotqa_dev.jsonl
```

7405 questions, all `level: hard` (5918 `bridge`, 1487 `comparison`). The API rate
limits partway through, so re-run any offset whose response is not JSON.

## Jericho

The interpreter comes from the `jericho` package; the game files do not, and are
not in this repository. The manifest lists the 56 games of the Jericho suite whose
score Jericho can read, and the roms go beside it in `data/jericho/roms/`:

```
mkdir -p jericho/roms
curl -s "https://api.github.com/repos/BYU-PCCL/z-machine-games/git/trees/master?recursive=1" \
  | jq -r '.tree[].path | select(startswith("jericho-game-suite/")) | select(contains("/") and (split("/") | length == 2))' \
  | while read -r path; do
      curl -sL -o "jericho/roms/$(basename "$path")" \
        "https://raw.githubusercontent.com/BYU-PCCL/z-machine-games/master/${path}"
    done
```

That fetches 57 files, 8.5 MB. One of them, `lgop.z3`, is deliberately not in the
manifest: Jericho reports a maximum score of 0 for it and knows no walkthrough, so
neither the progress rate nor a victory could ever be scored. The other 56 all
load and report a score range.

### The trial budget

`max_steps` for Jericho is 100 rather than the 30 every other task uses. The
number comes from walking each game's own walkthrough - an oracle, so a ceiling
no agent can beat - and recording the progress rate reached at each budget:

| budget | mean progress | median | games won | games still on zero |
|---|---|---|---|---|
| 30 | 18.0% | 10.0% | 2/56 | 7/56 |
| 50 | 30.4% | 19.2% | 5/56 | 4/56 |
| 75 | 47.6% | 30.0% | 13/56 | 0/56 |
| 100 | 56.2% | 42.4% | 19/56 | 0/56 |
| 150 | 66.5% | 71.7% | 24/56 | 0/56 |

At 30 moves seven games cannot score at all, so those tasks contribute nothing
whatever the memory module does, and `done` is near-dead with 2 of 56 winnable
by a perfect player. The zero floor disappears at 75. 100 clears it with margin,
makes 19 games winnable so `done` carries signal again, and leaves the median
game at 42% so the score has room to move in both directions.

The turns are linear in the budget - 100 trials over 56 games, 10 seeds and 10
arms is about 560k agent turns against 168k at 30 - but the tokens are not, and
that is the number that matters. Every turn re-sends the whole trajectory, so
prompt tokens grow with the square of the budget. Measured on one game:

| trials | prompt tokens for one task |
|---|---|
| 6 | 40,902 |
| 12 | 94,305 |
| 25 | 216,989 |

which fits `76*n^2 + 6789*n`. So 30 trials is about 272k prompt tokens per task
and 100 is about 1.44M: **5.3x the tokens for 3.3x the turns**. Over the whole
sweep that is roughly 1.5B prompt tokens at 30 against 8.1B at 100. Measured with
a 0.5B model, whose replies are short, so treat those as a floor.

A single call stays well inside a normal context window - about 11k tokens at
trial 30 and 22k at trial 100 - so this is cost rather than a limit. 75 is the
cheapest budget at which no game is stuck on zero, at 3.4x the tokens of 30.

Four of the games open above zero — `advent` on 36 of 350, `detective` on 10 of
360, `deephome` on 1 of 300 and `ludicorp` on 1 of 150 — which is why the progress
rate is measured from the opening score rather than from nothing.

## BabyAI

Nothing to download. `minigrid` generates each gridworld from a level name and a
seed, so the manifest holds those instead of the data: 20 of the 96 registered
BabyAI levels, spread across the competence ladder, at 10 seeds each for 200 tasks.

It is written seed-major - every level at seed 0, then every level at seed 1 - so
that a `--max_tasks` prefix is a spread across the levels rather than many seeds of
the first one. `tasks/tests/test_babyai_env.py` asserts that ordering.

To regenerate it, or to change the levels or the number of seeds:

```python
import json
import gymnasium as gym
import minigrid  # registers the BabyAI levels

LEVELS = [...]           # level ids, all of which must be in gym.registry
SEEDS = range(10)

with open('babyai/babyai_levels.jsonl', 'w') as out:
    for seed in SEEDS:
        for level in LEVELS:
            out.write(json.dumps({'id': f'{level}-seed{seed}', 'level': level, 'seed': seed}) + '\n')
```

The full list of registered levels is `sorted(k for k in gym.registry if k.startswith('BabyAI-'))`.

### The trial budget

BabyAI stays at 30, unlike Jericho. A mission is pass/fail, so the only question
is whether a perfect player fits in the budget; `minigrid.utils.baby_ai_bot`
solves 190 of the 200 tasks and needs a median of 9 actions, so 30 leaves most
tasks about three times the oracle's budget.

The tail is long, though, and four levels carry nearly all of it - `Unlock`
(median 72 actions), `GoTo` (60), `UnblockPickup` (24, up to 193) and
`SynthS5R2` (13, up to 168). They are large mazes where even the oracle wanders,
so they measure exploration rather than memory. Raising the budget does not fix
them: 30 trials puts 74% of tasks within a perfect player's reach and 50 trials
only 86%, for two thirds more compute. Replacing those four levels would buy more
than raising the budget.

The 10 tasks the oracle does not solve are all `PutNextS5N2Carrying`, where the
agent starts already holding something and the bot asserts it is empty-handed.
The level itself is solvable - dropping first and then handing over to the bot
completes it - so this is a limitation of the measurement, not of the task.

## SWE-bench

One task is one SWE-bench Verified instance: a repository at the commit before an
issue was fixed, the issue text, and the tests the fix has to make pass. Three
things are needed, and only the first is in this repository:

- `data/swebench/verified.jsonl` — the manifest, an instance id and its
  repository per row.
- the task repository, cloned separately, which holds each instance's issue
  text, tests and `eval.sh`.
- one container image per instance, built for aarch64 on Isambard.

Both of the latter are `containers/isambard.md`.

### What the manifest names, and what it leaves out

Verified is 500 instances across 12 repositories. The manifest holds the 333
from the four whose images build on aarch64 — django (231), sympy (75), pytest
(19) and requests (8). The exclusion is a build fact rather than a sampling
choice: `pydata__xarray-7229` pins the CDAT stack, which has no linux-aarch64
build in any channel, and that is the shape of the failure to expect.

The rows are ordered so that each repository is spread evenly through the file —
each instance sits at its fractional position through its own repository's
instances — which makes any prefix a proportional sample of the four rather than
231 django instances followed by everything else. `max_tasks` for swebench is 20,
because 20 instances is 20 images and about 50 GB.

Regenerating it needs the Verified instance list, which is the dataset's own
row order:

```
for off in 0 100 200 300 400; do
  curl -s "https://datasets-server.huggingface.co/rows?dataset=SWE-bench%2FSWE-bench_Verified&config=default&split=test&offset=$off&length=100" \
    | jq -r '.rows[].row | [.instance_id, .repo] | @tsv'
done > verified.tsv
```

then, keeping the four repositories and spreading them:

```
python - <<'EOF' > swebench/verified.jsonl
import json
BUILDS = {'django/django', 'sympy/sympy', 'pytest-dev/pytest', 'psf/requests'}
by_repo = {}
for line in open('verified.tsv'):
    instance, repo = line.split('\t')[:2]
    if repo in BUILDS:
        by_repo.setdefault(repo, []).append(instance)
for instances in by_repo.values():
    instances.sort()
for _position, repo, instance in sorted(
    ((rank + 0.5) / len(instances), repo, instance)
    for repo, instances in sorted(by_repo.items())
    for rank, instance in enumerate(instances)
):
    print(json.dumps({'id': instance, 'repo': repo}))
EOF
```

### The reward is a ladder, not a fraction

The median Verified instance has one FAIL_TO_PASS test (mean 3.0, max 438), so
scoring the fraction of target tests passing would be binary in practice and
would score nothing for the work of getting as far as a patch that runs. The
rungs are in `tasks/envs/swebench_grading.py`:

| reward | what the episode reached |
|---|---|
| 0.0 | the repository is unchanged |
| 0.25 | a non-empty diff, but its tests could not be run |
| 0.5 | the tests ran, and either none of the target tests pass or something regressed |
| 0.5–0.75 | the tests ran and some of the target tests pass |
| 1.0 | resolved: every target test passes and nothing that passed before regressed |

PASS_TO_PASS is a gate rather than a deduction: a patch that makes the target
test pass by breaking tests that passed before is not a fix, and SWE-bench scores
it unresolved. `done` is reserved for `resolved`, as Jericho reserves it for
victory.

The verdict is read per test name against the instance's `tests.json`, never off
the summary line. `eval.sh` runs whole test files, which can hold tests that are
not graded: on `psf__requests-2931` both the patched and the unpatched run report
81 errors, all of them tests wanting a live httpbin and none of them graded.

Each repository's runner prints its own status lines, and `task.yaml` names the
parser SWE-bench itself grades with. Three formats cover the four repositories:
unittest's verbose output (django), pytest's `-rA` summary (pytest, requests) and
sympy's own runner. An instance naming any other parser raises rather than
scoring zero, which is what a fifth repository will do the day its images build.

### The trial budget

`max_steps` is 30, the same as every dataset except Jericho, and it is not
calibrated. Jericho's 100 came from walking each game's own walkthrough; the
equivalent here is replaying a known-good sequence of shell commands per
instance and recording the reward reached at each budget, which has not been
done. 30 turns is enough to read a file, edit it and run one test file, and
prompt tokens grow with the square of the budget.

### Two things the agent can see that a benchmark run might not want it to

- The container's git history is the repository's, so the commit that fixed the
  issue is reachable with `git log --all` in some images. SWE-bench's own harness
  has the same property.
- Nothing from the task directory is mounted, so `gold.patch` and `test.patch`
  are not in the container. Grading hands `eval.sh` to `bash -c` instead.
