# Dataset
## ALFWorld

The manifest, 134 tasks, is checked in as `data/alfworld/alfworld_tasks_suffix.json`:

```
curl -o alfworld_tasks_suffix.json https://raw.githubusercontent.com/LeapLabTHU/ExpeL/e41ec9a24823e7b560c561ab191441b56d9bcefc/data/alfworld/alfworld_tasks_suffix.json
```

Every row of it names the game file to play, so the games are the only other
download. The archive unpacks to `json_2.1.1/{train,valid_seen,valid_train,valid_unseen}`,
4027 games in all, which is the path the manifest and `tasks/env_configs/alfworld_config.yaml`
both expect - run it from `data/alfworld`:

```
curl -L -o alfworld.zip https://github.com/alfworld/alfworld/releases/download/0.4.2/json_2.1.3_tw-pddl.zip
unzip -q alfworld.zip && rm alfworld.zip
```

Those are `game.tw-pddl` files and nothing else. The rest of ALFRED is not needed:
`AlfredTWEnv.collect_game_files` would want a `traj_data.json` beside each game and
reports `0 games` without one, but `AlfworldEnv.set_env` assigns `game_files` from
the manifest, so the collected split is never what gets played. The handcoded
expert does read `traj_data.json` - it is only attached to the `train` split, and
the experiments run `eval_out_of_distribution`.

### The simulator

`alfworld` is in neither `pyproject.toml` nor `requirements.txt`: both its
compiled dependencies lack an aarch64 wheel, so `uv sync` would fail on Isambard
and on any other Arm machine. Install it separately. On x86_64 that is the whole
of it:

```bash
uv pip install alfworld
```

On aarch64, two things fail.

**TextWorld** builds from its sdist, and `setup.sh` unpacks
`inform7-compilers_6M62_$(uname -m).tar.gz`. Inform 7 6M62 ships `i386`,
`x86_64`, `ppc` and `armv6lhf` - no `aarch64`, and `armv6lhf` is 32-bit ARM,
which Neoverse-V2 cannot run. Inform is the compiler for *authoring* a game;
playing a pre-generated `.tw-pddl` goes through the PDDL engine and never calls
it, so let that step fail:

```bash
curl -sLO https://files.pythonhosted.org/packages/source/t/textworld/textworld-1.7.0.tar.gz
tar xzf textworld-1.7.0.tar.gz
sed -i '/inform7-\(compilers\|interpreters\)_6M62_/ s/$/ || true/' textworld-1.7.0/setup.sh

uv pip install setuptools wheel Cython numpy
uv pip install --no-build-isolation ./textworld-1.7.0
uv pip install --no-deps alfworld
```

**fast-downward-textworld**, the PDDL planner behind `textworld[pddl]`, compiles
the whole planner with CMake. On a login node that ends in `g++: internal
compiler error: Killed (program cc1plus)` - the OOM killer, not the
architecture - so build it on a compute node:

```bash
srun --nodes=1 --gpus=1 --time=00:30:00 \
  uv pip install --no-build-isolation fast-downward-textworld==20.6.4
```

Then check the whole path with `TASK=alfworld sbatch slurm/smoke_test.sh`, which
prints the simulator version and the number of games it can see before it runs.

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
