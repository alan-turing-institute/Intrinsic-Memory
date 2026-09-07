# G-Memory: Tracing Hierarchical Memory for Multi-Agent Systems

## AZURE Usage

### Building the image

Login to Azure
```bash
az acr login --name intrinsic
```

Build the image
```bash
docker build -t intrinsic.azurecr.io/g-memory:<tag> .
```

Push the image
```bash
docker push intrinsic.azurecr.io/g-memory:<tag>
```


### Running the containers
Put secrets into `template/parameters.json`
- imagePassword can be found in the Azure Portal > Container registry > intrinsic > Settings > Access Keys
- environmentVariable5 is the Foundry API Key

`template/generate_templates.py` fills the numbered slots, which `entrypoint.sh` reads as named variables:

| Slot | | Slot | |
|---|---|---|---|
| 0 | `--task` | 4 | `OPENAI_API_BASE` |
| 1 | `--mas_memory` | 5 | `OPENAI_API_KEY` (Foundry) |
| 2 | `--seed` | 6 | storage connection string |
| 3 | `--model` | 7 | `--mas_type`, currently always `autogen` |

`entrypoint.sh` passes no `--use_validator` and no `--db_dir`, so the container runs the plain configuration into `./.db` inside the image.

Check it works

```bash
az deployment group validate --resource-group intrinsic-memory --template-file template/template.json --parameters template/parameters.json
```

Deploy the container
```bash
az deployment group create --resource-group intrinsic-memory --template-file template/template.json --parameters template/parameters.json

```

## 🖥️ HPC Usage (Slurm)

Slurm job scripts for running experiments on an HPC cluster (developed against the BriCS/Isambard-AI environment) live in `slurm/`. They follow two patterns:

**Self-contained: serve a model + run experiments in one job**
`slurm/alfworld_experiment.sh`, `slurm/babyai_experiment.sh`, `slurm/fever_experiment.sh`, `slurm/hotpotqa_experiment.sh`, `slurm/jericho_experiment.sh`, `slurm/pddl_experiment.sh`, `slurm/sciworld_experiment.sh`, `slurm/single_node_serve.sh`

Each of these scripts:
1. Requests a node with 4 GPUs (`#SBATCH --gpus=4 --exclusive`) and loads cluster modules (`module load brics/nccl`).
2. Starts a local `vllm serve` process in the background for a model (default `openai/gpt-oss-120b`), polling `/health` until it's ready.
3. Points `OPENAI_API_BASE` at the local vLLM server (`http://localhost:8000/v1`) and runs `uv run tasks/run.py`, sweeping over every memory module and 10 seeds for one task (alfworld/babyai/fever/hotpotqa/jericho/pddl/sciworld) or fever, pddl and sciworld at once (`single_node_serve.sh`).
4. Kills the vLLM process once `run.py` finishes.

Alongside the seven is one `slurm/crosstask.sh`: every dataset's intrinsic modules again, run with `--intrinsic_cross_task` so the memory is kept across the tasks of the dataset instead of starting each task from nothing. It gives each dataset its own `tasks/run.py`, concurrently against the one server, because the sweep is a Cartesian product and a single call over every dataset would pair each with every other dataset's hand-written template. Point it at the same results directory as the seven — the two arms are told apart by the `intrinsic_cross_task` column, not by the file they land in.

The datasets do not cost the same, so the starting token budget is per dataset in `slurm/generate_slurm.py` (`MAX_TOKENS_OVERRIDES`, with `DEFAULT_MAX_TOKENS` for the rest). `TIME_LIMIT` and `MAX_NUM_SEQS` are one value for every job. Edit the generator and rerun it; the scripts themselves are gitignored.

```bash
export DB_DIR=/projects/<project>/results/experiment-2026-09
sbatch slurm/alfworld_experiment.sh   # 10 arms x 10 seeds
sbatch slurm/babyai_experiment.sh     # 10 arms x 10 seeds
sbatch slurm/fever_experiment.sh      # 10 arms x 10 seeds
sbatch slurm/hotpotqa_experiment.sh   # 10 arms x 10 seeds
sbatch slurm/jericho_experiment.sh    # 10 arms x 10 seeds
sbatch slurm/pddl_experiment.sh       # 10 arms x 10 seeds
sbatch slurm/sciworld_experiment.sh   # 10 arms x 10 seeds
sbatch slurm/crosstask.sh             # every dataset's 3 intrinsic arms, cross-task
```

`DB_DIR` defaults to `$HOME/GMemory/.db-experiment`, and every script echoes where it is writing. Use a directory no earlier run wrote to: a run refuses to append to a results file whose header is not its schema.

**Check the whole path in half an hour first**
`slurm/smoke_test.sh` has the same shape as those - serve, then run - but for one task, two memory modules, one seed and two tasks of the dataset (`--max_tasks 2 --max_trials 3`). It then prints what the run wrote and fails if the two result rows are not there — including a wrong `--model`, which otherwise fails once per experiment rather than once. It also probes whether the filesystem grants `flock`, which is what the results file's append lock needs, whether Wikipedia answers, whether `java` is on `PATH` for ScienceWorld, and whether ALFWorld's simulator and games are installed. Worth a submission before any 24-hour job, and after any change to the cluster, the model or the environment.

`TASK` picks the dataset (default `fever`) and `VENV` the environment, so a dataset whose simulator is not in the shared `.venv` can be checked against one of its own without disturbing a job already queued against it:

```bash
TASK=alfworld VENV=~/alfworld-test-venv sbatch slurm/smoke_test.sh
```

**Size the real jobs before submitting them**
`slurm/calibrate.sh` sits between the smoke test and a 24-hour job: one dataset, every arm, one seed, twenty tasks at the full trial budget, in a 2-hour allocation. It prints the result table, the tokens per task and any failed tasks. That is what sizes the real jobs — every experiment in a job runs concurrently against one throughput-bound server, so the wall clock is total tokens divided by what the server sustains, and `tokens_per_task x episodes / throughput` is the estimate. Twenty tasks also takes `g-memory` past its twentieth, where `merge_insights` runs.

**Attach to an already-running vLLM/Ray cluster**
`slurm/experiment.sh` doesn't start its own model server. It expects a vLLM/Ray serving job already running elsewhere on the cluster and resolves that job's head node from its Slurm job ID:
```bash
sbatch slurm/experiment.sh <ray_jobid>
```
It looks up the node running `<ray_jobid>`, resolves its IP, and points `OPENAI_API_BASE` there before running a single `tasks/run.py` experiment (sciworld / autogen / g-memory by default — edit the script to change task, memory, or model).

These scripts hardcode several specific paths and values that need updating for another cluster:

| Where | What it is | How to change it |
|---|---|---|
| `cd ~/GMemory` (all scripts) | Path to this repo's checkout, activated with a `uv`-managed `.venv` (`uv run tasks/run.py ...`) rather than the conda env from Setup below | Point at wherever you clone this repo, and switch to `conda activate GMemory` + `python tasks/run.py` if you're not using `uv` |
| `cd ~/vllm_test` (every serving script) | A separate directory/venv used only to launch `vllm serve`, kept apart from the experiment venv above so vLLM's own dependencies don't clash with this repo's | Point at your own vLLM-serving venv, or drop this `cd`/`activate` pair if you serve models a different way |
| `module load brics/nccl` | Cluster environment module providing NCCL (GPU communication library needed for tensor-parallel vLLM) | Replace with your cluster's NCCL module, or drop it if your cluster's default environment already provides NCCL |
| `YAML_CONFIG="/projects/public/brics/distributed_vllm/GPT-OSS_Hopper.yaml"` | vLLM server config (tensor-parallel/batch settings) stored in BriCS's shared project space | Replace with your own vLLM config path, or drop `--config $YAML_CONFIG` and pass the equivalent `vllm serve` flags directly |
| `HF_HOME=/projects/public/brics/hf` | Shared HuggingFace cache directory on BriCS's project space | Point at your own HF cache dir, or unset to fall back to the default `~/.cache/huggingface` |
| `MODEL_PATH=$HF_HOME/hub/models--openai--gpt-oss-120b/snapshots/<hash>/` | Resolved local snapshot path for the served model's weights inside `HF_HOME` | Update the snapshot hash to match your own cache, or pass a HF Hub model name directly instead of a local path |
| `MODEL_NAME` (`openai/gpt-oss-120b`, or `Qwen/Qwen3.6-35B-A3B` in `single_node_serve.sh`) | The model tag `vllm serve` registers and the tag `tasks/run.py` requests via the OpenAI-compatible API | Set to whichever model you're serving — each script now serves and queries one name, where `single_node_serve.sh` used to serve one and ask for another |
| `TIKTOKEN_ENCODINGS_BASE="/projects/public/brics/distributed_vllm/etc/encodings"` | Local copy of tiktoken's tokenizer encodings, used to avoid downloading them from the internet on restricted compute nodes | Point at your own local encodings cache, or drop the variable if your compute nodes have internet access |

Beyond this table, `#SBATCH --output=out/...` in every script writes logs to a `out/` directory relative to wherever you run `sbatch` from — create it first (`mkdir -p out`) or change the path.

`OPENAI_API_BASE` needs its scheme: `http://host:8000`, not `host:8000`. A base URL without one parses as a relative path with no host, so requests never reach the server; the settings loader refuses it rather than letting a job run against nothing.

## ⚙️ tasks/run.py Flags

Flags marked **(sweep)** accept multiple values (`nargs='+'`). Any flag given more than one value sweeps: `run.py` builds the full Cartesian product of experiment configs, and — if `--num_workers` > 1 and there is more than one config — runs them in parallel via a `ProcessPoolExecutor`.

| Flag | Default | Description |
|---|---|---|
| `--task` (sweep) | `alfworld` | One or more of `alfworld`, `babyai`, `fever`, `hotpotqa`, `jericho`, `pddl`, `sciworld` |
| `--mas_type` | **required** | One of `autogen`, `dylan`, `macnet` |
| `--mas_memory` (sweep) | **required** | One or more memory modules: `empty`, `voyager`, `memorybank`, `chatdev`, `generative`, `metagpt`, `g-memory`, `intrinsicmemory-pddl`, `intrinsicmemory-fever`, `intrinsicmemory-hotpotqa`, `intrinsicmemory-jericho`, `intrinsicmemory-babyai`, `intrinsicmemory-sciworld`, `intrinsicmemory-alfworld`, `intrinsicmemory-llm-structured-template`, `intrinsicmemory-notemplate` |
| `--reasoning` | `io` | Reasoning module |
| `--model` (sweep) | `gpt-3.5-turbo-0125` | LLM model name, as recognized by your `OPENAI_API_BASE` backend |
| `--max_trials` | each task's `max_steps` | Trials one episode gets. Unset, the budget comes from that task's entry in `tasks/configs.yaml` (30 for all but Jericho, which is 100); given, it overrides every task in the sweep |
| `--max_tasks` | each task's `max_tasks` | How many tasks of the dataset a run covers. Unset, from `tasks/configs.yaml` (only FEVER and HotpotQA set one, both at 200) and otherwise the whole dataset — 56 games for Jericho and 200 level-and-seed pairs for BabyAI; given, it overrides every task in the sweep. `--max_tasks 2` is what makes a smoke run short |
| `--successful_topk` | `1` | Number of successful trajectories retrieved from memory |
| `--failed_topk` | `0` | Number of failed trajectories retrieved from memory |
| `--insights_topk` | `3` | Number of insights retrieved from memory |
| `--threshold` | `0.0` | Similarity threshold for trajectory retrieval |
| `--hop` | `1` | Hop count for graph-based trajectory similarity |
| `--use_projector` | off | Enable the role projector, which tailors retrieved insights per agent. Only `g-memory` implements projection |
| `--use_validator` | off | Add a validator agent that checks the solver's action format before it is taken, re-prompting the solver on a rejection. Only `autogen` acts on it |
| `--intrinsic_cross_task` | off | Keep an `intrinsicmemory-*` module's memory across the tasks of a dataset instead of starting each task from an empty one. No effect on the other modules, which accumulate across tasks either way. It is a column in every result file, so the two arms are distinguishable |
| `--max_tokens` | `2048` | Ceiling on the tokens generated per response, sent as `max_completion_tokens`. A reasoning model spends it on its reasoning and its answer together, so a model that thinks before answering needs more than a model that does not. The Slurm scripts set it per dataset |
| `--max_tokens_ceiling` | `8192` | Largest `max_completion_tokens` a retry may climb to. A reasoning model that spends the whole budget reasoning answers nothing at all, and the same request again answers nothing again, so the budget doubles up to here instead. Set it to `--max_tokens` to hold every call to one budget |
| `--temperature` | `0.1` | Sampling temperature for any call that does not set its own. The workflows set 0 for the calls that have to parse |
| `--request_timeout` | `300.0` | Seconds one request may take. The openai client's own default is 600, which multiplied by its retries and the retry loops above it lets one action block for around 90 minutes against a server that has stopped answering |
| `--log_responses` | off | Echo every LLM response and memory-update prompt to stderr. Off because one Slurm job writes one `.out` file, from every worker at once, over the order of 100,000 requests — it is all in the per-experiment log files either way |
| `--seed` (sweep) | `42` | One or more random seeds. Each seed gets its own memory persistence directory, so concurrent seeds of one config never share a graph or vector store |
| `--num_workers` | `os.cpu_count() - 32` (min 1) | Worker processes for running the experiment sweep in parallel |
| `--resume` | off | Skip the experiments already recorded in the overall results file, rather than appending a second row for each of them. The identity columns plus `seed` are the key, so a job killed at its wall clock is continued by resubmitting with this |
| `--db_dir` | `./.db` | Where results, logs and memory persistence for this run go. Point every job of one experiment set at the same one: they append to one `overall_results.csv` under a lock on the file |
| `--overall_results_filename` | `overall_results.csv` | Name of the per-set results file, in `--db_dir` |
| `--failed_tasks_filename` | `failed_tasks.csv` | Name of the failed-task file, in each experiment's own directory |
| `--failed_experiments_filename` | `failed_experiments.csv` | Name of the failed-experiment file, in `--db_dir` |

Two settings are per-task rather than flags, in `tasks/configs.yaml`: `max_steps` (the trial budget above) and `few_shots_num`. FEVER and HotpotQA also have `max_tasks: 200`, which cuts each to its first 200 claims or questions.
BabyAI's manifest is ordered seed-major, so a `--max_tasks` prefix of it is a spread across the levels rather than many seeds of the easiest one.
Jericho's `max_steps` is 100 rather than 30, because its games score over a much longer arc than one 30-move episode covers — `data/data.md` has the measurements behind the number. `tasks/configs.yaml` also holds `embedding_model` and `embedding_device`, which is `cpu`: left to itself the embedding model takes `cuda:0`, and on a node serving vLLM every GPU is the server's.

Example sweep (2 tasks × 3 memories × 3 seeds = 18 experiments, parallelized across 8 workers):
```bash
python tasks/run.py --task fever pddl --mas_type autogen \
    --mas_memory empty g-memory intrinsicmemory-notemplate \
    --seed 11 22 33 --model <your model> --num_workers 8
```

### 📊 What a run writes

Under `--db_dir`, with per-experiment files in `<db_dir>/<model>/<task>/<mas_type>/<mas_memory>/`:

| File | One row is |
|---|---|
| `<task>-<memory>-task_results.csv` | one completed task, raw: that episode's own reward, done and trials, and the tokens that episode spent |
| `<task>-<memory>-seed_<n>-progress.csv` | one completed task, as means over the tasks scored so far. **Deleted once that experiment finishes** — it is a crash backup, and the two files below say everything it did |
| `overall_results.csv` (at `<db_dir>/`) | one finished experiment |
| `failed_tasks.csv` | one task that could not be run, with its error |
| `failed_experiments.csv` (at `<db_dir>/`) | one experiment that could not be run, with its error |

The task file is the raw material — anything the means hide, like variance or cost per task, is computable from it, and the reverse is not true:

```
model,task,mas_type,mas_memory,use_validator,intrinsic_cross_task,max_trials,task_id,reward,done,
trials,completion_tokens,prompt_tokens,intrinsic_completion_tokens,intrinsic_prompt_tokens,seed
```

`trials` is empty for an episode cut short because an agent could not act — how many turns that task needed was never established, and `0` would read as a task that took no turns. The token columns are that episode's spend, so they sum to the run total unless a task failed part-way, since a task with no row still spent what it spent.

`overall_results.csv` and the progress file share the aggregate schema:

```
model,task,mas_type,mas_memory,use_validator,intrinsic_cross_task,max_trials,mean_reward,mean_done,
mean_trials,tasks_scored,completion_tokens,prompt_tokens,intrinsic_completion_tokens,
intrinsic_prompt_tokens,seed
```

`tasks_scored` is how many episodes the means are over, so a progress row states its own progress. A task whose episode could not run goes in `failed_tasks.csv` and is left out of the means rather than scored as zero. `intrinsic_*_tokens` are the share spent by the memory module's own LLM calls, and are non-zero only for the `intrinsicmemory-*` modules.

Every file opens with the same identity columns and ends with `seed`. A run refuses to append to a file whose header is not its own schema, and says which is which — so an old `--db_dir` fails at the start rather than producing a CSV nobody can read. Results written before September 2026 use a different, per-file column order and have no header at all.

`<db_dir>/<model>/` is the model name made into one path component: `openai/gpt-oss-120b` becomes `openai--gpt-oss-120b`.

## 👋 Introduction
This repo is the official implementation of [***G-Memory: Tracing Hierarchical Memory for Multi-Agent Systems***](https://arxiv.org/abs/2506.07398).

Our method, G-Memory, empowers multi-agent systems with a hierarchical memory architecture that continuously evolves through interaction. Inspired by organizational memory theory, G-Memory captures generalizable insights and agent-specific collaboration trajectories across tasks using a structured graph-based design. When a new task arrives, it retrieves relevant past experiences and distilled knowledge to inform agent behavior and coordination. As agents complete tasks, G-Memory updates its memory hierarchy with new interactions, enabling teams to adapt and improve over time.

![alt text](assets/method.png)

## 🌎 Setup

With `uv` (what the Slurm scripts and the Dockerfile use):
```
uv sync
uv run tasks/run.py ...
```

With conda:
```
conda create -n GMemory python=3.12
conda activate GMemory
pip install -r requirements.txt
```

`requirements.txt` and `pyproject.toml` carry the same pins by hand; neither is generated from the other yet.

ALFWorld's simulator is in neither of them: TextWorld and its PDDL planner have no aarch64 wheels, so listing `alfworld` would break `uv sync` on Isambard and on every other Arm machine. It installs separately, and on aarch64 needs two extra steps — the ALFWorld section of `data/data.md` has both.

### ✅ Tests and lint

The test suite runs offline — it stubs the simulators, the vector store and the LLM — in a dev environment without CUDA:
```
uv run --only-group dev pytest
uv run --only-group dev ruff check .
```
Plain `uv run pytest` installs the full dependency set, which needs a CUDA platform. Tests marked `network` need a live LLM endpoint and are deselected by default; run them with `-m network`.

## 🚀 Quick Start

### 🌳 Environments
Please download the ALFWorld, PDDL, FEVER datasets and place it in the data folder.
- 🏠 [ALFWorld](https://github.com/alfworld/alfworld)
- 🐹 [PDDL](https://github.com/hkust-nlp/AgentBoard)
- 🌡️ [FEVER](https://github.com/awslabs/fever)
- 🔎 [HotpotQA](https://hotpotqa.github.io/)
- 📖 [Jericho](https://github.com/microsoft/jericho) (game roms: [z-machine-games](https://github.com/BYU-PCCL/z-machine-games))
- 🧱 [BabyAI](https://github.com/Farama-Foundation/Minigrid)

The file structure should be organized as follows:
```
data
└── alfworld
    └── alfworld_tasks_suffix.json
    └── json_2.1.1
        └── valid_unseen/.../game.tw-pddl   # 4027 games, downloaded separately
└── pddl
    └── test.jsonl
└── fever
    └── fever_dev.jsonl
└── hotpotqa
    └── hotpotqa_dev.jsonl
└── jericho
    └── jericho_games.jsonl
    └── roms
        └── 905.z5 ... ztuu.z5      # the 56 game files, downloaded separately
└── babyai
    └── babyai_levels.jsonl
└── sciworld
    └── test.jsonl
```

Each dataset is parsed only when a task asks for it, so a FEVER-only run does not need the other six manifests present.

FEVER and HotpotQA both drive live Wikipedia through their `Search` and `Lookup` actions, so those two need outbound network from wherever the run happens; the other five are self-contained once their simulator is installed.

The manifests are checked in, but the data they name is not. Jericho's lists 56 games and needs the rom files themselves under `data/jericho/roms/`; ALFWorld's names a `.tw-pddl` game file per task and needs those under `data/alfworld/json_2.1.1/`. `data/data.md` has both commands. BabyAI needs no download at all — `minigrid` generates each gridworld from the level name and seed in the manifest.

### 🔑 Add API keys in template.env and change its name to .env
```
OPENAI_API_BASE = "" # the BASE_URL of OpenAI LLM backend
OPENAI_API_KEY = ""  # for OpenAI LLM backend
```

### 🔎 Choices Overview
- Available memories: ***Empty, ChatDev, MetaGPT, Voyager, Generative, MemoryBank, G-Memory***
- Added by this fork: ***nine intrinsic memory variants*** — `intrinsicmemory-notemplate`, `-pddl`, `-fever`, `-hotpotqa`, `-jericho`, `-babyai`, `-sciworld`, `-alfworld` and `-llm-structured-template`. Each keeps one agent-authored memory that an LLM rewrites as the episode goes; they differ only in the template their system prompt asks for, which is what the experiments compare.
- Available MAS: ***AutoGen, DyLAN, MacNet***
- `--mas_type autogen` also takes `--use_validator`, which adds a third agent reviewing the solver's action format.

### ▶️ How to Run
- Option 1: Run with Shell Script. Simply execute the following script:
    ```
    ./run_mas.sh
    ```
- Option 2: Run with Python Command. You can also launch specific tasks via command-line:
    ```
    python tasks/run.py --task alfworld --reasoning io --mas_memory g-memory --max_trials 30 --mas_type autogen --model <your model here>
    python tasks/run.py --task pddl --reasoning io --mas_memory g-memory --max_trials 30 --mas_type autogen --model <your model here>
    python tasks/run.py --task fever --reasoning io --mas_memory g-memory --max_trials 15 --mas_type autogen --model <your model here>
    ```

## 🫡 Citation
If you find this repository helpful, a citation to our paper would be greatly appreciated:
```
@article{zhang2025g-memory,
  title={G-Memory: Tracing Hierarchical Memory for Multi-Agent Systems},
  author={Zhang, Guibin and Fu, Muxin and Wan, Guancheng and Yu, Miao and Wang, Kun and Yan, Shuicheng},
  journal={arXiv preprint arXiv:2506.07398},
  year={2025}
}
```

## 🙏 Acknowledgement
- We sincerely thank [ExpeL](https://github.com/LeapLabTHU/ExpeL) for providing their prompt designs.
- We also extend our heartfelt thanks to [AgentSquare](https://github.com/tsinghua-fib-lab/AgentSquare) for their dataset environments and baseline implementations.

