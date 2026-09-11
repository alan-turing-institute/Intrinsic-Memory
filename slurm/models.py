"""The models a sweep can be pointed at, and what each needs to be served.

A model is a whole serving configuration, not just a name: the two here need
different vLLM builds, different weights locations and different flags, and
getting one of those wrong costs a whole allocation before anything says so.

Add a model by adding a `Model` here, then bring it up once with
`slurm/generate_slurm.py smoke --model <slug>` before pointing a sweep at it.
"""
from dataclasses import dataclass, field

from config import PROJECT_DIR

SHARED_HF_HOME = "/projects/public/brics/hf"
GPT_OSS_SNAPSHOT = "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"


@dataclass(frozen=True)
class Model:
    """One model's serving configuration.

    `slug` names it on the command line and in the generated scripts' directory.
    `name` is what `vllm serve` registers and what `tasks/run.py --model` asks
    for; the two have to agree, because GPTChat sizes its token budget from the
    `max_model_len` on the card under that exact id.

    `served` is what `vllm serve` is given - a local snapshot path or a Hub id.
    `vllm_dir` holds the `.venv` that serves; it is not the experiment venv.

    `hf_home` is exported for the whole job, so it has to be writable: the
    experiment processes fetch the retriever's embedding model through it, long
    after `vllm serve` has read the weights `served` names.

    The size fields are this model's defaults, not fixed values: what a
    checkpoint and a node can carry differ per model, so they cannot be one
    number for all of them, and `generate_slurm.py` takes a flag for each.

    The three generation fields are what a run is told to ask for, rather than
    what the server is started with. Left None, `tasks/run.py` uses its own
    defaults, which are sized for a model that answers without thinking first.
    """

    slug: str
    name: str
    served: str
    vllm_dir: str
    hf_home: str
    max_model_len: int
    max_num_batched_tokens: int
    max_num_seqs: int
    yaml_config: str = ""
    reasoning_parser: str = ""
    tiktoken_encodings: str = ""
    extra_serve_flags: tuple[str, ...] = ()
    extra_env: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    max_tokens: int | None = None
    max_tokens_ceiling: int | None = None
    thinking_token_budget: int | None = None
    notes: str = ""


GPT_OSS_120B = Model(
    slug="gpt-oss-120b",
    name="openai/gpt-oss-120b",
    served=f"{SHARED_HF_HOME}/hub/models--openai--gpt-oss-120b/snapshots/{GPT_OSS_SNAPSHOT}/",
    vllm_dir="~/vllm_test",
    hf_home=f"{PROJECT_DIR}/hf",
    # These three override the shared GPT-OSS_Hopper.yaml, which sets them to
    # 8192, 10240 and off. On this node the KV cache holds 3,730,336 tokens, 227
    # of them at max_model_len, so a queue deeper than that is a number the
    # server cannot honour.
    max_model_len=16384,
    max_num_batched_tokens=32768,
    max_num_seqs=256,
    yaml_config="/projects/public/brics/distributed_vllm/GPT-OSS_Hopper.yaml",
    tiktoken_encodings="/projects/public/brics/distributed_vllm/etc/encodings",
    extra_serve_flags=("--enable-prefix-caching",),
    notes="vLLM 0.15.1. Needs no reasoning parser: vLLM splits this model's reasoning off unasked.",
)

QWEN36_35B_A3B = Model(
    slug="qwen3.6-35b-a3b",
    name="Qwen/Qwen3.6-35B-A3B",
    served="Qwen/Qwen3.6-35B-A3B",
    vllm_dir="~/vllm_qwen36",
    # Not the shared cache and not home: one bf16 checkpoint of this size is
    # 72 GB, and home is quota'd well below that.
    hf_home=f"{PROJECT_DIR}/hf",
    max_model_len=65536,
    max_num_batched_tokens=8192,
    max_num_seqs=512,
    # Without a parser this model puts its whole chain of thought in `content`,
    # ending in a bare `</think>`, and process_action is handed the reasoning
    # rather than the answer.
    reasoning_parser="qwen3",
    extra_serve_flags=(
        "--enable-prefix-caching",
        # A multimodal checkpoint: vLLM sizes its profiling run for the largest
        # image and video batch the model accepts, and the experiment sends text.
        """--limit-mm-per-prompt '{"image": 0, "video": 0}'""",
        # What makes thinking_token_budget on a request do anything: the budget is
        # enforced by injecting this end delimiter, and vLLM has no default pair.
        """--reasoning-config '{"reasoning_start_str": "<think>", """
        """"reasoning_end_str": "</think>"}'""",
    ),
    # vLLM's default top-k/top-p sampler is FlashInfer's, which JIT-compiles its
    # kernels on first use and so needs nvcc. Compute nodes have neither nvcc on
    # PATH nor /usr/local/cuda, and the build fails inside the memory-profiling
    # run rather than at startup. The PyTorch-native sampler needs no toolchain.
    extra_env=(("VLLM_USE_FLASHINFER_SAMPLER", "0"),),
    # Measured: left to think freely this model spent every one of 96,993
    # completion tokens per task reasoning, and answered nothing at all. A
    # thinking budget bounds that, and the rest of max_tokens is what it has to
    # answer in once the budget closes its reasoning block.
    max_tokens=8192,
    max_tokens_ceiling=32768,
    thinking_token_budget=1024,
    notes=(
        "vLLM 0.28.0 built for CUDA 12.9, from https://wheels.vllm.ai/0.28.0/cu129 with torch "
        "from https://download.pytorch.org/whl/cu129 - the PyPI wheel pulls torch cu130, and the "
        "GH200 driver 565.57.01 tops out at CUDA 12.7."
    ),
)

MODELS = {model.slug: model for model in (GPT_OSS_120B, QWEN36_35B_A3B)}
DEFAULT_MODEL = GPT_OSS_120B.slug
