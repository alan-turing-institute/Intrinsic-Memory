# SWE-bench images on Isambard

SWE-bench grades a patch by running a repository's own tests inside a
per-instance image. The published images are `linux/x86_64` only — 4,503 of them
under the `swebench` Docker Hub namespace, none for arm64 — and Isambard-AI is
aarch64, with no qemu handler registered:

```
$ ls /proc/sys/fs/binfmt_misc/
register  status
$ podman-hpc run --rm --platform linux/amd64 ubuntu:jammy uname -m
runc create failed … Exec format error: '/sbin/ldconfig'
```

So every instance image is built natively here. `swebench_arm64.py` rewrites a
task Dockerfile for aarch64, `build_swebench_image.sh` builds and migrates it,
and `swebench_build.sbatch` runs that as a job.

## Build it as a job, not on a login node

`loginctl enable-linger` is denied for user accounts, so logind kills every user
process when the last session closes — a `setsid --fork` build and a tmux
session were both killed mid-build. Compute nodes have outbound network and
`podman-hpc`, and `podman-hpc migrate` writes to `$SCRATCH`, so a batch job can
do the whole thing and outlive the session that submitted it.

```bash
cd ~/swebench-arm64
sbatch containers/swebench_build.sbatch django__django-16485
```

A login-node build works only while you stay logged in, which is fine for
watching one build and useless for a queue of them.

## One-time setup

```bash
mkdir -p ~/swebench-arm64 && cd ~/swebench-arm64
git clone --depth 1 https://github.com/SWE-bench/swe-bench-tasks   # 98 MB, 2,519 tasks
uv venv --python 3.12 .venv                                        # the rewriter needs 3.10+
cp <repo>/containers/{swebench_arm64.py,build_swebench_image.sh,swebench_build.sbatch} .
```

The login node's system python is 3.6, which is why the venv is not optional.

## What the rewriter changes

SWE-bench 5.x builds images from a *task repo* — one directory per instance
holding a `Dockerfile`, an `eval.sh`, the gold patch and the test patch.

| In the published Dockerfile | Rewritten to |
|---|---|
| `FROM --platform=linux/amd64 ubuntu:jammy` | `--platform=linux/arm64` |
| `Miniconda3-py311_23.11.0-2-Linux-x86_64.sh` | `…-Linux-aarch64.sh` |
| `- python=3.11.10=he870216_0` | `- python=3.11.10` |
| `RUN <<EOF_…` with a shebang body | `COPY <tag>.sh` + `RUN bash /tmp/<tag>.sh` |

The third row is the one that matters: the embedded `environment.yml` pins a
conda *build string* per package and those hashes exist only for `linux-64`.
Versions are kept, build strings dropped, and six packages with no aarch64 build
under their linux-64 names are dropped outright — `_libgcc_mutex`,
`_openmp_mutex`, `ld_impl_linux-64`, `libgcc-ng`, `libgomp`, `libstdcxx-ng` —
since conda pulls the aarch64 equivalents in as dependencies. The `pip:` section
is untouched: its pins are `==` and PyPI resolves them per platform.

The fourth row is a buildah limitation, not an architecture one. A heredoc `RUN`
whose body starts with a shebang is mounted as a file rather than piped to the
shell, and runc here refuses the mount:

```
invalid mount … /dev/pipes/buildahheredoc…: bind mounts cannot have any
filesystem-specific options applied
```

`--isolation=chroot` exits 127 instead, and a `SHELL ["/bin/bash", "-c"]`
directive is ignored for heredocs, so `set -euo pipefail` still reaches
`/bin/sh` and dies on `Illegal option -o pipefail`. Heredocs with no shebang
build fine and are left alone.

## Checking an image is a faithful instance

The aarch64 solve gives the same package versions, not the same builds, so
every instance has to be shown to reproduce the benchmark's verdicts: its
FAIL_TO_PASS tests must fail before the gold patch and pass after it, with
PASS_TO_PASS passing throughout.

```bash
TASK=~/swebench-arm64/swe-bench-tasks/tasks/django__django-16485
IMAGE=localhost/sweb.eval.arm64.django__django-16485:latest

# before: the target test must fail
podman-hpc run --rm -v $TASK:/task:ro $IMAGE bash /task/eval.sh

# after: it must pass
podman-hpc run --rm -v $TASK:/task:ro $IMAGE \
    bash -c 'cd /testbed && git apply /task/gold.patch && bash /task/eval.sh'
```

`eval.sh` applies the test patch itself and brackets the test output with
`>>>>> Start Test Output`.

**Read the verdict against `tests.json`, never off the summary line.** `eval.sh`
runs whole test files, which can hold tests that are not graded: on
`psf__requests-2931` both runs report 81 errors, all of them tests that want a
live httpbin, and none of them in FAIL_TO_PASS or PASS_TO_PASS. The patched run
says `86 passed … 81 errors` and the instance is nonetheless faithful — all 84
PASS_TO_PASS pass and the one FAIL_TO_PASS test flips.

On `django__django-16485` the first run ends

```
ValueError: valid range for prec is [1, MAX_PREC]
Ran 10 tests … FAILED (errors=1)
```

which is the issue the instance is about, and the second run ends `Ran 10 tests
… OK`. That image took 6m23s to build and is 2.56 GB.

## Limits worth knowing

- **Not every instance will build.** Of five repositories tried, four built:
  django, sympy, pytest and requests, each in about four minutes and 2.3-2.6 GB.
  `pydata__xarray-7229` cannot: its environment pins `cdms2`, `cdtime` and
  `libcdms`, the CDAT stack, which has no linux-aarch64 build in any channel.
  That is the shape of the failure to expect — a dependency that was never built
  for this architecture — and the answer is to pick another instance rather than
  fight it. django alone is 231 of the 500 Verified instances.
- **A login node's local image store is erased at the end of a session.** The
  migrated squashfs on `$SCRATCH` is what survives, and what compute nodes read;
  `podman-hpc images` shows `R/O = true` for those.
- **The login node rate-limits reconnections.** A poll loop that opens an ssh
  connection every 30 seconds gets `Connection reset by peer`. Reuse one
  connection (`ControlMaster`/`ControlPersist`) and poll minutes apart.
