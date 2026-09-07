"""Rewrite a SWE-bench task Dockerfile so that it builds on aarch64.

    python containers/swebench_arm64.py <task_dir> --out-dir <build context>

The task Dockerfiles published in SWE-bench/swe-bench-tasks are x86_64 in three
places - the base image platform, the Miniconda installer, and the conda build
strings pinned in the embedded environment.yml - and they drive their setup
through heredoc `RUN` blocks, which buildah on Isambard cannot run.
"""

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Packages whose linux-64 name has no linux-aarch64 build. Dropping them leaves
# the compilers and runtimes to conda's own solve, which picks the aarch64 ones.
X86_ONLY_CONDA_PACKAGES = frozenset({
    '_libgcc_mutex',
    '_openmp_mutex',
    'ld_impl_linux-64',
    'libgcc-ng',
    'libgomp',
    'libstdcxx-ng',
})

_PLATFORM = re.compile(r'--platform=linux/amd64')
_MINICONDA = re.compile(r'(Miniconda3-[\w.\-]+-Linux-)x86_64(\.sh)')
_CONDA_DEP = re.compile(r'^(\s*)-\s([A-Za-z0-9_.\-]+)=([^=\s]+)=([^=\s]+)\s*$')
_HEREDOC_RUN = re.compile(
    r'^RUN <<(?P<tag>[A-Za-z_][\w]*)\n(?P<body>.*?)\n(?P=tag)\s*?$',
    re.DOTALL | re.MULTILINE,
)
_SHEBANG = re.compile(r'^#!')


@dataclass
class Rewritten:
    """An aarch64 Dockerfile, and the scripts its build context now needs."""

    dockerfile: str
    scripts: dict[str, str] = field(default_factory=dict)

    def write(self, out_dir: Path) -> Path:
        """Write the Dockerfile and its scripts into a build context."""
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, script in self.scripts.items():
            (out_dir / name).write_text(script)

        path = out_dir / 'Dockerfile.aarch64'
        path.write_text(self.dockerfile)

        return path


def _rewrite_conda_dependency(line: str) -> str | None:
    """One `  - name=version=build` line with its build string dropped."""
    match = _CONDA_DEP.match(line)
    if match is None:
        return line

    indent, package, version, _build = match.groups()
    if package in X86_ONLY_CONDA_PACKAGES:
        return None

    return f'{indent}- {package}={version}'


def _script_name(tag: str) -> str:
    """`EOF_a88cf31974aa` names `a88cf31974aa.sh`, so a task's scripts are stable."""
    return f'{tag.split("_", 1)[-1]}.sh'


def to_aarch64(dockerfile: str) -> Rewritten:
    """The same Dockerfile, buildable on aarch64, and any scripts it now needs."""
    dockerfile = _PLATFORM.sub('--platform=linux/arm64', dockerfile)
    dockerfile = _MINICONDA.sub(r'\1aarch64\2', dockerfile)

    lines = (_rewrite_conda_dependency(line) for line in dockerfile.splitlines())
    dockerfile = '\n'.join(line for line in lines if line is not None) + '\n'

    scripts: dict[str, str] = {}

    def extract(match: re.Match) -> str:
        body = match.group('body')
        if not _SHEBANG.match(body):
            return match.group(0)

        name = _script_name(match.group('tag'))
        scripts[name] = body + '\n'

        return f'COPY {name} /tmp/{name}\nRUN bash /tmp/{name}'

    return Rewritten(_HEREDOC_RUN.sub(extract, dockerfile), scripts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('task_dir', type=Path, help='a tasks/<instance_id> directory')
    parser.add_argument('--out-dir', type=Path, default=None,
                        help='build context to write Dockerfile.aarch64 and its scripts into')
    args = parser.parse_args(argv)

    source = args.task_dir / 'Dockerfile'
    if not source.is_file():
        parser.error(f'{source} does not exist')

    rewritten = to_aarch64(source.read_text())
    if args.out_dir is None:
        sys.stdout.write(rewritten.dockerfile)
    else:
        print(rewritten.write(args.out_dir))

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
