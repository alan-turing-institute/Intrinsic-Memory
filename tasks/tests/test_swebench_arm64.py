"""A SWE-bench task Dockerfile, rewritten for aarch64.

The published task Dockerfiles are x86_64 in three places, and Isambard has no
qemu handler registered, so an amd64 image cannot run there at all. They also
drive their setup through heredoc `RUN` blocks with a shebang, which buildah
mounts as a file and runc then refuses.
"""

from containers.swebench_arm64 import to_aarch64

TASK_DOCKERFILE = """
FROM --platform=linux/amd64 ubuntu:jammy

RUN wget 'https://repo.anaconda.com/miniconda/Miniconda3-py311_23.11.0-2-Linux-x86_64.sh' -O miniconda.sh \\
    && bash miniconda.sh -b -p /opt/miniconda3

RUN <<EOF_a88cf31974aa
#!/bin/bash
set -euxo pipefail
cat <<'EOF_c1418fe7bce4' > /root/environment.yml
name: testbed
channels:
  - defaults
  - conda-forge
dependencies:
  - _libgcc_mutex=0.1=main
  - ld_impl_linux-64=2.40=h12ee557_0
  - python=3.11.10=he870216_0
  - sqlite=3.45.3=h5eee18b_0
  - pip:
      - asgiref==3.8.1
      - numpy==2.1.2
EOF_c1418fe7bce4
conda env create -f /root/environment.yml
EOF_a88cf31974aa

RUN <<EOF_nosheb
echo no shebang here
EOF_nosheb

WORKDIR /testbed/
"""


def test_the_base_image_is_built_for_arm():
    rewritten = to_aarch64(TASK_DOCKERFILE).dockerfile

    assert '--platform=linux/arm64' in rewritten
    assert 'linux/amd64' not in rewritten


def test_the_miniconda_installer_is_the_aarch64_one():
    rewritten = to_aarch64(TASK_DOCKERFILE).dockerfile

    assert 'Miniconda3-py311_23.11.0-2-Linux-aarch64.sh' in rewritten
    assert 'x86_64.sh' not in rewritten


def test_conda_pins_keep_their_version_and_lose_their_build_string():
    scripts = to_aarch64(TASK_DOCKERFILE).scripts
    environment = '\n'.join(scripts.values())

    assert '  - python=3.11.10\n' in environment, 'the version pin must survive'
    assert 'he870216_0' not in environment, 'the build string is linux-64 only'
    assert '  - sqlite=3.45.3\n' in environment


def test_conda_packages_with_no_aarch64_build_are_dropped():
    everything = to_aarch64(TASK_DOCKERFILE)
    everything = everything.dockerfile + '\n'.join(everything.scripts.values())

    assert 'ld_impl_linux-64' not in everything
    assert '_libgcc_mutex' not in everything


def test_the_pip_section_is_left_alone():
    """Its pins are `==` and PyPI resolves them per platform."""
    environment = '\n'.join(to_aarch64(TASK_DOCKERFILE).scripts.values())

    assert '      - asgiref==3.8.1' in environment
    assert '      - numpy==2.1.2' in environment


def test_a_heredoc_run_with_a_shebang_becomes_a_script_the_build_copies():
    rewritten = to_aarch64(TASK_DOCKERFILE)

    assert 'a88cf31974aa.sh' in rewritten.scripts
    assert 'COPY a88cf31974aa.sh /tmp/a88cf31974aa.sh' in rewritten.dockerfile
    assert 'RUN bash /tmp/a88cf31974aa.sh' in rewritten.dockerfile
    assert 'RUN <<EOF_a88cf31974aa' not in rewritten.dockerfile


def test_the_extracted_script_is_the_heredoc_body_and_stops_at_its_terminator():
    script = to_aarch64(TASK_DOCKERFILE).scripts['a88cf31974aa.sh']

    assert script.startswith('#!/bin/bash\nset -euxo pipefail\n')
    assert script.rstrip().endswith('conda env create -f /root/environment.yml')
    assert 'EOF_a88cf31974aa' not in script


def test_a_nested_heredoc_survives_inside_the_script():
    script = to_aarch64(TASK_DOCKERFILE).scripts['a88cf31974aa.sh']

    assert "cat <<'EOF_c1418fe7bce4' > /root/environment.yml" in script
    assert script.count('EOF_c1418fe7bce4') == 2


def test_a_heredoc_run_with_no_shebang_is_left_where_it_is():
    """Those buildah pipes to the shell, which works."""
    rewritten = to_aarch64(TASK_DOCKERFILE)

    assert 'RUN <<EOF_nosheb' in rewritten.dockerfile
    assert 'nosheb.sh' not in rewritten.scripts


def test_everything_else_is_untouched():
    rewritten = to_aarch64(TASK_DOCKERFILE).dockerfile

    assert 'WORKDIR /testbed/' in rewritten
    assert 'bash miniconda.sh -b -p /opt/miniconda3' in rewritten
