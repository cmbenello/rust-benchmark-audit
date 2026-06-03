# IF you change the base image, you need to rebuild all images (run with --force_rebuild)
_DOCKERFILE_BASE = r"""
FROM --platform={platform} ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

ENV RUSTUP_DIST_SERVER="https://static.rust-lang.org"
ENV RUSTUP_UPDATE_ROOT="https://static.rust-lang.org/rustup"

# Use AWS regional Ubuntu mirror for much faster apt from EC2 in us-east-2
RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|http://us-east-2.ec2.archive.ubuntu.com/ubuntu/|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu/|http://us-east-2.ec2.archive.ubuntu.com/ubuntu/|g' /etc/apt/sources.list

RUN for i in 1 2 3 4 5; do \
      apt update && apt install -y --fix-missing \
        wget git build-essential libffi-dev libtiff-dev jq curl locales locales-all tzdata \
      && break; \
      echo "apt retry $i failed, sleeping"; sleep 15; \
    done && rm -rf /var/lib/apt/lists/*

RUN for i in 1 2 3 4 5; do \
      apt-get update && apt-get install -y --fix-missing libssl-dev && break; \
      echo "apt retry $i failed, sleeping"; sleep 15; \
    done

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH=/root/.cargo/bin:$PATH
RUN git clone https://github.com/riverLaugh/LTS.git
RUN cd LTS && git checkout main && cargo install --path . 

RUN git clone --bare https://github.com/rust-lang/crates.io-index-archive.git
ENV CARGO_REGISTRY_GIT_DIR=/crates.io-index-archive.git

RUN adduser --disabled-password --gecos 'dog' nonroot


RUN for i in 1 2 3 4 5; do \
      apt-get update && apt-get install -y --fix-missing \
        pkg-config python3-pip cmake protobuf-compiler \
      && break; \
      echo "apt retry $i failed, sleeping"; sleep 15; \
    done

"""

_DOCKERFILE_ENV = r"""FROM --platform={platform} rustb.base.{arch}:latest

RUN mkdir -p ~/.cargo && \
    cat <<EOF > ~/.cargo/config
[net]
git-fetch-with-cli = true
EOF

COPY ./setup_env.sh /root/
RUN chmod +x /root/setup_env.sh
RUN /bin/bash -c "source ~/.bashrc && /root/setup_env.sh"
WORKDIR /testbed/

# Automatically activate the testbed environment
# RUN echo "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed" > /root/.bashrc
"""

_DOCKERFILE_INSTANCE = r"""FROM --platform={platform} {env_image_name}
RUN cd ../LTS && git pull && git checkout main && cargo install --path . 
RUN cd ../testbed/
COPY ./setup_repo.sh /root/
RUN /bin/bash /root/setup_repo.sh

WORKDIR /testbed/
"""

_DOCKERFILE_ENV_asterinas = r"""
FROM --platform={platform} asterinas/asterinas:{tag}
# ENV RUSTUP_DIST_SERVER=https://mirrors.tuna.tsinghua.edu.cn/rustup
# ENV RUSTUP_UPDATE_ROOT=https://mirrors.tuna.tsinghua.edu.cn/rustup/rustup

ENV RUSTUP_DIST_SERVER=https://mirror.sjtu.edu.cn/rust-static
ENV RUSTUP_UPDATE_ROOT=https://mirror.sjtu.edu.cn/rust-static/rustup

RUN mkdir -p ~/.cargo && \
    cat <<EOF > ~/.cargo/config
[source.crates-io]
replace-with = "sjtu"

[source.tuna]
registry = "https://mirrors.tuna.tsinghua.edu.cn/crates.io-index"


[source.ustc]
registry = "git://mirrors.ustc.edu.cn/crates.io-index"


[source.sjtu]
registry = "https://mirrors.sjtug.sjtu.edu.cn/git/crates.io-index"

# rustcc社区
[source.rustcc]
registry = "git://crates.rustcc.cn/crates.io-index"
EOF

COPY ./setup_env.sh /root/
RUN chmod +x /root/setup_env.sh
RUN /bin/bash -c "source ~/.bashrc && /root/setup_env.sh"
"""



def get_dockerfile_base(platform, arch):
    if arch == "arm64":
        conda_arch = "aarch64"
    else:
        conda_arch = arch
    return _DOCKERFILE_BASE.format(platform=platform, conda_arch=conda_arch)


def get_dockerfile_env(platform, arch):
    return _DOCKERFILE_ENV.format(platform=platform, arch=arch)


def get_dockerfile_instance(platform, env_image_name):
    return _DOCKERFILE_INSTANCE.format(platform=platform, env_image_name=env_image_name)


def get_dockerfile_env_asterinas(platform, tag):
    return _DOCKERFILE_ENV_asterinas.format(platform=platform, tag=tag)