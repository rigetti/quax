# Copyright 2026 Rigetti & Co, LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM ubuntu:26.04

USER root

RUN apt-get update -qq \
    && apt-get install -y -qq \
    # Basics
    curl \
    wget \
    make \
    emacs \
    less \
    unzip \
    build-essential \
    software-properties-common \
    libpq-dev \
    git \
    bash-completion \
    openssh-client \
    sudo \
    nodejs \
    npm \
    pandoc \
    && rm -rf /var/lib/apt/lists/*

# Python 3.12 from the deadsnakes PPA: 26.04's own Python is newer. Link python and python3 to it in
# /usr/local/bin, which comes first on PATH, leaving /usr/bin/python3 to the system tools (eg, apt) that need it.
RUN add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update -qq \
    && apt-get install -y -qq \
    python3.12-dev \
    python3.12-venv \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python3

# Chrome, for plotly's image export through kaleido. Ubuntu's chromium-browser is a snap, which cannot run in a
# container; apt installs Chrome's own package with its dependencies.
RUN wget -q https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb \
    && apt-get update -qq \
    && apt-get install -y -qq ./google-chrome-stable_current_amd64.deb \
    && rm -rf google-chrome-stable_current_amd64.deb /var/lib/apt/lists/*

# Create the docker user 'rigetti'
ARG UID=1000
ARG GID=1000
RUN userdel -r ubuntu
RUN addgroup --gid ${GID} rigetti \
    && adduser \
    --disabled-password \
    --gecos rigetti \
    --gid ${GID} \
    --uid ${UID} \
    --home /home/rigetti/ \
    rigetti \
    && usermod -aG sudo rigetti \
    && echo "rigetti ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/rigetti \
    && passwd -d rigetti

# Install qcs-cli, aws-cli
RUN wget https://qcs-cli-golang.s3-us-west-2.amazonaws.com/latest/linux/qcs \
    && chmod +x qcs && mv qcs /usr/local/bin/qcs \
    && curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip -q awscliv2.zip \
    && ./aws/install > /dev/null \
    && rm -r aws awscliv2.zip

# Create our project directory and switch to it
RUN mkdir -p /home/rigetti/quax/src && chown -R rigetti:rigetti /home/rigetti/

USER rigetti

# Created here so the devcontainer's named volumes for them (see
# .devcontainer/docker-compose.extend.yml) are owned by rigetti
RUN mkdir -p /home/rigetti/.vscode-server/bin /home/rigetti/.vscode-server/extensions

WORKDIR /home/rigetti/quax/

# Install poetry and rust
RUN curl -sSL https://install.python-poetry.org | python3 - \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y >/dev/null 2>&1
ENV PATH="${PATH}:/home/rigetti/.local/bin:/home/rigetti/.cargo/bin"

# Install the package
COPY --chown=rigetti:rigetti pyproject.toml poetry.lock* README.md* ./
COPY --chown=rigetti:rigetti --chmod=775 src/quax/__init__.py ./src/quax/
RUN poetry install \
    --no-interaction \
    --no-ansi \
    --no-cache;
        
# Set up and activate the venv
RUN mkdir ${HOME}/.venv/ && ln -s $(poetry env info -p) /home/rigetti/.venv/quax
ENV VIRTUAL_ENV=/home/rigetti/.venv/quax
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN poetry run python -m ipykernel install --user --name=quax

# Install Claude Code
RUN curl -fsSL https://claude.ai/install.sh | bash

# Keep Claude Code's config and state (login, session history) in the project directory, which persists: the rest of
# the container is reset whenever the devcontainer is rebuilt
ENV CLAUDE_CONFIG_DIR=/home/rigetti/quax/.claude-home

# Default to 64-bit JAX, and the highest matmul precision
ENV JAX_ENABLE_X64=1
ENV JAX_DEFAULT_MATMUL_PRECISION=highest
ENV SHELL=/bin/bash
