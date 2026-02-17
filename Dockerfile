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

# FROM nvidia/cuda:12.6.3-devel-ubuntu24.04
FROM ubuntu:24.04

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
    # python
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    chromium-browser \
    pandoc \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf $(which python3.12) $(which python || echo '/usr/local/bin/python') \
    && ln -sf $(which python3.12) $(which python3 || echo '/usr/local/bin/python3')

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
    --no-cache 

# Set up and activate the venv
RUN mkdir ${HOME}/.venv/ && ln -s $(poetry env info -p) /home/rigetti/.venv/quax
ENV VIRTUAL_ENV=/home/rigetti/.venv/quax
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN poetry run python -m ipykernel install --user --name=quax

# Default to 64-bit JAX
ENV JAX_ENABLE_X64=1
ENV SHELL=/bin/bash
