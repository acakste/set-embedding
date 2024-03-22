# Platform parameter not needed if not on Mac M1
FROM --platform=linux/amd64 python:3.10-buster AS base

# Consistent text encoding
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

# Python, don't write bytecode!
ENV PYTHONDONTWRITEBYTECODE 1

# Install Poetry
# https://python-poetry.org/docs/#ci-recommendations
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VIRTUALENVS_IN_PROJECT 1

RUN python3 -m venv $POETRY_HOME
RUN $POETRY_HOME/bin/pip install poetry==1.3.2
RUN $POETRY_HOME/bin/poetry --version
RUN pip3 install torch==2.2.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
ENV PATH=$POETRY_HOME/bin:${PATH}

RUN mkdir /python_env /app
WORKDIR /python_env

COPY src/pyproject.toml ./

# Configure git to access private repositories
ARG GITHUB_TOKEN
RUN git config --global url."https://${GITHUB_TOKEN}@github.com".insteadOf "ssh://git@github.com"

RUN poetry lock
RUN poetry install --no-root

# Remove a few GB of pip and poetry cache
RUN rm -rf /root/.cache

# Remove git configuration
RUN git config --global --remove-section url."https://${GITHUB_TOKEN}@github.com"

# Making sure that the virtual env is the first python to be found in PATH
ENV PATH=/python_env/.venv/bin:${PATH}

WORKDIR /app

FROM base AS app
ARG GIT_SHORT_SHA
ENV GIT_SHA=${GIT_SHORT_SHA}

COPY --from=base /python_env /python_env
# making sure that the virtual env is the first python to be found in PATH
ENV PATH=/python_env/.venv/bin:${PATH}

# Copy this code to the image
COPY src /app/src
COPY secrets /app/secrets

ENV PYTHONPATH /app

#ENTRYPOINT ["python", "src/training/data/dataloader.py"]
