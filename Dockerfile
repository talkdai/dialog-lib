FROM python:3.11-slim
LABEL org.opencontainers.image.source https://github.com/talkdai/dialog
LABEL org.opencontainers.image.licenses MIT

RUN useradd --user-group --system --create-home --no-log-init talkd
RUN export PATH="/home/talkd/.local/bin:$PATH"

ENV PIP_DEFAULT_TIMEOUT=100
ENV PIP_DISABLE_PIP_VERSION_CHECK=on
ENV PIP_NO_CACHE_DIR=on
ENV PYTHONFAULTHANDLER=1
ENV PYTHONHASHSEED=random
ENV PYTHONUNBUFFERED=1

USER dialog
WORKDIR /app

COPY poetry.lock pyproject.toml README.md /app/
COPY pytest.ini /app/dialog_lib/

USER root
RUN apt update -y && apt upgrade -y && apt install gcc libpq-dev postgresql-client -y
RUN pip install -U pip poetry

COPY /etc /app/etc
COPY /dialog_lib /app/dialog_lib

RUN  poetry config virtualenvs.create false && \
    poetry install

USER root

RUN apt update && apt -y upgrade && apt -y install libpq-dev

RUN chmod +x /app/etc/run-tests.sh

WORKDIR /app/dialog_lib