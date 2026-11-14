# Sandbox mínimo do Odysseus V2 para isolamento de ShellTool.
# Base oficial Python 3.12 slim + git (necessário para testes que
# executam `git status` no workspace).
FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Utilizador sem privilégios (a política de isolamento pode definir
# --user em runtime, mas convém existir para o default).
RUN useradd -m -u 1000 sandbox || true

WORKDIR /work
