set dotenv-load := true

# create dev environment
[no-cd]
install:
    curl -LsSf https://astral.sh/uv/install.sh | sh

# list justfile recipes
default:
    just --list

# clean dev environment
[no-cd]
clean:
    rm -rf .venv

# install dependencies
[no-cd]
setup:
    uv venv
    source .venv/bin/activate
    uv sync

# run linting
lint:
    ruff format .
    ruff check --fix .

# run tests
test:
    pytest
