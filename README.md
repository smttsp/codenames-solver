# samet_tamplate

This repo is for template for my projects.


## Installation - Mac

### install uv

This installs `uv` on your system. `uv` is a tool that helps you to install the 
dependencies for the project.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install the environment

The following command will create the environment for the project.

```bash
uv venv
```

and sync will install it. (you can skip `uv venv` if you do `uv sync`)

```bash
uv sync
```

To activate the environment, you can run the following command:

```bash
source .venv/bin/activate
```

### Install the dependencies

To install the dependencies, you can run the following command:

```bash
uv add <package-name>
```

and to remove the dependencies, you can run the following command:

```bash
uv remove <package-name>
```
