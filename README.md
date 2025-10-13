# Agent Evaluation Project

This is a Python project for evaluating agent performance.

## Project Structure

```
├── .venv/             # Virtual environment
├── notebook.ipynb     # Jupyter notebook for analysis
├── pyproject.toml     # Project configuration
├── README.md          # This file
├── .gitignore         # Git ignore file
├── .python-version    # Python version specification
└── uv.lock           # Dependency lock file
```

## Setup

1. Clone the repository:

```bash
git clone git@github.com:CSI-Genomics-and-Data-Analytics-Core/ai-agent-evaluation.git
cd ai-agent-evaluation
```

2. Install uv (if not already installed):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. Create virtual environment and install dependencies:

```bash
uv sync
```

4. Run the Jupyter notebook:

```bash
uv run jupyter notebook notebook.ipynb
```
