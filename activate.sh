#!/bin/bash
# Activate the uv virtual environment for this project
echo "Activating uv virtual environment..."
source .venv/bin/activate
echo "Environment activated! You can now run:"
echo "  jupyter lab    # Start JupyterLab"
echo "  jupyter notebook    # Start Jupyter Notebook"
echo "  python -m ipykernel install --user --name=agent-evaluation    # Install kernel for other Jupyter instances"