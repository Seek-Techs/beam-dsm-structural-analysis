"""Execute Beam_Analysis_Clean.ipynb top to bottom in a fresh kernel and report any error cell.

Usage:  python3 run_notebook.py            (the notebook is executed in place and saved with its outputs)
"""
import sys

import nbformat
from nbclient import NotebookClient

nb = nbformat.read("Beam_Analysis_Clean.ipynb", as_version=4)
NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": "."}}).execute()
nbformat.write(nb, "Beam_Analysis_Clean.ipynb")
errors = sum(any(o.get("output_type") == "error" for o in c.get("outputs", [])) for c in nb.cells if c.cell_type == "code")
print(f"notebook executed: {sum(c.cell_type == 'code' for c in nb.cells)} code cells, {errors} with errors")
sys.exit(1 if errors else 0)
