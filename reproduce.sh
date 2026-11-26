#!/usr/bin/env bash
# Reproduce every result quoted in the book, from the folder that contains beam_dsm.py.
set -e
python3 --version
python3 -c "import numpy, matplotlib, pytest; print('numpy', numpy.__version__, '| matplotlib', matplotlib.__version__, '| pytest', pytest.__version__)"
echo; echo "== 1. Automated tests (all)"
python3 -m pytest -q
echo; echo "== 2. Automated tests (the original 54, without the exercise and figure tests)"
python3 -m pytest -q --ignore=tests/test_exercises_ch04_06.py --ignore=tests/test_exercises_ch07_11.py --ignore=tests/test_figure_2_1.py
echo; echo "== 3. Benchmark register (individual numerical comparisons)"
python3 run_benchmarks.py
echo; echo "== 4. Notebook"
python3 run_notebook.py
echo; echo "== 5. Figure 2.1 (PNG, SVG, PDF)"
python3 figures/make_figure_2_1.py figures/output
