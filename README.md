# Beam DSM package

Companion package to *Python for Structural Analysis: Continuous Beam Analysis Using the Direct Stiffness Method*
(A Step-by-Step Practical Guide for Civil Engineering Students).

## Install and reproduce

    pip install -r requirements.txt     # Python 3.10 or later; produced with Python 3.13, NumPy 2.5.3
    ./reproduce.sh                      # runs everything below and prints the results

or each step separately, from this folder:

    python3 -m pytest -q                # 99 automated tests
    python3 run_benchmarks.py           # 115 benchmark checks; writes benchmark_results.json
    python3 run_notebook.py             # executes Beam_Analysis_Clean.ipynb end to end
    python3 figures/make_figure_2_1.py figures/output     # Figure 2.1 as PNG, SVG and PDF

## Three different kinds of evidence (they are not interchangeable)

1. **99 automated tests** (pytest, folder `tests/`). They check that the code behaves as intended: 54 original tests (the
   six benchmark cases, the second worked example and its exact reference, 5 injected faults each tried on both
   examples, input validation, reactions, diagrams, deflection), 16 tests of the exercise solutions of Chapters 4 to 6, 22 tests of the exercise solutions and function-card tasks of
   Chapters 7 to 11, and 7 tests that Figure 2.1 agrees with the model. `python3 -m pytest -q
   --ignore=tests/test_exercises_ch04_06.py --ignore=tests/test_exercises_ch07_11.py --ignore=tests/test_figure_2_1.py`
   runs the original 54 alone.
2. **115 benchmark checks** (`run_benchmarks.py`). Each is one numerical comparison of a calculated quantity with a
   reference value inside a stated tolerance: 85 for the six benchmark cases, 30 for the second worked example.
3. **Independent analytical verification.** The reference values come from calculations that share no code with the
   package: the Three-Moment equations for the five-support beam, and an exact force-method solution in rational
   arithmetic (`tests/reference_force_method.py`) for the second beam. They are what the tests and checks compare against.

Passing tests show that the code does what the tests expect; independent verification is what shows the expected values are right.

## Scope
Constant EI (normalised to 1 in the worked example), vertical-only supports, element UDLs, nodal and internal downward
point loads. Reactions, shears and moments do not depend on EI; displacements do (they scale as 1/EI). Reported maximum
deflections require the mesh-refinement check (`find_converged_maximum_deflection`).

Exercise solutions (book Appendix B): `exercise_solutions_ch04_06.py`.
