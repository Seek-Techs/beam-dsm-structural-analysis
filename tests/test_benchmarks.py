"""The six-case benchmark register (book Chapter 19) as automated tests."""
import pytest

import beam_dsm
from benchmarks import register

RECORDS = register(beam_dsm)


@pytest.mark.parametrize("test_id", [1, 2, 3, 4, 5, 6])
def test_benchmark_case(test_id):
    failed = [r for r in RECORDS if r["test"] == test_id and not r["passed"]]
    assert not failed, "\n".join(
        f"{r['representation']}: {r['quantity']} calc={r['calculated']} ref={r['reference']}" for r in failed
    )


def test_register_is_complete():
    assert {r["test"] for r in RECORDS} == {1, 2, 3, 4, 5, 6}
    assert len(RECORDS) >= 80
