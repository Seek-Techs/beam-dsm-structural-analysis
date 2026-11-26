"""Figure 2.1 agrees with the verified model: node positions, supports, loads, dimensions and labels.

The expected values are typed here from the problem statement of Section 1.7, not read from beam_dsm, so a change to
the model or to the figure that makes them disagree is detected.
"""
import importlib.util
import os
import re
import xml.etree.ElementTree as ET
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import pytest
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.text import Annotation

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("fig21", os.path.join(HERE, "..", "figures", "make_figure_2_1.py"))
fig21 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fig21)

NODES = [0, 6, 8, 10, 12.5, 14.1, 17, 22]
SUPPORT_X = [0, 6, 12.5, 17, 22]
POINT_LOADS = {8: "40 kN", 10: "60 kN", 14.1: "100 kN"}
UDL_BLOCKS = [(0, 6, "60 kN/m"), (12.5, 14.1, "20 kN/m"), (14.1, 17, "20 kN/m"), (17, 22, "80 kN/m")]
SPANS = ["span AB = 6 m", "span BC = 6.5 m", "span CD = 4.5 m", "span DE = 5 m"]
ELEMENT_LENGTHS = [6, 2, 2, 2.5, 1.6, 2.9, 5]


@pytest.fixture(scope="module")
def drawn():
    fig, ax = fig21.build()
    yield fig, ax
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_node_markers_at_node_coordinates(drawn):
    _, ax = drawn
    xs = sorted(l.get_xdata()[0] for l in ax.lines if isinstance(l, Line2D) and l.get_marker() == "o")
    assert xs == pytest.approx(NODES)


def test_support_triangles_at_supported_nodes(drawn):
    _, ax = drawn
    apexes = sorted(p.get_xy()[0][0] for p in ax.patches
                    if isinstance(p, Polygon) and len(p.get_xy()) in (3, 4) and p.get_facecolor()[:3] == (1.0, 1.0, 1.0))
    assert apexes == pytest.approx(SUPPORT_X)


def test_distributed_load_blocks_and_values(drawn):
    _, ax = drawn
    blocks = sorted((round(p.get_xy()[:, 0].min(), 6), round(p.get_xy()[:, 0].max(), 6))
                    for p in ax.patches if isinstance(p, Polygon) and len(p.get_xy()) == 5 and p.get_edgecolor()[0] < 0.3)
    flat = [v for pair in blocks for v in pair]
    assert flat == pytest.approx([v for a, b, _ in UDL_BLOCKS for v in (a, b)])
    labels = Counter(t.get_text() for t in ax.texts if t.get_text().endswith("kN/m"))
    assert labels == Counter(w for _, _, w in UDL_BLOCKS)


def test_point_load_arrows_and_labels(drawn):
    _, ax = drawn
    arrows = sorted(a.xy[0] for a in ax.texts
                    if isinstance(a, Annotation) and a.arrowprops and a.arrowprops.get("color") == "#b03030")
    assert arrows == pytest.approx(sorted(POINT_LOADS))
    for t in ax.texts:
        if t.get_text() in POINT_LOADS.values():
            x = t.get_position()[0]
            assert POINT_LOADS[min(POINT_LOADS, key=lambda k: abs(k - x))] == t.get_text()
    assert Counter(t.get_text() for t in ax.texts if t.get_text().endswith(" kN")) == Counter(POINT_LOADS.values())
    # downward: the arrow tip is below its tail
    for a in ax.texts:
        if isinstance(a, Annotation) and a.arrowprops and a.arrowprops.get("color") == "#b03030":
            assert a.xy[1] < a.xyann[1]


def test_node_labels_and_support_letters(drawn):
    _, ax = drawn
    texts = [t.get_text() for t in ax.texts]
    for n, x in enumerate(NODES):
        assert f"node {n}\nx = {x:g} m" in texts
    for letter in "ABCDE":
        assert letter in texts


def test_dimension_labels(drawn):
    _, ax = drawn
    texts = [t.get_text() for t in ax.texts]
    for s in SPANS:
        assert s in texts
    for e, length in enumerate(ELEMENT_LENGTHS, start=1):
        assert f"element {e}\n{length:g} m" in texts
    assert abs(sum(ELEMENT_LENGTHS) - 22) < 1e-12


def test_exported_files_and_svg_text(tmp_path):
    fig21.main(str(tmp_path))
    for ext in ("png", "svg", "pdf"):
        assert (tmp_path / f"figure_2_1_beam_idealisation.{ext}").stat().st_size > 1000
    root = ET.parse(tmp_path / "figure_2_1_beam_idealisation.svg").getroot()
    text = " ".join("".join(el.itertext()) for el in root.iter() if el.tag.endswith("text"))
    text = re.sub(r"\s+", " ", text)
    for needle in ["x = 14.1 m", "x = 12.5 m", "100 kN", "span BC = 6.5 m", "80 kN/m", "60 kN/m", "1050 kN"]:
        assert needle in text, needle
