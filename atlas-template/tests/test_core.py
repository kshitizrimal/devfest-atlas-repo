"""Small checks of the pure parts: pointers, passages, search, the lens, the number check.
    uv run pytest tests"""
import os
import tempfile

os.environ["ATLAS_DATA"] = tempfile.mkdtemp()          # never touch a real library

import numpy as np                                         # noqa: E402

from atlas import store                                    # noqa: E402
from atlas.ask import ascii_digits, numbers_ok             # noqa: E402
from atlas.ingest import block_id, group                   # noqa: E402
from atlas.lens import steer                               # noqa: E402


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_pointer_address():
    assert block_id("a1b2c3", 12, 3) == "a1b2c3#p12#b3"


def test_passages_never_cross_a_page():
    blocks = [{"page": 1, "text": "a" * 500}, {"page": 1, "text": "b" * 300}, {"page": 1, "text": "c" * 300}, {"page": 2, "text": "d"}]
    assert [len(g) for g in group(blocks)] == [2, 1, 1]
    assert {b["page"] for g in group(blocks) for b in g if g[0]["page"] == 2} == {2}


def test_search_ranks_by_cosine_and_scopes_to_an_item():
    store.put_vectors("passage", [("x#c1", "x", unit([1, 0, 0])), ("x#c2", "x", unit([0, 1, 0])), ("y#c1", "y", unit([1, 0.1, 0]))])
    assert [k for k, *_ in store.search(["passage"], unit([1, 0, 0]), k=2)] == ["x#c1", "y#c1"]
    assert [k for k, *_ in store.search(["passage"], unit([1, 0, 0]), item_id="y")] == ["y#c1"]


def test_lens_pulls_the_question():
    q, lens = unit([1, 0]), unit([0, 1])
    v = steer(q, [lens])
    assert abs(np.linalg.norm(v) - 1) < 1e-6 and v @ lens > v @ q
    assert np.allclose(steer(q, []), q)


def test_numbers_are_checked_across_scripts():
    assert ascii_digits("२५० and 12,800") == "250 and 12800"
    assert numbers_ok("२५० जनाको मृत्यु", "the total number of deaths has reached 250") is None
    assert numbers_ok("Six roads, 6 of them, were closed", "four roads were blocked") == "6 is not in the cited source"
