"""
The backend contract.

Three very different things -- a prompted VLM, a prompted LLM, a frozen encoder
-- have to produce records `analyze.py` can pool. These tests pin the shared
shape without loading any real model.
"""

import numpy as np
import pytest

import backends as BK
from prompts import build_image_prompt, build_text_prompt, parse_choice


class FakeAdapter:
    """Replies with a fixed string; records what it was asked."""

    def __init__(self, reply="2"):
        self.reply = reply
        self.calls = []

    def generate(self, prompt, image_paths, max_tokens):
        self.calls.append((prompt, list(image_paths), max_tokens))
        return self.reply


@pytest.fixture
def item():
    texts = {lvl: f"scene text {lvl}" for lvl in ("L1", "L2", "L3")}
    return {
        "item_id": "s0_s000_t090", "condition": "allocentric",
        "delta_azimuth": 90, "foil_family": "metric", "foil_param": 4.0,
        "appearance_changed": True, "foil_displacement_m": 4.0,
        "min_foil_displacement_m": 4.0, "n_peaks": 4,
        "distinctiveness": "distinct", "answer_index": 1,
        "study": {"image": "s0/study.png", "text": texts},
        "options": [{"image": f"s0/opt{i}.png", "is_target": i == 1,
                     "text": {k: f"{v} opt{i}" for k, v in texts.items()},
                     "variant": f"v{i}", "foil_op": None,
                     "max_displacement_m": 0.0 if i == 1 else 4.0}
                    for i in range(4)],
        "text_distinguishable": {"L1": True, "L2": True, "L3": True},
    }


def test_generative_backend_sends_five_images_on_the_image_channel(item):
    ad = FakeAdapter("2")
    b = BK.GenerativeBackend("fake", ad)
    pred = b.predict(item, "V", "/bank")
    prompt, paths, _ = ad.calls[0]
    assert len(paths) == 5                     # study + four options
    assert paths[0].endswith("study.png")
    assert prompt == build_image_prompt()
    assert pred.choice == 1                    # "2" -> 0-based 1


def test_generative_backend_sends_no_images_on_a_text_channel(item):
    ad = FakeAdapter("3")
    b = BK.GenerativeBackend("fake", ad)
    b.predict(item, "L2", "/bank")
    prompt, paths, _ = ad.calls[0]
    assert paths == []
    assert prompt == build_text_prompt(item, "L2")


def test_generative_backend_survives_a_backend_exception(item):
    class Boom:
        def generate(self, *a):
            raise RuntimeError("cuda melted")

    pred = BK.GenerativeBackend("fake", Boom()).predict(item, "V", "/bank")
    assert pred.choice is None
    assert "cuda melted" in pred.error


def test_parse_choice_round_trip():
    assert [parse_choice(x) for x in ["1", "Option 3", "the answer is 4", "", "z"]] \
        == [0, 2, 3, None, None]
