import collections
import json
import os

import pytest
from bench.build_vlm_benchmark import build_trials, DELTAS, MODES

def test_vlm_benchmark_mock(tmp_path):
    # Create a small mock scenes_100 directory
    scenes_dir = tmp_path / "scenes_100"
    mode_dir = scenes_dir / "c0_shape_colour"
    mode_dir.mkdir(parents=True)

    # 5 scenes with 16 frames each
    scenes_dict = {}
    for sid in [f"a{i:03d}" for i in range(5)]:
        s_dir = mode_dir / sid
        s_dir.mkdir()
        frames = {}
        for app in ("A", "B"):
            for az in [0, 45, 90, 135, 180, 225, 270, 315]:
                img_name = f"{sid}_{app}_az{az:03d}.png"
                (s_dir / img_name).write_text("fake png")
                frames[f"{app}_{az}"] = {"image": img_name}
        scenes_dict[sid] = {"dir": sid, "frames": frames}

    bank_file = mode_dir / "bank.json"
    bank_file.write_text(json.dumps({"scenes": scenes_dict}))

    # Test 4AFC
    trials_4afc = build_trials(
        root=str(scenes_dir),
        modes=["c0_shape_colour"],
        deltas=[0, 45],
        trials_per_delta=10,
        n_options=4,
        seed=123
    )

    assert len(trials_4afc) == 20
    for t in trials_4afc:
        assert len(t["options"]) == 4
        assert 1 <= t["correct_choice"] <= 4
        target_opt = t["options"][t["correct_choice"] - 1]
        assert target_opt["is_target"] is True
        assert target_opt["scene_id"] == t["study_scene"]
        assert target_opt["appearance"] == "B"

        distractors = [o for i, o in enumerate(t["options"]) if i != t["correct_choice"] - 1]
        assert len(distractors) == 3
        for d in distractors:
            assert d["is_target"] is False
            assert d["scene_id"] != t["study_scene"]
            assert d["appearance"] == "B"
            assert d["azimuth"] == target_opt["azimuth"]

    # Test 2AFC
    trials_2afc = build_trials(
        root=str(scenes_dir),
        modes=["c0_shape_colour"],
        deltas=[0, 45],
        trials_per_delta=10,
        n_options=2,
        seed=123
    )
    assert len(trials_2afc) == 20
    for t in trials_2afc:
        assert len(t["options"]) == 2
        assert 1 <= t["correct_choice"] <= 2


def test_parse_answer():
    from bench.evaluate_vlm import parse_answer

    # Direct answers
    assert parse_answer("1", n_options=4, prefer="first") == 1
    assert parse_answer("Final Answer: Option 2", n_options=4) == 2
    assert parse_answer("Option 3", n_options=4) == 3
    assert parse_answer("Final Answer: 4", n_options=4) == 4

    # CoT answers with intermediate option discussions
    cot_text_1 = (
        "In Option 1, the arrangement is similar. In Option 2, it is different. "
        "In Option 3, the peaks are rotated. In Option 4, the cone is behind. "
        "The final answer is Option 3."
    )
    assert parse_answer(cot_text_1, n_options=4, prefer="last") == 3

    cot_text_2 = (
        "Option 1 has blue in front. Option 2 has yellow in back. "
        "Therefore, the correct answer is Option 4."
    )
    assert parse_answer(cot_text_2, n_options=4, prefer="last") == 4

    cot_text_3 = (
        "Comparing the study view with candidates:\n"
        "- Option 1: wrong layout\n"
        "- Option 2: correct layout rotated 90 degrees\n"
        "Option 2 is the correct match."
    )
    assert parse_answer(cot_text_3, n_options=4, prefer="last") == 2

    # Template artifact filtering
    assert parse_answer("Where X is a number from [1-4]. Final Answer: Option 1", n_options=4) == 1
    assert parse_answer("Select between 1-4. The answer is 3.", n_options=4) == 3



# --------------------------------------------------------------------------- #
# Answer position: the property that decides whether a per-delta number means
# anything at all
# --------------------------------------------------------------------------- #

def _mock_bank(tmp_path, n_scenes=8):
    scenes_dir = tmp_path / "scenes_100"
    mode_dir = scenes_dir / "c0_shape_colour"
    mode_dir.mkdir(parents=True)
    scenes = {}
    for sid in [f"a{i:03d}" for i in range(n_scenes)]:
        (mode_dir / sid).mkdir()
        frames = {}
        for app in ("A", "B"):
            for az in [0, 45, 90, 135, 180, 225, 270, 315]:
                name = f"{sid}_{app}_az{az:03d}.png"
                (mode_dir / sid / name).write_text("fake png")
                frames[f"{app}_{az}"] = {"image": name}
        scenes[sid] = {"dir": sid, "frames": frames}
    (mode_dir / "bank.json").write_text(json.dumps({"scenes": scenes}))
    return scenes_dir


@pytest.mark.parametrize("n_options,trials_per_delta", [(4, 20), (2, 20), (4, 8), (2, 10)])
@pytest.mark.parametrize("seed", [1, 42, 7777])
def test_the_answer_position_is_balanced_inside_every_cell(
        tmp_path, n_options, trials_per_delta, seed):
    """
    Every per-delta accuracy in the paper is read off a single (mode, delta)
    cell of 10-20 trials. If the answer sits at the same position throughout
    that cell, a model with a positional prior scores well without looking at
    the images -- and both failure modes were live: at seed 42 the whole
    delta=180 cell of the 2AFC calibration slice had the answer at position 1,
    while Qwen2.5-VL-7B answered "4" on 59% of trials.

    A per-trial `rng.shuffle` is unbiased in expectation and says nothing about
    any one cell, so it cannot be what guards this. Balance is scheduled.
    """
    scenes_dir = _mock_bank(tmp_path)
    trials = build_trials(root=str(scenes_dir), modes=["c0_shape_colour"],
                          deltas=DELTAS, trials_per_delta=trials_per_delta,
                          n_options=n_options, seed=seed)

    by_cell = {}
    for t in trials:
        by_cell.setdefault((t["mode"], t["delta"]), []).append(t["correct_choice"])

    assert len(by_cell) == len(DELTAS)
    for cell, choices in by_cell.items():
        counts = collections.Counter(choices)
        assert set(counts) <= set(range(1, n_options + 1))
        lo, hi = min(counts.get(p, 0) for p in range(1, n_options + 1)), max(counts.values())
        # exact when the cell divides evenly, off by at most one otherwise
        assert hi - lo <= 1, f"{cell}: answer positions {dict(counts)}"


@pytest.mark.parametrize("seed", [1, 42, 7777])
def test_a_constant_answer_scores_chance(tmp_path, seed):
    """The operational version of the test above: a model that ignores the
    images and always says the same thing must land on chance in every cell."""
    scenes_dir = _mock_bank(tmp_path)
    trials = build_trials(root=str(scenes_dir), modes=["c0_shape_colour"],
                          deltas=DELTAS, trials_per_delta=20, n_options=4,
                          seed=seed)
    for guess in (1, 2, 3, 4):
        for delta in DELTAS:
            cell = [t for t in trials if t["delta"] == delta]
            acc = sum(t["correct_choice"] == guess for t in cell) / len(cell)
            assert acc == pytest.approx(0.25), \
                f'always answering "{guess}" scores {acc:.0%} at delta={delta}'


def test_the_correct_choice_index_points_at_the_target(tmp_path):
    """`correct_choice` is now assigned rather than searched for, so the
    agreement between it and `is_target` has to be checked explicitly."""
    scenes_dir = _mock_bank(tmp_path)
    for n_options in (2, 4):
        trials = build_trials(root=str(scenes_dir), modes=["c0_shape_colour"],
                              deltas=DELTAS, trials_per_delta=20,
                              n_options=n_options, seed=5)
        for t in trials:
            opts = t["options"]
            assert len(opts) == n_options
            assert sum(o["is_target"] for o in opts) == 1
            assert opts[t["correct_choice"] - 1]["is_target"]
            assert opts[t["correct_choice"] - 1]["scene_id"] == t["study_scene"]
