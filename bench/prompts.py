"""
The question every model is asked, and how its reply is read.

Kept in one module so the MLX runner on the laptop, the CUDA runner on the
cluster and the worked-example dump all pose *identical* prompts. If they
drifted apart the runs would not be comparable, which is the whole point of
running on more than one machine.

The task statement has to name what the foils actually do. v3 introduced a
three-family ladder -- a peak's *shape* is substituted (`identity`), peaks
*exchange* places (`binding`), or peaks are *displaced* (`metric`) -- but the
instruction text still said only that peaks "have been MOVED". A model told to
look for movement is being scored on the wrong question in two families out of
three, so the wording below enumerates all three. It is the same sentence for
every item, so it leaks nothing about which family an individual item is.
"""

import re

TASK = (
    "You are looking at an alpine valley with a small lake at its centre, "
    "ringed by several mountain peaks.\n\n"
    "First you are given the STUDY view of the valley.\n"
    "Then four OPTIONS, each taken from a different compass direction than the "
    "study view.\n\n"
    "Exactly ONE option shows the SAME valley as the study view: the same peaks, "
    "with the same shapes, standing in the same places, simply seen from "
    "elsewhere.\n"
    "Each of the other three has been altered in one of these ways:\n"
    "  - one peak has been replaced by a differently shaped peak;\n"
    "  - two or more peaks have exchanged positions with each other;\n"
    "  - one or more peaks have been moved to different positions.\n\n"
    "Which option is the same valley?\n"
    "Answer with a single digit: 1, 2, 3 or 4. Do not explain."
)

# Kept under the old name: `dump_examples.py` and older result dumps refer to it.
INSTRUCTIONS = TASK

_CHANNEL_NOTE = {
    "L1": ("Each view is given as an overhead map of the valley in a fixed "
           "world frame.\nNote that this frame does not turn with the viewer."),
    "L2": ("Each view is given as a list of peaks as they appear from that "
           "viewpoint,\nleft to right, with the bearing, distance and apparent "
           "width each one\nsubtends from there. Those numbers are measured from "
           "the viewer, so they\nchange when the viewpoint changes even though "
           "the valley does not."),
    "L3": ("Each view is given as a short prose description of what is visible "
           "from that\nviewpoint, in the order the peaks appear from left to "
           "right."),
}


def build_text_prompt(item, level):
    parts = [TASK]
    if level in _CHANNEL_NOTE:
        parts += ["", _CHANNEL_NOTE[level]]
    parts += ["", "=== STUDY VIEW ===", item["study"]["text"][level]]
    for i, opt in enumerate(item["options"], 1):
        parts += ["", f"=== OPTION {i} ===", opt["text"][level]]
    parts += ["", "Answer with a single digit (1-4):"]
    return "\n".join(parts)


def build_image_prompt():
    return (TASK + "\n\nThe first image is the STUDY view. The next four "
            "images are OPTION 1, OPTION 2, OPTION 3 and OPTION 4 in that order.\n"
            "Answer with a single digit (1-4):")


def parse_choice(text, n_options=4, prefer="first"):
    """
    Read the chosen option out of a reply, as a 0-based index.

    `prefer="first"` is the original behaviour and stays the default so v2/v3
    numbers remain comparable: with `max_tokens` in the low teens a compliant
    model's reply *is* the digit. A reasoning model asked to think first will
    instead emit digits all the way through its working and only settle at the
    end, so those runs pass `prefer="last"`. Which one a run used is recorded
    alongside the result -- it is a scoring decision, not an implementation
    detail, and reporting it is what keeps two runs comparable.
    """
    if not text:
        return None
    hits = re.findall(r"\b([1-%d])\b" % n_options, text)
    if not hits:
        hits = re.findall(r"([1-%d])" % n_options, text)
    if not hits:
        return None
    return int(hits[-1 if prefer == "last" else 0]) - 1
