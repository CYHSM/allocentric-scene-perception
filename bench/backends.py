"""
One interface over every way of answering a 4MT item.

Three shapes of model have to be compared on identical items:

* a **generative** VLM/LLM, prompted and parsed (MLX on the laptop, transformers
  on the cluster -- the same class, two adapters, because MLX wants file paths
  and transformers wants PIL objects);
* a frozen **encoder**, which answers by nearest-neighbour in embedding space
  with no prompt at all;
* anything added later.

Before this existed the three runners were ~80% copy-paste, which is how the
embedding runner silently lost resume support and the perception probe grew a
third copy of model loading.

The interface has to accommodate one awkward fact: encoders are fastest when
they embed the *whole corpus* once, deduplicated and batched, whereas a
generative model works one item at a time. Hence `precompute(items, bank_dir)`,
which the encoder uses to do its single pass and the generative backends ignore.

Every backend takes an item and returns a `Prediction`, so the harness does not
care which family a model belongs to.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Prediction:
    choice: int | None
    raw: str = ""
    error: str | None = None
    extras: dict = field(default_factory=dict)


class Backend:
    name = "base"

    def precompute(self, items, bank_dir):
        """Optional whole-corpus pass. Encoders embed here; others do nothing."""

    def predict(self, item, channel, bank_dir):
        raise NotImplementedError

    def ask(self, prompt, image_paths=(), max_tokens=16):
        """Single free-form question. Used by the perception probe."""
        raise NotImplementedError(f"{self.name} cannot answer free-form questions")


# --------------------------------------------------------------------------- #
# Generative backends
# --------------------------------------------------------------------------- #

class GenerativeBackend(Backend):
    """A prompted model that replies with a digit."""


    def __init__(self, model_id, adapter, max_tokens=12, answer_from="first"):
        self.name = model_id
        self.model_id = model_id
        self.adapter = adapter
        self.max_tokens = max_tokens
        # "last" for models that reason before answering; see prompts.parse_choice.
        self.answer_from = answer_from

    def predict(self, item, channel, bank_dir):
        from prompts import build_image_prompt, build_text_prompt, parse_choice

        if channel == "V":
            if not getattr(self.adapter, "accepts_images", True):
                raise ValueError(
                    f"{self.name} is a text-only backend; channel V needs images. "
                    f"Run it on L1, L2 or L3.")
            paths = [os.path.join(bank_dir, item["study"]["image"])] + [
                os.path.join(bank_dir, o["image"]) for o in item["options"]]
            prompt = build_image_prompt()
        else:
            paths = []
            prompt = build_text_prompt(item, channel)
        try:
            reply = self.adapter.generate(prompt, paths, self.max_tokens)
            err = None
        except Exception as exc:                       # keep the sweep alive
            reply, err = "", f"{type(exc).__name__}: {exc}"
        return Prediction(parse_choice(reply, prefer=self.answer_from),
                          reply[:400], err,
                          extras={"answer_from": self.answer_from})

    def ask(self, prompt, image_paths=(), max_tokens=16):
        return self.adapter.generate(prompt, list(image_paths), max_tokens)


class _MLXAdapter:
    """mlx-vlm: takes image *paths*."""

    accepts_images = True

    def __init__(self, model_id):
        from mlx_vlm import generate, load
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_config

        self._generate, self._template = generate, apply_chat_template
        self.model, self.processor = load(model_id)
        self.config = load_config(model_id)

    def generate(self, prompt, image_paths, max_tokens):
        p = self._template(self.processor, self.config, prompt,
                           num_images=len(image_paths))
        r = self._generate(self.model, self.processor, p, image_paths,
                           max_tokens=max_tokens, verbose=False)
        return r.text if hasattr(r, "text") else str(r)


class _HFAdapter:
    """transformers on CUDA: takes PIL objects."""

    accepts_images = True

    def __init__(self, model_id, max_pixels=640 * 440):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        assert torch.cuda.is_available(), \
            "no CUDA - check the torch wheel (CLUSTER.md 4.4)"
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id, max_pixels=max_pixels)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="auto")
        self.model.eval()

    def generate(self, prompt, image_paths, max_tokens):
        from PIL import Image

        images = [Image.open(p).convert("RGB") for p in image_paths] or None
        content = [{"type": "image"} for _ in image_paths]
        content.append({"type": "text", "text": prompt})
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images, return_tensors="pt")
        inputs = {k: (v.to(self.model.device) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}
        with self.torch.inference_mode():
            gen = self.model.generate(**inputs, max_new_tokens=max_tokens,
                                      do_sample=False)
        return self.processor.decode(gen[0][inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True)



class _HFTextAdapter:
    """
    A local text-only causal LM -- the pure-language arm of the benchmark.

    The point of running one is that the L1/L2/L3 channels hand a model the
    *same scene* the vision channel shows, already extracted into words. A VLM
    that scores at chance on V while a comparable LM scores well on L2 has a
    read-out failure, not a knowledge failure; that contrast is only available
    if a model with no vision tower at all can sit in the same table.
    """

    accepts_images = False

    def __init__(self, model_id, dtype="bfloat16", device_map="auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=getattr(torch, dtype), device_map=device_map)
        self.model.eval()

    def generate(self, prompt, image_paths, max_tokens):
        if image_paths:
            raise ValueError("text-only backend was handed images")
        msgs = [{"role": "user", "content": prompt}]
        # Reasoning models default their chat template to thinking mode, which
        # opens with "<think>" and does not reach an answer inside a 12-token
        # budget: Qwen3-8B produced `choice: null` on every single item until
        # this was turned off. The task asks for a bare digit, so thinking is
        # switched off where the template supports it, and a run that wants
        # reasoning uses --answer_from last with a larger --max_tokens instead.
        text = None
        for kwargs in ({"enable_thinking": False}, {}):
            try:
                text = self.tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True, **kwargs)
                break
            except Exception:
                continue
        if text is None:                        # base model with no chat template
            text = prompt
        inputs = self.tok(text, return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            gen = self.model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=False,
                pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        return self.tok.decode(gen[0][inputs["input_ids"].shape[1]:],
                               skip_special_tokens=True)


class _APIAdapter:
    """
    Any OpenAI-compatible /chat/completions endpoint, over stdlib HTTP.

    Deliberately dependency-free: a benchmark that requires a vendor SDK to be
    installed before anyone can run it against a hosted model is a benchmark
    fewer people run. Base URL and key come from the environment
    (`FMBENCH_API_BASE`, `FMBENCH_API_KEY`) or from `--api_base`.

    Images are sent as base64 data URLs, so the same adapter serves channel V
    and the text channels.

    NOTE: written against the documented chat-completions schema but not
    exercised against a live endpoint in this repo -- no key is configured here.
    Check one run against a known-good model before trusting a sweep.
    """

    accepts_images = True

    def __init__(self, model_id, base=None, key=None, timeout=120):
        self.model_id = model_id
        self.base = (base or os.environ.get("FMBENCH_API_BASE", "")).rstrip("/")
        self.key = key or os.environ.get("FMBENCH_API_KEY", "")
        self.timeout = timeout
        if not self.base:
            raise ValueError("set FMBENCH_API_BASE (or pass --api_base) to the "
                             "endpoint root, e.g. https://api.example.com/v1")

    def generate(self, prompt, image_paths, max_tokens):
        import base64
        import json as _json
        import urllib.request

        content = []
        for p in image_paths:
            with open(p, "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode()
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"}})
        content.append({"type": "text", "text": prompt})
        body = _json.dumps({
            "model": self.model_id,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }).encode()
        req = urllib.request.Request(
            f"{self.base}/chat/completions", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.key}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            out = _json.loads(r.read())
        return out["choices"][0]["message"]["content"]


# --------------------------------------------------------------------------- #
# Embedding backend
# --------------------------------------------------------------------------- #

class EmbeddingBackend(Backend):
    """
    A frozen vision encoder answering by nearest neighbour.

    No prompt, no language model: embed the study view and the four options and
    pick the closest. Because it can perturb the study vector directly, this is
    """


    def __init__(self, model_id, batch=64, workers=8):
        self.name = model_id
        self.model_id = model_id
        self.batch = batch
        self.workers = workers
        self.feats = None
        self.index = {}

    def precompute(self, items, bank_dir):
        import timm
        import torch
        from PIL import Image


        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = timm.create_model(self.model_id, pretrained=True,
                                  num_classes=0).eval().to(dev)
        cfg = timm.data.resolve_data_config({}, model=model)
        tf = timm.data.create_transform(**cfg)

        paths = sorted({os.path.join(bank_dir, p) for it in items
                        for p in [it["study"]["image"]]
                        + [o["image"] for o in it["options"]]})
        self.index = {p: i for i, p in enumerate(paths)}
        feats = np.zeros((len(paths), model.num_features), dtype=np.float32)

        # Decoding and resizing ~14k PNGs is the bottleneck here, not the
        # forward pass -- an A100 sits at 0% while one thread runs PIL. Both
        # release the GIL, so a small thread pool is enough; ordering is
        # preserved because `map` yields in submission order.
        def load(p):
            return tf(Image.open(p).convert("RGB"))

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            with torch.inference_mode():
                for s in range(0, len(paths), self.batch):
                    chunk = paths[s:s + self.batch]
                    batch = torch.stack(list(pool.map(load, chunk))).to(dev)
                    feats[s:s + len(chunk)] = model(batch).float().cpu().numpy()
        # Noise is scaled to the corpus spread, not to a single vector: one
        # L2-normalised embedding has no meaningful per-dimension std of its own.
        self.feats = feats
        del model

    def _unit(self, vec):
        n = np.linalg.norm(vec)
        return vec / (n + 1e-8)

    def predict(self, item, channel, bank_dir):

        q = self.feats[self.index[os.path.join(bank_dir, item["study"]["image"])]]
        q = self._unit(np.asarray(q, dtype=np.float64))
        sims = [float(q @ self._unit(
            self.feats[self.index[os.path.join(bank_dir, o["image"])]].astype(np.float64)))
            for o in item["options"]]
        order = sorted(sims, reverse=True)
        # Where the *correct* option ranks, not just whether it won. A model
        # with no viewpoint-invariant signal ranks it uniformly; a below-chance
        # score with a centre-heavy rank distribution means something else --
        # the target sits at the centre of the foil cloud, so the best of three
        # perturbations beats it more often than one time in four. Recording
        # the rank is what tells those two apart, and it is free here.
        target = sims[item["answer_index"]]
        return Prediction(
            choice=int(np.argmax(sims)),
            raw=" ".join(f"{s:.4f}" for s in sims),
            extras={"sim_margin": float(order[0] - order[1]),
                    "target_rank": int(sum(s > target for s in sims)) + 1,
                    "target_sim": float(target)})


# --------------------------------------------------------------------------- #

BACKENDS = ("mlx", "hf", "hflm", "api", "timm")


def make_backend(kind, model_id, **kw):
    gen = dict(max_tokens=kw.get("max_tokens", 12),
               answer_from=kw.get("answer_from", "first"))
    if kind == "mlx":
        return GenerativeBackend(model_id, _MLXAdapter(model_id), **gen)
    if kind == "hf":
        return GenerativeBackend(
            model_id, _HFAdapter(model_id,
                                 max_pixels=kw.get("max_pixels", 640 * 440)), **gen)
    if kind == "hflm":
        return GenerativeBackend(model_id, _HFTextAdapter(model_id), **gen)
    if kind == "api":
        return GenerativeBackend(
            model_id, _APIAdapter(model_id, base=kw.get("api_base")), **gen)
    if kind == "timm":
        return EmbeddingBackend(model_id, batch=kw.get("batch", 64),
                                workers=kw.get("workers", 8))
    raise ValueError(f"unknown backend {kind!r}; expected one of {BACKENDS}")


def auto_backend(model_id, **kw):
    """Pick MLX on the laptop, transformers on a CUDA box."""
    try:
        import mlx.core  # noqa: F401
        return make_backend("mlx", model_id, **kw)
    except ImportError:
        return make_backend("hf", model_id, **kw)
