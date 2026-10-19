"""Read the same decision prefixes under real, direct and random interventions."""
import json
from pathlib import Path
import time

import numpy as np

from chatlab.extension_api import read_steering_vector
from .analysis import Aggregate, FORMAT


def examples(rows):
    result = []
    for row in rows:
        point, text, recovered = (list(row) + [None] * 3)[:3]
        if not str(text or "").strip():
            continue
        point = str(point or "Decision").strip()
        if recovered not in (None, "", "0", "1", 0, 1):
            raise ValueError("Returned to task must be 0, 1, or empty (unknown).")
        result.append({"point": point, "text": str(text),
                       "recovered": None if recovered in (None, "") else int(recovered)})
    if not 1 <= len(result) <= 64 or len({r["point"] for r in result}) > 4:
        raise ValueError("Give 1–64 examples across at most four decision points.")
    if any(len(r["text"]) > 32000 or len(r["point"]) > 200 for r in result):
        raise ValueError("Each prefix may have at most 32,000 characters; point names at most 200.")
    return result


def addition(vector):
    def hook(hidden):
        import torch
        step = torch.as_tensor(vector, device=hidden.device, dtype=hidden.dtype)
        moved = hidden + step
        if not torch.isfinite(moved).all():
            raise ValueError("Steering overflowed the activation precision; use a smaller strength.")
        return moved
    return hook


def run(session, rows, paths, blocks, multiplier=1, seed=0):
    samples = examples(rows)
    if not paths or not 1 <= len(paths) <= 6:
        raise ValueError("Open one to six steering vector files.")
    vectors = [read_steering_vector(path) for path in paths]
    blocks = sorted(set(int(x.strip()) for x in blocks.split(",") if x.strip()))
    if not 1 <= len(blocks) <= 8 or any(b < 0 or b >= session.block_count for b in blocks):
        raise ValueError("Choose one to eight valid decoder block indices, separated by commas.")
    if not np.isfinite(multiplier) or int(seed) != seed or not 0 <= seed < 2 ** 32:
        raise ValueError("Use a finite strength multiplier and a whole seed between 0 and 4294967295.")
    lens = session.jacobian_lens()
    if lens is None or set(blocks) - set(lens["layers"]):
        raise ValueError("Import a Jacobian lens in Chat fitted at every selected block first.")
    encoded = [session.example_ids(row["text"]) for row in samples]
    for ids in encoded:
        if not ids or len(ids) > min(4096, session.position_limit or 4096):
            raise ValueError("Every example must contain 1–4096 tokens and fit the model's context window.")
    for vector in vectors:
        session.check_steering(dict(vector, enabled=True, strength=1.0))
        if min(blocks) < vector["layer"]:
            raise ValueError("Read blocks at or after each vector's injection block.")
    result = {"format": FORMAT, "created": time.time(), "model_id": session.model_id,
              "model_revision": session.model_revision, "precision": session.precision,
              "lens": lens, "blocks": blocks, "examples": samples,
              "multiplier": multiplier, "seed": int(seed), "vectors": []}
    vocabulary_text = {}
    # Keep only streaming sums and one example's ranks; never retain example × vocabulary tensors.
    for vi, vector in enumerate(vectors):
        step = np.asarray(vector["vector"], dtype=float) * vector["strength"] * multiplier
        if not np.isfinite(step).all():
            raise ValueError("The scaled steering vector is not finite.")
        random = np.random.default_rng(int(seed) + vi).normal(size=step.shape)
        norm = np.linalg.norm(step)
        if not np.isfinite(norm):
            raise ValueError("The steering vector norm overflows; use a smaller strength.")
        random *= norm / np.linalg.norm(random)
        entry = {"name": Path(paths[vi]).stem.replace("_", " "), "vector": vector, "points": {}}
        # Finish one point at a time so adding groups never multiplies vocabulary-sized working memory.
        for point in dict.fromkeys(s["point"] for s in samples):
            modes = {mode: Aggregate() for mode in ("Total", "Computed")}
            with session.transformers_model():
                for number, (sample, ids) in enumerate(zip(samples, encoded), 1):
                    if sample["point"] != point:
                        continue
                    yield f"Vector {vi + 1}/{len(vectors)}, example {number}/{len(samples)}: reading vocabulary ranks."
                    def read(hooks, wanted=blocks):
                        with session.block_hooks(hooks):
                            return session.lens_ranks(ids, wanted, [len(ids) - 1])[:, 0]
                    base = read({})
                    steered = read({vector["layer"]: addition(step)})
                    control = read({vector["layer"]: addition(random)})
                    direct = np.concatenate([read({b: addition(step)}, [b]) for b in blocks])
                    random_direct = np.concatenate([read({b: addition(random)}, [b]) for b in blocks])
                    modes["Total"].add(steered, base, control)
                    # Each intervention has its own direct-addition baseline.
                    modes["Computed"].add(steered, direct, control, random_direct)
            entry["points"][point] = {}
            for mode, aggregate in modes.items():
                block_rows = {}
                for index, block in enumerate(blocks):
                    rs = aggregate.rows(index)
                    for row in rs:
                        token = row["id"]
                        if token not in vocabulary_text:
                            vocabulary_text[token] = session.decode([token])
                        row["token"] = vocabulary_text[token]
                    block_rows[str(block)] = rs
                entry["points"][point][mode] = {"n": aggregate.n, "blocks": block_rows, "traces": aggregate.traces()}
            del modes, aggregate
        result["vectors"].append(entry)
    return result


def read_result(path):
    if Path(path).stat().st_size > 128 * 1024 ** 2:
        raise ValueError("Comparison results must be smaller than 128 MiB.")
    def reject_nonfinite(_value):
        raise ValueError("Comparison JSON must contain finite numbers.")
    value = json.loads(Path(path).read_text(), parse_constant=reject_nonfinite)
    if not isinstance(value, dict) or value.get("format") != FORMAT:
        raise ValueError("Open a chatlab-steering-comparison-1 result.")
    # Validate the complete renderable structure before making it visible.
    if not isinstance(value.get("vectors"), list) or not 1 <= len(value["vectors"]) <= 6:
        raise ValueError("A result needs one to six vectors.")
    if not isinstance(value.get("blocks"), list) or not 1 <= len(value["blocks"]) <= 8:
        raise ValueError("A result needs one to eight blocks.")
    if any(type(b) is not int or not 0 <= b < 4096 for b in value["blocks"]) or len(set(value["blocks"])) != len(value["blocks"]):
        raise ValueError("Invalid decoder block indices.")
    if not isinstance(value.get("model_id"), str) or not value["model_id"].strip():
        raise ValueError("A result needs a model ID.")
    samples = examples([[e["point"], e["text"], e["recovered"]] for e in value["examples"]])
    value["examples"] = samples
    points = {sample["point"] for sample in samples}
    for vector in value["vectors"]:
        if not isinstance(vector.get("name"), str) or not vector.get("points"):
            raise ValueError("Each vector needs a name and decision points.")
        if set(vector["points"]) != points:
            raise ValueError("Every vector must contain the same example decision points.")
        for point, modes in vector["points"].items():
            for mode in ("Total", "Computed"):
                group = modes[mode]
                if type(group["n"]) is not int or group["n"] != sum(s["point"] == point for s in samples):
                    raise ValueError("Invalid example count.")
                for trace in group["traces"].values():
                    if not isinstance(trace, list) or len(trace) != len(value["blocks"]) or any(
                            type(x) not in (int, float) or not np.isfinite(x) or x <= 0 for x in trace):
                        raise ValueError("Invalid block rank-change trace.")
                for block in value["blocks"]:
                    for row in group["blocks"][str(block)]:
                        if not isinstance(row["token"], str) or type(row["id"]) is not int:
                            raise ValueError("Invalid vocabulary token.")
                        if str(row["id"]) not in group["traces"]:
                            raise ValueError("A vocabulary token is missing its block trace.")
                        for field in ("factor", "top50", "base50", "top10", "base10", "random50", "t", "random_t"):
                            number = row[field]
                            if number is None and field in ("t", "random_t"):
                                continue
                            if type(number) not in (int, float) or not np.isfinite(number):
                                raise ValueError("Invalid rank statistic.")
                            if field == "factor" and number <= 0:
                                raise ValueError("Rank factors must be positive.")
                            if field not in ("factor", "t", "random_t") and not 0 <= number <= 1:
                                raise ValueError("Top-k shares must lie between zero and one.")
    return value
