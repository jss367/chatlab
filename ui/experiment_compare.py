"""Configure both conditions before running a sequential comparison."""

from __future__ import annotations

import contextlib
import threading

import gradio as gr

import experiment_runs
import settings
from model_runtime import MLX_KIND, TEXT_KIND, cache_status
from ui import runtime
from ui.common import failure_status
from ui.compare import fill_slot
from ui.models_page import stream_load
from ui.panel import as_plain_text

_PAIR_LOCK = threading.Lock()


def next_difference(held, previous):
    readings = (held or {}).get("reading", {}).get("readings", [])
    ranked = sorted((item for item in readings if item["scored"]),
                    key=lambda item: (-item["surprise_bits"], item["position"]))
    if not ranked:
        return None, "Fill both slots with runs that share scored text first."
    positions = [item["position"] for item in ranked]
    offset = (positions.index(previous) + 1) % len(ranked) if previous in positions else 0
    item = ranked[offset]
    return item["position"], (
        f"**Difference {offset + 1} of {len(ranked)} · span {item['position']}**\n\n"
        f"{as_plain_text(item['text'])}\n\n"
        f"A: **{item['left_surprise']:.2f} bits** · B: **{item['right_surprise']:.2f} bits** · "
        f"gap: **{item['surprise_bits']:.2f} bits**.\n\n"
        f"A tokens {item['left_range'][0] + 1}–{item['left_range'][1]}; "
        f"B tokens {item['right_range'][0] + 1}–{item['right_range'][1]}."
    )


def build():
    with gr.Accordion("Set up both runs", open=False):
        gr.Markdown("Configure A and B, then run them in order. Models must already be downloaded. "
                    "Completed runs are saved in Experiments. MLX models use their stored precision; choose Current. "
                    "Other settings come from the controls below and the Chat tab.")
        preset = gr.Dropdown(["Custom", "Two seeds", "Steering off versus on", "Full precision versus 4-bit"],
                             value="Custom", label="Comparison preset")
        conditions = []
        with gr.Row():
            for side in ("A", "B"):
                with gr.Column():
                    gr.Markdown(f"**Condition {side}**")
                    conditions.extend([
                        gr.Textbox(label=f"Model {side}", placeholder="Blank uses the model loaded when you start"),
                        gr.Dropdown(["Current", *settings.WEIGHT_PRECISIONS], value="Current", label=f"Precision {side}"),
                        gr.Slider(0, 2, value=0.7, step=0.05, label=f"Temperature {side}"),
                        gr.Number(value=42 if side == "A" else 43, precision=0, minimum=0, label=f"Seed {side}"),
                        gr.Checkbox(value=False, label=f"Apply Chat steering vector to {side}"),
                        gr.Number(value=1, label=f"Steering strength {side}"),
                    ])
        run = gr.Button("Run comparison", variant="primary")
        load_status = gr.HTML()
        preset.change(apply_preset, preset, [conditions[1], conditions[3], conditions[4],
                                            conditions[7], conditions[9], conditions[10]])
    return conditions, run, load_status


def apply_preset(preset):
    if preset == "Custom":
        return (gr.skip(),) * 6
    return (
        "full" if preset == "Full precision versus 4-bit" else "Current", 42, False,
        "4-bit" if preset == "Full precision versus 4-bit" else "Current",
        43 if preset == "Two seeds" else 42, preset == "Steering off versus on",
    )


def run_pair(*values):
    """Publish completed sides immediately; never replace a side with a partial run."""
    skip = gr.skip()
    if not _PAIR_LOCK.acquire(blocking=False):
        yield (skip, skip, "Another comparison is running.", skip, skip, skip, skip, skip)
        return

    def frame(status, *, left=skip, right=skip, busy=True, load=skip):
        return (left, right, status, gr.update(interactive=not busy),
                gr.update(interactive=not busy), gr.update(visible=busy),
                gr.update(interactive=not busy), load)

    try:
        conditions = [list(values[:6]), list(values[6:12])]
        shared = list(values[12:])
        occupied = runtime.MANAGER.occupant
        if occupied:
            yield (skip, skip, f"Wait until the model is free; it is {occupied}.", skip, skip, skip, skip, skip)
            return
        initial = runtime.MANAGER.loaded_model()
        kinds = []
        # Resolve blank model IDs once, before A changes what is in memory.
        for condition in conditions:
            condition[0] = (condition[0] or "").strip() or initial.model_id
            if not condition[0]:
                raise ValueError("Choose a model for both conditions or load one first.")
            if condition[4] and not shared[14]:
                raise ValueError("Import a steering vector in Chat before running this preset.")
            if int(condition[3]) < 0:
                raise ValueError("Seeds must be zero or greater.")
            status = cache_status(condition[0])
            if not status.present or status.missing_files or status.unsupported or status.kind not in (TEXT_KIND, MLX_KIND):
                raise ValueError(f"Download a supported text model first: {condition[0]}.")
            if status.kind == MLX_KIND and condition[1] != "Current":
                raise ValueError("MLX models use the precision stored in their repository. Choose Current or compare two different conversions.")
            if condition[1] == "Current":
                condition[1] = initial.precision if initial.precision in settings.WEIGHT_PRECISIONS else "full"
            kinds.append(status.kind)
        yield frame("Starting comparison…")
        for side, condition, kind in zip(("A", "B"), conditions, kinds, strict=True):
            model, precision, temperature, seed, enabled, strength = condition
            current = runtime.MANAGER.loaded_model()
            # Full weights are reported by dtype; quantized weights by their width.
            same_precision = kind == MLX_KIND or current.precision == precision or (
                precision == "full" and current.precision not in ("4-bit", "8-bit"))
            if current.model_id != model or not same_precision:
                claimed, occupied = runtime.MANAGER.claim_exclusive_load(model)
                if claimed is None:
                    raise ValueError(f"Cannot load condition {side}: the model is {occupied}.")
                try:
                    path = runtime.MANAGER.find_cached(model)
                    with contextlib.closing(stream_load(model, path, precision, kind)) as loading:
                        for card in loading:
                            yield frame(f"Loading condition {side}…", load=card)
                finally:
                    runtime.MANAGER.release_load(claimed[1])
            current = runtime.MANAGER.loaded_model()
            if current.model_id != model or current.load_id is None:
                raise ValueError(f"Condition {side}'s model did not load.")
            if kind != MLX_KIND and precision in ("4-bit", "8-bit") and current.precision != precision:
                raise ValueError(f"Condition {side} could not use {precision} weights on this device. Choose full precision or a supported quantized model.")
            args = list(shared)
            args[6], args[11], args[12] = temperature, seed, False
            args[15], args[16] = enabled, strength
            completed = None
            with contextlib.closing(fill_slot(side, *args, expected_load_id=current.load_id)) as filling:
                for result, status, *_ in filling:
                    if isinstance(result, dict) and result.get("metrics"):
                        completed = result
                    else:
                        yield frame(status)
            if completed is None:
                yield frame(f"Condition {side} did not finish. {status} Completed results are kept.", busy=False)
                return
            completed["requested_precision"] = precision
            # Persist before yielding, so Stop at the completed frame cannot
            # discard a run that the interface already reported as saved.
            try:
                experiment_runs.save(completed, f"Condition {side}: {completed['prompt'][:65]}")
            except (OSError, ValueError) as error:
                yield frame(failure_status(f"Condition {side} finished but could not be saved", str(error)),
                            busy=False, **{"left" if side == "A" else "right": completed})
                return
            yield frame(f"Condition {side} complete and saved.", **{"left" if side == "A" else "right": completed})
        yield frame("Comparison complete. Both runs are saved in Experiments.", busy=False, load="")
    except (OSError, ValueError, RuntimeError) as error:
        yield frame(failure_status("Comparison stopped; completed results are kept", str(error)), busy=False)
    finally:
        _PAIR_LOCK.release()
