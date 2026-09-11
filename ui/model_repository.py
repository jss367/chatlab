"""Check a model's Hub metadata without downloading its weights."""

from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import gradio as gr

import mlx_runtime
from model_runtime import cache_folder, format_bytes, mlx_snapshot_bits, snapshot_folder, validate_model_id


UNCHECKED = "**Repository not checked** · Choose **Check model** to verify this ID on Hugging Face."


def token_scope(hf_token: str | None) -> str:
    """Identify the request credentials without retaining the token in results."""

    return hashlib.sha256((hf_token or "").strip().encode()).hexdigest()


def check_model_repository(model_id: str, hf_token: str | None):
    """Scope results to the ID and credentials used for the request."""

    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError
    import httpx

    cleaned = (model_id or "").strip()
    token = (hf_token or "").strip() or None
    result = {"model_id": cleaned, "token_scope": token_scope(token)}
    try:
        validate_model_id(cleaned)
    except ValueError:
        yield {**result, "status": "invalid", "detail": "Enter an ID in organization/model-name format."}
        return
    yield {**result, "status": "checking", "detail": "Checking Hugging Face…"}
    try:
        info = HfApi().model_info(
            cleaned, token=token,
            files_metadata=True, timeout=10,
        )
    except GatedRepoError:
        yield {**result, "status": "restricted", "detail": "Repository found · Access required. Accept its terms on Hugging Face and enter your token under Access token."}
        return
    except RepositoryNotFoundError:
        # The Hub deliberately does not distinguish private repos from absent ones.
        yield {**result, "status": "missing", "detail": "Repository not found or private. Check the spelling, or enter a token under Access token and check again."}
        return
    except HfHubHTTPError as error:
        code = getattr(error.response, "status_code", None)
        detail = (
            "Access could not be verified. Check your token under Access token and try again."
            if code in (401, 403)
            else "Could not reach Hugging Face. Try Check model again; downloaded models can still load from disk."
        )
        yield {**result, "status": "error", "detail": detail}
        return
    except (httpx.HTTPError, OSError):
        yield {**result, "status": "error", "detail": "Could not reach Hugging Face. Try Check model again; downloaded models can still load from disk."}
        return

    files = info.siblings or []
    sizes = [file.size for file in files]
    total = sum(sizes) if sizes and all(size is not None for size in sizes) else None
    filenames = [file.rfilename for file in files]
    # model_info exposes only a subset of config.json, omitting quantization.
    # Inspect the small config at the same revision, never the model weights.
    config = None
    access_restricted = False
    if "config.json" in filenames:
        try:
            # A config in the normal Hub cache would make this metadata check
            # appear as an incomplete model download in the local inventory.
            with TemporaryDirectory(prefix="chatlab-model-check-") as cache:
                config_path = hf_hub_download(
                    cleaned, "config.json", revision=info.sha, token=token,
                    etag_timeout=10, cache_dir=cache,
                )
                loaded = json.loads(Path(config_path).read_text())
            if isinstance(loaded, dict):
                config = loaded
        except GatedRepoError:
            access_restricted = True
        except HfHubHTTPError as error:
            access_restricted = getattr(error.response, "status_code", None) in (401, 403)
        except (httpx.HTTPError, OSError, ValueError):
            # Repository existence was already confirmed. A failed config
            # lookup must not turn it into a missing or inaccessible repo.
            pass
    quant = mlx_runtime.mlx_quantization(config)
    is_mlx = bool(quant and config is not None and "model_type" in config)
    bits = quant["bits"] if is_mlx else None
    mlx_tagged = info.library_name == "mlx" or "mlx" in (info.tags or [])
    unsupported = bool(filenames) and all(
        not name.endswith((".safetensors", ".bin")) for name in filenames
    )
    if is_mlx:
        format_name = f"MLX · {bits}-bit weights"
        supported = mlx_runtime.mlx_supports(config.get("model_type"))
        compatibility = (
            "Text chat architecture supported by the installed MLX runtime. Loading has not been tested."
            if supported else "MLX architecture support could not be confirmed in this installation."
        )
    elif info.library_name == "diffusers":
        format_name = "Image model"
        compatibility = "Use the Images page after loading."
    else:
        format_name = "Transformers" if info.library_name == "transformers" or (mlx_tagged and config is not None) else "Format not confirmed"
        compatibility = "Repository existence is confirmed; loading compatibility has not been tested."
    if config is None and "config.json" in filenames:
        compatibility += " Configuration could not be verified."
    if unsupported:
        compatibility = "No supported weight files found. ChatLab cannot load GGUF-only or other exported formats."
    yield {
        **result, "status": "found", "format": format_name, "mlx": is_mlx,
        "bits": bits, "download_bytes": total, "gated": bool(info.gated),
        "private": bool(info.private), "unsupported": unsupported,
        "access_restricted": access_restricted, "config_verified": config is not None,
        "compatibility": compatibility,
    }


def matching_repository(model_id: str, result: dict | None, hf_token: str | None = None) -> dict:
    if (
        result and result.get("model_id") == (model_id or "").strip()
        and result.get("token_scope") == token_scope(hf_token)
    ):
        return result
    return {}


def repository_view(
    model_id: str, result: dict | None, hf_token: str | None = None,
    selected: str | None = None,
):
    """Render results only for the current model ID and credentials."""

    # Match the load actions: a cached row takes precedence while its ID is
    # still being copied into the textbox. Local evidence needs no Hub access.
    chosen = (selected or model_id or "").strip()
    result = matching_repository(chosen, result, hf_token)
    try:
        snapshot = snapshot_folder(cache_folder(chosen)) if chosen else None
        local_bits = mlx_snapshot_bits(snapshot) if snapshot is not None else None
    except (OSError, ValueError):
        local_bits = None
    local_note = (
        f"\n\n**Cached checkpoint:** MLX · {local_bits}-bit weights. Precision is fixed by the cached checkpoint."
        if local_bits is not None else ""
    )
    precision = gr.update(visible=not (local_bits is not None or result.get("mlx", False)))
    if not result:
        return UNCHECKED + local_note, precision
    status = result["status"]
    if status != "found":
        return html.escape(result["detail"]) + local_note, precision
    name = html.escape(result["model_id"])
    lines = [f"**Repository found** · [View on Hugging Face](https://huggingface.co/{name})"]
    size = result.get("download_bytes")
    lines.append(f"**{result['format']}** · " + (
        f"{format_bytes(size)} of repository files" if size is not None else "Download size unavailable"
    ))
    lines.append(result["compatibility"])
    if result.get("mlx") and result.get("bits"):
        lines.append("Precision is fixed by this checkpoint; no extra quantization is needed.")
    if result.get("access_restricted"):
        lines.append("**Access required:** accept the model's terms on Hugging Face and provide an authorized token under **Access token**.")
    elif result.get("gated"):
        lines.append(
            "Gated repository · Access to the configuration was verified."
            if result.get("config_verified") else "Gated repository · File access has not been verified."
        )
    elif result.get("private"):
        lines.append("Private repository · Your saved or entered token provided access.")
    return "\n\n".join(lines) + local_note, precision
