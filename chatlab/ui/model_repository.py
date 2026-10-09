"""Check a model's Hub metadata without downloading its weights."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from chatlab import mlx_runtime
from chatlab.model_cache import (
    validate_model_id,
)


CONFIG_PROBE_MAX_BYTES = 1024 * 1024


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
    config_size = next((file.size for file in files if file.rfilename == "config.json"), None)
    config_note = "Configuration could not be verified."
    probe_config = isinstance(config_size, int) and 0 <= config_size <= CONFIG_PROBE_MAX_BYTES
    if config_size is None:
        config_note = "Configuration was not checked because its download size is unknown."
    elif not probe_config:
        config_note = "Configuration was not checked because it exceeds the 1 MiB preview limit."
    if probe_config:
        try:
            # A config in the normal Hub cache would make this metadata check
            # appear as an incomplete model download in the local inventory.
            with TemporaryDirectory(prefix="chatlab-model-check-") as cache:
                config_path = hf_hub_download(
                    cleaned, "config.json", revision=info.sha, token=token,
                    etag_timeout=10, cache_dir=cache,
                )
                with Path(config_path).open("rb") as config_file:
                    content = config_file.read(CONFIG_PROBE_MAX_BYTES + 1)
                if len(content) > CONFIG_PROBE_MAX_BYTES:
                    config_note = "Configuration was not checked because it exceeds the 1 MiB preview limit."
                    raise ValueError("Configuration exceeds the preview limit")
                loaded = json.loads(content)
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
    architecture_unavailable = False
    if is_mlx:
        format_name = f"MLX · {bits}-bit weights"
        supported = mlx_runtime.mlx_supports(config.get("model_type"))
        architecture_unavailable = not supported
        compatibility = (
            "Text chat architecture supported by the installed MLX runtime. Loading has not been tested."
            if supported else "This MLX architecture is unavailable in the installed runtime. You can download the files, but ChatLab cannot load them in this installation."
        )
    elif info.library_name == "diffusers":
        format_name = "Image model"
        compatibility = "Use the Images page after loading."
    elif "adapter_config.json" in filenames and "config.json" not in filenames:
        format_name = "LoRA adapter"
        compatibility = (
            "Downloading also fetches the base model the adapter was trained on. "
            "The adapter is merged into that model's full-precision weights, so it "
            "needs the memory of the whole base model."
        )
    else:
        format_name = "Transformers" if info.library_name == "transformers" or (mlx_tagged and config is not None) else "Format not confirmed"
        compatibility = "Repository existence is confirmed; loading compatibility has not been tested."
    if config is None and "config.json" in filenames:
        compatibility += " " + config_note
    if unsupported:
        compatibility += " No supported weight files found. ChatLab cannot load GGUF-only or other exported formats."
    yield {
        **result, "status": "found", "format": format_name, "mlx": is_mlx,
        "bits": bits, "download_bytes": total, "gated": bool(info.gated),
        "private": bool(info.private), "unsupported": unsupported,
        "architecture_unavailable": architecture_unavailable,
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
