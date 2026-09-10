"""Small offline starter catalog, with metadata checked against HF on 2026-09-09.

Download estimates cover the entire snapshot: ChatLab currently downloads all
repository files, including alternate formats. Keep these separate from loaded
weight sizes. Model IDs link to the source cards in the UI.
"""

from model_runtime import HubModel, TEXT_KIND, IMAGE_KIND


DISCOVERY_ORDERS = ("Recommended", "Popular", "Trending", "New")
STARTER_MODELS = {
    TEXT_KIND: (
        HubModel(
            model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
            parameters=134_515_008,
            pipeline_tag="text-generation",
            library="transformers",
            license="apache-2.0",
            summary="Smallest starter — explore token predictions with a compact chat model.",
            download_bytes=1_969_841_187,
        ),
        HubModel(
            model_id="Qwen/Qwen3-0.6B",
            parameters=751_632_384,
            pipeline_tag="text-generation",
            library="transformers",
            license="apache-2.0",
            summary="Compact reasoning — try a thinking model with modest memory needs.",
            download_bytes=1_519_209_243,
        ),
        HubModel(
            model_id="allenai/Olmo-3-7B-Think",
            parameters=7_298_011_136,
            pipeline_tag="text-generation",
            library="transformers",
            license="apache-2.0",
            summary="Deeper exploration — ChatLab’s default reasoning model; needs more memory.",
            download_bytes=14_605_886_999,
        ),
    ),
    IMAGE_KIND: (
        HubModel(
            model_id="stabilityai/sd-turbo",
            pipeline_tag="text-to-image",
            library="diffusers",
            license="sai-nc-community",
            summary="Quick image experiments — start with 1 step, guidance 0, and 512 × 512 pixels.",
            download_bytes=12_957_331_338,
        ),
    ),
}


def recommended_models(query: str, kind: str) -> list[HubModel]:
    """Filter a bundled shortlist without contacting the Hub."""

    words = query.casefold().split()
    return [
        model for model in STARTER_MODELS.get(kind, ())
        if all(word in f"{model.model_id} {model.summary}".casefold() for word in words)
    ]
