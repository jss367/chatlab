"""Thinking controls for Qwen3 checkpoints with a switchable chat template."""

from jinja2 import Environment, TemplateError, meta

THINKING_MODES = ("default", "on", "off")
THINKING_CHOICES = [("Model default", "default"), ("On", "on"), ("Off", "off")]


def supports_thinking(model, tokenizer) -> bool:
    """Require both a supported architecture and a template that accepts the flag.

    The original Qwen3 templates use ``enable_thinking``. Later thinking-only
    and instruct-only releases must not inherit a switch from their name.
    MLX keeps the architecture on ``args`` instead of ``config``.
    """

    config = getattr(model, "config", None)
    if config is None:
        config = getattr(model, "args", None)
    model_type = getattr(config, "model_type", None)
    if model_type not in ("qwen3", "qwen3_moe"):
        return False
    template = getattr(tokenizer, "chat_template", None)
    if isinstance(template, dict):
        template = template.get("default")
    if not isinstance(template, str):
        return False
    try:
        return "enable_thinking" in meta.find_undeclared_variables(Environment().parse(template))
    except TemplateError:
        return False
