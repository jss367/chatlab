"""The blocking tradeoff, drawn as plain SVG in the app's own chart styles."""
import html

_WIDTH = 640.0
_HEIGHT = 240.0
_PAD_LEFT = 42.0
_PAD_RIGHT = 12.0
_PAD_TOP = 16.0
_PAD_BOTTOM = 32.0

EMPTY = ('<div class="viz-empty">The blocking tradeoff appears after a probability '
         'run over labeled cases, or after importing predictions that carry '
         'per-label probabilities.</div>')


def _x(rate):
    return _PAD_LEFT + rate * (_WIDTH - _PAD_LEFT - _PAD_RIGHT)


def _y(rate):
    return _HEIGHT - _PAD_BOTTOM - rate * (_HEIGHT - _PAD_TOP - _PAD_BOTTOM)


def tradeoff_chart(blocking):
    """Unsafe actions caught against ordinary actions blocked, at every threshold.

    Both classes have to be present for the picture to mean anything: a set
    with no unsafe case has no recall to trade, and one with nothing else has
    nothing to lose. The operating point marked is the threshold with the
    widest gap between the two rates, which is where a deployer would start.
    """
    if not blocking or blocking.get("auc") is None:
        return EMPTY
    curve = blocking["curve"]
    path = " ".join(
        f"{'M' if index == 0 else 'L'}{_x(point['false_block_rate']):.1f},{_y(point['unsafe_recall']):.1f}"
        for index, point in enumerate(curve)
    )
    grid = "".join(
        f'<line class="viz-grid" x1="{_PAD_LEFT}" x2="{_WIDTH - _PAD_RIGHT}" '
        f'y1="{_y(value):.1f}" y2="{_y(value):.1f}" />'
        f'<text class="viz-tick" x="{_PAD_LEFT - 6}" y="{_y(value) + 3.5:.1f}" '
        f'text-anchor="end">{value:.0%}</text>'
        for value in (0.0, 0.25, 0.5, 0.75, 1.0)
    )
    ticks = "".join(
        f'<text class="viz-tick" x="{_x(value):.1f}" y="{_HEIGHT - 14:.1f}" '
        f'text-anchor="middle">{value:.0%}</text>'
        for value in (0.0, 0.25, 0.5, 0.75, 1.0)
    )
    point = blocking["operating_point"]
    # The curve begins at a point no threshold in [0, 1] expresses, because an
    # inclusive comparison blocks a case certain of unsafe even at 1.0. Say what
    # it does rather than print a number the slider cannot be moved to.
    label = ("blocks nothing" if point["threshold"] is None
             else f'P(unsafe) ≥ {point["threshold"]:.2f}')
    marker = (
        f'<circle class="viz-peak-dot" cx="{_x(point["false_block_rate"]):.1f}" '
        f'cy="{_y(point["unsafe_recall"]):.1f}" r="4" />'
        f'<text class="viz-peak-label" x="{_x(point["false_block_rate"]) + 8:.1f}" '
        f'y="{max(_y(point["unsafe_recall"]) - 8, 12):.1f}">'
        f'{label}</text>'
    )
    caption = (f"{blocking['unsafe']:,} unsafe and {blocking['other']:,} other cases · "
               f"AUC {blocking['auc']:.3f}")
    return (
        '<figure class="viz-root" id="blocking-tradeoff">'
        f'<figcaption class="viz-title">Unsafe actions caught against ordinary actions blocked'
        f'<span class="viz-sub">{html.escape(caption)}</span></figcaption>'
        f'<svg viewBox="0 0 {_WIDTH:g} {_HEIGHT:g}" role="img" '
        f'aria-label="{html.escape(caption)}">'
        f'{grid}'
        f'<line class="viz-line-faint" x1="{_x(0):.1f}" y1="{_y(0):.1f}" '
        f'x2="{_x(1):.1f}" y2="{_y(1):.1f}" />'
        f'<path class="viz-line" d="{path}" />'
        f'{marker}'
        f'<line class="viz-axis" x1="{_PAD_LEFT}" x2="{_WIDTH - _PAD_RIGHT}" '
        f'y1="{_y(0):.1f}" y2="{_y(0):.1f}" />'
        f'{ticks}'
        '</svg>'
        '<div class="viz-note">Vertical: share of unsafe actions blocked. '
        'Horizontal: share of allowed and unrelated actions blocked with them. '
        'The faint diagonal is what a coin would score.</div>'
        '</figure>'
    )
