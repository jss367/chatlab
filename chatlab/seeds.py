"""The seed a generation is handed, whoever asked for it.

A module of its own, below the interface, because the interface is not the
only thing that picks one: the local API answers requests with the reader's
saved sampling settings, a randomized seed included, and it has no business
importing a page to do it. Both read the same rule from here, so a seed the
API chose and a seed the Chat page chose can be compared as the same kind of
number.
"""

from __future__ import annotations

import random


# The ceiling on a seed ChatLab picks for itself. A seed the reader locks is
# only floored (see :func:`resolve_seed`), so this bounds what "randomize"
# draws rather than what a reader may type.
SEED_LIMIT = 2**31 - 1


def resolve_seed(seed, randomize: bool) -> int:
    """Pick the seed for one generation, inside the range NumPy will accept.

    ``np.random.default_rng()`` rejects negative integers, so a locked seed of
    ``-1`` used to fail every generation with "expected non-negative integer"
    and produce no reply at all. The number input is constrained to 0 and above,
    but the clamp lives here as well: this is the only place the value is turned
    into the one the generator is handed, and it can still arrive out of range
    from the API, from a browser that ignores the constraint, or from a float
    the input rounded. Non-numeric and missing values keep falling back to 0.
    """

    if randomize:
        return random.randrange(SEED_LIMIT)
    try:
        # OverflowError covers infinities, which int() refuses to convert.
        value = int(seed)
    except (OverflowError, TypeError, ValueError):
        return 0
    return max(value, 0)
