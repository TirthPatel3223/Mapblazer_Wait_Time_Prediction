"""Canonical entity keys.

There must be exactly ONE way to turn a human ride name into a key. The v1 codebase had
two: the training scripts used a regex that stripped punctuation, while the evaluation
scripts used a plain space/slash replacement. Any ride whose name contained an
apostrophe, colon, comma, '!' or '&' therefore hashed to a path the training run had
never written, the lookup fell into a bare `except: pass`, and the ride was silently
dropped from every reported metric.

43 of 120 rides were being discarded that way -- and they were disproportionately the
marquee attractions (Rise of the Resistance, Guardians, Toy Story Mania), which carry
roughly twice the mean wait of the rides that survived.

This module is the single implementation. Import it; never re-derive a key inline.
"""

from __future__ import annotations

import re

_STRIP_PUNCT = re.compile(r"[^\w\s-]")


def canonical_ride_key(name: str | None) -> str:
    """Return the canonical key for an attraction name.

    Drops punctuation, trims, converts whitespace runs to single underscores.

        >>> canonical_ride_key("Guardians of the Galaxy - Mission: BREAKOUT!")
        'Guardians_of_the_Galaxy_-_Mission_BREAKOUT'
        >>> canonical_ride_key("Soarin' Around the World")
        'Soarin_Around_the_World'
        >>> canonical_ride_key("CraZanity ")
        'CraZanity'
    """
    if not name:
        return ""
    safe = _STRIP_PUNCT.sub("", str(name)).strip().replace(" ", "_")
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe


def canonical_park_key(name: str | None) -> str:
    """Return the canonical key for a park name.

    Parks avoided the v1 bug because their names carry no punctuation, but they get the
    same treatment so that a future park with an ampersand cannot reintroduce it.
    """
    return canonical_ride_key(name)


def entity_key(park: str | None, ride: str | None) -> str:
    """Fully-qualified key. Two parks may legitimately share a ride name."""
    return f"{canonical_park_key(park)}::{canonical_ride_key(ride)}"
