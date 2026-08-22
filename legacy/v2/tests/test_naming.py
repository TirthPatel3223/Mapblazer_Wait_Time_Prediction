"""Regression tests for the entity-resolution defect.

v1 sanitised ride names two different ways. Training wrote artefacts under a regex that
stripped punctuation; the evaluation and KPI scripts looked them up with a plain
space/slash replacement. Names containing an apostrophe, colon, comma, '!' or '&'
therefore never resolved, were swallowed by a bare `except: pass`, and vanished from
every reported metric -- 43 of 120 rides, and the highest-wait ones at that.

These tests pin the surviving implementation and prove the previously-lost rides resolve.
"""

import re

import pytest

from themepark.naming import canonical_park_key, canonical_ride_key, entity_key


def legacy_training_sanitizer(name: str) -> str:
    """Verbatim from v1 train_prophet_models.py / train_xgboost_global.py."""
    if not name:
        return ""
    safe = re.sub(r"[^\w\s-]", "", name).strip().replace(" ", "_")
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe


def legacy_eval_sanitizer(name: str) -> str:
    """Verbatim from v1 evaluate_test_data.py / calc_kpis.py -- the broken one."""
    return name.replace(" ", "_").replace("/", "_")


# The marquee attractions that v1 silently dropped, with their raw source names.
PREVIOUSLY_DROPPED_RIDES = [
    "Guardians of the Galaxy - Mission: BREAKOUT!",
    "Soarin' Around the World",
    "Star Wars: Rise of the Resistance",
    "Toy Story Midway Mania!",
    "WEB SLINGERS: A Spider-Man Adventure",
    "Tiana's Bayou Adventure",
    "Monsters, Inc. Mike & Sulley to the Rescue",
    "Goofy's Sky School",
    "Peter Pan's Flight",
    "Mickey & Minnie's Runaway Railway",
    "Pirates' Lair on Tom Sawyer Island",
    "Davy Crockett's Explorer Canoes",
    "Snow White's Enchanted Wish",
    "Roger Rabbit's Car Toon Spin",
    "Mr. Toad's Wild Ride",
    "Pinocchio's Daring Journey",
    "it's a small world Holiday",
    "Chip 'n Dale's GADGETcoaster",
    "The Simpson's Ride",
    "Jessie's Critter Carousel",
]


@pytest.mark.parametrize("name", PREVIOUSLY_DROPPED_RIDES)
def test_previously_dropped_rides_now_resolve(name):
    """Each of these produced a lookup miss in v1. All must now agree with training."""
    assert canonical_ride_key(name) == legacy_training_sanitizer(name)


@pytest.mark.parametrize("name", PREVIOUSLY_DROPPED_RIDES)
def test_these_are_exactly_the_names_the_legacy_bug_lost(name):
    """Guard the premise: if these agreed under v1, the bug story would be wrong."""
    assert legacy_eval_sanitizer(name) != legacy_training_sanitizer(name)


def test_canonical_key_is_the_training_convention():
    """Existing artefacts were written under the training regex; stay compatible."""
    for name in [*PREVIOUSLY_DROPPED_RIDES, "Space Mountain", "Jungle Cruise", "Autopia"]:
        assert canonical_ride_key(name) == legacy_training_sanitizer(name)


def test_known_keys():
    assert canonical_ride_key("Guardians of the Galaxy - Mission: BREAKOUT!") == (
        "Guardians_of_the_Galaxy_-_Mission_BREAKOUT"
    )
    assert canonical_ride_key("Soarin' Around the World") == "Soarin_Around_the_World"
    assert canonical_ride_key("Monsters, Inc. Mike & Sulley to the Rescue") == (
        "Monsters_Inc_Mike_Sulley_to_the_Rescue"
    )


def test_whitespace_is_normalised():
    """'CraZanity ' with its trailing space is real, and so are double spaces."""
    assert canonical_ride_key("CraZanity ") == "CraZanity"
    assert canonical_ride_key("  Space   Mountain  ") == "Space_Mountain"


def test_idempotent():
    """Keying an already-keyed name must not change it, or round-tripping corrupts."""
    for name in PREVIOUSLY_DROPPED_RIDES:
        once = canonical_ride_key(name)
        assert canonical_ride_key(once) == once


def test_empty_and_none():
    assert canonical_ride_key("") == ""
    assert canonical_ride_key(None) == ""


def test_entity_key_disambiguates_shared_ride_names():
    """Two parks can legitimately run a ride with the same name."""
    a = entity_key("Disneyland", "Space Mountain")
    b = entity_key("Disney California Adventure Park", "Space Mountain")
    assert a != b
    assert canonical_park_key("Disney California Adventure Park") in b
