# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for ``ws_to_map_graph_data`` (World tab — Map sub-tab)."""
from __future__ import annotations

from example_worlds.macbeth import world_state as macbeth_ws
from shadow_loom.models import (
    Affordance,
    EventNode,
    NarrativeObject,
    ObjectStateSnapshot,
)

from shadow_loom_ui.viz_helpers import ws_to_map_graph_data


def _categories(nodes):
    return {n["id"]: n["category"] for n in nodes}


def test_map_emits_three_categories():
    nodes, links, cats = ws_to_map_graph_data(macbeth_ws, fabula_anchor=10_000)
    # PR 7 of EventNode.at_location_id added an "Event" category for
    # the in-window event glyphs; the original three categories must
    # still appear at the same indexes for back-compat with existing
    # node ``category`` ints.
    assert [c["name"] for c in cats] == [
        "Location", "Entity", "Object", "Event (★)",
    ]
    seen = {c["name"] for c in cats}
    cat_index = {c["name"]: i for i, c in enumerate(cats)}
    by_cat = {n["category"] for n in nodes}
    # Macbeth has at least one entity, location, object.
    assert cat_index["Location"] in by_cat
    assert cat_index["Entity"] in by_cat
    assert cat_index["Object"] in by_cat
    assert {"Location", "Entity", "Object"}.issubset(seen)


def test_map_layer_toggles_drop_categories():
    nodes_full, _, _ = ws_to_map_graph_data(macbeth_ws, fabula_anchor=10_000)
    nodes_no_obj, _, _ = ws_to_map_graph_data(
        macbeth_ws, fabula_anchor=10_000, show_objects=False,
    )
    nodes_no_ent, _, _ = ws_to_map_graph_data(
        macbeth_ws, fabula_anchor=10_000, show_entities=False, show_objects=False,
    )
    assert any(n["category"] == 2 for n in nodes_full)
    assert not any(n["category"] == 2 for n in nodes_no_obj)
    assert not any(n["category"] in (1, 2) for n in nodes_no_ent)


def test_map_locked_edges_can_be_hidden():
    # Find any locked spatial edge in the fixture.
    has_locked = any(se.is_locked for se in macbeth_ws.spatial_topology)
    if not has_locked:
        return  # nothing to assert for this fixture
    _, links_show, _ = ws_to_map_graph_data(
        macbeth_ws, fabula_anchor=10_000, show_locked=True,
    )
    _, links_hide, _ = ws_to_map_graph_data(
        macbeth_ws, fabula_anchor=10_000, show_locked=False,
    )
    n_dashed_show = sum(
        1 for l in links_show
        if l.get("lineStyle", {}).get("type") == "dashed"
        and not l.get("_sl_attractor")
        and not l.get("_sl_channel_arc")
    )
    n_dashed_hide = sum(
        1 for l in links_hide
        if l.get("lineStyle", {}).get("type") == "dashed"
        and not l.get("_sl_attractor")
        and not l.get("_sl_channel_arc")
    )
    assert n_dashed_hide < n_dashed_show


def test_map_object_follows_owner_through_snapshots():
    """Object held by entity at t=10 should anchor to that entity, not its initial location."""
    ws = macbeth_ws.model_copy(deep=True)
    # Inject a fresh narrative object that starts in a location, then
    # gets picked up by an entity at fabula_time=5.
    locs = list(ws.locations.keys())
    ents = list(ws.entities.keys())
    assert locs and ents
    loc_id, ent_id = locs[0], ents[0]
    obj = NarrativeObject(
        id="OBJ_TEST_RELIC",
        name="Test Relic",
        location_id=loc_id,
        owner_id=None,
        properties={},
        affordances=[Affordance(action="pick_up", target_type="Entity")],
        state_timeline=[
            ObjectStateSnapshot(
                fabula_time=5,
                triggered_by=None,
                owner_id=ent_id,
                set_location_null=True,
            ),
        ],
    )
    ws.objects[obj.id] = obj

    # Before pickup: anchored to location.
    nodes_before, links_before, _ = ws_to_map_graph_data(ws, fabula_anchor=4)
    obj_link_before = next(
        (l for l in links_before if l.get("target") == obj.id), None,
    )
    assert obj_link_before is not None
    assert obj_link_before["source"] == loc_id

    # After pickup: anchored to the owning entity.
    nodes_after, links_after, _ = ws_to_map_graph_data(ws, fabula_anchor=10_000)
    obj_link_after = next(
        (l for l in links_after if l.get("target") == obj.id), None,
    )
    assert obj_link_after is not None
    assert obj_link_after["source"] == ent_id


def test_map_channel_arc_only_on_utterance_tick():
    """A channel arc must only appear when an utterance fires within \u00b1window of cursor."""
    ws = macbeth_ws.model_copy(deep=True)
    ents = list(ws.entities.keys())
    assert len(ents) >= 2
    speaker, addressee = ents[0], ents[1]
    utt = EventNode(
        id="EVT_TEST_UTTERANCE",
        fabula_time=500,
        syuzhet_index=999,
        event_type="utterance",
        actor_ids=[speaker],
        target_ids=[],
        description="test utterance",
        content="hello",
        speaker_id=speaker,
        addressee_ids=[addressee],
        truth_value="true",
    )
    ws.events.append(utt)

    # Cursor on the utterance tick: arc present.
    _, links_on, _ = ws_to_map_graph_data(ws, fabula_anchor=500, channel_window=0)
    assert any(
        l.get("_sl_channel_arc") and l.get("source") == speaker and l.get("target") == addressee
        for l in links_on
    )

    # Cursor off the utterance tick (window=0): no arc.
    _, links_off, _ = ws_to_map_graph_data(ws, fabula_anchor=499, channel_window=0)
    assert not any(
        l.get("_sl_channel_arc") and l.get("source") == speaker and l.get("target") == addressee
        for l in links_off
    )

    # Cursor off-tick but within widened window: arc reappears.
    _, links_widen, _ = ws_to_map_graph_data(ws, fabula_anchor=499, channel_window=2)
    assert any(
        l.get("_sl_channel_arc") and l.get("source") == speaker and l.get("target") == addressee
        for l in links_widen
    )


# =====================================================================
# Event glyphs + co-presence highlights (PR 7 of EventNode.at_location_id)
# =====================================================================


def _build_copresence_ws():
    from shadow_loom.models import (
        Entity, EntityStateSnapshot, EventNode, Location, TraitVector,
        WorldStateV1,
    )
    return WorldStateV1(
        locations={
            "LOC_BED": Location(
                id="LOC_BED",
                name="Bedchamber", description="d", ambient_state={}),
            "LOC_HALL": Location(
                id="LOC_HALL",
                name="Hall", description="d", ambient_state={}),
        },
        objects={},
        entities={
            "ENT_M": Entity(
                id="ENT_M", name="Macbeth", location_id="LOC_BED",
                status="healthy",
                traits={"x": TraitVector(value=0.5, inertia=0.5)},
                state_timeline=[],
            ),
            "ENT_D": Entity(
                id="ENT_D", name="Duncan", location_id="LOC_BED",
                status="healthy",
                traits={"x": TraitVector(value=0.5, inertia=0.5)},
                state_timeline=[],
            ),
            # Banquo lives in the Hall — phantom-witness candidate.
            "ENT_B": Entity(
                id="ENT_B", name="Banquo", location_id="LOC_HALL",
                status="healthy",
                traits={"x": TraitVector(value=0.5, inertia=0.5)},
                state_timeline=[],
            ),
        },
        events=[
            EventNode(
                id="EVT_MURDER", fabula_time=10, syuzhet_index=10,
                event_type="outcome", actor_ids=["ENT_M"],
                target_ids=["ENT_D"], description="murder",
                at_location_id="LOC_BED",
            ),
        ],
        causal_topology=[],
        world_traits={},
    )


def test_map_emits_event_glyph_at_anchor():
    ws = _build_copresence_ws()
    nodes, links, cats = ws_to_map_graph_data(ws, fabula_anchor=10)
    cat_event = next(i for i, c in enumerate(cats) if c["name"].startswith("Event"))
    glyphs = [n for n in nodes if n.get("category") == cat_event]
    assert len(glyphs) == 1
    assert glyphs[0]["name"] == "\u2605"
    # The glyph is anchored to the event's location via an attractor link.
    assert any(
        l.get("source") == "LOC_BED"
        and l.get("target") == glyphs[0]["id"]
        for l in links
    )


def test_map_event_glyph_disabled_when_show_events_false():
    ws = _build_copresence_ws()
    nodes, _, cats = ws_to_map_graph_data(ws, fabula_anchor=10, show_events=False)
    cat_event = next(i for i, c in enumerate(cats) if c["name"].startswith("Event"))
    assert not any(n.get("category") == cat_event for n in nodes)


def test_map_bound_participants_get_yellow_border():
    ws = _build_copresence_ws()
    nodes, _, _ = ws_to_map_graph_data(ws, fabula_anchor=10)
    macbeth = next(n for n in nodes if n["id"] == "ENT_M")
    duncan = next(n for n in nodes if n["id"] == "ENT_D")
    assert macbeth["itemStyle"]["borderColor"] == "#facc15"
    assert duncan["itemStyle"]["borderColor"] == "#facc15"


def test_map_event_outside_window_skipped():
    ws = _build_copresence_ws()
    nodes, _, cats = ws_to_map_graph_data(ws, fabula_anchor=100, event_window=0)
    cat_event = next(i for i, c in enumerate(cats) if c["name"].startswith("Event"))
    assert not any(n.get("category") == cat_event for n in nodes)
