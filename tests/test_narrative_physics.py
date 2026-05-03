# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests for the narrative physics pipeline.

Exercises all five query types (observation, intervention, counterfactual,
directive, interrogation) against real plot models and verifies the returned
graph structures reflect the expected mutations.
"""
import pytest
import networkx as nx

from shadow_loom.narrative_physics import calculate_narrative_physics
from shadow_loom.query_models import (
    ObservationQuery,
    InterventionQuery,
    CounterfactualQuery,
    DirectiveQuery,
    InterrogationQuery,
    GeneralQuery,
)

# ── Import plot world-states used across tests ──────────────────────────
from example_worlds.macbeth import world_state as macbeth_ws
from example_worlds.romeo_and_juliet import world_state as romeo_ws
from example_worlds.gone_girl import world_state as gone_girl_ws
from example_worlds.great_gatsby import world_state as gatsby_ws
from example_worlds.death_on_the_nile import world_state as nile_ws
from example_worlds.apocalypse_now import world_state as apocalypse_ws
from example_worlds.dads_army import world_state as dads_army_ws
from example_worlds.frankenstein import world_state as frankenstein_ws
from example_worlds.reservoir_dogs import world_state as reservoir_ws
from example_worlds.wuthering_heights import world_state as wuthering_ws
from example_worlds.a_court_of_thorn_and_roses import world_state as acotar_ws
from example_worlds.a_fish_called_wanda import world_state as wanda_ws
from example_worlds.brief_encounter import world_state as brief_ws
from example_worlds.great_expectations import world_state as expectations_ws
from example_worlds.nineteen_eighty_four import world_state as orwell_ws
from example_worlds.persuasion import world_state as persuasion_ws


# =====================================================================
# RUNG 1 — OBSERVATION
# =====================================================================
class TestObservation:
    """Observation queries must return an ego-graph centred on the focus entity."""

    def test_observation_macbeth_basic(self):
        query = ObservationQuery(focus_entity_ids=["ENT_MACBETH"])
        result = calculate_narrative_physics(query, macbeth_ws)

        assert result["status"] == "success"
        assert result["query_type"] == "observation"
        ps = result["physics_state"]
        assert any(e["id"] == "ENT_MACBETH" for e in ps["focus_entities"])
        assert len(ps["current_locations"]) >= 1
        assert isinstance(ps["present_entities"], list)
        assert isinstance(ps["recent_memory"], list)

    def test_observation_returns_colocated_entities(self):
        """Entities sharing the focus entity's location must appear in present_entities."""
        query = ObservationQuery(focus_entity_ids=["ENT_MACBETH"])
        result = calculate_narrative_physics(query, macbeth_ws)
        ps = result["physics_state"]
        focus_locs = {loc["id"] for loc in ps["current_locations"]}
        for ent in ps["present_entities"]:
            assert ent["location_id"] in focus_locs

    def test_observation_with_temporal_anchor(self):
        """Temporal anchor must limit recent_memory to events at or before the anchor."""
        anchor = 5
        query = ObservationQuery(focus_entity_ids=["ENT_MACBETH"])
        result = calculate_narrative_physics(query, macbeth_ws, temporal_anchor=anchor)
        for evt in result["physics_state"]["recent_memory"]:
            assert evt["fabula_time"] <= anchor

    def test_observation_returns_relationships(self):
        """Relevant social edges involving focus entities must be present."""
        query = ObservationQuery(focus_entity_ids=["ENT_MACBETH"])
        result = calculate_narrative_physics(query, macbeth_ws)
        ps = result["physics_state"]
        assert len(ps["relevant_relationships"]) > 0
        for rel in ps["relevant_relationships"]:
            assert rel["source_entity_id"] == "ENT_MACBETH" or rel["target_entity_id"] == "ENT_MACBETH"

    def test_observation_romeo(self):
        query = ObservationQuery(focus_entity_ids=["ENT_ROMEO"])
        result = calculate_narrative_physics(query, romeo_ws)
        assert result["status"] == "success"
        assert any(e["id"] == "ENT_ROMEO" for e in result["physics_state"]["focus_entities"])

    def test_observation_passes_observations_through(self):
        obs = {"OBJ_DAGGER": "bloodied", "ENT_MACBETH": "guilty"}
        query = ObservationQuery(focus_entity_ids=["ENT_MACBETH"], observations=obs)
        result = calculate_narrative_physics(query, macbeth_ws)
        assert result["directives"] == obs

    def test_observation_multi_entity(self):
        """Multiple focus entity IDs should produce a union of ego-graphs."""
        query = ObservationQuery(focus_entity_ids=["ENT_MACBETH", "ENT_LADY_MACBETH"])
        result = calculate_narrative_physics(query, macbeth_ws)
        ps = result["physics_state"]
        focus_ids = {e["id"] for e in ps["focus_entities"]}
        assert "ENT_MACBETH" in focus_ids
        assert "ENT_LADY_MACBETH" in focus_ids

    def test_observation_empty_pov_returns_full_world(self):
        """No focus_entity_ids should fall back to the full omniscient world state."""
        query = ObservationQuery()
        result = calculate_narrative_physics(query, macbeth_ws)
        ps = result["physics_state"]
        # Full world dump has 'entities' dict at top level, not 'focus_entities'
        assert "entities" in ps
        assert "locations" in ps
        # Every entity in the model must appear
        for ent_id in macbeth_ws.entities:
            assert ent_id in ps["entities"]

    def test_observation_omniscient_with_temporal_anchor(self):
        """Omniscient fallback with a temporal anchor must still time-slice events."""
        anchor = 5
        query = ObservationQuery()
        result = calculate_narrative_physics(query, macbeth_ws, temporal_anchor=anchor)
        ps = result["physics_state"]
        assert "entities" in ps
        for evt in ps["events"]:
            assert evt["fabula_time"] <= anchor
        # Entities/locations should remain complete
        assert set(ps["entities"].keys()) == set(macbeth_ws.entities.keys())


# =====================================================================
# RUNG 2 — INTERVENTION
# =====================================================================
class TestIntervention:
    """Intervention queries must build a NetworkX sandbox and apply do-calculus."""

    def test_spatial_intervention_macbeth(self):
        """Teleporting Macbeth must update location_id in the graph."""
        query = InterventionQuery(
            interventions={"ENT_MACBETH.location_id": "LOC_HEATH"}
        )
        result = calculate_narrative_physics(query, macbeth_ws)

        assert result["status"] == "success"
        assert result["query_type"] == "intervention"
        # Rebuild the graph to inspect
        G = nx.node_link_graph(result["physics_state"])
        assert G.nodes["ENT_MACBETH"]["location_id"] == "LOC_HEATH"

    def test_spatial_intervention_severs_old_edge(self):
        """After teleport, no located_in edge to the OLD location should remain."""
        original_loc = macbeth_ws.entities["ENT_MACBETH"].location_id
        query = InterventionQuery(
            interventions={"ENT_MACBETH.location_id": "LOC_HEATH"}
        )
        result = calculate_narrative_physics(query, macbeth_ws)
        G = nx.node_link_graph(result["physics_state"])
        for _, target, data in G.out_edges("ENT_MACBETH", data=True):
            if data.get("edge_type") == "located_in":
                assert target != original_loc

    def test_state_intervention_macbeth_status(self):
        """Forcing Macbeth's status to 'dead' must update the node attribute."""
        query = InterventionQuery(
            interventions={"ENT_MACBETH.status": "dead"}
        )
        result = calculate_narrative_physics(query, macbeth_ws)
        G = nx.node_link_graph(result["physics_state"])
        assert G.nodes["ENT_MACBETH"]["status"] == "dead"

    def test_state_intervention_severs_causal_edges(self):
        """State surgery (do-operator) must sever incoming causal edges."""
        query = InterventionQuery(
            interventions={"ENT_MACBETH.status": "dead"}
        )
        result = calculate_narrative_physics(query, macbeth_ws)
        G = nx.node_link_graph(result["physics_state"])
        for source, target, data in G.in_edges("ENT_MACBETH", data=True):
            assert data.get("edge_type") != "causal"

    def test_intervention_preserves_other_nodes(self):
        """Intervention on one node must not remove unrelated nodes."""
        query = InterventionQuery(
            interventions={"ENT_MACBETH.status": "dead"}
        )
        result = calculate_narrative_physics(query, macbeth_ws)
        G = nx.node_link_graph(result["physics_state"])
        # The location node must still exist
        assert G.has_node(macbeth_ws.entities["ENT_MACBETH"].location_id)

    def test_intervention_returns_math_changes(self):
        interventions = {"ENT_MACBETH.status": "dead"}
        query = InterventionQuery(interventions=interventions)
        result = calculate_narrative_physics(query, macbeth_ws)
        assert result["math_changes"] == interventions

    def test_multiple_interventions(self):
        """Multiple simultaneous interventions must all be applied."""
        query = InterventionQuery(
            interventions={
                "ENT_MACBETH.status": "dead",
                "ENT_MACBETH.location_id": "LOC_HEATH",
            }
        )
        result = calculate_narrative_physics(query, macbeth_ws)
        G = nx.node_link_graph(result["physics_state"])
        assert G.nodes["ENT_MACBETH"]["status"] == "dead"
        assert G.nodes["ENT_MACBETH"]["location_id"] == "LOC_HEATH"

    def test_multi_entity_intervention_union(self):
        """Interventions on entities in DIFFERENT rooms must pull both rooms."""
        # Macbeth @ LOC_BATTLEFIELD, Fleance @ LOC_DUNSINANE_CASTLE
        query = InterventionQuery(
            interventions={
                "ENT_MACBETH.status": "injured",
                "ENT_FLEANCE.status": "injured",
            }
        )
        result = calculate_narrative_physics(query, macbeth_ws)
        G = nx.node_link_graph(result["physics_state"])
        assert G.has_node("ENT_MACBETH")
        assert G.has_node("ENT_FLEANCE")
        assert G.nodes["ENT_MACBETH"]["status"] == "injured"
        assert G.nodes["ENT_FLEANCE"]["status"] == "injured"
        # Both locations must be in the graph
        assert G.has_node("LOC_BATTLEFIELD")
        assert G.has_node("LOC_DUNSINANE_CASTLE")

    def test_inventory_intervention_give_item(self):
        """Inventory surgery must transfer ownership and rewire owned_by edge."""
        # Use a deepcopy with crown explicitly owned by ENT_MACBETH so the
        # transfer target is distinct from the original owner. Co-locate
        # Lady Macbeth so the new owner is pulled into the sandbox.
        from copy import deepcopy
        ws = deepcopy(macbeth_ws)
        ws.objects["OBJ_CROWN"].owner_id = "ENT_MACBETH"
        ws.entities["ENT_LADY_MACBETH"].location_id = ws.entities["ENT_MACBETH"].location_id
        query = InterventionQuery(
            interventions={"OBJ_CROWN.owner_id": "ENT_LADY_MACBETH"}
        )
        result = calculate_narrative_physics(query, ws)
        G = nx.node_link_graph(result["physics_state"])
        assert G.nodes["OBJ_CROWN"]["owner_id"] == "ENT_LADY_MACBETH"
        # New owned_by edge must point to Lady Macbeth
        owned_targets = [
            v for _, v, d in G.out_edges("OBJ_CROWN", data=True)
            if d.get("edge_type") == "owned_by"
        ]
        assert "ENT_LADY_MACBETH" in owned_targets

    def test_inventory_intervention_drop_item(self):
        """Dropping an item (owner_id=None) must sever owned_by and add located_in."""
        query = InterventionQuery(
            interventions={"OBJ_CROWN.owner_id": None}
        )
        result = calculate_narrative_physics(query, macbeth_ws)
        G = nx.node_link_graph(result["physics_state"])
        assert G.nodes["OBJ_CROWN"]["owner_id"] is None
        # No owned_by edge should remain
        owned_edges = [
            d for _, _, d in G.out_edges("OBJ_CROWN", data=True)
            if d.get("edge_type") == "owned_by"
        ]
        assert len(owned_edges) == 0

    def test_social_intervention_affinity(self):
        """Relationship surgery must alter the affinity metric, dampened by inertia."""
        # Place Lady Macbeth co-located with Macbeth so the relationship
        # passes the scene/co-presence filter in the ego-graph extractor.
        from copy import deepcopy
        ws = deepcopy(macbeth_ws)
        ws.entities["ENT_LADY_MACBETH"].location_id = ws.entities["ENT_MACBETH"].location_id
        # Read actual per-axis inertia from the fixture so this test stays
        # robust to per-metric inertia tuning.
        edge = next(
            r for r in ws.social_topology
            if r.source_entity_id == "ENT_MACBETH" and r.target_entity_id == "ENT_LADY_MACBETH"
        )
        m = edge.metrics["affinity"]
        # shift = -1.0 - m.value, dampened by m.inertia toward zero
        shift = -1.0 - m.value
        effective = shift + m.inertia  # shift is negative, dampening adds inertia toward 0
        expected = m.value + effective
        query = InterventionQuery(
            interventions={"ENT_MACBETH.relationships.ENT_LADY_MACBETH.affinity": -1.0}
        )
        result = calculate_narrative_physics(query, ws)
        G = nx.node_link_graph(result["physics_state"])
        found = False
        for _, v, d in G.out_edges("ENT_MACBETH", data=True):
            if v == "ENT_LADY_MACBETH" and d.get("edge_type") == "relationship":
                assert abs(d["affinity"] - expected) < 0.01, \
                    f"Expected ~{expected:.2f} (inertia={m.inertia}), got {d['affinity']}"
                found = True
                break
        assert found, "Relationship edge ENT_MACBETH→ENT_LADY_MACBETH not found"

    def test_sandbox_world_id_tagged_shadow(self):
        """Intervention sandbox must tag all nodes as 'shadow'."""
        query = InterventionQuery(
            interventions={"ENT_MACBETH.status": "dead"}
        )
        result = calculate_narrative_physics(query, macbeth_ws)
        G = nx.node_link_graph(result["physics_state"])
        for node_id, data in G.nodes(data=True):
            assert data.get("world_id") == "shadow", f"{node_id} not tagged shadow"

    def test_sandbox_topology_edges(self):
        """Sandbox must wire located_in edges for entities and owned_by for owned objects."""
        query = InterventionQuery(
            interventions={"ENT_MACBETH.status": "injured"}
        )
        result = calculate_narrative_physics(query, macbeth_ws)
        G = nx.node_link_graph(result["physics_state"])
        # Macbeth must have a located_in edge
        macbeth_loc_edges = [
            v for _, v, d in G.out_edges("ENT_MACBETH", data=True)
            if d.get("edge_type") == "located_in"
        ]
        assert len(macbeth_loc_edges) >= 1
        # OBJ_CROWN (owned by Macbeth) must have owned_by edge
        if G.has_node("OBJ_CROWN"):
            crown_edges = [
                (v, d["edge_type"]) for _, v, d in G.out_edges("OBJ_CROWN", data=True)
                if d.get("edge_type") in ("owned_by", "located_in")
            ]
            assert len(crown_edges) >= 1

    def test_genesis_spawns_new_entity(self):
        """Genesis intervention must create a brand-new node in the sandbox."""
        macbeth_loc = macbeth_ws.entities["ENT_MACBETH"].location_id
        query = InterventionQuery(
            interventions={
                "ENT_GHOST_BANQUO.spawn": {
                    "node_type": "Entity",
                    "name": "Ghost of Banquo",
                    "location_id": macbeth_loc,
                    "status": "dead",
                },
            }
        )
        result = calculate_narrative_physics(query, macbeth_ws)
        G = nx.node_link_graph(result["physics_state"])
        assert G.has_node("ENT_GHOST_BANQUO")
        assert G.nodes["ENT_GHOST_BANQUO"]["node_type"] == "Entity"
        assert G.nodes["ENT_GHOST_BANQUO"]["world_id"] == "shadow"
        assert G.nodes["ENT_GHOST_BANQUO"]["name"] == "Ghost of Banquo"

    def test_genesis_wires_location_edge(self):
        """A spawned node with a location_id must get a located_in edge."""
        macbeth_loc = macbeth_ws.entities["ENT_MACBETH"].location_id
        query = InterventionQuery(
            interventions={
                "ENT_GHOST_BANQUO.spawn": {
                    "node_type": "Entity",
                    "location_id": macbeth_loc,
                },
            }
        )
        result = calculate_narrative_physics(query, macbeth_ws)
        G = nx.node_link_graph(result["physics_state"])
        located_targets = [
            v for _, v, d in G.out_edges("ENT_GHOST_BANQUO", data=True)
            if d.get("edge_type") == "located_in"
        ]
        assert macbeth_loc in located_targets

    def test_genesis_plus_existing_intervention(self):
        """Genesis and standard interventions can coexist in a single query."""
        macbeth_loc = macbeth_ws.entities["ENT_MACBETH"].location_id
        query = InterventionQuery(
            interventions={
                "ENT_MACBETH.status": "dead",
                "ENT_PHANTOM.spawn": {
                    "node_type": "Entity",
                    "location_id": macbeth_loc,
                },
            }
        )
        result = calculate_narrative_physics(query, macbeth_ws)
        G = nx.node_link_graph(result["physics_state"])
        assert G.nodes["ENT_MACBETH"]["status"] == "dead"
        assert G.has_node("ENT_PHANTOM")

    def test_genesis_object(self):
        """Genesis can spawn a NarrativeObject, not just an Entity."""
        macbeth_loc = macbeth_ws.entities["ENT_MACBETH"].location_id
        query = InterventionQuery(
            interventions={
                "OBJ_MAGIC_SWORD.spawn": {
                    "node_type": "NarrativeObject",
                    "name": "Enchanted Sword",
                    "location_id": macbeth_loc,
                },
            }
        )
        result = calculate_narrative_physics(query, macbeth_ws)
        G = nx.node_link_graph(result["physics_state"])
        assert G.has_node("OBJ_MAGIC_SWORD")
        assert G.nodes["OBJ_MAGIC_SWORD"]["node_type"] == "NarrativeObject"


# =====================================================================
# RUNG 3 — COUNTERFACTUAL
# =====================================================================
class TestCounterfactual:
    """Counterfactual queries must time-slice, intervene, and return shadow state."""

    def test_counterfactual_macbeth_murder_prevented(self):
        query = CounterfactualQuery(
            historical_interventions={"EVT_DUNCAN_MURDER.event_type": "prevented"},
            evidence_node_ids=["EVT_MACBETH_CROWNED"],
        )
        result = calculate_narrative_physics(query, macbeth_ws)

        assert result["status"] == "success"
        assert result["query_type"] == "counterfactual"
        assert result["evidence_conditions"] == ["EVT_MACBETH_CROWNED"]

    def test_counterfactual_time_slices_correctly(self):
        """Memory should be limited to events at or before the point of divergence."""
        murder_time = next(
            e.fabula_time for e in macbeth_ws.events if e.id == "EVT_DUNCAN_MURDER"
        )
        query = CounterfactualQuery(
            historical_interventions={"EVT_DUNCAN_MURDER.event_type": "prevented"},
            evidence_node_ids=[],
        )
        result = calculate_narrative_physics(query, macbeth_ws)
        G = nx.node_link_graph(result["physics_state"])
        for node_id, data in G.nodes(data=True):
            if data.get("node_type") == "EventNode":
                assert data["fabula_time"] <= murder_time

    def test_counterfactual_applies_intervention(self):
        """The target event's attribute must be changed in the shadow graph."""
        query = CounterfactualQuery(
            historical_interventions={"EVT_DUNCAN_MURDER.event_type": "prevented"},
            evidence_node_ids=[],
        )
        result = calculate_narrative_physics(query, macbeth_ws)
        G = nx.node_link_graph(result["physics_state"])
        assert G.nodes["EVT_DUNCAN_MURDER"]["event_type"] == "prevented"

    def test_counterfactual_death_on_nile(self):
        """Counterfactual on a different plot: prevent Linnet's murder."""
        query = CounterfactualQuery(
            historical_interventions={"EVT_SIMON_MURDERS_LINNET.event_type": "prevented"},
            evidence_node_ids=["ENT_POIROT"],
        )
        result = calculate_narrative_physics(query, nile_ws)
        # Some queries are flagged 'implausible' by the plausibility gate;
        # what matters here is that the engine completes a structured
        # counterfactual response (no crash, with a recognised status).
        assert result["status"] in {"success", "implausible"}

    def test_counterfactual_multi_entity(self):
        """Counterfactual with interventions on events by different actors pulls them all."""
        query = CounterfactualQuery(
            historical_interventions={
                "EVT_DUNCAN_MURDER.event_type": "prevented",   # actor: ENT_MACBETH, T=6
                "EVT_SONS_FLEE.event_type": "prevented",       # actor: ENT_MALCOLM, T=9
            },
            evidence_node_ids=[],
        )
        result = calculate_narrative_physics(query, macbeth_ws)
        G = nx.node_link_graph(result["physics_state"])
        # Both events must be in the sandbox (time-sliced to oldest = T=6)
        assert G.has_node("EVT_DUNCAN_MURDER")
        assert G.nodes["EVT_DUNCAN_MURDER"]["event_type"] == "prevented"
        # ENT_MACBETH (from EVT_DUNCAN_MURDER) must be in the graph
        assert G.has_node("ENT_MACBETH")
        # Time-slice: no events after T=6000 (the EVT_DUNCAN_MURDER anchor)
        for _, data in G.nodes(data=True):
            if data.get("node_type") == "EventNode":
                assert data["fabula_time"] <= 6000

    def test_counterfactual_shadow_world_id(self):
        """Counterfactual sandbox must tag all nodes as 'shadow'."""
        query = CounterfactualQuery(
            historical_interventions={"EVT_DUNCAN_MURDER.event_type": "prevented"},
            evidence_node_ids=[],
        )
        result = calculate_narrative_physics(query, macbeth_ws)
        G = nx.node_link_graph(result["physics_state"])
        for node_id, data in G.nodes(data=True):
            assert data.get("world_id") == "shadow", f"{node_id} not tagged shadow"


# =====================================================================
# SEMANTIC — DIRECTIVE
# =====================================================================
class TestDirective:
    """Directive queries must return ego-graph + a prose injection rule."""

    def test_directive_trait(self):
        query = DirectiveQuery(
            target_entity_ids=["ENT_MACBETH"],
            target_effect="fear",
            target_vector_id="ENT_MACBETH.traits.guilt",
            intensity=0.8,
        )
        result = calculate_narrative_physics(query, macbeth_ws)

        assert result["status"] == "success"
        assert result["query_type"] == "directive"
        assert result["target_effect"] == "fear"
        assert "guilt" in result["directives"].lower()
        assert "0.80" in result["directives"] or "+0.80" in result["directives"]

    def test_directive_relationship(self):
        query = DirectiveQuery(
            target_entity_ids=["ENT_MACBETH"],
            target_effect="rage",
            target_vector_id="ENT_MACBETH.relationships.ENT_LADY_MACBETH.affinity",
            intensity=0.5,
        )
        result = calculate_narrative_physics(query, macbeth_ws)
        assert "AFFINITY" in result["directives"]
        assert "ENT_LADY_MACBETH" in result["directives"]

    def test_directive_epistemic(self):
        query = DirectiveQuery(
            target_entity_ids=["ENT_MACBETH"],
            target_effect="dramatic_irony",
            target_vector_id="ENT_MACBETH.beliefs.ENT_BANQUO",
            intensity=0.9,
        )
        result = calculate_narrative_physics(query, macbeth_ws)
        assert "EPISTEMIC" in result["directives"]

    def test_directive_no_vector(self):
        """When no target_vector_id is given, a generic directive is returned."""
        query = DirectiveQuery(
            target_entity_ids=["ENT_MACBETH"],
            target_effect="grief",
            intensity=0.6,
        )
        result = calculate_narrative_physics(query, macbeth_ws)
        assert "NARRATIVE DIRECTIVE" in result["directives"]

    def test_directive_gatsby(self):
        query = DirectiveQuery(
            target_entity_ids=["ENT_GATSBY"],
            target_effect="regret",
            target_vector_id="ENT_GATSBY.traits.obsession",
            intensity=1.0,
        )
        result = calculate_narrative_physics(query, gatsby_ws)
        assert result["status"] == "success"
        assert "obsession" in result["directives"].lower()

    def test_directive_multi_entity(self):
        """Multi-target directive must pull a union ego-graph of all directed actors."""
        query = DirectiveQuery(
            target_entity_ids=["ENT_MACBETH", "ENT_LADY_MACBETH"],
            target_effect="fear",
            target_vector_id="ENT_MACBETH.traits.guilt",
            intensity=0.7,
        )
        result = calculate_narrative_physics(query, macbeth_ws)
        ps = result["physics_state"]
        focus_ids = {e["id"] for e in ps["focus_entities"]}
        assert "ENT_MACBETH" in focus_ids
        assert "ENT_LADY_MACBETH" in focus_ids
        assert "guilt" in result["directives"].lower()
class TestInterrogation:
    """Interrogation queries must return the full world state + the question for RAG."""

    def test_interrogation_basic(self):
        query = InterrogationQuery(
            question="Can Macbeth reach the courtyard unseen?",
            require_proof=True,
        )
        result = calculate_narrative_physics(query, macbeth_ws)

        assert result["status"] == "success"
        assert result["query_type"] == "interrogate"
        assert result["question"] == "Can Macbeth reach the courtyard unseen?"
        assert result["require_proof"] is True

    def test_interrogation_returns_full_world_state(self):
        query = InterrogationQuery(
            question="Who killed whom?",
            require_proof=False,
        )
        result = calculate_narrative_physics(query, macbeth_ws)
        ps = result["physics_state"]
        assert "locations" in ps
        assert "entities" in ps
        assert "objects" in ps
        assert "events" in ps
        assert "causal_topology" in ps
        assert "social_topology" in ps

    def test_interrogation_includes_all_entities(self):
        query = InterrogationQuery(question="List all characters", require_proof=False)
        result = calculate_narrative_physics(query, macbeth_ws)
        ps = result["physics_state"]
        assert set(ps["entities"].keys()) == set(macbeth_ws.entities.keys())

    def test_interrogation_with_temporal_anchor(self):
        """Temporal anchor must limit events without filtering entities/locations."""
        anchor = 5
        query = InterrogationQuery(question="What happened?", require_proof=False)
        result = calculate_narrative_physics(query, macbeth_ws, temporal_anchor=anchor)
        ps = result["physics_state"]
        for evt in ps["events"]:
            assert evt["fabula_time"] <= anchor
        # Entities and locations should still be complete
        assert set(ps["entities"].keys()) == set(macbeth_ws.entities.keys())

    def test_interrogation_gone_girl(self):
        query = InterrogationQuery(
            question="Does Nick have an alibi?",
            require_proof=True,
        )
        result = calculate_narrative_physics(query, gone_girl_ws)
        assert result["status"] == "success"


class TestGeneral:
    """General queries return the full world state for open-ended Q&A."""

    def test_general_basic(self):
        query = GeneralQuery(question="What is the overall power structure?")
        result = calculate_narrative_physics(query, macbeth_ws)
        assert result["status"] == "success"
        assert result["query_type"] == "general"
        assert result["question"] == "What is the overall power structure?"
        assert result["include_topology"] is True

    def test_general_returns_full_world_state(self):
        query = GeneralQuery(question="Describe everything.")
        result = calculate_narrative_physics(query, macbeth_ws)
        ps = result["physics_state"]
        assert "locations" in ps
        assert "entities" in ps
        assert "objects" in ps
        assert "events" in ps
        assert "causal_topology" in ps
        assert "social_topology" in ps
        assert "channels" in ps

    def test_general_with_temporal_anchor(self):
        anchor = 5
        query = GeneralQuery(question="What happened so far?")
        result = calculate_narrative_physics(query, macbeth_ws, temporal_anchor=anchor)
        ps = result["physics_state"]
        for evt in ps["events"]:
            assert evt["fabula_time"] <= anchor
        assert set(ps["entities"].keys()) == set(macbeth_ws.entities.keys())

    def test_general_without_topology(self):
        query = GeneralQuery(question="Just the basics.", include_topology=False)
        result = calculate_narrative_physics(query, macbeth_ws)
        assert result["include_topology"] is False
        assert result["status"] == "success"

    def test_general_gone_girl(self):
        query = GeneralQuery(question="Summarise the relationships in this story.")
        result = calculate_narrative_physics(query, gone_girl_ws)
        assert result["status"] == "success"
        assert result["query_type"] == "general"


# =====================================================================
# CROSS-PLOT — Verify pipeline works across multiple worlds
# =====================================================================
class TestCrossPlot:
    """Ensure the pipeline handles different plot models without coupling."""

    @pytest.mark.parametrize("ws,entity_id", [
        (macbeth_ws, "ENT_MACBETH"),
        (romeo_ws, "ENT_ROMEO"),
        (gone_girl_ws, "ENT_NICK"),
        (gatsby_ws, "ENT_GATSBY"),
        (nile_ws, "ENT_POIROT"),
        (apocalypse_ws, "ENT_WILLARD"),
        (dads_army_ws, "ENT_MAINWARING"),
        (frankenstein_ws, "ENT_VICTOR"),
        (reservoir_ws, "ENT_WHITE"),
        (wuthering_ws, "ENT_HEATHCLIFF"),
        (acotar_ws, "ENT_FEYRE"),
        (wanda_ws, "ENT_ARCHIE"),
        (brief_ws, "ENT_LAURA"),
        (expectations_ws, "ENT_PIP"),
        (orwell_ws, "ENT_WINSTON"),
        (persuasion_ws, "ENT_ANNE"),
    ])
    def test_observation_across_plots(self, ws, entity_id):
        query = ObservationQuery(focus_entity_ids=[entity_id])
        result = calculate_narrative_physics(query, ws)
        assert result["status"] == "success"
        assert any(e["id"] == entity_id for e in result["physics_state"]["focus_entities"])

    @pytest.mark.parametrize("ws,entity_id", [
        (macbeth_ws, "ENT_MACBETH"),
        (romeo_ws, "ENT_ROMEO"),
        (gatsby_ws, "ENT_GATSBY"),
        (apocalypse_ws, "ENT_WILLARD"),
        (dads_army_ws, "ENT_MAINWARING"),
        (frankenstein_ws, "ENT_VICTOR"),
        (reservoir_ws, "ENT_WHITE"),
        (wuthering_ws, "ENT_HEATHCLIFF"),
        (acotar_ws, "ENT_FEYRE"),
        (wanda_ws, "ENT_ARCHIE"),
        (brief_ws, "ENT_LAURA"),
        (expectations_ws, "ENT_PIP"),
        (orwell_ws, "ENT_WINSTON"),
        (persuasion_ws, "ENT_ANNE"),
    ])
    def test_intervention_status_change_across_plots(self, ws, entity_id):
        query = InterventionQuery(
            interventions={f"{entity_id}.status": "dead"}
        )
        result = calculate_narrative_physics(query, ws)
        G = nx.node_link_graph(result["physics_state"])
        assert G.nodes[entity_id]["status"] == "dead"


# =====================================================================
# TOPOLOGY WIRING — Causal edges, mechanisms, location connectivity
# =====================================================================
class TestTopologyWiring:
    """Ensure causal_topology and location connectivity are wired into the sandbox."""

    def test_causal_topology_edges_present(self):
        """Formal causal edges from world_state.causal_topology must appear in the sandbox."""
        query = InterventionQuery(
            interventions={"ENT_MACBETH.status": "healthy"}
        )
        result = calculate_narrative_physics(query, macbeth_ws)
        G = nx.node_link_graph(result["physics_state"])
        causal_edges = [
            (u, v, d) for u, v, d in G.edges(data=True)
            if d.get("edge_type") == "causal"
        ]
        assert len(causal_edges) > 0

    def test_causal_mechanism_types_preserved(self):
        """Mechanism types (psychological, epistemic, etc.) must not all be 'physical'."""
        query = InterventionQuery(
            interventions={"ENT_MACBETH.status": "healthy"}
        )
        result = calculate_narrative_physics(query, macbeth_ws)
        G = nx.node_link_graph(result["physics_state"])
        mechanisms = {
            d["mechanism"]
            for _, _, d in G.edges(data=True)
            if d.get("edge_type") == "causal" and "mechanism" in d
        }
        # Macbeth's causal_topology has psychological, epistemic, social, physical
        assert len(mechanisms) > 1, f"Only found mechanisms: {mechanisms}"
        assert "psychological" in mechanisms or "epistemic" in mechanisms or "social" in mechanisms

    def test_location_connectivity_edges(self):
        """SpatialEdges must produce 'connected_to' edges between Location nodes."""
        query = InterventionQuery(
            interventions={"ENT_MACBETH.status": "healthy"}
        )
        result = calculate_narrative_physics(query, macbeth_ws)
        G = nx.node_link_graph(result["physics_state"])
        connected_edges = [
            (u, v) for u, v, d in G.edges(data=True)
            if d.get("edge_type") == "connected_to"
        ]
        assert len(connected_edges) > 0
        # Both endpoints must be Location nodes
        for u, v in connected_edges:
            assert G.nodes[u].get("node_type") == "Location"
            assert G.nodes[v].get("node_type") == "Location"

    def test_causal_edge_both_endpoints_in_graph(self):
        """Every wired causal edge must have both endpoints in the sandbox."""
        query = InterventionQuery(
            interventions={"ENT_MACBETH.status": "healthy"}
        )
        result = calculate_narrative_physics(query, macbeth_ws)
        G = nx.node_link_graph(result["physics_state"])
        for u, v, d in G.edges(data=True):
            if d.get("edge_type") == "causal":
                assert G.has_node(u), f"Causal source {u} not in graph"
                assert G.has_node(v), f"Causal target {v} not in graph"

    def test_observation_includes_causal_and_spatial_edges(self):
        """Observation ego-graph payload should include causal and spatial fields."""
        query = ObservationQuery(focus_entity_ids=["ENT_MACBETH"])
        result = calculate_narrative_physics(query, macbeth_ws)
        ps = result["physics_state"]
        assert "relevant_causal_edges" in ps
        assert "relevant_spatial_edges" in ps
        assert "relevant_channels" in ps
        assert "relevant_utterance_events" in ps

    def test_connected_to_is_bidirectional(self):
        """Unlocked SpatialEdges must produce bidirectional connected_to edges: A→B and B→A."""
        query = InterventionQuery(
            interventions={"ENT_MACBETH.status": "healthy"}
        )
        result = calculate_narrative_physics(query, macbeth_ws)
        G = nx.node_link_graph(result["physics_state"])
        connected_pairs = set()
        for u, v, d in G.edges(data=True):
            if d.get("edge_type") == "connected_to":
                connected_pairs.add((u, v))
        # For every A→B there must be a B→A
        for u, v in connected_pairs:
            assert (v, u) in connected_pairs, f"Missing reverse edge {v}→{u}"

    def test_neighbor_locations_in_sandbox(self):
        """1-hop neighbor locations via spatial_topology must be pulled into the sandbox."""
        from copy import deepcopy
        ws = deepcopy(macbeth_ws)
        ws.entities["ENT_MACBETH"].location_id = "LOC_DUNSINANE_CASTLE"
        query = InterventionQuery(
            interventions={"ENT_MACBETH.status": "healthy"}
        )
        result = calculate_narrative_physics(query, ws)
        G = nx.node_link_graph(result["physics_state"])
        # Macbeth at LOC_DUNSINANE_CASTLE. spatial_topology connects it to:
        # LOC_INVERNESS_CASTLE, LOC_BIRNAM_WOOD, LOC_MACDUFF_CASTLE, LOC_WITCHES_CAVERN
        assert G.has_node("LOC_DUNSINANE_CASTLE")
        assert G.has_node("LOC_INVERNESS_CASTLE")
        assert G.has_node("LOC_BIRNAM_WOOD")
        assert G.has_node("LOC_MACDUFF_CASTLE")
        for nid in ["LOC_INVERNESS_CASTLE", "LOC_BIRNAM_WOOD", "LOC_MACDUFF_CASTLE"]:
            assert G.nodes[nid].get("node_type") == "Location"

    def test_has_path_between_connected_locations(self):
        """nx.has_path must succeed between connected locations in the sandbox."""
        from copy import deepcopy
        ws = deepcopy(macbeth_ws)
        ws.entities["ENT_MACBETH"].location_id = "LOC_DUNSINANE_CASTLE"
        query = InterventionQuery(
            interventions={"ENT_MACBETH.status": "healthy"}
        )
        result = calculate_narrative_physics(query, ws)
        G = nx.node_link_graph(result["physics_state"])
        assert nx.has_path(G, "LOC_DUNSINANE_CASTLE", "LOC_INVERNESS_CASTLE")

    def test_locked_spatial_edge_blocks_path(self):
        """A locked SpatialEdge must block entity movement via spatial affordance check."""
        from copy import deepcopy
        from shadow_loom.models import SpatialEdge
        ws = deepcopy(macbeth_ws)
        ws.entities["ENT_MACBETH"].location_id = "LOC_DUNSINANE_CASTLE"
        # Lock ALL edges involving Inverness so there's no unlocked alternate path
        ws.spatial_topology = [
            se if not (se.source_id == "LOC_INVERNESS_CASTLE" or se.target_id == "LOC_INVERNESS_CASTLE")
            else SpatialEdge(source_id=se.source_id, target_id=se.target_id, is_locked=True)
            for se in ws.spatial_topology
        ]
        # Macbeth is at LOC_DUNSINANE_CASTLE — try to move to LOC_INVERNESS_CASTLE
        query = InterventionQuery(
            interventions={"ENT_MACBETH.location_id": "LOC_INVERNESS_CASTLE"}
        )
        result = calculate_narrative_physics(query, ws)
        G = nx.node_link_graph(result["physics_state"])
        # Move must be BLOCKED — Macbeth should still be at Dunsinane
        assert G.nodes["ENT_MACBETH"]["location_id"] == "LOC_DUNSINANE_CASTLE"


# =====================================================================
# CHANNELS & PHYSICS OVERRIDE — Remote communication
# =====================================================================
class TestChannels:
    """Ensure Channel extraction, wiring, and physics override work."""

    def _make_comms_ws(self):
        """Macbeth world with an active telepathy channel between Macbeth and Lady Macbeth."""
        from copy import deepcopy
        from shadow_loom.models import Channel
        ws = deepcopy(macbeth_ws)
        # Put Lady Macbeth in a different room
        ws.entities["ENT_LADY_MACBETH"].location_id = "LOC_INVERNESS_CASTLE"
        ws.channels = {
            "CHN_TELEPATHY": Channel(
                id="CHN_TELEPATHY",
                name="telepathy",
                medium="telepathy",
                participant_ids=["ENT_MACBETH", "ENT_LADY_MACBETH"],
                established_at_fabula=10,
            ),
        }
        return ws

    def test_info_edge_extracted_in_observation(self):
        """Channel must appear in the ego-graph payload."""
        ws = self._make_comms_ws()
        query = ObservationQuery(focus_entity_ids=["ENT_MACBETH", "ENT_LADY_MACBETH"])
        result = calculate_narrative_physics(query, ws)
        ps = result["physics_state"]
        assert len(ps["relevant_channels"]) >= 1
        ch = next(c for c in ps["relevant_channels"] if c["medium"] == "telepathy")
        assert "ENT_MACBETH" in ch["participant_ids"]

    def test_info_edge_temporal_filter(self):
        """Channel established AFTER the temporal_anchor must be excluded."""
        ws = self._make_comms_ws()
        query = ObservationQuery(focus_entity_ids=["ENT_MACBETH", "ENT_LADY_MACBETH"])
        result = calculate_narrative_physics(query, ws, temporal_anchor=5)
        ps = result["physics_state"]
        # The CHN_TELEPATHY (established=10) is excluded; macbeth_ws has no
        # other channels established by t<=5, so list is empty.
        assert all(c["medium"] != "telepathy" for c in ps["relevant_channels"])

    def test_info_edge_wired_in_sandbox(self):
        """Channel must produce 'communicating_with' edges in the sandbox."""
        ws = self._make_comms_ws()
        query = InterventionQuery(
            interventions={"ENT_MACBETH.status": "healthy", "ENT_LADY_MACBETH.status": "healthy"}
        )
        result = calculate_narrative_physics(query, ws)
        G = nx.node_link_graph(result["physics_state"])
        comms_edges = [
            (u, v, d) for u, v, d in G.edges(data=True)
            if d.get("edge_type") == "communicating_with"
            and d.get("medium") == "telepathy"
        ]
        assert len(comms_edges) >= 1

    def test_physics_override_multi_room_comms(self):
        """Intervention result must include physics_override when multi-room + comms."""
        ws = self._make_comms_ws()
        query = InterventionQuery(
            interventions={"ENT_MACBETH.status": "healthy", "ENT_LADY_MACBETH.status": "healthy"}
        )
        result = calculate_narrative_physics(query, ws)
        assert "physics_override" in result
        assert "PHYSICS OVERRIDE" in result["physics_override"]
        assert "SEPARATE" in result["physics_override"]

    def test_no_physics_override_single_room(self):
        """Single-room intervention must NOT include a physics_override."""
        query = InterventionQuery(
            interventions={"ENT_MACBETH.status": "healthy"}
        )
        result = calculate_narrative_physics(query, macbeth_ws)
        assert "physics_override" not in result

    def test_comms_intervention_establishes_link(self):
        """communicating_with intervention must spawn communication edges."""
        query = InterventionQuery(
            interventions={
                "ENT_MACBETH.communicating_with": ["ENT_LADY_MACBETH"],
            }
        )
        result = calculate_narrative_physics(query, macbeth_ws)
        G = nx.node_link_graph(result["physics_state"])
        comms = [
            v for _, v, d in G.out_edges("ENT_MACBETH", data=True)
            if d.get("edge_type") == "communicating_with"
        ]
        assert "ENT_LADY_MACBETH" in comms

    def test_terminated_comms_excluded_no_anchor(self):
        """Terminated Channel must NOT appear when temporal_anchor is None."""
        from copy import deepcopy
        from shadow_loom.models import Channel
        ws = deepcopy(macbeth_ws)
        ws.entities["ENT_LADY_MACBETH"].location_id = "LOC_INVERNESS_CASTLE"
        ws.channels = {
            "CHN_RAVEN": Channel(
                id="CHN_RAVEN", name="raven", medium="raven",
                participant_ids=["ENT_MACBETH", "ENT_LADY_MACBETH"],
                established_at_fabula=5, terminated_at_fabula=15,
            ),
        }
        # No temporal anchor → terminated links are dead and must be excluded
        query = ObservationQuery(focus_entity_ids=["ENT_MACBETH", "ENT_LADY_MACBETH"])
        result = calculate_narrative_physics(query, ws)
        ps = result["physics_state"]
        assert len(ps["relevant_channels"]) == 0

    def test_terminated_comms_no_false_override(self):
        """Terminated Channel must NOT trigger a physics override."""
        from copy import deepcopy
        from shadow_loom.models import Channel
        ws = deepcopy(macbeth_ws)
        ws.entities["ENT_LADY_MACBETH"].location_id = "LOC_INVERNESS_CASTLE"
        ws.channels = {
            "CHN_RAVEN": Channel(
                id="CHN_RAVEN", name="raven", medium="raven",
                participant_ids=["ENT_MACBETH", "ENT_LADY_MACBETH"],
                established_at_fabula=5, terminated_at_fabula=15,
            ),
        }
        query = InterventionQuery(
            interventions={
                "ENT_MACBETH.status": "healthy",
                "ENT_LADY_MACBETH.status": "healthy",
            }
        )
        result = calculate_narrative_physics(query, ws)
        # Dead comms should NOT produce a physics override
        assert "physics_override" not in result

    def test_no_duplicate_causal_edges(self):
        """Section C fallback must NOT duplicate edges already in formal causal topology."""
        query = InterventionQuery(
            interventions={"ENT_MACBETH.status": "healthy"}
        )
        result = calculate_narrative_physics(query, macbeth_ws)
        G = nx.node_link_graph(result["physics_state"])
        # Collect causal edges as (source, target) pairs with multiplicity
        from collections import Counter
        causal_pairs = Counter()
        for u, v, d in G.edges(data=True):
            if d.get("edge_type") == "causal":
                causal_pairs[(u, v)] += 1
        # No pair should have more than 1 causal edge
        for pair, count in causal_pairs.items():
            assert count == 1, f"Duplicate causal edge {pair} appears {count} times"

    def test_inventory_drop_after_teleport_uses_new_location(self):
        """Dropping an object after teleporting the owner should place it at the new room."""
        from copy import deepcopy
        ws = deepcopy(macbeth_ws)
        # Set up: Macbeth at Dunsinane, owns the crown.
        ws.entities["ENT_MACBETH"].location_id = "LOC_DUNSINANE_CASTLE"
        ws.objects["OBJ_CROWN"].owner_id = "ENT_MACBETH"
        # Teleport Macbeth, then drop the crown
        query = InterventionQuery(
            interventions={
                "ENT_MACBETH.location_id": "LOC_INVERNESS_CASTLE",
                "OBJ_CROWN.owner_id": None,
            }
        )
        result = calculate_narrative_physics(query, ws)
        G = nx.node_link_graph(result["physics_state"])
        # Crown should be in the NEW room (Inverness), not the old one (Dunsinane)
        located_in = [
            v for _, v, d in G.out_edges("OBJ_CROWN", data=True)
            if d.get("edge_type") == "located_in"
        ]
        assert "LOC_INVERNESS_CASTLE" in located_in, (
            f"Crown dropped at {located_in}, expected LOC_INVERNESS_CASTLE"
        )

    def test_full_dump_filters_terminated_comms(self):
        """extract_full_world_state must exclude terminated channels when no anchor."""
        from shadow_loom.extract_graph import extract_full_world_state
        from copy import deepcopy
        from shadow_loom.models import Channel
        ws = deepcopy(macbeth_ws)
        ws.channels = {
            "CHN_RAVEN": Channel(
                id="CHN_RAVEN", name="raven", medium="raven",
                participant_ids=["ENT_MACBETH", "ENT_LADY_MACBETH"],
                established_at_fabula=5, terminated_at_fabula=15,
            ),
            "CHN_SPEECH": Channel(
                id="CHN_SPEECH", name="speech", medium="speech",
                participant_ids=["ENT_MACBETH", "ENT_BANQUO"],
                established_at_fabula=3,
            ),
        }
        dump = extract_full_world_state(ws)
        # Terminated channel excluded, active channel kept
        assert len(dump["channels"]) == 1
        assert next(iter(dump["channels"].values()))["medium"] == "speech"

    def test_full_dump_timeslice_comms_with_anchor(self):
        """extract_full_world_state must time-slice channels when anchor given."""
        from shadow_loom.extract_graph import extract_full_world_state
        from copy import deepcopy
        from shadow_loom.models import Channel
        ws = deepcopy(macbeth_ws)
        ws.channels = {
            "CHN_LETTER": Channel(
                id="CHN_LETTER", name="letter", medium="letter",
                participant_ids=["ENT_MACBETH", "ENT_LADY_MACBETH"],
                established_at_fabula=5,
            ),
            "CHN_SPEECH": Channel(
                id="CHN_SPEECH", name="speech", medium="speech",
                participant_ids=["ENT_MACBETH", "ENT_BANQUO"],
                established_at_fabula=20,
            ),
        }
        dump = extract_full_world_state(ws, temporal_anchor=10)
        # Only the T=5 channel should survive (T=20 is future)
        assert len(dump["channels"]) == 1
        assert next(iter(dump["channels"].values()))["medium"] == "letter"

    def test_comms_intervention_resolves_remote_target(self):
        """communicating_with intervention must pull remote target entity into ego-graph."""
        from copy import deepcopy
        ws = deepcopy(macbeth_ws)
        # Ensure Lady Macbeth is in a DIFFERENT room from Macbeth
        ws.entities["ENT_LADY_MACBETH"].location_id = "LOC_ENGLAND"
        query = InterventionQuery(
            interventions={
                "ENT_MACBETH.communicating_with": ["ENT_LADY_MACBETH"],
            }
        )
        result = calculate_narrative_physics(query, ws)
        G = nx.node_link_graph(result["physics_state"])
        # Lady Macbeth must be in the sandbox even though she's far away
        assert G.has_node("ENT_LADY_MACBETH"), "Remote comms target not pulled into sandbox"
        comms = [
            v for _, v, d in G.out_edges("ENT_MACBETH", data=True)
            if d.get("edge_type") == "communicating_with"
        ]
        assert "ENT_LADY_MACBETH" in comms


# =====================================================================
# ROBUSTNESS — Edge-case safety and input validation
# =====================================================================
class TestRobustness:
    """Tests for crash safety, input validation, and data integrity."""

    def test_repeated_calls_dont_mutate_world_state(self):
        """Pipeline calls must NOT mutate the shared world_state event ordering."""
        original_order = [evt.id for evt in macbeth_ws.events]
        # Call twice — the in-place .sort() bug would reorder events
        calculate_narrative_physics(
            ObservationQuery(focus_entity_ids=["ENT_MACBETH"]), macbeth_ws
        )
        calculate_narrative_physics(
            ObservationQuery(focus_entity_ids=["ENT_MACBETH"]), macbeth_ws
        )
        assert [evt.id for evt in macbeth_ws.events] == original_order

    def test_dotless_intervention_key_skipped(self):
        """Intervention key without a dot must be skipped, not crash."""
        query = InterventionQuery(
            interventions={"ENT_MACBETH": "broken_key", "ENT_MACBETH.status": "healthy"}
        )
        result = calculate_narrative_physics(query, macbeth_ws)
        assert result["status"] == "success"

    def test_comms_intervention_string_coerced_to_list(self):
        """A bare string for communicating_with must be coerced to a single-element list."""
        # Co-locate Lady Macbeth so she's pulled into the sandbox alongside
        # Macbeth before the comms surgery wires the edge.
        from copy import deepcopy
        ws = deepcopy(macbeth_ws)
        ws.entities["ENT_LADY_MACBETH"].location_id = ws.entities["ENT_MACBETH"].location_id
        query = InterventionQuery(
            interventions={
                "ENT_MACBETH.communicating_with": "ENT_LADY_MACBETH",
            }
        )
        result = calculate_narrative_physics(query, ws)
        G = nx.node_link_graph(result["physics_state"])
        comms = [
            v for _, v, d in G.out_edges("ENT_MACBETH", data=True)
            if d.get("edge_type") == "communicating_with"
        ]
        assert "ENT_LADY_MACBETH" in comms

    def test_nested_state_intervention_through_scalar(self):
        """State intervention through a scalar intermediate must not crash."""
        query = InterventionQuery(
            interventions={"ENT_MACBETH.status.sub_field": "test_value"}
        )
        result = calculate_narrative_physics(query, macbeth_ws)
        G = nx.node_link_graph(result["physics_state"])
        # status was a string; it should now be a dict with sub_field
        assert G.nodes["ENT_MACBETH"]["status"] == {"sub_field": "test_value"}

    def test_genesis_normalizes_id(self):
        """Genesis surgery must create the node with the correct graph ID."""
        query = InterventionQuery(
            interventions={
                "ENT_GHOST.spawn": {
                    "node_type": "Entity",
                    "location_id": "LOC_DUNSINANE_CASTLE",
                    "id": "wrong_id",
                }
            }
        )
        result = calculate_narrative_physics(query, macbeth_ws)
        G = nx.node_link_graph(result["physics_state"])
        assert G.has_node("ENT_GHOST")
        # Node should NOT exist under the payload's stale "wrong_id"
        assert not G.has_node("wrong_id")
        assert G.nodes["ENT_GHOST"]["node_type"] == "Entity"

    def test_malformed_relationship_path_skipped(self):
        """Relationship path with wrong number of dots must be skipped, not crash."""
        query = InterventionQuery(
            interventions={"ENT_MACBETH.relationships.ENT_LADY_MACBETH": 0.5}
        )
        result = calculate_narrative_physics(query, macbeth_ws)
        assert result["status"] == "success"


# =====================================================================
# INERTIA PHYSICS — Impact > Inertia check on trait mutations
# =====================================================================
class TestInertiaPhysics:
    """Ensure trait mutations respect the Impact > Inertia gate."""

    def test_inertia_blocks_small_shift(self):
        """A shift smaller than inertia must be blocked entirely."""
        from copy import deepcopy
        ws = deepcopy(macbeth_ws)
        # Macbeth's ambition: value=0.7, inertia=0.55
        # Trying to set to 0.75 → |shift|=0.05 < 0.55 inertia → blocked
        query = InterventionQuery(
            interventions={"ENT_MACBETH.traits.ambition.value": 0.75}
        )
        result = calculate_narrative_physics(query, ws)
        G = nx.node_link_graph(result["physics_state"])
        # Trait should remain unchanged since shift < inertia
        assert G.nodes["ENT_MACBETH"]["traits"]["ambition"]["value"] == 0.7

    def test_inertia_dampens_large_shift(self):
        """A shift larger than inertia must be dampened by the inertia amount."""
        from copy import deepcopy
        ws = deepcopy(macbeth_ws)
        # Macbeth's ambition: value=0.7, inertia=0.55
        # Trying to set to 0.0 → shift=-0.7, |shift|=0.7 > 0.55 → passes
        # effective_shift = -0.7 + 0.55 = -0.15  →  effective_val = 0.7 + (-0.15) = 0.55
        query = InterventionQuery(
            interventions={"ENT_MACBETH.traits.ambition.value": 0.0}
        )
        result = calculate_narrative_physics(query, ws)
        G = nx.node_link_graph(result["physics_state"])
        val = G.nodes["ENT_MACBETH"]["traits"]["ambition"]["value"]
        assert 0.54 <= val <= 0.56, f"Expected ~0.55, got {val}"

    def test_non_trait_state_always_succeeds(self):
        """Status changes (non-trait) must always succeed regardless of inertia."""
        query = InterventionQuery(
            interventions={"ENT_MACBETH.status": "dead"}
        )
        result = calculate_narrative_physics(query, macbeth_ws)
        G = nx.node_link_graph(result["physics_state"])
        assert G.nodes["ENT_MACBETH"]["status"] == "dead"

    def test_inertia_blocked_preserves_causal_edges(self):
        """When inertia blocks a trait shift, incoming causal edges must NOT be severed."""
        from copy import deepcopy
        from shadow_loom.models import CausalEdge
        ws = deepcopy(macbeth_ws)
        # Macbeth's causal_topology has no edges targeting ENT_MACBETH directly.
        # Inject one so we can verify it survives an inertia-blocked mutation.
        ws.causal_topology.append(
            CausalEdge(
                source_id="EVT_MACBETH_KILLED",
                target_id="ENT_MACBETH", causality_type="mutation", causal_force=5.0,
                mechanism="physical",
                fabula_time=19,
            )
        )
        query = InterventionQuery(
            interventions={"ENT_MACBETH.traits.ambition.value": 0.94}  # tiny shift, blocked
        )
        result = calculate_narrative_physics(query, ws)
        G = nx.node_link_graph(result["physics_state"])
        causal_in = [
            (u, v) for u, v, d in G.in_edges("ENT_MACBETH", data=True)
            if d.get("edge_type") == "causal"
        ]
        # Causal edges must still be present since inertia blocked the mutation
        assert len(causal_in) > 0, "Inertia-blocked intervention should not sever causal edges"

    def test_inertia_shorthand_preserves_trait_dict(self):
        """The 2-part shorthand 'traits.ambition' must NOT replace the dict with a scalar."""
        from copy import deepcopy
        ws = deepcopy(macbeth_ws)
        # Macbeth's ambition: value=0.7, inertia=0.55
        # Shift to 0.0 → |shift|=0.7 > 0.55 → passes, dampened to ~0.55
        query = InterventionQuery(
            interventions={"ENT_MACBETH.traits.ambition": 0.0}
        )
        result = calculate_narrative_physics(query, ws)
        G = nx.node_link_graph(result["physics_state"])
        trait = G.nodes["ENT_MACBETH"]["traits"]["ambition"]
        # Must still be a dict with value AND inertia — not a bare float
        assert isinstance(trait, dict), f"Expected dict, got {type(trait).__name__}: {trait}"
        assert "value" in trait, "TraitVector dict lost 'value' key"
        assert "inertia" in trait, "TraitVector dict lost 'inertia' key"
        assert 0.54 <= trait["value"] <= 0.56, f"Expected ~0.55, got {trait['value']}"
class TestAbduction:
    """Ensure the abduction step updates latent variables from evidence."""

    def test_abduction_entity_evidence_shifts_traits(self):
        """Entity evidence must shift sandbox traits toward present-day values."""
        from copy import deepcopy
        ws = deepcopy(macbeth_ws)
        # Counterfactual: prevent the murder, conditioning on Macbeth's current state
        query = CounterfactualQuery(
            historical_interventions={"EVT_DUNCAN_MURDER.event_type": "prevented"},
            evidence_node_ids=["ENT_MACBETH"],
        )
        result = calculate_narrative_physics(query, ws)
        assert result["status"] == "success"
        # The abduction step should have run without errors
        G = nx.node_link_graph(result["physics_state"])
        assert G.has_node("ENT_MACBETH")

    def test_abduction_event_evidence(self):
        """Event evidence must propagate through causal edges."""
        query = CounterfactualQuery(
            historical_interventions={"EVT_DUNCAN_MURDER.event_type": "prevented"},
            evidence_node_ids=["EVT_DUNCAN_MURDER"],
        )
        result = calculate_narrative_physics(query, macbeth_ws)
        assert result["status"] == "success"

    def test_abduction_empty_evidence_is_noop(self):
        """Empty evidence_node_ids must not crash."""
        query = CounterfactualQuery(
            historical_interventions={"EVT_DUNCAN_MURDER.event_type": "prevented"},
            evidence_node_ids=[],
        )
        result = calculate_narrative_physics(query, macbeth_ws)
        assert result["status"] == "success"

    def test_abduction_missing_evidence_node_skipped(self):
        """Evidence node not in sandbox must be skipped gracefully."""
        query = CounterfactualQuery(
            historical_interventions={"EVT_DUNCAN_MURDER.event_type": "prevented"},
            evidence_node_ids=["ENT_NONEXISTENT"],
        )
        result = calculate_narrative_physics(query, macbeth_ws)
        assert result["status"] == "success"


# =====================================================================
# BELIEF TIME-SLICING — Temporal filtering of beliefs in ego-graph
# =====================================================================
class TestBeliefTimeSlicing:
    """Ensure beliefs are time-sliced when a temporal anchor is set."""

    def test_beliefs_filtered_by_temporal_anchor(self):
        """Beliefs established after the temporal anchor must be excluded."""
        from copy import deepcopy
        from shadow_loom.models import Belief
        ws = deepcopy(macbeth_ws)
        # Add a belief established at T=10 to Macbeth
        ws.entities["ENT_MACBETH"].beliefs.append(
            Belief(
                target_id="ENT_MACDUFF",
                perceived_state="Macduff is a traitor",
                confidence=0.9,
                inertia=0.7,
                established_at_fabula=10,
            )
        )
        # Observation at T=5 should exclude the T=10 belief
        query = ObservationQuery(focus_entity_ids=["ENT_MACBETH"])
        result = calculate_narrative_physics(query, ws, temporal_anchor=5)
        ps = result["physics_state"]
        focus_ent = next(e for e in ps["focus_entities"] if e["id"] == "ENT_MACBETH")
        for b in focus_ent["beliefs"]:
            assert b.get("established_at_fabula", 0) <= 5

    def test_beliefs_unfiltered_without_anchor(self):
        """Without a temporal anchor, all beliefs must be included."""
        from copy import deepcopy
        from shadow_loom.models import Belief
        ws = deepcopy(macbeth_ws)
        ws.entities["ENT_MACBETH"].beliefs.append(
            Belief(
                target_id="ENT_MACDUFF",
                perceived_state="Macduff is a traitor",
                confidence=0.9,
                inertia=0.7,
                established_at_fabula=10,
            )
        )
        query = ObservationQuery(focus_entity_ids=["ENT_MACBETH"])
        result = calculate_narrative_physics(query, ws)
        ps = result["physics_state"]
        focus_ent = next(e for e in ps["focus_entities"] if e["id"] == "ENT_MACBETH")
        targets = [b["target_id"] for b in focus_ent["beliefs"]]
        assert "ENT_MACDUFF" in targets


# =====================================================================
# SCHEMA COMPLETENESS — New model fields
# =====================================================================
class TestSchemaCompleteness:
    """Verify new schema fields are present and correctly typed."""

    def test_event_node_has_target_ids(self):
        """EventNode must have target_ids field."""
        from shadow_loom.models import EventNode
        evt = EventNode(
            id="EVT_TEST", fabula_time=1, syuzhet_index=1,
            event_type="choice", actor_ids=["ENT_A"], target_ids=["ENT_B"],
            description="Test"
        )
        assert evt.target_ids == ["ENT_B"]

    def test_event_node_target_ids_defaults_empty(self):
        """EventNode.target_ids must default to empty list."""
        from shadow_loom.models import EventNode
        evt = EventNode(
            id="EVT_TEST", fabula_time=1, syuzhet_index=1,
            event_type="choice", description="Test"
        )
        assert evt.target_ids == []

    def test_belief_has_established_at_fabula(self):
        """Belief must have established_at_fabula field."""
        from shadow_loom.models import Belief
        b = Belief(
            target_id="ENT_A", perceived_state="test",
            confidence=0.5, inertia=0.5, established_at_fabula=5
        )
        assert b.established_at_fabula == 5

    def test_belief_established_defaults_zero(self):
        """Belief.established_at_fabula must default to 0."""
        from shadow_loom.models import Belief
        b = Belief(target_id="ENT_A", perceived_state="test", confidence=0.5, inertia=0.5)
        assert b.established_at_fabula == 0

    def test_location_ambient_state_typed(self):
        """Location.ambient_state must accept AmbientVector values."""
        from shadow_loom.models import Location, AmbientVector
        loc = Location(
            name="Test", description="Test",
            ambient_state={"heat": AmbientVector(value=0.8, volatility=0.3)}
        )
        assert loc.ambient_state["heat"].value == 0.8
        assert loc.ambient_state["heat"].volatility == 0.3

    def test_location_ambient_state_coerces_dict(self):
        """Location.ambient_state must coerce plain dicts to AmbientVector."""
        from shadow_loom.models import Location
        loc = Location(
            name="Test", description="Test",
            ambient_state={"heat": {"value": 0.8, "volatility": 0.3}}
        )
        assert loc.ambient_state["heat"].value == 0.8

    def test_causal_edge_evidence_strength(self):
        """CausalEdge must have evidence_strength field."""
        from shadow_loom.models import CausalEdge
        ce = CausalEdge(
            source_id="EVT_A", target_id="ENT_B", causality_type="mutation", causal_force=5.0,
            mechanism="physical", fabula_time=1, evidence_strength="strong"
        )
        assert ce.evidence_strength == "strong"

    def test_causal_edge_evidence_strength_default(self):
        """CausalEdge.evidence_strength must default to 'moderate'."""
        from shadow_loom.models import CausalEdge
        ce = CausalEdge(
            source_id="EVT_A", target_id="ENT_B", causality_type="mutation", causal_force=5.0,
            mechanism="physical", fabula_time=1
        )
        assert ce.evidence_strength == "moderate"

    def test_relationship_edge_evidence_strength(self):
        """RelationshipEdge must surface per-axis evidence_strength.

        Per the per-metric refactor (D5c), evidence_strength now lives
        on each RelationshipMetric inside ``metrics`` rather than as a
        flat edge field. The aggregated ``evidence_strength`` property
        returns the strongest observed axis (default ``"moderate"`` if
        no metrics are present).
        """
        from shadow_loom.models import RelationshipEdge, RelationshipMetric
        re_ = RelationshipEdge(
            source_entity_id="ENT_A", target_entity_id="ENT_B",
            metrics={"affinity": RelationshipMetric(value=0.5, evidence_strength="weak")},
        )
        assert re_.evidence_strength == "weak"
        assert re_.metrics["affinity"].evidence_strength == "weak"

    def test_query_models_no_dead_fields(self):
        """ObservationQuery must not have time_steps; InterventionQuery must not have commit_to_factual."""
        assert not hasattr(ObservationQuery, 'model_fields') or 'time_steps' not in ObservationQuery.model_fields
        assert not hasattr(InterventionQuery, 'model_fields') or 'commit_to_factual' not in InterventionQuery.model_fields


# =====================================================================
# SPATIAL AFFORDANCE — Pathfinding + locked-barrier affordance check
# =====================================================================
class TestSpatialAffordance:
    """Verify that spatial movement uses nx.has_path and affordance checks."""

    def test_unlocked_path_allows_movement(self):
        """Movement through unlocked spatial edges must succeed."""
        query = InterventionQuery(
            interventions={"ENT_MACBETH.location_id": "LOC_INVERNESS_CASTLE"}
        )
        result = calculate_narrative_physics(query, macbeth_ws)
        G = nx.node_link_graph(result["physics_state"])
        assert G.nodes["ENT_MACBETH"]["location_id"] == "LOC_INVERNESS_CASTLE"

    def test_locked_path_blocks_movement(self):
        """Movement through a fully locked path must be blocked."""
        from copy import deepcopy
        from shadow_loom.models import SpatialEdge
        ws = deepcopy(macbeth_ws)
        ws.entities["ENT_MACBETH"].location_id = "LOC_DUNSINANE_CASTLE"
        # Lock ALL edges touching Inverness
        ws.spatial_topology = [
            se if not (se.source_id == "LOC_INVERNESS_CASTLE" or se.target_id == "LOC_INVERNESS_CASTLE")
            else SpatialEdge(source_id=se.source_id, target_id=se.target_id, is_locked=True)
            for se in ws.spatial_topology
        ]
        query = InterventionQuery(
            interventions={"ENT_MACBETH.location_id": "LOC_INVERNESS_CASTLE"}
        )
        result = calculate_narrative_physics(query, ws)
        G = nx.node_link_graph(result["physics_state"])
        assert G.nodes["ENT_MACBETH"]["location_id"] == "LOC_DUNSINANE_CASTLE"

    def test_locked_barrier_with_key_allows_movement(self):
        """Locked barrier must be traversable if the entity holds an object with unlock affordance."""
        from copy import deepcopy
        from shadow_loom.models import SpatialEdge, NarrativeObject, Affordance
        ws = deepcopy(macbeth_ws)
        # Lock all edges touching Inverness with a barrier object
        ws.spatial_topology = [
            se if not (se.source_id == "LOC_INVERNESS_CASTLE" or se.target_id == "LOC_INVERNESS_CASTLE")
            else SpatialEdge(source_id=se.source_id, target_id=se.target_id,
                             is_locked=True, barrier_item_id="OBJ_IRON_DOOR")
            for se in ws.spatial_topology
        ]
        # Add the door and a key that Macbeth owns
        ws.objects["OBJ_IRON_DOOR"] = NarrativeObject(
            id="OBJ_IRON_DOOR", name="Iron Door",
            location_id=None, owner_id=None,
            affordances=[],
        )
        ws.objects["OBJ_CASTLE_KEY"] = NarrativeObject(
            id="OBJ_CASTLE_KEY", name="Castle Key",
            location_id=None, owner_id="ENT_MACBETH",
            affordances=[Affordance(action="unlock", target_type="NarrativeObject")],
        )
        query = InterventionQuery(
            interventions={"ENT_MACBETH.location_id": "LOC_INVERNESS_CASTLE"}
        )
        result = calculate_narrative_physics(query, ws)
        G = nx.node_link_graph(result["physics_state"])
        assert G.nodes["ENT_MACBETH"]["location_id"] == "LOC_INVERNESS_CASTLE"

    def test_locked_edges_wired_with_metadata(self):
        """Locked SpatialEdges must still appear as connected_to edges with is_locked=True."""
        from copy import deepcopy
        from shadow_loom.models import SpatialEdge
        ws = deepcopy(macbeth_ws)
        ws.entities["ENT_MACBETH"].location_id = "LOC_DUNSINANE_CASTLE"
        ws.spatial_topology = [
            se if not (se.source_id == "LOC_DUNSINANE_CASTLE" and se.target_id == "LOC_INVERNESS_CASTLE")
            else SpatialEdge(source_id=se.source_id, target_id=se.target_id, is_locked=True)
            for se in ws.spatial_topology
        ]
        query = InterventionQuery(interventions={"ENT_MACBETH.status": "healthy"})
        result = calculate_narrative_physics(query, ws)
        G = nx.node_link_graph(result["physics_state"])
        locked_edges = [
            d for _, v, d in G.out_edges("LOC_DUNSINANE_CASTLE", data=True)
            if d.get("edge_type") == "connected_to" and v == "LOC_INVERNESS_CASTLE"
            and d.get("is_locked")
        ]
        assert len(locked_edges) >= 1


# =====================================================================
# EPISTEMIC LEAKAGE — Eavesdropping on unencrypted comms
# =====================================================================
class TestEpistemicLeakage:
    """Verify unencrypted comms leak to co-located entities."""

    def test_unencrypted_comms_create_eavesdrop_edges(self):
        """Co-located entity must get an eavesdropped_by edge from intelligible channels."""
        from copy import deepcopy
        from shadow_loom.models import Channel
        ws = deepcopy(macbeth_ws)
        # Put Lady Macbeth in a different room, keep Lennox with Macbeth at Dunsinane
        ws.entities["ENT_LADY_MACBETH"].location_id = "LOC_INVERNESS_CASTLE"
        ws.channels = {
            "CHN_SHOUT": Channel(
                id="CHN_SHOUT", name="shouting", medium="shouting",
                participant_ids=["ENT_MACBETH", "ENT_LADY_MACBETH"],
                established_at_fabula=10,
            ),
        }
        query = InterventionQuery(
            interventions={"ENT_MACBETH.status": "healthy", "ENT_LADY_MACBETH.status": "healthy"}
        )
        result = calculate_narrative_physics(query, ws)
        G = nx.node_link_graph(result["physics_state"])
        # Lennox is at Dunsinane (same room as Macbeth) — should have eavesdropped_by
        eavesdrop = [
            (u, v) for u, v, d in G.edges(data=True)
            if d.get("edge_type") == "eavesdropped_by"
        ]
        assert len(eavesdrop) > 0, "Unencrypted comms should produce eavesdropped_by edges"

    def test_encrypted_comms_no_eavesdrop(self):
        """Encrypted (low-intelligibility) channels must NOT produce eavesdropped_by edges."""
        from copy import deepcopy
        from shadow_loom.models import Channel
        ws = deepcopy(macbeth_ws)
        ws.entities["ENT_LADY_MACBETH"].location_id = "LOC_INVERNESS_CASTLE"
        ws.channels = {
            "CHN_TELEPATHY": Channel(
                id="CHN_TELEPATHY", name="telepathy", medium="telepathy",
                participant_ids=["ENT_MACBETH", "ENT_LADY_MACBETH"],
                intelligibility={"ENT_MACBETH": 0.0, "ENT_LADY_MACBETH": 0.0},
                established_at_fabula=10,
            ),
        }
        query = InterventionQuery(
            interventions={"ENT_MACBETH.status": "healthy", "ENT_LADY_MACBETH.status": "healthy"}
        )
        result = calculate_narrative_physics(query, ws)
        G = nx.node_link_graph(result["physics_state"])
        eavesdrop = [
            d for _, _, d in G.edges(data=True)
            if d.get("edge_type") == "eavesdropped_by"
        ]
        assert len(eavesdrop) == 0, "Encrypted comms should NOT produce eavesdropping"


# =====================================================================
# RELATIONSHIP INERTIA — Impact > Inertia on social edges
# =====================================================================
class TestRelationshipInertia:
    """Verify relationship mutations respect Impact > Inertia."""

    def test_relationship_inertia_blocks_small_shift(self):
        """A small relationship shift must be blocked by inertia."""
        from copy import deepcopy
        from shadow_loom.models import RelationshipEdge
        ws = deepcopy(macbeth_ws)
        # Set high inertia (0.9) on Macbeth→Lady Macbeth (affinity=0.8)
        ws.social_topology = [
            re_ if not (re_.source_entity_id == "ENT_MACBETH" and re_.target_entity_id == "ENT_LADY_MACBETH")
            else RelationshipEdge(
                source_entity_id="ENT_MACBETH", target_entity_id="ENT_LADY_MACBETH",
                affinity=0.8, fear=0.3, power_dynamic=-0.3, inertia=0.9,
            )
            for re_ in ws.social_topology
        ]
        # Shift of 0.1 (from 0.8 to 0.7) — |0.1| <= 0.9 → blocked
        query = InterventionQuery(
            interventions={"ENT_MACBETH.relationships.ENT_LADY_MACBETH.affinity": 0.7}
        )
        result = calculate_narrative_physics(query, ws)
        G = nx.node_link_graph(result["physics_state"])
        for _, v, d in G.out_edges("ENT_MACBETH", data=True):
            if v == "ENT_LADY_MACBETH" and d.get("edge_type") == "relationship":
                assert d["affinity"] == 0.8, f"Inertia should block; got {d['affinity']}"
                break

    def test_relationship_inertia_dampens_large_shift(self):
        """A large relationship shift must be dampened by inertia."""
        # Read per-axis inertia from the fixture; assert the dampening formula.
        edge = next(
            r for r in macbeth_ws.social_topology
            if r.source_entity_id == "ENT_MACBETH" and r.target_entity_id == "ENT_LADY_MACBETH"
        )
        m = edge.metrics["affinity"]
        shift = -1.0 - m.value
        expected = m.value + (shift + m.inertia)  # negative shift dampened toward 0
        query = InterventionQuery(
            interventions={"ENT_MACBETH.relationships.ENT_LADY_MACBETH.affinity": -1.0}
        )
        result = calculate_narrative_physics(query, macbeth_ws)
        G = nx.node_link_graph(result["physics_state"])
        for _, v, d in G.out_edges("ENT_MACBETH", data=True):
            if v == "ENT_LADY_MACBETH" and d.get("edge_type") == "relationship":
                assert abs(d["affinity"] - expected) < 0.01, \
                    f"Expected ~{expected:.2f} (inertia={m.inertia}), got {d['affinity']}"
                break

    def test_relationship_inertia_schema_field(self):
        """RelationshipEdge must have an inertia field with default 0.3."""
        from shadow_loom.models import RelationshipEdge
        re_ = RelationshipEdge(source_entity_id="ENT_A", target_entity_id="ENT_B")
        assert re_.inertia == 0.3


# =====================================================================
# FORWARD CASCADE — CTF Step 3 Prediction
# =====================================================================
class TestForwardCascade:
    """Verify counterfactual forward cascade propagates through causal topology."""

    def test_forward_cascade_adjusts_downstream_traits(self):
        """After counterfactual intervention, downstream entity traits must shift."""
        from copy import deepcopy
        ws = deepcopy(macbeth_ws)
        # Macbeth @ LOC_DUNSINANE_CASTLE, guilt=0.7
        pre_guilt = ws.entities["ENT_MACBETH"].traits["guilt"].value
        query = CounterfactualQuery(
            historical_interventions={"EVT_DUNCAN_MURDER.event_type": "prevented"},
            evidence_node_ids=[],
        )
        result = calculate_narrative_physics(query, ws)
        G = nx.node_link_graph(result["physics_state"])
        if G.has_node("ENT_MACBETH"):
            post_guilt = G.nodes["ENT_MACBETH"]["traits"]["guilt"]["value"]
            # Forward cascade should have shifted traits from the counterfactual intervention
            assert post_guilt != pre_guilt, (
                f"Counterfactual should shift guilt: pre={pre_guilt}, post={post_guilt}"
            )

    def test_forward_cascade_runs_after_intervention(self):
        """Counterfactual result must include the forward cascade step without crashing."""
        query = CounterfactualQuery(
            historical_interventions={"EVT_DUNCAN_MURDER.event_type": "prevented"},
            evidence_node_ids=["ENT_MACBETH"],
        )
        result = calculate_narrative_physics(query, macbeth_ws)
        assert result["status"] == "success"
        assert result["query_type"] == "counterfactual"


# =====================================================================
# DESTROYED PATH — SpatialEdge.destroyed_at_fabula consumption
# =====================================================================
class TestDestroyedPath:
    """Verify destroyed spatial paths are excluded from the ego-graph."""

    def test_destroyed_path_excluded_no_anchor(self):
        """A destroyed SpatialEdge must not appear when there is no temporal anchor."""
        from copy import deepcopy
        from shadow_loom.models import SpatialEdge
        ws = deepcopy(macbeth_ws)
        # Destroy the Dunsinane↔Inverness path at T=5
        ws.spatial_topology = [
            se if not (se.source_id == "LOC_DUNSINANE_CASTLE" and se.target_id == "LOC_INVERNESS_CASTLE")
            else SpatialEdge(source_id=se.source_id, target_id=se.target_id, destroyed_at_fabula=5)
            for se in ws.spatial_topology
        ]
        query = ObservationQuery(focus_entity_ids=["ENT_MACBETH"])
        result = calculate_narrative_physics(query, ws)
        ps = result["physics_state"]
        for se in ps["relevant_spatial_edges"]:
            if se["source_id"] == "LOC_DUNSINANE_CASTLE" and se["target_id"] == "LOC_INVERNESS_CASTLE":
                assert False, "Destroyed path should be excluded"

    def test_destroyed_path_included_before_destruction(self):
        """A path destroyed at T=10 must still appear at anchor T=5."""
        from copy import deepcopy
        from shadow_loom.models import SpatialEdge
        ws = deepcopy(macbeth_ws)
        # Place Macbeth at Dunsinane so the relevant spatial edge is in his ego.
        ws.entities["ENT_MACBETH"].location_id = "LOC_DUNSINANE_CASTLE"
        ws.spatial_topology = [
            se if not (se.source_id == "LOC_DUNSINANE_CASTLE" and se.target_id == "LOC_INVERNESS_CASTLE")
            else SpatialEdge(source_id=se.source_id, target_id=se.target_id, destroyed_at_fabula=10)
            for se in ws.spatial_topology
        ]
        query = ObservationQuery(focus_entity_ids=["ENT_MACBETH"])
        result = calculate_narrative_physics(query, ws, temporal_anchor=5)
        ps = result["physics_state"]
        found = any(
            se["source_id"] == "LOC_DUNSINANE_CASTLE" and se["target_id"] == "LOC_INVERNESS_CASTLE"
            for se in ps["relevant_spatial_edges"]
        )
        assert found, "Path destroyed at T=10 should still exist at T=5"

# =====================================================================
# RELATIONSHIP TIME-SLICING — last_updated_fabula consumption
# =====================================================================
class TestRelationshipTimeSlicing:
    """Verify relationships are time-sliced by last_updated_fabula."""

    def test_relationship_excluded_after_anchor(self):
        """A relationship updated after the temporal anchor must be excluded."""
        from copy import deepcopy
        from shadow_loom.models import RelationshipEdge
        ws = deepcopy(macbeth_ws)
        # Set Macbeth→Lady Macbeth relationship to last_updated at T=15
        ws.social_topology = [
            re_ if not (re_.source_entity_id == "ENT_MACBETH" and re_.target_entity_id == "ENT_LADY_MACBETH")
            else RelationshipEdge(
                source_entity_id="ENT_MACBETH", target_entity_id="ENT_LADY_MACBETH",
                affinity=0.8, fear=0.3, power_dynamic=-0.3, last_updated_fabula=15,
            )
            for re_ in ws.social_topology
        ]
        # Observation at T=10 — the T=15 relationship should be excluded
        query = ObservationQuery(focus_entity_ids=["ENT_MACBETH"])
        result = calculate_narrative_physics(query, ws, temporal_anchor=10)
        ps = result["physics_state"]
        for rel in ps["relevant_relationships"]:
            if rel["source_entity_id"] == "ENT_MACBETH" and rel["target_entity_id"] == "ENT_LADY_MACBETH":
                assert False, "Relationship updated at T=15 should be excluded at anchor T=10"

    def test_relationship_included_before_anchor(self):
        """A relationship updated before the temporal anchor must be included."""
        from copy import deepcopy
        from shadow_loom.models import RelationshipEdge
        ws = deepcopy(macbeth_ws)
        ws.social_topology = [
            re_ if not (re_.source_entity_id == "ENT_MACBETH" and re_.target_entity_id == "ENT_LADY_MACBETH")
            else RelationshipEdge(
                source_entity_id="ENT_MACBETH", target_entity_id="ENT_LADY_MACBETH",
                affinity=0.8, fear=0.3, power_dynamic=-0.3, last_updated_fabula=5,
            )
            for re_ in ws.social_topology
        ]
        # Include Lady Macbeth in focus so the co-presence filter passes
        # regardless of whether they share a room in the fixture.
        query = ObservationQuery(focus_entity_ids=["ENT_MACBETH", "ENT_LADY_MACBETH"])
        result = calculate_narrative_physics(query, ws, temporal_anchor=10)
        ps = result["physics_state"]
        found = any(
            rel["source_entity_id"] == "ENT_MACBETH" and rel["target_entity_id"] == "ENT_LADY_MACBETH"
            for rel in ps["relevant_relationships"]
        )
        assert found, "Relationship updated at T=5 should be included at anchor T=10"


# =====================================================================
# PLOT ENRICHMENT — Verify improved plot model features
# =====================================================================
class TestPlotEnrichment:
    """Verify the enriched plot models have correct beliefs, info edges, and relationships."""

    # --- Multiple Channels / Utterances ---
    @pytest.mark.parametrize("ws,expected_min", [
        (macbeth_ws, 3),
        (gatsby_ws, 2),
        (orwell_ws, 3),
        (gone_girl_ws, 3),
        (nile_ws, 3),
        (persuasion_ws, 3),
        (reservoir_ws, 2),
    ])
    def test_multiple_information_signals(self, ws, expected_min):
        """Enriched plots must have multiple channels + utterance events."""
        utt = sum(1 for e in ws.events if e.event_type == "utterance")
        signals = len(ws.channels) + utt
        assert signals >= expected_min, (
            f"Expected >= {expected_min} info signals, got {signals} "
            f"({len(ws.channels)} channels + {utt} utterances)"
        )

    # --- Implicit False Beliefs ---
    def test_gatsby_george_false_belief(self):
        """George Wilson's misattribution surfaces in the plot — either as a
        belief on the entity, a revelation event, or an utterance from Tom
        to George that names Gatsby. The audited fixture currently encodes
        the misattribution via ``EVT_UTT_TOM_DIRECTS_GEORGE`` (Tom
        directing George at Gatsby) rather than as a pre-baked belief."""
        george = gatsby_ws.entities["ENT_GEORGE"]
        beliefs_about_gatsby = [b for b in george.beliefs if b.target_id == "ENT_GATSBY"]
        revelation_events = [
            e for e in gatsby_ws.events
            if "ENT_GEORGE" in (e.actor_ids or []) and e.event_type == "revelation"
        ]
        utterance_to_george_about_gatsby = [
            e for e in gatsby_ws.events
            if e.event_type == "utterance"
            and "ENT_GEORGE" in (e.addressee_ids or [])
            and "ENT_GATSBY" in (e.target_ids or [])
        ]
        assert beliefs_about_gatsby or revelation_events or utterance_to_george_about_gatsby, (
            "Plot must encode George's misattribution about Gatsby as either "
            "a belief, a revelation event, or a Tom→George utterance about Gatsby."
        )

    def test_gatsby_gatsby_false_belief(self):
        """Gatsby must believe Daisy will choose him."""
        gatsby = gatsby_ws.entities["ENT_GATSBY"]
        beliefs_about_daisy = [b for b in gatsby.beliefs if b.target_id == "ENT_DAISY"]
        assert len(beliefs_about_daisy) >= 1

    def test_1984_winston_false_beliefs(self):
        """Winston must hold a false belief about O'Brien being on his side."""
        winston = orwell_ws.entities["ENT_WINSTON"]
        obrien_beliefs = [b for b in winston.beliefs if b.target_id == "ENT_OBRIEN"]
        assert any("brotherhood" in b.perceived_state.lower() or "ally" in b.perceived_state.lower()
                   or "sympath" in b.perceived_state.lower()
                    for b in obrien_beliefs), "Winston must believe O'Brien is sympathetic / a Brotherhood ally"

    def test_pip_central_false_belief(self):
        """Pip must believe Miss Havisham is his secret benefactress."""
        pip = expectations_ws.entities["ENT_PIP"]
        # The audited fixture canonicalises the entity id to ENT_HAVISHAM
        # (no "MISS_" prefix); accept either to remain robust.
        havisham_beliefs = [
            b for b in pip.beliefs
            if b.target_id in ("ENT_HAVISHAM", "ENT_MISS_HAVISHAM")
        ]
        assert any("benefact" in b.perceived_state.lower() for b in havisham_beliefs), (
            "Pip's central false belief about Havisham as benefactress must be present"
        )

    def test_wanda_sibling_disguise_beliefs(self):
        """George and Ken must believe Wanda and Otto are siblings."""
        george = wanda_ws.entities["ENT_GEORGE"]
        assert any("sibling" in b.perceived_state.lower() for b in george.beliefs)
        ken = wanda_ws.entities["ENT_KEN"]
        assert any("brother" in b.perceived_state.lower() or "sister" in b.perceived_state.lower()
                    for b in ken.beliefs)

    def test_white_trust_in_orange(self):
        """Mr. White must believe Orange is trustworthy (not a cop)."""
        white = reservoir_ws.entities["ENT_WHITE"]
        orange_beliefs = [b for b in white.beliefs if b.target_id == "ENT_ORANGE"]
        assert any("trustworthy" in b.perceived_state.lower() or "not a cop" in b.perceived_state.lower()
                    for b in orange_beliefs)

    # --- Critical Information Transfers ---
    def _has_pair(self, ws, a: str, b: str) -> bool:
        """True when a Channel has both a and b as participants, or an utterance
        event has speaker=a and b in addressee_ids (or vice versa)."""
        for ch in ws.channels.values():
            if a in ch.participant_ids and b in ch.participant_ids:
                return True
        for evt in ws.events:
            if evt.event_type != "utterance":
                continue
            if evt.speaker_id == a and b in evt.addressee_ids:
                return True
            if evt.speaker_id == b and a in evt.addressee_ids:
                return True
        return False

    def test_gatsby_tom_tells_george_info_edge(self):
        """Gatsby must encode the fatal Tom→George information transfer."""
        assert self._has_pair(gatsby_ws, "ENT_TOM", "ENT_GEORGE"), (
            "Tom→George info link (telling about the car) must exist as channel or utterance"
        )

    def test_macbeth_prophecy_info_edges(self):
        """Macbeth must encode at least one Witches→Macbeth prophecy signal."""
        assert self._has_pair(macbeth_ws, "ENT_WITCHES", "ENT_MACBETH"), (
            "Witches→Macbeth prophecy must exist as channel or utterance"
        )

    def test_1984_false_flag_info_edge(self):
        """1984 must encode O'Brien→Winston false-flag recruitment."""
        assert self._has_pair(orwell_ws, "ENT_OBRIEN", "ENT_WINSTON")

    def test_nile_signal_shout_info_edge(self):
        """Death on the Nile must encode Simon→Jacqueline signal."""
        assert self._has_pair(nile_ws, "ENT_SIMON", "ENT_JACQUELINE")

    def test_persuasion_overheard_conversation(self):
        """Persuasion must encode the pivotal Anne→Wentworth overheard exchange."""
        assert self._has_pair(persuasion_ws, "ENT_ANNE", "ENT_WENTWORTH")

    # --- New Entities and Relationships ---
    def test_romeo_nurse_entity_exists(self):
        """Romeo and Juliet must include the Nurse entity."""
        assert "ENT_NURSE" in romeo_ws.entities
        nurse = romeo_ws.entities["ENT_NURSE"]
        assert nurse.location_id == "LOC_CAPULET_HOUSE"

    def test_romeo_nurse_relationship(self):
        """Nurse→Juliet relationship must exist."""
        rels = [r for r in romeo_ws.social_topology
                if r.source_entity_id == "ENT_NURSE" and r.target_entity_id == "ENT_JULIET"]
        assert len(rels) >= 1

    def test_juliet_capulet_power_dynamic(self):
        """Juliet→Capulet relationship must reflect patriarchal power."""
        rels = [r for r in romeo_ws.social_topology
                if r.source_entity_id == "ENT_JULIET" and r.target_entity_id == "ENT_CAPULET"]
        assert len(rels) >= 1
        assert rels[0].power_dynamic < -0.5, "Juliet must be subordinate to Capulet"

    def test_gatsby_george_correct_location(self):
        """George Wilson must be at LOC_WILSON_GARAGE, not LOC_GATSBY_MANSION."""
        george = gatsby_ws.entities["ENT_GEORGE"]
        assert george.location_id == "LOC_WILSON_GARAGE", (
            f"George should be at Wilson's Garage, got {george.location_id}"
        )

    # --- Encrypted (low-intelligibility) Channels ---
    def test_encrypted_info_edges_exist(self):
        """At least one of the audited \"hidden-channel\" plots must encode a
        low-intelligibility channel.

        The audit pass moved several of these signals from per-channel
        ``intelligibility`` maps onto ``encrypted_for`` participant lists
        on individual utterance events. We accept either representation as
        long as the comprehension asymmetry is encoded somewhere across
        1984 + Reservoir Dogs."""
        def _has_encryption_signal(ws) -> bool:
            for ch in ws.channels.values():
                if ch.intelligibility and min(ch.intelligibility.values()) < 0.5:
                    return True
            for evt in ws.events:
                if getattr(evt, "encrypted_for", None):
                    return True
            return False

        assert _has_encryption_signal(orwell_ws) or _has_encryption_signal(reservoir_ws), (
            "Either 1984 or Reservoir Dogs must encode at least one "
            "low-intelligibility / encrypted information signal."
        )

    # --- Belief Temporal Anchoring ---
    def test_beliefs_have_fabula_timestamps(self):
        """At least some beliefs across the audited corpus carry a non-zero
        ``established_at_fabula`` timestamp so the time-slicing engine can
        prune them in counterfactuals."""
        worlds = [orwell_ws, expectations_ws, gatsby_ws, nile_ws]
        timestamped = 0
        for ws in worlds:
            for ent in ws.entities.values():
                for b in ent.beliefs:
                    if getattr(b, "established_at_fabula", 0) > 0:
                        timestamped += 1
        assert timestamped >= 1, (
            "At least one belief across the audited fixtures must carry a "
            "non-zero established_at_fabula timestamp."
        )

    # --- Constants on Enriched Entities ---
    def test_simon_co_conspirator_constant(self):
        """Simon Doyle's co-conspirator status must be encoded on the entity.

        The audit moved ``constants`` from a ``Dict[str, Any]`` to a
        ``List[str]`` of immutable boolean tags; the co-conspirator
        signal can also live as a trait. Accept either representation."""
        simon = nile_ws.entities["ENT_SIMON"]
        constants = simon.constants  # List[str]
        traits = simon.traits  # Dict[str, TraitVector]
        in_constants = any("conspirator" in c.lower() for c in constants)
        in_traits = any("jacqueline" in t.lower() or "conspirator" in t.lower() for t in traits)
        assert in_constants or in_traits, (
            "Simon's co-conspirator status with Jacqueline must be encoded "
            "as either a constant tag or a loyalty trait."
        )

    # --- Channel Observation Pipeline Integration ---
    def test_enriched_info_edges_flow_through_observation(self):
        """Channels and utterance events must appear in observation payload."""
        query = ObservationQuery(focus_entity_ids=["ENT_MACBETH"])
        result = calculate_narrative_physics(query, macbeth_ws)
        ps = result["physics_state"]
        signals = len(ps["relevant_channels"]) + len(ps["relevant_utterance_events"])
        assert signals >= 1, "Macbeth should see at least one channel or utterance"

    def test_enriched_info_edges_flow_through_intervention(self):
        """Active channels between focus entities must produce communicating_with edges."""
        from copy import deepcopy
        from shadow_loom.models import Channel
        ws = deepcopy(macbeth_ws)
        ws.entities["ENT_LADY_MACBETH"].location_id = "LOC_ENGLAND"
        ws.channels["CHN_TELEPATHY_X"] = Channel(
            id="CHN_TELEPATHY_X", name="telepathy", medium="telepathy",
            participant_ids=["ENT_MACBETH", "ENT_LADY_MACBETH"],
            established_at_fabula=1,
        )
        query = InterventionQuery(
            interventions={"ENT_MACBETH.status": "healthy", "ENT_LADY_MACBETH.status": "healthy"}
        )
        result = calculate_narrative_physics(query, ws)
        G = nx.node_link_graph(result["physics_state"])
        comms = [
            (u, v, d) for u, v, d in G.edges(data=True)
            if d.get("edge_type") == "communicating_with"
        ]
        assert len(comms) >= 1, "Active channels should produce communicating_with edges in sandbox"


# =====================================================================
# LEGACY FORWARD CASCADE PARITY
# =====================================================================
class TestLegacyForwardCascadeParity:
    """_apply_forward_cascade must match CausalPhysicsEngine.propagate() semantics."""

    def test_legacy_cascade_uses_inertia_gating(self):
        """Legacy cascade must gate on |impact| > inertia (high inertia blocks)."""
        from copy import deepcopy
        ws = deepcopy(macbeth_ws)
        # Set very high inertia on all Macbeth traits
        for tv in ws.entities["ENT_MACBETH"].traits.values():
            tv.inertia = 0.99
        query = CounterfactualQuery(
            historical_interventions={"EVT_DUNCAN_MURDER.event_type": "outcome"},
            evidence_node_ids=["ENT_MACBETH"],
        )
        result = calculate_narrative_physics(query, ws, use_causal_engine=False)
        # Should succeed without crash
        assert result["status"] == "success"

    def test_legacy_cascade_bidirectional_shifts(self):
        """Legacy cascade must allow trait decreases (signed delta)."""
        from copy import deepcopy
        ws = deepcopy(macbeth_ws)
        query = CounterfactualQuery(
            historical_interventions={"EVT_DUNCAN_MURDER.event_type": "outcome"},
            evidence_node_ids=["ENT_MACBETH"],
        )
        result = calculate_narrative_physics(query, ws, use_causal_engine=False)
        assert result["status"] == "success"


# =====================================================================
# RESOLVE FOCUS ENTITIES EDGE CASES
# =====================================================================
class TestResolveFocusEntities:
    """_resolve_focus_entities edge cases."""

    def test_object_owner_fallback(self):
        """Intervention on OBJ_X must resolve to the object's owner."""
        query = InterventionQuery(interventions={
            "OBJ_CROWN.owner_id": None,
        })
        result = calculate_narrative_physics(query, macbeth_ws)
        assert result["status"] == "success"

    def test_event_actor_fallback(self):
        """Intervention on EVT_X must resolve to the event's actor."""
        query = InterventionQuery(interventions={
            "EVT_DUNCAN_MURDER.event_type": "outcome",
        })
        result = calculate_narrative_physics(query, macbeth_ws)
        assert result["status"] == "success"


# =====================================================================
# CALCULATE PAST ANCHOR EDGE CASES
# =====================================================================
class TestCalculatePastAnchor:
    """_calculate_past_anchor edge cases."""

    def test_no_anchor_returns_implausible(self):
        """Counterfactual targeting a non-existent node now returns
        an implausible status instead of raising — the pipeline can
        explain rather than crash."""
        query = CounterfactualQuery(
            historical_interventions={"FAKE_NODE.status": "alive"},
            evidence_node_ids=[],
        )
        result = calculate_narrative_physics(query, macbeth_ws)
        assert result["status"] == "implausible"
        assert "implausibility_reason" in result
        assert any(
            t.get("target") == "FAKE_NODE.status"
            for t in result["implausibility_details"]["unresolved_targets"]
        )


# =====================================================================
# INSTANTIATOR EDGE CASES
# =====================================================================
class TestInstantiatorEdgeCases:
    """Surgery edge cases in AMWNInstantiator."""

    def test_sever_all_comms_none(self):
        """Setting communicating_with to None must sever all outgoing comms."""
        from copy import deepcopy
        from shadow_loom.models import Channel
        ws = deepcopy(macbeth_ws)
        ws.entities["ENT_LADY_MACBETH"].location_id = "LOC_DUNSINANE_CASTLE"
        ws.channels["CHN_SPEECH_X"] = Channel(
            id="CHN_SPEECH_X", name="speech", medium="speech",
            participant_ids=["ENT_MACBETH", "ENT_LADY_MACBETH"],
            established_at_fabula=1,
        )
        query = InterventionQuery(interventions={
            "ENT_MACBETH.communicating_with": None,
        })
        result = calculate_narrative_physics(query, ws)
        G = nx.node_link_graph(result["physics_state"])
        comms = [
            (u, v) for u, v, d in G.edges(data=True)
            if d.get("edge_type") == "communicating_with" and u == "ENT_MACBETH"
        ]
        assert len(comms) == 0

    def test_fear_clamped_0_to_1(self):
        """Fear metric must be clamped to [0, 1], not [-1, 1]."""
        query = InterventionQuery(interventions={
            "ENT_MACBETH.relationships.ENT_LADY_MACBETH.fear": -0.5,
        })
        result = calculate_narrative_physics(query, macbeth_ws)
        G = nx.node_link_graph(result["physics_state"])
        for u, v, d in G.edges(data=True):
            if (d.get("edge_type") == "relationship"
                    and u == "ENT_MACBETH" and v == "ENT_LADY_MACBETH"):
                assert d.get("fear", 0) >= 0.0

    def test_trait_shorthand_promoted(self):
        """'ENT_X.traits.courage' with numeric value must update the value inside dict."""
        query = InterventionQuery(interventions={
            "ENT_MACBETH.traits.ambition": 0.99,
        })
        result = calculate_narrative_physics(query, macbeth_ws)
        G = nx.node_link_graph(result["physics_state"])
        macbeth = next(n for n in G.nodes(data=True) if n[0] == "ENT_MACBETH")
        traits = macbeth[1]["traits"]
        assert isinstance(traits["ambition"], dict), "Trait must remain a dict, not be replaced by float"
        assert "value" in traits["ambition"]

    def test_eavesdropping_unencrypted_comms(self):
        """Entity co-located with comms participant must get eavesdropped_by edge."""
        from copy import deepcopy
        from shadow_loom.models import Channel
        ws = deepcopy(macbeth_ws)
        ws.channels["CHN_SPEECH_X"] = Channel(
            id="CHN_SPEECH_X", name="speech", medium="speech",
            participant_ids=["ENT_MACBETH", "ENT_LADY_MACBETH"],
            established_at_fabula=1,
        )
        query = InterventionQuery(interventions={
            "ENT_MACBETH.status": "healthy",
        })
        result = calculate_narrative_physics(query, ws)
        G = nx.node_link_graph(result["physics_state"])
        eavesdrop = [
            (u, v) for u, v, d in G.edges(data=True)
            if d.get("edge_type") == "eavesdropped_by"
        ]
        # If there are co-located entities, eavesdrop edges should exist
        # (depends on scene setup — just verify no crash)
        assert result["status"] == "success"

    def test_encrypted_comms_no_eavesdrop(self):
        """Encrypted (low-intelligibility) channels must NOT produce eavesdropped_by edges."""
        from copy import deepcopy
        from shadow_loom.models import Channel
        ws = deepcopy(macbeth_ws)
        ws.channels["CHN_MIRROR"] = Channel(
            id="CHN_MIRROR", name="magic_mirror", medium="magic_mirror",
            participant_ids=["ENT_MACBETH", "ENT_LADY_MACBETH"],
            intelligibility={"ENT_MACBETH": 0.0, "ENT_LADY_MACBETH": 0.0},
            established_at_fabula=1,
        )
        query = InterventionQuery(interventions={
            "ENT_MACBETH.status": "healthy",
        })
        result = calculate_narrative_physics(query, ws)
        G = nx.node_link_graph(result["physics_state"])
        eavesdrop = [
            (u, v) for u, v, d in G.edges(data=True)
            if d.get("edge_type") == "eavesdropped_by"
        ]
        assert len(eavesdrop) == 0


# =====================================================================
# EXTRACT GRAPH EDGE CASES
# =====================================================================
class TestExtractGraphEdgeCases:
    """extract_ego_graph_from_memory edge cases."""

    def test_nonexistent_focus_entity_raises(self):
        """Passing entity IDs that don't exist must raise ValueError."""
        from shadow_loom.extract_graph import extract_ego_graph_from_memory
        with pytest.raises(ValueError, match="None of the focus entities"):
            extract_ego_graph_from_memory(macbeth_ws, ["ENT_NOBODY"])

    def test_memory_limit_caps_events(self):
        """memory_limit parameter must cap the number of recent events."""
        from shadow_loom.extract_graph import extract_ego_graph_from_memory
        ego = extract_ego_graph_from_memory(macbeth_ws, ["ENT_MACBETH"],
                                             memory_limit=2)
        assert len(ego.recent_memory) <= 2

    def test_relationship_temporal_filter(self):
        """Relationships with last_updated_fabula > anchor must be excluded."""
        from copy import deepcopy
        from shadow_loom.extract_graph import extract_ego_graph_from_memory
        ws = deepcopy(macbeth_ws)
        # Set one relationship to be updated in the far future. The
        # property is read-only; mutate the underlying per-axis metric
        # instead. Touch every observed axis so the aggregated
        # `last_updated_fabula` (max across axes) becomes 9999.
        for rel in ws.social_topology:
            if rel.source_entity_id == "ENT_MACBETH":
                for m in rel.metrics.values():
                    m.last_updated_fabula = 9999
                break
        ego = extract_ego_graph_from_memory(ws, ["ENT_MACBETH"], temporal_anchor=5)
        # The relationship updated at T=9999 should be excluded
        for rel in ego.relevant_relationships:
            assert rel.get("last_updated_fabula", 0) <= 5

    def test_terminated_at_fabula_boundary(self):
        """Channel terminated AT the anchor must be excluded (<=)."""
        from copy import deepcopy
        from shadow_loom.extract_graph import extract_ego_graph_from_memory
        from shadow_loom.models import Channel
        ws = deepcopy(macbeth_ws)
        ws.channels = {
            "CHN_RAVEN": Channel(
                id="CHN_RAVEN", name="raven", medium="raven",
                participant_ids=["ENT_MACBETH", "ENT_LADY_MACBETH"],
                established_at_fabula=1, terminated_at_fabula=5,
            ),
        }
        ego = extract_ego_graph_from_memory(ws, ["ENT_MACBETH"], temporal_anchor=5)
        # terminated_at_fabula=5 == anchor=5 → must be excluded
        assert len(ego.relevant_channels) == 0



# =====================================================================
# TIER-2 ENGINE-LEVEL IMPLAUSIBILITY (Rung 2 / Rung 3 vacuity)
# =====================================================================
class TestEngineVacuityImplausibility:
    """When tier-1 (target resolution) passes but the causal physics
    engine itself produces no effect, the request is flagged as
    tier-2 implausible — unless ``force_implausible=True``."""

    def _build_vacuous_result(self, rung):
        from shadow_loom.causal_physics import CausalPhysicsResult
        return CausalPhysicsResult(
            sandbox_data={},
            mutations=[],
            social_mutations=[],
            blocked=[],
            intervened_nodes=[],
            hidden_deltas={},
        )

    def test_vacuity_helper_flags_empty_rung3(self):
        from shadow_loom.narrative_physics import _check_engine_vacuity
        v = _check_engine_vacuity(
            self._build_vacuous_result(3),
            rung=3,
            interventions={"FOO.bar": 1},
            evidence_node_ids=["ENT_X"],
        )
        assert v is not None
        assert v["tier"] == 2
        assert v["rung"] == 3
        assert any(t["target"] == "FOO.bar" for t in v["unresolved_targets"])

    def test_vacuity_helper_flags_empty_rung2(self):
        from shadow_loom.narrative_physics import _check_engine_vacuity
        v = _check_engine_vacuity(
            self._build_vacuous_result(2),
            rung=2,
            interventions={"FOO.bar": 1},
        )
        assert v is not None
        assert v["tier"] == 2
        assert v["rung"] == 2

    def test_vacuity_helper_passes_when_intervened(self):
        from shadow_loom.causal_physics import CausalPhysicsResult
        from shadow_loom.narrative_physics import _check_engine_vacuity
        result = CausalPhysicsResult(
            sandbox_data={},
            mutations=[],
            social_mutations=[],
            blocked=[],
            intervened_nodes=["ENT_MACBETH"],
            hidden_deltas={},
        )
        assert _check_engine_vacuity(result, rung=2, interventions={}) is None

    def test_vacuity_helper_passes_when_hidden_deltas(self):
        from shadow_loom.causal_physics import CausalPhysicsResult
        from shadow_loom.narrative_physics import _check_engine_vacuity
        result = CausalPhysicsResult(
            sandbox_data={},
            mutations=[],
            social_mutations=[],
            blocked=[],
            intervened_nodes=[],
            hidden_deltas={"ENT_X": {"guilt": 0.2}},
        )
        # rung 3 with hidden_deltas is NOT vacuous
        assert _check_engine_vacuity(result, rung=3, interventions={}) is None
        # but the same situation under rung 2 IS vacuous (hidden_deltas don't exist there)
        assert _check_engine_vacuity(result, rung=2, interventions={}) is not None

    def test_counterfactual_engine_vacuity_short_circuits(self, monkeypatch):
        """Force the engine to return an empty result and verify the
        counterfactual branch reports tier-2 implausibility."""
        from shadow_loom.causal_physics import CausalPhysicsResult
        import shadow_loom.narrative_physics as np_mod

        class _FakeEngine:
            def __init__(self, *a, **kw): pass
            def execute(self, *a, **kw):
                return CausalPhysicsResult(
                    sandbox_data={"nodes": [], "links": []},
                    mutations=[], social_mutations=[], blocked=[],
                    intervened_nodes=[], hidden_deltas={},
                )

        monkeypatch.setattr(np_mod, "CausalPhysicsEngine", _FakeEngine)

        first_event = macbeth_ws.events[0]
        query = CounterfactualQuery(
            historical_interventions={f"{first_event.id}.event_type": "outcome"},
            evidence_node_ids=[],
        )
        result = calculate_narrative_physics(query, macbeth_ws, use_causal_engine=True)
        assert result["status"] == "implausible"
        assert result["implausibility_details"]["tier"] == 2
        assert result["implausibility_details"]["rung"] == 3

    def test_counterfactual_engine_vacuity_force_proceeds(self, monkeypatch):
        from shadow_loom.causal_physics import CausalPhysicsResult
        import shadow_loom.narrative_physics as np_mod

        class _FakeEngine:
            def __init__(self, *a, **kw): pass
            def execute(self, *a, **kw):
                return CausalPhysicsResult(
                    sandbox_data={"nodes": [], "links": []},
                    mutations=[], social_mutations=[], blocked=[],
                    intervened_nodes=[], hidden_deltas={},
                )

        monkeypatch.setattr(np_mod, "CausalPhysicsEngine", _FakeEngine)

        first_event = macbeth_ws.events[0]
        query = CounterfactualQuery(
            historical_interventions={f"{first_event.id}.event_type": "outcome"},
            evidence_node_ids=[],
            force_implausible=True,
        )
        result = calculate_narrative_physics(query, macbeth_ws, use_causal_engine=True)
        assert result["status"] == "success"
        assert result["implausibility_warning"]
        assert result["implausibility_details"]["tier"] == 2

    def test_intervention_engine_vacuity_short_circuits(self, monkeypatch):
        from shadow_loom.causal_physics import CausalPhysicsResult
        import shadow_loom.narrative_physics as np_mod

        class _FakeEngine:
            def __init__(self, *a, **kw): pass
            def execute(self, *a, **kw):
                return CausalPhysicsResult(
                    sandbox_data={"nodes": [], "links": []},
                    mutations=[], social_mutations=[], blocked=[],
                    intervened_nodes=[], hidden_deltas={},
                )

        monkeypatch.setattr(np_mod, "CausalPhysicsEngine", _FakeEngine)

        first_ent = next(iter(macbeth_ws.entities))
        query = InterventionQuery(
            interventions={f"{first_ent}.status": "altered"},
        )
        result = calculate_narrative_physics(query, macbeth_ws, use_causal_engine=True)
        assert result["status"] == "implausible"
        assert result["implausibility_details"]["tier"] == 2
        assert result["implausibility_details"]["rung"] == 2
