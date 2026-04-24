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
)

# ── Import plot world-states used across tests ──────────────────────────
from tests.test_plot_models.macbeth import world_state as macbeth_ws
from tests.test_plot_models.romeo_and_juliet import world_state as romeo_ws
from tests.test_plot_models.gone_girl import world_state as gone_girl_ws
from tests.test_plot_models.great_gatsby import world_state as gatsby_ws
from tests.test_plot_models.death_on_the_nile import world_state as nile_ws


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
        """Relevant social edges from focus→co-located entities must be present."""
        query = ObservationQuery(focus_entity_ids=["ENT_MACBETH"])
        result = calculate_narrative_physics(query, macbeth_ws)
        ps = result["physics_state"]
        for rel in ps["relevant_relationships"]:
            assert rel["source_entity_id"] == "ENT_MACBETH"

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
        # Macbeth @ LOC_DUNSINANE_CASTLE, Fleance @ LOC_ENGLAND
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
        assert G.has_node("LOC_DUNSINANE_CASTLE")
        assert G.has_node("LOC_ENGLAND")

    def test_inventory_intervention_give_item(self):
        """Inventory surgery must transfer ownership and rewire owned_by edge."""
        # OBJ_CROWN is owned by ENT_MACBETH
        query = InterventionQuery(
            interventions={"OBJ_CROWN.owner_id": "ENT_LADY_MACBETH"}
        )
        result = calculate_narrative_physics(query, macbeth_ws)
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
        """Relationship surgery must alter the affinity metric on the edge."""
        query = InterventionQuery(
            interventions={"ENT_MACBETH.relationships.ENT_LADY_MACBETH.affinity": -1.0}
        )
        result = calculate_narrative_physics(query, macbeth_ws)
        G = nx.node_link_graph(result["physics_state"])
        # Find the relationship edge
        found = False
        for _, v, d in G.out_edges("ENT_MACBETH", data=True):
            if v == "ENT_LADY_MACBETH" and d.get("edge_type") == "relationship":
                assert d["affinity"] == -1.0
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
        assert result["status"] == "success"

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
        # Time-slice: no events after T=6
        for _, data in G.nodes(data=True):
            if data.get("node_type") == "EventNode":
                assert data["fabula_time"] <= 6

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
        """connected_locations must produce 'connected_to' edges between Location nodes."""
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
        assert "connected_locations" in ps

    def test_connected_to_is_bidirectional(self):
        """Spatial edges must be bidirectional: A→B and B→A."""
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
        """1-hop neighbor locations must be pulled into the sandbox as Location nodes."""
        query = InterventionQuery(
            interventions={"ENT_MACBETH.status": "healthy"}
        )
        result = calculate_narrative_physics(query, macbeth_ws)
        G = nx.node_link_graph(result["physics_state"])
        # Macbeth is at LOC_DUNSINANE_CASTLE, which connects to
        # LOC_INVERNESS_CASTLE, LOC_BIRNAM_WOOD, LOC_MACDUFF_CASTLE
        assert G.has_node("LOC_DUNSINANE_CASTLE")
        assert G.has_node("LOC_INVERNESS_CASTLE")
        assert G.has_node("LOC_BIRNAM_WOOD")
        assert G.has_node("LOC_MACDUFF_CASTLE")
        for nid in ["LOC_INVERNESS_CASTLE", "LOC_BIRNAM_WOOD", "LOC_MACDUFF_CASTLE"]:
            assert G.nodes[nid].get("node_type") == "Location"

    def test_has_path_between_connected_locations(self):
        """nx.has_path must succeed between connected locations in the sandbox."""
        query = InterventionQuery(
            interventions={"ENT_MACBETH.status": "healthy"}
        )
        result = calculate_narrative_physics(query, macbeth_ws)
        G = nx.node_link_graph(result["physics_state"])
        assert nx.has_path(G, "LOC_DUNSINANE_CASTLE", "LOC_INVERNESS_CASTLE")
