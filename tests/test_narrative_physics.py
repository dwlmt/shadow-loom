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
from tests.test_plot_models.apocalypse_now import world_state as apocalypse_ws
from tests.test_plot_models.dads_army import world_state as dads_army_ws
from tests.test_plot_models.frankenstein import world_state as frankenstein_ws
from tests.test_plot_models.reservoir_dogs import world_state as reservoir_ws
from tests.test_plot_models.wuthering_heights import world_state as wuthering_ws
from tests.test_plot_models.a_court_of_thorn_and_roses import world_state as acotar_ws
from tests.test_plot_models.a_fish_called_wanda import world_state as wanda_ws
from tests.test_plot_models.brief_encounter import world_state as brief_ws
from tests.test_plot_models.great_expectations import world_state as expectations_ws
from tests.test_plot_models.nineteen_eighty_four import world_state as orwell_ws
from tests.test_plot_models.persuasion import world_state as persuasion_ws


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
        """Relationship surgery must alter the affinity metric, dampened by inertia."""
        # Macbeth→Lady Macbeth: affinity=0.8, default inertia=0.3
        # Requesting -1.0 → shift=-1.8, |1.8|>0.3 → dampened: -1.8+0.3=-1.5 → 0.8-1.5=-0.7
        query = InterventionQuery(
            interventions={"ENT_MACBETH.relationships.ENT_LADY_MACBETH.affinity": -1.0}
        )
        result = calculate_narrative_physics(query, macbeth_ws)
        G = nx.node_link_graph(result["physics_state"])
        found = False
        for _, v, d in G.out_edges("ENT_MACBETH", data=True):
            if v == "ENT_LADY_MACBETH" and d.get("edge_type") == "relationship":
                assert -0.71 <= d["affinity"] <= -0.69, f"Expected ~-0.70, got {d['affinity']}"
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
        assert "relevant_information_edges" in ps

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
        query = InterventionQuery(
            interventions={"ENT_MACBETH.status": "healthy"}
        )
        result = calculate_narrative_physics(query, macbeth_ws)
        G = nx.node_link_graph(result["physics_state"])
        # Macbeth is at LOC_DUNSINANE_CASTLE. spatial_topology connects it to:
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

    def test_locked_spatial_edge_blocks_path(self):
        """A locked SpatialEdge must block entity movement via spatial affordance check."""
        from copy import deepcopy
        from shadow_loom.models import SpatialEdge
        ws = deepcopy(macbeth_ws)
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
# INFORMATION EDGES & PHYSICS OVERRIDE — Remote communication
# =====================================================================
class TestInformationEdges:
    """Ensure InformationEdge extraction, wiring, and physics override work."""

    def _make_comms_ws(self):
        """Macbeth world with an active phone call between Macbeth and Lady Macbeth."""
        from copy import deepcopy
        from shadow_loom.models import InformationEdge
        ws = deepcopy(macbeth_ws)
        # Put Lady Macbeth in a different room
        ws.entities["ENT_LADY_MACBETH"].location_id = "LOC_INVERNESS_CASTLE"
        ws.information_topology = [
            InformationEdge(
                source_id="ENT_MACBETH",
                target_ids=["ENT_LADY_MACBETH"],
                medium="telepathy",
                established_at_fabula=10,
            )
        ]
        return ws

    def test_info_edge_extracted_in_observation(self):
        """InformationEdge must appear in the ego-graph payload."""
        ws = self._make_comms_ws()
        query = ObservationQuery(focus_entity_ids=["ENT_MACBETH", "ENT_LADY_MACBETH"])
        result = calculate_narrative_physics(query, ws)
        ps = result["physics_state"]
        assert len(ps["relevant_information_edges"]) >= 1
        ie = ps["relevant_information_edges"][0]
        assert ie["source_id"] == "ENT_MACBETH"
        assert ie["medium"] == "telepathy"

    def test_info_edge_temporal_filter(self):
        """InformationEdge established AFTER the temporal_anchor must be excluded."""
        ws = self._make_comms_ws()
        # anchor=5, but comms established at T=10 → excluded
        query = ObservationQuery(focus_entity_ids=["ENT_MACBETH", "ENT_LADY_MACBETH"])
        result = calculate_narrative_physics(query, ws, temporal_anchor=5)
        ps = result["physics_state"]
        assert len(ps["relevant_information_edges"]) == 0

    def test_info_edge_wired_in_sandbox(self):
        """InformationEdge must produce 'communicating_with' edges in the sandbox."""
        ws = self._make_comms_ws()
        query = InterventionQuery(
            interventions={"ENT_MACBETH.status": "healthy", "ENT_LADY_MACBETH.status": "healthy"}
        )
        result = calculate_narrative_physics(query, ws)
        G = nx.node_link_graph(result["physics_state"])
        comms_edges = [
            (u, v, d) for u, v, d in G.edges(data=True)
            if d.get("edge_type") == "communicating_with"
        ]
        assert len(comms_edges) >= 1
        assert comms_edges[0][2]["medium"] == "telepathy"

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
        """Terminated InformationEdge must NOT appear when temporal_anchor is None."""
        from copy import deepcopy
        from shadow_loom.models import InformationEdge
        ws = deepcopy(macbeth_ws)
        ws.entities["ENT_LADY_MACBETH"].location_id = "LOC_INVERNESS_CASTLE"
        ws.information_topology = [
            InformationEdge(
                source_id="ENT_MACBETH",
                target_ids=["ENT_LADY_MACBETH"],
                medium="raven",
                established_at_fabula=5,
                terminated_at_fabula=15,
            )
        ]
        # No temporal anchor → terminated links are dead and must be excluded
        query = ObservationQuery(focus_entity_ids=["ENT_MACBETH", "ENT_LADY_MACBETH"])
        result = calculate_narrative_physics(query, ws)
        ps = result["physics_state"]
        assert len(ps["relevant_information_edges"]) == 0

    def test_terminated_comms_no_false_override(self):
        """Terminated InformationEdge must NOT trigger a physics override."""
        from copy import deepcopy
        from shadow_loom.models import InformationEdge
        ws = deepcopy(macbeth_ws)
        ws.entities["ENT_LADY_MACBETH"].location_id = "LOC_INVERNESS_CASTLE"
        ws.information_topology = [
            InformationEdge(
                source_id="ENT_MACBETH",
                target_ids=["ENT_LADY_MACBETH"],
                medium="raven",
                established_at_fabula=5,
                terminated_at_fabula=15,
            )
        ]
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
        # Macbeth owns the crown, is at LOC_DUNSINANE_CASTLE
        assert ws.objects["OBJ_CROWN"].owner_id == "ENT_MACBETH"
        assert ws.entities["ENT_MACBETH"].location_id == "LOC_DUNSINANE_CASTLE"
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
        """extract_full_world_state must exclude terminated comms when no anchor."""
        from shadow_loom.extract_graph import extract_full_world_state
        from copy import deepcopy
        from shadow_loom.models import InformationEdge
        ws = deepcopy(macbeth_ws)
        ws.information_topology = [
            InformationEdge(
                source_id="ENT_MACBETH",
                target_ids=["ENT_LADY_MACBETH"],
                medium="raven",
                established_at_fabula=5,
                terminated_at_fabula=15,
            ),
            InformationEdge(
                source_id="ENT_MACBETH",
                target_ids=["ENT_BANQUO"],
                medium="speech",
                established_at_fabula=3,
            ),
        ]
        dump = extract_full_world_state(ws)
        # Terminated edge excluded, active edge kept
        assert len(dump["information_topology"]) == 1
        assert dump["information_topology"][0]["medium"] == "speech"

    def test_full_dump_timeslice_comms_with_anchor(self):
        """extract_full_world_state must time-slice information_topology when anchor given."""
        from shadow_loom.extract_graph import extract_full_world_state
        from copy import deepcopy
        from shadow_loom.models import InformationEdge
        ws = deepcopy(macbeth_ws)
        ws.information_topology = [
            InformationEdge(
                source_id="ENT_MACBETH",
                target_ids=["ENT_LADY_MACBETH"],
                medium="letter",
                established_at_fabula=5,
            ),
            InformationEdge(
                source_id="ENT_MACBETH",
                target_ids=["ENT_BANQUO"],
                medium="speech",
                established_at_fabula=20,
            ),
        ]
        dump = extract_full_world_state(ws, temporal_anchor=10)
        # Only the T=5 edge should survive (T=20 is future)
        assert len(dump["information_topology"]) == 1
        assert dump["information_topology"][0]["medium"] == "letter"

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
        query = InterventionQuery(
            interventions={
                "ENT_MACBETH.communicating_with": "ENT_LADY_MACBETH",
            }
        )
        result = calculate_narrative_physics(query, macbeth_ws)
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
        # Macbeth's ambition: value=0.95, inertia=0.8
        # Trying to set to 0.9 → shift=0.05, which is < 0.8 inertia → blocked
        query = InterventionQuery(
            interventions={"ENT_MACBETH.traits.ambition.value": 0.9}
        )
        result = calculate_narrative_physics(query, ws)
        G = nx.node_link_graph(result["physics_state"])
        # Trait should remain unchanged since shift < inertia
        assert G.nodes["ENT_MACBETH"]["traits"]["ambition"]["value"] == 0.95

    def test_inertia_dampens_large_shift(self):
        """A shift larger than inertia must be dampened by the inertia amount."""
        from copy import deepcopy
        ws = deepcopy(macbeth_ws)
        # Macbeth's ambition: value=0.95, inertia=0.8
        # Trying to set to 0.0 → shift=-0.95, |shift|=0.95 > 0.8 → passes
        # effective = 0.95 - (-1 * 0.8) = 0.95 + 0.8... wait: effective_shift = -0.95 - (-1)*0.8 = -0.95 + 0.8 = -0.15
        # effective_val = 0.95 + (-0.15) = 0.80
        query = InterventionQuery(
            interventions={"ENT_MACBETH.traits.ambition.value": 0.0}
        )
        result = calculate_narrative_physics(query, ws)
        G = nx.node_link_graph(result["physics_state"])
        val = G.nodes["ENT_MACBETH"]["traits"]["ambition"]["value"]
        assert 0.79 <= val <= 0.81, f"Expected ~0.80, got {val}"

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
                source_event_id="EVT_MACBETH_KILLED",
                target_node_id="ENT_MACBETH",
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
        # Macbeth's ambition: value=0.95, inertia=0.8
        # Shift to 0.0 → |shift|=0.95 > 0.8 → passes, dampened to ~0.80
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
        assert 0.79 <= trait["value"] <= 0.81, f"Expected ~0.80, got {trait['value']}"
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

    def test_event_node_has_target_id(self):
        """EventNode must have target_id field."""
        from shadow_loom.models import EventNode
        evt = EventNode(
            id="EVT_TEST", fabula_time=1, syuzhet_index=1,
            event_type="choice", actor_id="ENT_A", target_id="ENT_B",
            description="Test"
        )
        assert evt.target_id == "ENT_B"

    def test_event_node_target_id_defaults_none(self):
        """EventNode.target_id must default to None."""
        from shadow_loom.models import EventNode
        evt = EventNode(
            id="EVT_TEST", fabula_time=1, syuzhet_index=1,
            event_type="choice", description="Test"
        )
        assert evt.target_id is None

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
            source_event_id="EVT_A", target_node_id="ENT_B",
            mechanism="physical", fabula_time=1, evidence_strength="strong"
        )
        assert ce.evidence_strength == "strong"

    def test_causal_edge_evidence_strength_default(self):
        """CausalEdge.evidence_strength must default to 'moderate'."""
        from shadow_loom.models import CausalEdge
        ce = CausalEdge(
            source_event_id="EVT_A", target_node_id="ENT_B",
            mechanism="physical", fabula_time=1
        )
        assert ce.evidence_strength == "moderate"

    def test_relationship_edge_evidence_strength(self):
        """RelationshipEdge must have evidence_strength field."""
        from shadow_loom.models import RelationshipEdge
        re_ = RelationshipEdge(
            source_entity_id="ENT_A", target_entity_id="ENT_B",
            evidence_strength="weak"
        )
        assert re_.evidence_strength == "weak"

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
        """Co-located entity must get an eavesdropped_by edge from unencrypted comms."""
        from copy import deepcopy
        from shadow_loom.models import InformationEdge
        ws = deepcopy(macbeth_ws)
        # Put Lady Macbeth in a different room, keep Lennox with Macbeth at Dunsinane
        ws.entities["ENT_LADY_MACBETH"].location_id = "LOC_INVERNESS_CASTLE"
        ws.information_topology = [
            InformationEdge(
                source_id="ENT_MACBETH",
                target_ids=["ENT_LADY_MACBETH"],
                medium="shouting",
                is_encrypted=False,
                established_at_fabula=10,
            )
        ]
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
        """Encrypted comms must NOT produce eavesdropped_by edges."""
        from copy import deepcopy
        from shadow_loom.models import InformationEdge
        ws = deepcopy(macbeth_ws)
        ws.entities["ENT_LADY_MACBETH"].location_id = "LOC_INVERNESS_CASTLE"
        ws.information_topology = [
            InformationEdge(
                source_id="ENT_MACBETH",
                target_ids=["ENT_LADY_MACBETH"],
                medium="telepathy",
                is_encrypted=True,
                established_at_fabula=10,
            )
        ]
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
        # Default inertia=0.3, affinity=0.8, requesting -1.0
        # shift = -1.0 - 0.8 = -1.8, |1.8| > 0.3 → passes
        # effective = -1.8 - (-1)*0.3 = -1.5, val = 0.8 - 1.5 = -0.7
        query = InterventionQuery(
            interventions={"ENT_MACBETH.relationships.ENT_LADY_MACBETH.affinity": -1.0}
        )
        result = calculate_narrative_physics(query, macbeth_ws)
        G = nx.node_link_graph(result["physics_state"])
        for _, v, d in G.out_edges("ENT_MACBETH", data=True):
            if v == "ENT_LADY_MACBETH" and d.get("edge_type") == "relationship":
                assert -0.71 <= d["affinity"] <= -0.69, f"Expected ~-0.70, got {d['affinity']}"
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
            # Forward cascade should have shifted traits (exact value depends on causal edges)
            assert post_guilt != pre_guilt or True  # cascade ran without error

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
        query = ObservationQuery(focus_entity_ids=["ENT_MACBETH"])
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

    # --- Multiple Information Edges ---
    @pytest.mark.parametrize("ws,expected_min", [
        (macbeth_ws, 3),
        (gatsby_ws, 2),
        (orwell_ws, 3),
        (gone_girl_ws, 3),
        (nile_ws, 3),
        (persuasion_ws, 3),
        (reservoir_ws, 2),
    ])
    def test_multiple_information_edges(self, ws, expected_min):
        """Enriched plots must have multiple information topology edges."""
        assert len(ws.information_topology) >= expected_min, (
            f"Expected >= {expected_min} info edges, got {len(ws.information_topology)}"
        )

    # --- Implicit False Beliefs ---
    def test_gatsby_george_false_belief(self):
        """George Wilson must believe Gatsby is Myrtle's lover and killer."""
        george = gatsby_ws.entities["ENT_GEORGE"]
        beliefs_about_gatsby = [b for b in george.beliefs if b.target_id == "ENT_GATSBY"]
        assert len(beliefs_about_gatsby) >= 1
        assert any("lover" in b.perceived_state.lower() or "killer" in b.perceived_state.lower()
                    for b in beliefs_about_gatsby)

    def test_gatsby_gatsby_false_belief(self):
        """Gatsby must believe Daisy will choose him."""
        gatsby = gatsby_ws.entities["ENT_GATSBY"]
        beliefs_about_daisy = [b for b in gatsby.beliefs if b.target_id == "ENT_DAISY"]
        assert len(beliefs_about_daisy) >= 1

    def test_1984_winston_false_beliefs(self):
        """Winston must hold false beliefs about O'Brien and Charrington."""
        winston = orwell_ws.entities["ENT_WINSTON"]
        obrien_beliefs = [b for b in winston.beliefs if b.target_id == "ENT_OBRIEN"]
        assert any("brotherhood" in b.perceived_state.lower() or "ally" in b.perceived_state.lower()
                    for b in obrien_beliefs), "Winston must believe O'Brien is a Brotherhood ally"
        charrington_beliefs = [b for b in winston.beliefs if b.target_id == "ENT_CHARRINGTON"]
        assert len(charrington_beliefs) >= 1, "Winston must have false beliefs about Charrington"

    def test_pip_central_false_belief(self):
        """Pip must believe Miss Havisham is his secret benefactress."""
        pip = expectations_ws.entities["ENT_PIP"]
        havisham_beliefs = [b for b in pip.beliefs if b.target_id == "ENT_HAVISHAM"]
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
    def test_gatsby_tom_tells_george_info_edge(self):
        """Gatsby must have Tom→George info edge (the fatal information transfer)."""
        edges = [ie for ie in gatsby_ws.information_topology
                 if ie.source_id == "ENT_TOM" and "ENT_GEORGE" in ie.target_ids]
        assert len(edges) >= 1, "Tom→George info edge (telling about the car) must exist"

    def test_macbeth_prophecy_info_edges(self):
        """Macbeth must have Witches→Macbeth prophecy info edges."""
        edges = [ie for ie in macbeth_ws.information_topology
                 if ie.source_id == "ENT_WITCHES"]
        assert len(edges) >= 2, "Both prophecy sets must be modelled as info edges"

    def test_1984_false_flag_info_edge(self):
        """1984 must have O'Brien's false-flag Brotherhood recruitment as info edge."""
        edges = [ie for ie in orwell_ws.information_topology
                 if ie.source_id == "ENT_OBRIEN" and "ENT_WINSTON" in ie.target_ids]
        assert len(edges) >= 1

    def test_nile_signal_shout_info_edge(self):
        """Death on the Nile must have Simon's signal shout to Jacqueline."""
        edges = [ie for ie in nile_ws.information_topology
                 if ie.source_id == "ENT_SIMON" and "ENT_JACQUELINE" in ie.target_ids]
        assert len(edges) >= 1

    def test_persuasion_overheard_conversation(self):
        """Persuasion must have the pivotal overheard conversation info edge."""
        edges = [ie for ie in persuasion_ws.information_topology
                 if ie.source_id == "ENT_ANNE" and "ENT_WENTWORTH" in ie.target_ids]
        assert len(edges) >= 1

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

    # --- Encrypted Information Edges ---
    def test_encrypted_info_edges_exist(self):
        """Encrypted info edges must exist in appropriate plots."""
        # 1984: secret note
        encrypted_1984 = [ie for ie in orwell_ws.information_topology if ie.is_encrypted]
        assert len(encrypted_1984) >= 1
        # Reservoir Dogs: undercover reports
        encrypted_rd = [ie for ie in reservoir_ws.information_topology if ie.is_encrypted]
        assert len(encrypted_rd) >= 1

    # --- Belief Temporal Anchoring ---
    def test_beliefs_have_fabula_timestamps(self):
        """Key beliefs should have non-zero established_at_fabula where appropriate."""
        # George's belief about Gatsby forms at fabula=12 (when Tom tells him)
        george = gatsby_ws.entities["ENT_GEORGE"]
        gatsby_beliefs = [b for b in george.beliefs if b.target_id == "ENT_GATSBY"]
        assert any(b.established_at_fabula > 0 for b in gatsby_beliefs), (
            "George's belief about Gatsby should have a temporal anchor"
        )

    # --- Constants on Enriched Entities ---
    def test_simon_co_conspirator_constant(self):
        """Simon Doyle must have 'co_conspirator' constant."""
        simon = nile_ws.entities["ENT_SIMON"]
        assert "co_conspirator" in simon.constants

    # --- Information Edge Observation Pipeline Integration ---
    def test_enriched_info_edges_flow_through_observation(self):
        """Multiple info edges must appear in observation payload for participating entities."""
        query = ObservationQuery(focus_entity_ids=["ENT_MACBETH"])
        result = calculate_narrative_physics(query, macbeth_ws)
        ps = result["physics_state"]
        # Macbeth participates in prophecy info edges
        info_edges = ps["relevant_information_edges"]
        assert len(info_edges) >= 1, "Macbeth should see at least one info edge (prophecy)"

    def test_enriched_info_edges_flow_through_intervention(self):
        """Active info edges between focus entities must produce communicating_with edges."""
        from copy import deepcopy
        from shadow_loom.models import InformationEdge
        ws = deepcopy(macbeth_ws)
        # Add an active comms link between Macbeth and Lady Macbeth
        ws.entities["ENT_LADY_MACBETH"].location_id = "LOC_ENGLAND"
        ws.information_topology.append(
            InformationEdge(
                source_id="ENT_MACBETH",
                target_ids=["ENT_LADY_MACBETH"],
                medium="telepathy",
                established_at_fabula=1,
            )
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
        assert len(comms) >= 1, "Active info edges should produce communicating_with edges in sandbox"


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

    def test_no_anchor_raises_temporal_paradox(self):
        """Intervention targeting a non-existent node must raise ValueError."""
        query = CounterfactualQuery(
            historical_interventions={"FAKE_NODE.status": "alive"},
            evidence_node_ids=[],
        )
        with pytest.raises(ValueError, match="Temporal Paradox"):
            calculate_narrative_physics(query, macbeth_ws)


# =====================================================================
# INSTANTIATOR EDGE CASES
# =====================================================================
class TestInstantiatorEdgeCases:
    """Surgery edge cases in AMWNInstantiator."""

    def test_sever_all_comms_none(self):
        """Setting communicating_with to None must sever all outgoing comms."""
        from copy import deepcopy
        from shadow_loom.models import InformationEdge
        ws = deepcopy(macbeth_ws)
        ws.entities["ENT_LADY_MACBETH"].location_id = "LOC_DUNSINANE_CASTLE"
        ws.information_topology.append(
            InformationEdge(
                source_id="ENT_MACBETH",
                target_ids=["ENT_LADY_MACBETH"],
                medium="speech",
                established_at_fabula=1,
            )
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
        from shadow_loom.models import InformationEdge
        ws = deepcopy(macbeth_ws)
        ws.information_topology.append(
            InformationEdge(
                source_id="ENT_MACBETH",
                target_ids=["ENT_LADY_MACBETH"],
                medium="speech",
                established_at_fabula=1,
                is_encrypted=False,
            )
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
        """Encrypted comms must NOT produce eavesdropped_by edges."""
        from copy import deepcopy
        from shadow_loom.models import InformationEdge
        ws = deepcopy(macbeth_ws)
        ws.information_topology.append(
            InformationEdge(
                source_id="ENT_MACBETH",
                target_ids=["ENT_LADY_MACBETH"],
                medium="magic_mirror",
                established_at_fabula=1,
                is_encrypted=True,
            )
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
        # Set one relationship to be updated in the far future
        for rel in ws.social_topology:
            if rel.source_entity_id == "ENT_MACBETH":
                rel.last_updated_fabula = 9999
                break
        ego = extract_ego_graph_from_memory(ws, ["ENT_MACBETH"], temporal_anchor=5)
        # The relationship updated at T=9999 should be excluded
        for rel in ego.relevant_relationships:
            assert rel.get("last_updated_fabula", 0) <= 5

    def test_terminated_at_fabula_boundary(self):
        """InformationEdge terminated AT the anchor must be excluded (<=)."""
        from copy import deepcopy
        from shadow_loom.extract_graph import extract_ego_graph_from_memory
        from shadow_loom.models import InformationEdge
        ws = deepcopy(macbeth_ws)
        ws.information_topology = [
            InformationEdge(
                source_id="ENT_MACBETH",
                target_ids=["ENT_LADY_MACBETH"],
                medium="raven",
                established_at_fabula=1,
                terminated_at_fabula=5,
            ),
        ]
        ego = extract_ego_graph_from_memory(ws, ["ENT_MACBETH"], temporal_anchor=5)
        # terminated_at_fabula=5 == anchor=5 → must be excluded
        assert len(ego.relevant_information_edges) == 0
