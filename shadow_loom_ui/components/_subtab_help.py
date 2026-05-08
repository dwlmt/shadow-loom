# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Per-sub-tab help registry.

The main tabs (Story, Explorer, World, Causality, Reasoning, Audit,
Research, Editor, Export) each render a ``help_popover`` in their
header explaining the tab as a whole. Many of those tabs then split
into sub-tabs (e.g. *Causality > Affective Dashboard*, *World > Causal
Edges*) where users still need an in-place explanation of *what this
specific view shows* and *how to read it*.

Rather than scatter the prose across the panel files, every sub-tab
help body lives here keyed by a stable ``"<tab>.<subtab>"`` string.
The :func:`subtab_help` helper renders a compact right-aligned info
icon at the top of the panel that opens the corresponding markdown
card on click.
"""

from __future__ import annotations

from typing import Dict, Tuple

from nicegui import ui

from shadow_loom_ui.components.help_popover import help_popover


# Keyed "<tab>.<subtab>" → (title, body_md). Bodies are markdown.
_HELP: Dict[str, Tuple[str, str]] = {
    # ---------------- Reasoning ----------------
    "reasoning.events": (
        "Events — fabula timeline & inspector",
        (
            "An ordered list of every event in the current world state,"
            " with a compact inspector for the selected event.\n\n"
            "### What it shows\n"
            "- Fabula time, syuzhet (reading) index, type, actors,"
            " targets, and natural-language description.\n"
            "- Click an event to deep-link the cursor across the other"
            " sub-tabs (the Trace, Belief lens, and Attribution panels"
            " all retarget to it).\n\n"
            "### When it's empty\n"
            "Run an extraction or load an example world to populate the"
            " event list."
        ),
    ),
    "reasoning.trace": (
        "Trace — last-query reasoning trace",
        (
            "The engine's step-by-step trace for the most recent rung-2"
            " (intervention) or rung-3 (counterfactual) query in this"
            " session.\n\n"
            "### What it shows\n"
            "- Each physics step: which entities were touched, which"
            " edges fired, which propagations were blocked.\n"
            "- The directive that drove generation, the audit"
            " convergence outcome, and the final prose.\n\n"
            "### When it's empty\n"
            "The trace populates only after a rung-2 or rung-3 query"
            " runs. Use the **command bar** (Intervene / What-If) or"
            " the *What-If Workbench* in the Causality tab."
        ),
    ),
    "reasoning.belief": (
        "Belief lens — per-entity epistemic state",
        (
            "What each entity *believes* about the others at the"
            " selected fabula tick — separated from what is actually"
            " true (objective state).\n\n"
            "### How to read it\n"
            "- Pick an observer; the lens shows that entity's belief"
            " set as confidence-weighted assertions about traits,"
            " locations, and relationships of others.\n"
            "- Divergence between belief and ground truth drives"
            " *dramatic irony* and *mystery* in the affective calculus."
        ),
    ),
    "reasoning.channels": (
        "Hidden channels — covert information flow",
        (
            "Standing information channels that aren't part of any"
            " single utterance — e.g. *body language*, *room acoustics*,"
            " *shared diary*, *psychic bond*.\n\n"
            "### What it shows\n"
            "- Each channel's medium, parties, fidelity, and any"
            " reception filters.\n"
            "- Channels that the engine is currently using to leak"
            " information between characters or to the reader."
        ),
    ),
    "reasoning.attribution": (
        "Why this? — attribution graph",
        (
            "For the selected event, the directed sub-graph of *direct*"
            " and *indirect* causes that the engine considers"
            " responsible for it occurring.\n\n"
            "### How to read it\n"
            "- Edges are coloured by causal type and weighted by"
            " causal force.\n"
            "- Hover any node for the raw event id and metadata.\n"
            "- Use this to debug *why* a generated outcome came out"
            " the way it did."
        ),
    ),
    "reasoning.foreshadow": (
        "Foreshadowing — anticipation→payoff matches",
        (
            "Pairs of earlier *anticipations* (utterances, beliefs,"
            " ambient cues) that point at later *payoffs* — the engine's"
            " explicit foreshadowing ledger.\n\n"
            "### What it shows\n"
            "- Each row is an anticipation→payoff arc with a"
            " confidence score and the syuzhet distance between them.\n"
            "- Use the audit tab's *Top fixes* to plant new"
            " foreshadowing where payoffs feel unearned."
        ),
    ),
    "reasoning.convergence": (
        "Convergence — audit-loop iteration chart",
        (
            "Per-iteration audit scores for the most recent generative"
            " query: shows whether the auditor accepted on the first"
            " pass or pushed back, and on what dimensions.\n\n"
            "### How to read it\n"
            "- X-axis: audit iteration (0 = initial draft).\n"
            "- Lines: plausibility, foreshadowing, emotional fit.\n"
            "- A plateau below threshold means the auditor gave up;"
            " a curve crossing the line means it converged."
        ),
    ),

    # ---------------- Causality (top-level sub-tabs) ----------------
    "causality.topology": (
        "Causal Topology — typed multi-aspect graph",
        (
            "The directed graph of cause→effect edges between events,"
            " with optional social, spatial, and information slices"
            " overlaid.\n\n"
            "### Controls\n"
            "- **Aspect** — which slice to draw (causal, social,"
            " spatial, information, or combined).\n"
            "- **Layout** — *Force* clusters tightly entangled events;"
            " *Circular* / *Cartesian* are useful for sparse graphs.\n"
            "- **Sankey Flow** shows the volume of causal flow between"
            " event clusters as a separate diagram.\n\n"
            "### Reading the diagram\n"
            "- Solid edges are causal; dashed are social; dotted are"
            " informational.\n"
            "- Edge thickness = strength / confidence.\n"
            "- Hover any node or edge for tooltips with raw ids."
        ),
    ),
    "causality.evolution": (
        "Evolution — topology over fabula time",
        (
            "How the graph grows tick-by-tick. Step the time cursor and"
            " watch new events and edges appear.\n\n"
            "### Sub-views\n"
            "- **Character** — a single entity's trait trajectory.\n"
            "- **Relationship** — affinity / fear / power between two"
            " entities over time.\n"
            "- **World Trait** — a global trait's magnitude over time.\n"
            "- **Causal Graph** — incremental graph snapshot at the"
            " selected tick.\n"
            "- **Physics** — engine-internal physics step traces."
        ),
    ),
    "causality.whatif": (
        "What-If Workbench — visual counterfactual builder",
        (
            "Build interventions or counterfactuals visually without"
            " writing prose:\n\n"
            "1. Pick a target node (event, entity, or trait).\n"
            "2. Specify the change (set value, block edge, add edge).\n"
            "3. Run.\n"
            "4. Inspect mutations, blocked propagations, and hidden"
            " deltas in the result panel.\n\n"
            "Use the *Causal Propagation* expansion to see the"
            " step-by-step physics."
        ),
    ),
    "causality.directive": (
        "Directive Builder — emotional directive without prose",
        (
            "Craft a directive (target effect, target entities,"
            " intensity) and inspect the assembled brief without"
            " running generation.\n\n"
            "Useful for understanding which levers the engine has and"
            " what a directive looks like before it hits the LLM."
        ),
    ),
    "affective": (
        "Affective Dashboard — heatmaps over syuzhet",
        (
            "Per-syuzhet-index heatmaps of *suspense*, *surprise*,"
            " *dramatic irony*, *love*, and *regret*.\n\n"
            "### How to read it\n"
            "- Brighter cells = higher emotional intensity at that"
            " reading position.\n"
            "- Spikes mark dramatic peaks; long sags often indicate"
            " pacing problems the audit will flag.\n"
            "- The *Event Timeline* below pins specific events to the"
            " heatmap columns."
        ),
    ),

    # ---------------- Causality > Evolution sub-views ---------------
    "causality.evolution.character": (
        "Character — single-entity trait trajectory",
        (
            "Trait values for the selected entity over fabula time."
            " Each line is one trait (e.g. *trust*, *fear*).\n\n"
            "Use this to see when a character changed and what the"
            " engine attributed the change to."
        ),
    ),
    "causality.evolution.relationship": (
        "Relationship — pairwise dynamics over time",
        (
            "Affinity, fear, and power between two selected entities"
            " plotted over fabula time, with the inertia (resistance"
            " to change) shown as a band."
        ),
    ),
    "causality.evolution.world": (
        "World Trait — global magnitude over time",
        (
            "How a single global trait (e.g. *dread*, *order*) evolves"
            " as events fire."
        ),
    ),
    "causality.evolution.causal": (
        "Causal Graph — snapshot at the cursor",
        (
            "The cumulative causal subgraph as it stands at the"
            " selected fabula tick. Step the cursor to watch edges"
            " appear in reading order."
        ),
    ),
    "causality.evolution.physics": (
        "Physics — engine-internal step traces",
        (
            "Low-level physics traces — what the propagation engine"
            " did at each step: mutations applied, edges fired,"
            " propagations blocked, with raw deltas."
        ),
    ),

    # ---------------- Causality > What-If propagation tabs ---------
    "causality.whatif.waterfall": (
        "Waterfall — propagation step-by-step",
        (
            "A waterfall chart of every mutation the physics engine"
            " applied, in order, including blocked propagations shown"
            " as cancelled bars."
        ),
    ),
    "causality.whatif.graph": (
        "Graph — propagation as a directed graph",
        (
            "The same propagation rendered as a node-edge graph:"
            " sources fan out to their downstream mutations."
        ),
    ),
    "causality.whatif.table": (
        "Table — propagation rows",
        (
            "Tabular dump of mutations and blocks: step, entity,"
            " trait, delta, from→to value. Sort/filter for debugging."
        ),
    ),

    # ---------------- Causality > Affective raw-data --------------
    "affective.events": (
        "Events — raw rows for the affective view",
        (
            "Tabular dump of every event scored by the affective"
            " calculus on this branch: id, fabula t, syuzhet index,"
            " type, actors, targets, description."
        ),
    ),
    "affective.affect": (
        "Affect — per-metric scores",
        (
            "Aggregated affective scores for the current branch:"
            " one row per metric (mystery, dramatic irony, suspense,"
            " surprise, love, regret) with the engine's value."
        ),
    ),

    # ---------------- World tab ----------------
    "world.entities": (
        "Entities — character roster",
        (
            "Every entity in the current world snapshot with status,"
            " location, trait/belief counts, and identity constants."
        ),
    ),
    "world.events": (
        "Events — chronological event log",
        (
            "All events in the current snapshot ordered by fabula"
            " time, with reading-order index, type, actors, targets,"
            " and description."
        ),
    ),
    "world.objects": (
        "Objects — narrative objects",
        (
            "Inventory of narrative objects (props, locations of"
            " significance) with location, owner, affordances, and"
            " arbitrary properties."
        ),
    ),
    "world.world_traits": (
        "World Traits — global atmosphere magnitudes",
        (
            "Global traits (e.g. *dread*, *order*, *time of day*)"
            " with their current magnitude and inertia (resistance"
            " to change)."
        ),
    ),
    "world.trait_stats": (
        "Trait Stats — distributional summaries",
        (
            "Per-trait box-plot summary across all entities: min,"
            " Q1, median, Q3, max, and outlier count. Useful for"
            " spotting traits that are uniformly extreme."
        ),
    ),
    "world.causal": (
        "Causal Edges — typed cause→effect ledger",
        (
            "All causal edges in the current snapshot with source,"
            " target, type, mechanism, force, and supporting"
            " evidence event ids."
        ),
    ),
    "world.spatial": (
        "Spatial Edges — locations & barriers",
        (
            "Spatial topology: which locations connect to which,"
            " whether the connection is locked, and any barrier"
            " description (door, wall, river)."
        ),
    ),
    "world.social": (
        "Social Edges — relationship graph",
        (
            "Directed social ties between entities with affinity,"
            " fear, power, and inertia. Asymmetric — A→B can differ"
            " from B→A."
        ),
    ),
    "world.info": (
        "Info Edges — channels & utterances",
        (
            "Information topology: standing channels (capabilities)"
            " and discrete utterances (transmissions) with their"
            " medium, parties, fidelity, and reception."
        ),
    ),
    "social.propositions": (
        "Propositions \u2014 first-class storyworld facts",
        (
            "Each row is a Proposition with kind, current truth"
            " value at the cursor, audience prior, narrative stakes,"
            " and counts of concerns and beliefs that reference it."
        ),
    ),
    "social.concerns": (
        "Concerns \u2014 per-entity desires and fears",
        (
            "Each row is one entity's concern about a proposition"
            " with polarity (desire/fear), salience, optional kind"
            " label, and whether it is currently active at the"
            " cursor."
        ),
    ),
    "social.beliefs": (
        "Beliefs \u2014 who thinks what about whom",
        (
            "Each row is one entity's belief about a target"
            " (entity / object / location): the perceived state,"
            " confidence, inertia, when it was acquired, and the"
            " channel or event that delivered it. Filtered to"
            " beliefs established at or before the fabula cursor."
        ),
    ),
    "social.relationships": (
        "Relationships \u2014 dyadic affinity, fear, power",
        (
            "Each row is one directed RelationshipEdge with the"
            " three core metrics, evidence strength, the number of"
            " axes actually observed (vs missing), and the fabula"
            " time of the most recent update."
        ),
    ),
}


def subtab_help(key: str) -> None:
    """Render a right-aligned info icon for the given sub-tab key.

    Silently no-ops on unknown keys so panel code can call it
    unconditionally without breaking when help copy is still TODO.
    """
    entry = _HELP.get(key)
    if not entry:
        return
    title, body_md = entry
    with ui.row().classes("w-full justify-end items-center gap-1 -mt-1 -mb-1"):
        help_popover(
            title=title,
            body_md=body_md,
            tooltip="What is this sub-tab?",
        )
