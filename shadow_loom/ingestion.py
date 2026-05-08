# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Text-to-WorldState Register-Hybrid Extraction Pipeline.

Five-step LLM extraction using PydanticAI + Ollama:
  Step 1 — Global Coreference Pre-Pass (three separate passes → GlobalRegister)
    Step 1a — Location extraction (LOC_ nodes)
    Step 1b — Object extraction (OBJ_ nodes, with location context)
    Step 1c — Entity extraction (ENT_ nodes, with location + object context)
  Step 2 — Semantic Scaffolding (Socratic QA per chunk)
  Step 3 — Decomposed Topology Extraction (Physics + Social + Consequences agents per chunk)
  Step 4 — Pydantic Propose-Critique-Repair (per-chunk result validation)
  Step 5 — Global Assembly + Mathematical Sorting + Validation + Correction
  Step 5b — Post-Assembly World Trait Timeline Extraction (single focused LLM pass)

All LLM system prompts are loaded from external markdown files
in the ``prompts/`` directory inside the package.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import re
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Set, Tuple

if TYPE_CHECKING:
    from shadow_loom.research import WorldFact

from pydantic import BaseModel, Field, model_validator
from pydantic_ai import Agent, ModelRetry, NativeOutput, PromptedOutput, RunContext

from shadow_loom.settings import get_settings as _get_settings, resolve_model as _resolve_model

from shadow_loom.models import (
    AmbientVector,
    Belief,
    CausalEdge,
    Channel,
    Concern,
    ConcernSnapshot,
    Entity,
    EntityStateSnapshot,
    EventNode,
    GlobalTrait,
    Location,
    NarrativeObject,
    Proposition,
    PropositionSnapshot,
    RelationshipEdge,
    SpatialEdge,
    TraitVector,
    WorldStateV1,
    WorldTraitSnapshot,
)
from shadow_loom._agent_logging import log_agent_output
from shadow_loom.narrative_style import infer_narrative_style

logger = logging.getLogger(__name__)

# =====================================================================
# Prompts directory — lives inside the package
# =====================================================================
_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def _load_prompt(filename: str) -> str:
    """Read a markdown prompt file from the ``prompts/`` directory."""
    path = _PROMPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


# =====================================================================
# Extraction-Specific Pydantic Models
# =====================================================================


class GlobalRegister(BaseModel):
    """Step 1 output: the static ontology (nouns) of the narrative."""
    locations: Dict[str, Location] = Field(
        description="All unique locations keyed by LOC_ IDs (e.g. LOC_INVERNESS_CASTLE).",
    )
    objects: Dict[str, NarrativeObject] = Field(
        description="All unique narrative objects keyed by OBJ_ IDs (e.g. OBJ_DAGGER).",
    )
    entities: Dict[str, Entity] = Field(
        description="All unique entities (characters, groups) keyed by ENT_ IDs (e.g. ENT_MACBETH).",
    )
    world_traits: Dict[str, GlobalTrait] = Field(
        default_factory=dict,
        description="World-level facts, laws, and conditions keyed by WORLD_ IDs.",
    )


class LocationRegister(BaseModel):
    """Step 1a output: all unique locations extracted from the narrative."""
    locations: Dict[str, Location] = Field(
        description="All unique locations keyed by LOC_ IDs (e.g. LOC_INVERNESS_CASTLE).",
    )


class ObjectRegister(BaseModel):
    """Step 1b output: all unique narrative objects extracted from the narrative."""
    objects: Dict[str, NarrativeObject] = Field(
        description="All unique narrative objects keyed by OBJ_ IDs (e.g. OBJ_DAGGER).",
    )


class EntityRegister(BaseModel):
    """Step 1c output: all unique entities (characters, groups) extracted from the narrative."""
    entities: Dict[str, Entity] = Field(
        description="All unique entities (characters, groups) keyed by ENT_ IDs (e.g. ENT_MACBETH).",
    )


class WorldTraitsRegister(BaseModel):
    """Step 1d output: world-level facts, laws, and conditions."""
    world_traits: Dict[str, GlobalTrait] = Field(
        description="All world-level traits keyed by WORLD_ IDs (e.g. WORLD_SURVEILLANCE_STATE).",
    )


class ChunkPropositionSnapshot(BaseModel):
    """Per-chunk wire format for a Proposition framing-shift snapshot.

    Mirrors :class:`shadow_loom.models.PropositionSnapshot` but adds the
    ``proposition_id`` discriminator so a single per-chunk payload can
    carry snapshots for many propositions. The reconciler (Phase C)
    routes each entry into ``Proposition.state_timeline`` and strips
    the discriminator before persisting.
    """
    proposition_id: str = Field(
        description="Catalogue PROP_ id this snapshot applies to.",
    )
    fabula_time: int
    triggered_by: str = Field(
        description="EVT_ id from this chunk that caused the framing shift.",
    )
    stakes: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    audience_default_prior: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    description: Optional[str] = None


class PropositionTruthCommit(BaseModel):
    """Per-chunk wire format for a Proposition ground-truth commitment.

    Folded into ``Proposition.truth_at_fabula`` by the reconciler.
    ``triggered_by`` lets the auditor explain *why* the commit fires
    at this fabula tick (typically an outcome / revelation event).
    """
    proposition_id: str = Field(
        description="Catalogue PROP_ id whose truth value commits.",
    )
    fabula_time: int = Field(
        description="Fabula tick at which the commit fires; must equal triggered_by's fabula_time.",
    )
    truth: bool = Field(
        description="True if the proposition resolves true; False if false.",
    )
    triggered_by: str = Field(
        description="EVT_ id from this chunk that resolves the proposition.",
    )


class ChunkConcernSnapshot(BaseModel):
    """Per-chunk wire format for a Concern drift snapshot.

    Mirrors :class:`shadow_loom.models.ConcernSnapshot` but adds the
    ``concern_id`` discriminator so a single per-chunk payload can
    carry snapshots for many concerns. The reconciler (Phase C) routes
    each entry into ``Concern.state_timeline``.
    """
    concern_id: str = Field(
        description="Catalogue CCN_ id this snapshot applies to.",
    )
    fabula_time: int
    triggered_by: str = Field(
        description="EVT_ id from this chunk that caused the drift.",
    )
    salience: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    polarity: Optional[Literal["desire", "fear"]] = None
    activation_fabula_window: Optional[List[int]] = None
    counter_concern_ids: Optional[List[str]] = None
    kind: Optional[str] = None


class ChunkAffectExtraction(BaseModel):
    """Step 3d output (Phase B4): per-chunk affect deltas.

    Owned exclusively by the Affect Agent; no other per-chunk
    extractor may emit any of these fields. Reconciled into
    ``WorldStateV1.propositions`` and ``Entity.concerns`` by the
    Phase C affect-unification reconciler.
    """
    proposition_snapshots: List[ChunkPropositionSnapshot] = Field(default_factory=list)
    proposition_truth_commits: List[PropositionTruthCommit] = Field(default_factory=list)
    concern_snapshots: List[ChunkConcernSnapshot] = Field(default_factory=list)
    new_concern_seeds: List["ConcernSeed"] = Field(default_factory=list)


class ChunkTopology(BaseModel):
    """Merged output per chunk: events + all edge types. Used by assembly.

    ``events`` includes both Physics-extracted events AND any
    ``event_type='utterance'`` events emitted by the Social Agent
    (utterance EventNodes are merged in by ``_extract_single_chunk*``).
    ``channels`` carries standing :class:`Channel` capabilities
    extracted by the Social Agent, replacing the legacy
    ``information_topology`` field.

    The ``new_*`` collections carry brand-new ontology nodes promoted
    from a Rung-2/3 sandbox after a ``.spawn`` (genesis) intervention.
    They are empty for plain ingestion chunks; the pipeline populates
    them when a query introduces characters / objects / locations /
    world-traits that did not exist in the canonical world state.

    The Phase B4 affect collections (``proposition_snapshots`` /
    ``proposition_truth_commits`` / ``concern_snapshots`` /
    ``new_concern_seeds``) carry the per-chunk Affect Agent's diff
    against the catalogue. They are consumed by the Phase C
    reconciler and otherwise pass through unchanged.
    """
    events: List[EventNode] = Field(default_factory=list)
    causal_topology: List[CausalEdge] = Field(default_factory=list)
    channels: Dict[str, Channel] = Field(default_factory=dict)
    social_topology: List[RelationshipEdge] = Field(default_factory=list)
    spatial_topology: List[SpatialEdge] = Field(default_factory=list)
    entity_updates: List["EntityUpdate"] = Field(default_factory=list)
    # Phase B4 affect outputs (default empty when affect stage skipped).
    proposition_snapshots: List[ChunkPropositionSnapshot] = Field(default_factory=list)
    proposition_truth_commits: List[PropositionTruthCommit] = Field(default_factory=list)
    concern_snapshots: List[ChunkConcernSnapshot] = Field(default_factory=list)
    new_concern_seeds: List["ConcernSeed"] = Field(default_factory=list)
    # Genesis spawns (post-physics promotion). Keyed by canonical id.
    new_entities: Dict[str, Entity] = Field(default_factory=dict)
    new_objects: Dict[str, NarrativeObject] = Field(default_factory=dict)
    new_locations: Dict[str, Location] = Field(default_factory=dict)
    new_world_traits: Dict[str, GlobalTrait] = Field(default_factory=dict)
    # Affect-layer genesis: brand-new propositions/concerns introduced
    # by this chunk (re-extracted prose, sandbox spawn, or directive
    # seed). The merge step inserts these into the world before folding
    # ``proposition_truth_commits`` / ``concern_snapshots`` so the
    # snapshots have a target to attach to.
    new_propositions: Dict[str, Proposition] = Field(default_factory=dict)
    new_concerns: Dict[str, List[Concern]] = Field(
        default_factory=dict,
        description="entity_id → list of brand-new Concern records to attach.",
    )

    # --- Deletion vocabulary (all default empty; consumed by the merge
    # step's deletion pass before additive sections run, so a single
    # merge can replace-then-add cleanly). All counters are recorded
    # into the resulting MergeChangeset.
    removed_event_ids: List[str] = Field(default_factory=list)
    removed_causal_edge_keys: List[Tuple[str, str, str, int]] = Field(
        default_factory=list,
        description="(source_id, target_id, causality_type, fabula_time) keys to drop from causal_topology.",
    )
    removed_social_dyad_keys: List[Tuple[str, str]] = Field(
        default_factory=list,
        description="(source_entity_id, target_entity_id) dyads to drop from social_topology.",
    )
    removed_spatial_keys: List[Tuple[str, str]] = Field(
        default_factory=list,
        description="(source_id, target_id) keys to drop from spatial_topology.",
    )
    removed_channel_ids: List[str] = Field(default_factory=list)
    removed_entity_ids: List[str] = Field(default_factory=list)
    removed_object_ids: List[str] = Field(default_factory=list)
    removed_location_ids: List[str] = Field(default_factory=list)
    removed_world_trait_ids: List[str] = Field(default_factory=list)
    removed_proposition_ids: List[str] = Field(default_factory=list)
    removed_concern_ids: List[Tuple[str, str]] = Field(
        default_factory=list,
        description="(entity_id, concern_id) pairs to drop from Entity.concerns.",
    )

    # Supersession (mainline-promoted counterfactual override). Maps
    # ``new_event_id → old_event_id``; the merge step marks the old
    # event with ``superseded_by_event_id`` and rewrites downstream
    # references (``triggered_by``, ``acquired_via_event_id``,
    # ``resolves_proposition_ids``) onto the new id.
    supersedes_event_ids: Dict[str, str] = Field(default_factory=dict)


class QAPair(BaseModel):
    """A single Socratic question-answer pair from semantic scaffolding."""
    category: Literal["who", "what", "where", "when", "why", "how"] = Field(
        description="The interrogative category of this QA pair.",
    )
    question: str = Field(description="The question about this chunk of text.")
    answer: str = Field(description="The answer, articulating implicit reasoning and hidden variables.")


class SocraticScaffold(BaseModel):
    """Step 2 output: semantic scaffolding QA pairs for a single chunk."""
    qa_pairs: List[QAPair] = Field(
        default_factory=list,
        description="Who/What/Where/When/Why/How pairs articulating the "
        "narrative logic of this chunk before structured extraction.",
    )


class PhysicsExtraction(BaseModel):
    """Step 3a output: events + causal/spatial edges from the Physics Agent."""
    events: List[EventNode] = Field(default_factory=list)
    causal_topology: List[CausalEdge] = Field(default_factory=list)
    spatial_topology: List[SpatialEdge] = Field(default_factory=list)
    entity_updates: List["EntityUpdate"] = Field(default_factory=list)


class SocialExtraction(BaseModel):
    """Step 3b output from the Social Agent.

    Replaces the legacy ``information_topology`` field with a two-part
    split that mirrors the model:
      * ``channels`` — standing :class:`Channel` capabilities
        (telephone, mind-link, classified pipeline, ongoing
        correspondence). Keyed by CHN_ id.
      * ``utterance_events`` — discrete
        :class:`EventNode` records with ``event_type='utterance'``
        modelling on-page speech-acts. These are merged into the
        global ``events`` list during chunk-topology assembly so they
        participate in normal causal/temporal physics.
    """
    channels: Dict[str, Channel] = Field(default_factory=dict)
    utterance_events: List[EventNode] = Field(default_factory=list)
    social_topology: List[RelationshipEdge] = Field(default_factory=list)


class ConsequencesExtraction(BaseModel):
    """Step 3c output: per-entity state deltas from the Consequences Agent.

    Run *after* the Physics Agent so it can ground each EntityUpdate in the
    actual events and mutation edges produced by Physics. Decoupling this
    from the Physics pass lets the LLM concentrate fully on the
    trait/belief/status accounting that previously had to share attention
    with event/edge extraction.
    """
    entity_updates: List["EntityUpdate"] = Field(default_factory=list)


class BeliefConfidenceUpdate(BaseModel):
    """Per-chunk overwrite of an existing belief's confidence (and optional inertia).

    Carried alongside :class:`EntityUpdate` so a Pearl-Rung-2 belief
    clamp (``BeliefMutation`` from the causal physics layer) can be
    persisted onto an existing :class:`Belief` without needing to
    create a new one. The merge step locates the belief by
    ``target_id`` (and ``proposition_id`` when set) on the entity's
    *current* belief list at merge time and overwrites its
    ``confidence`` (and ``inertia`` when supplied).
    """
    target_id: str = Field(description="target_id of the belief to update.")
    proposition_id: Optional[str] = Field(
        default=None,
        description="Optional PROP_ id discriminator when target_id is ambiguous.",
    )
    new_confidence: float = Field(ge=0.0, le=1.0)
    new_inertia: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class EntityUpdate(BaseModel):
    """Per-chunk delta: how an entity's state changed during this chunk."""
    entity_id: str = Field(description="ENT_ ID of the entity that changed.")
    fabula_time: int = Field(description="fabula_time when this change occurred.")
    triggered_by: Optional[str] = Field(default=None, description="EVT_ ID that caused this change.")
    trait_updates: Dict[str, TraitVector] = Field(
        default_factory=dict,
        description="Updated trait values. Only include traits that changed.",
    )
    new_beliefs: List[Belief] = Field(default_factory=list, description="New beliefs formed.")
    invalidated_belief_targets: List[str] = Field(
        default_factory=list,
        description="target_ids of beliefs shattered by this event.",
    )
    belief_confidence_updates: List[BeliefConfidenceUpdate] = Field(
        default_factory=list,
        description=(
            "Overwrite confidence (and optional inertia) on existing beliefs. "
            "Used by the Pearl-Rung-2 BeliefMutation bridge so do-surgery "
            "clamps persist without spawning duplicate beliefs."
        ),
    )
    new_status: Optional[Literal["healthy", "injured", "ill", "dead", "unconscious"]] = Field(
        default=None, description="New status if changed.",
    )
    new_location_id: Optional[str] = Field(default=None, description="New location if entity moved.")


# Resolve forward references now that EntityUpdate is defined.
# ``ChunkTopology`` and ``ChunkAffectExtraction`` additionally reference
# ``ConcernSeed`` (defined later for Phase A3), so their rebuild is
# deferred until after that class lands.
PhysicsExtraction.model_rebuild()
ConsequencesExtraction.model_rebuild()


class WorldTraitTimelineExtraction(BaseModel):
    """Post-assembly Step 5 output: inflection points for world traits."""
    timelines: Dict[str, List[WorldTraitSnapshot]] = Field(
        default_factory=dict,
        description=(
            "Mapping of WORLD_ ID → list of WorldTraitSnapshot entries. "
            "Each snapshot marks a moment where the world trait fundamentally "
            "changed (e.g., a war ends, a law is repealed, a regime falls)."
        ),
    )


# =====================================================================
# Canonical vocabularies — kept in sync with downstream physics engine
# (``shadow_loom.causal_physics.MECHANISM_TRAIT_MAP`` + the
# ``RelationshipEdge`` schema). Used by per-chunk output validators to
# auto-correct non-fatal LLM drift without paying a full ``ModelRetry``.
# =====================================================================
_CANONICAL_MECHANISMS: set[str] = {
    "physical", "physical_force",
    "psychological",
    "epistemic", "epistemic_revelation",
    "social", "social_coercion",
    "emotional",
    "informational",
    "betrayal",
}
# Custom labels we deliberately tolerate — they bypass the routing
# fallback (so they receive full impulse to all traits) and are common
# domain words the LLM legitimately reaches for.
_TOLERATED_MECHANISMS: set[str] = {
    "kinetic", "chemical", "seduction", "coercion", "deduction",
    "manipulation", "intimidation", "persuasion", "supernatural",
    "ritual", "biological", "environmental",
}
_RELATIONSHIP_METRICS: set[str] = {"affinity", "fear", "power_dynamic"}
# Common LLM synonyms that should map to a canonical relationship axis
# rather than triggering ``mutation_social`` edge rejection. The
# Pydantic ``RelationshipMetric.metrics`` keys are a closed Literal so
# the LLM can't inject a synonym there directly, but ``CausalEdge.
# trait_target`` is a free string and the ``mutation_social`` validator
# only accepts the canonical names. Without this map a perfectly valid
# ``mutation_social`` edge whose trait_target reads "trust" or "power"
# is silently dropped instead of repaired.
_RELATIONSHIP_METRIC_ALIASES: dict[str, str] = {
    # Direct truncations / informal forms.
    "power": "power_dynamic",
    "dominance": "power_dynamic",
    "authority": "power_dynamic",
    "control": "power_dynamic",
    "trust": "affinity",
    "love": "affinity",
    "friendship": "affinity",
    "rapport": "affinity",
    "intimacy": "affinity",
    # Negative-direction synonyms map to the same axis; the sign is
    # carried by ``trait_delta`` (the LLM is instructed to use a
    # negative delta for hostility / contempt, etc.).
    "hostility": "affinity",
    "contempt": "affinity",
    "resentment": "affinity",
    "fearfulness": "fear",
    "dread": "fear",
    "terror": "fear",
    "anxiety": "fear",
}
_VALID_STATUSES: set[str] = {"healthy", "injured", "ill", "dead", "unconscious"}


class ValidationIssue(BaseModel):
    """A single problem found during validation."""
    severity: Literal["error", "warning"] = Field(
        description="'error' = must fix, 'warning' = informational",
    )
    category: str = Field(
        description="e.g. 'hallucinated_id', 'broken_link', 'contradiction', 'duplicate', 'orphan'",
    )
    detail: str = Field(description="Human-readable description of the issue.")


class ValidationReport(BaseModel):
    """Step 3 output: structured audit of an assembled WorldStateV1."""
    is_valid: bool = Field(description="True if no errors were found (warnings are OK).")
    issues: List[ValidationIssue] = Field(
        default_factory=list,
        description="All issues found, both errors and warnings.",
    )
    suggestions: List[str] = Field(
        default_factory=list,
        description="Recommended fixes or improvements.",
    )
    repairs: List[str] = Field(
        default_factory=list,
        description=(
            "Human-readable log of every auto-repair / correction-patch "
            "change applied during the pipeline (in order). Surfaces "
            "silent fixes \u2014 e.g. nulled location_id, renamed "
            "duplicate EVT_ ids, fabula shifts, dropped self-loops \u2014 "
            "so downstream UIs and tests can audit them."
        ),
    )


def _ext_defaults() -> dict:
    return _get_settings().extraction_config()


class ExtractionConfig(BaseModel):
    """Runtime configuration for the extraction pipeline."""
    model: str = Field(
        default="ollama:qwen3.6:35b",
        description="PydanticAI model string (e.g. 'ollama:qwen3.6:35b', 'openai:gpt-4o').",
    )
    chunk_strategy: Literal["act_headings", "paragraph"] = Field(
        default="act_headings",
        description="How to split the input text into chunks for Step 2.",
    )
    output_retries: int = Field(
        default=5,
        description="Max retries for PydanticAI output validation.",
    )
    fabula_time_spacing: int = Field(
        default=1000,
        description="Base spacing between fabula_time values (e.g. 1000 → 1000, 2000, 3000). "
        "Gaps allow flashbacks and interstitial events to be inserted later.",
    )
    min_chunk_chars: int = Field(
        default=800,
        description="Minimum chunk size in characters. Adjacent small paragraphs are "
        "merged until they reach this threshold. Lowered from 1500 so each "
        "physics/social pass sees roughly one scene rather than a whole act, "
        "which produced anonymous events and missing dyads on dense texts.",
    )
    chunk_overlap_chars: int = Field(
        default=300,
        description="Number of trailing characters from the previous chunk to prepend "
        "as context for the next chunk. Helps maintain coreference across "
        "chunk boundaries.",
    )
    max_chunk_chars: int = Field(
        default=1200,
        ge=400,
        description=(
            "Hard cap on a single chunk's character length before it is "
            "subdivided on paragraph boundaries (audit fix #6). The "
            "previous fixed module-level cap of 1200 is preserved as the "
            "default, but raising this lets a stronger model digest "
            "scene-sized blocks in one pass; lowering it forces denser "
            "subdivision on dialogue-heavy texts where the LLM tends to "
            "lose participants."
        ),
    )
    enable_chunk_carry_over: bool = Field(
        default=False,
        description=(
            "When true, chunks are extracted **serially** in syuzhet "
            "order so that each chunk's deps see the prior chunk's "
            "event ids and standing channels (``previous_event_ids`` / "
            "``previous_chunk_channels``). Audit fix #2: the parallel "
            "default leaves both lists empty, which breaks coreference "
            "across chunk boundaries and lets the same standing "
            "capability be re-invented as a near-duplicate ``CHN_`` per "
            "chunk. Off by default because serial dispatch sacrifices "
            "the ``max_concurrent_chunks`` speedup; enable for "
            "long-form prose where continuity matters more than wall "
            "time."
        ),
    )
    scaffold_drift_retry: bool = Field(
        default=True,
        description=(
            "When true (audit fix #8), a chunk whose Physics output "
            "covers <50% of the entities the Socratic scaffold flagged "
            "as on-page triggers a single Physics retry that names the "
            "missed entities explicitly. Off-by-default during the "
            "scaffold-drift-only diagnostic, this is now active so the "
            "warning becomes a corrective action."
        ),
    )
    fabula_monotonicity_retry: bool = Field(
        default=True,
        description=(
            "When true (audit fix #8), a chunk with retrograde "
            "syuzhet/fabula event pairs (events that move *backwards* "
            "in story-time without any flashback marker) triggers a "
            "single Physics retry asking the agent to either re-emit "
            "the events with corrected fabula_time or mark them as "
            "deliberate flashbacks via negative fabula_time."
        ),
    )
    second_drift_pass: bool = Field(
        default=False,
        description=(
            "When true, after the Social agent has run (and after any "
            "social re-sync), re-evaluate scaffold drift counting BOTH "
            "physics event participants and social utterance/channel "
            "participants. If scaffold-flagged entities are still "
            "missing AND the first drift retry already executed, fire "
            "a second targeted Physics retry naming the still-missed "
            "entities. Off by default \u2014 the first drift pass is "
            "usually sufficient and a second pass costs an extra LLM "
            "round-trip per affected chunk."
        ),
    )
    pipeline_checkpoint_dir: Optional[str] = Field(
        default=None,
        description=(
            "If set, the orchestrator additionally persists the "
            "post-Step-1 ``GlobalRegister`` and the post-Step-2 list "
            "of ``ChunkTopology`` to "
            "``<pipeline_checkpoint_dir>/register_<hash>.json`` and "
            "``<pipeline_checkpoint_dir>/topologies_<hash>.json`` "
            "respectively (hash = sha256(text) + extraction "
            "fingerprint). On a subsequent run with the same text + "
            "config, these are loaded from disk and the matching "
            "pipeline step is skipped wholesale. Independent of the "
            "per-chunk ``checkpoint_dir``; both can be set together."
        ),
    )
    max_correction_retries: int = Field(
        default=5,
        description="Maximum correction passes after validation. Each pass feeds "
        "programmatic errors back to the LLM for targeted repair.",
    )
    validation_payload_max_chars: int = Field(
        default=600_000,
        description=(
            "Hard char cap on the WorldStateV1 JSON sent to the LLM "
            "validator (Step 3 Phase B). When the serialised state "
            "exceeds this, the validator switches to a compact "
            "projection (or, as a last resort, truncates). The default "
            "(~600K chars \u2248 ~150K tokens) is sized for a 256K-token "
            "context window with comfortable headroom for the system "
            "prompt and the validator's structured-output response. "
            "Lower this when targeting a smaller-context model."
        ),
    )
    correction_subgraph_threshold_chars: int = Field(
        default=400_000,
        description=(
            "When the WorldStateV1 JSON sent to the correction-patch "
            "agent exceeds this size, fall back to an error-relevant "
            "subgraph (events named in the errors + their immediate "
            "causal neighbours + ontology header) instead of the full "
            "state. The default (~400K chars \u2248 ~100K tokens) keeps "
            "the patch contract rich enough for the LLM to reason "
            "across the whole topology on a 256K-token model while "
            "still cutting over before the prompt would crowd out the "
            "patch response."
        ),
    )
    max_concurrent_chunks: int = Field(
        default=12,
        ge=1,
        description="Maximum number of chunks to extract in parallel during "
        "async topology extraction. Controls LLM request concurrency. Must be >= 1; "
        "a value of 0 would create ``asyncio.Semaphore(0)`` and hang every chunk.",
    )
    per_chunk_timeout_seconds: float = Field(
        default=1200.0,
        ge=0,
        description="Per-chunk soft timeout (seconds) for the entire "
        "Socratic\u2192Physics\u2192Social\u2192Consequences pipeline on a "
        "single chunk. When >0, a wedged LLM call is cancelled and that "
        "chunk yields an empty ChunkTopology so the rest of the run can "
        "proceed; the chunk's stage flags are all set to 1 so the "
        ">50%-of-chunks failure threshold still triggers if many chunks "
        "time out. Default 1200s (20 min) accommodates slow local models "
        "with multiple retries; set to 0 to disable.",
    )
    estimated_events_per_chunk: int = Field(
        default=10,
        description="Estimated events per chunk — used to pre-allocate syuzhet "
        "and fabula_time ranges for parallel extraction.",
    )
    enable_consequences_agent: bool = Field(
        default=True,
        description="If true, run a third per-chunk agent (Step 3c) that focuses "
        "exclusively on producing EntityUpdate records (trait/belief/status/location "
        "deltas) from the events Physics produced. When enabled, the Physics agent's "
        "own entity_updates output is discarded in favour of the Consequences output. "
        "Disable to fall back to the legacy two-agent (Physics + Social) split.",
    )

    enable_affect_agent: bool = Field(
        default=True,
        description=(
            "If true, run the Phase B4 per-chunk Affect agent (Step 3d) "
            "after Consequences. Emits proposition_snapshots, "
            "proposition_truth_commits, concern_snapshots and "
            "new_concern_seeds folded into Proposition.state_timeline / "
            "truth_at_fabula and Entity.concerns by the Phase C "
            "reconciler. The agent is gated per-chunk on the presence "
            "of an affect signal (catalogue PROP_ id touched by an "
            "event, belief, or seeded entity update); chunks with no "
            "signal skip the call entirely. Disable to fall back to "
            "static catalogue baselines (no per-chunk drift)."
        ),
    )

    enable_proposition_catalogue: bool = Field(
        default=True,
        description=(
            "If true, run the Phase A3 Proposition Catalogue agent over "
            "the full text after Step 1 ontology, producing a global "
            "PROP_ registry plus baseline ConcernSeed entries. The "
            "catalogue's PROP_ id list is threaded into every per-chunk "
            "extractor (Physics / Social / Consequences) so beliefs and "
            "utterances can carry canonical ``proposition_id`` / "
            "``asserts_proposition_id`` / ``denies_proposition_id`` / "
            "``resolves_proposition_ids`` at extraction time — removing "
            "the round-trip the post-pass belief-clustering stage would "
            "otherwise need. Disable to skip the catalogue and fall back "
            "to legacy post-pass clustering only."
        ),
    )

    # ------------------------------------------------------------------
    # Step 5c — entity concern extraction (Affect Unification, Step 2)
    # ------------------------------------------------------------------
    enable_concern_extraction: bool = Field(
        default=True,
        description=(
            "If true, after world-state assembly the pipeline runs a single "
            "focused LLM call per non-trivial entity to extract their standing "
            "``Concern`` ledger (fears + desires over the synthesised "
            "``Proposition`` register). Concerns drive the affect-unification "
            "layer's per-entity threat / hope surfacing. Cheap (~1 call/entity) "
            "and additive — disable to leave ``Entity.concerns`` empty for all "
            "entities (legacy behaviour). See "
            "``/memories/repo/affect-unification-plan.md`` Step 2."
        ),
    )
    concern_min_event_appearances: int = Field(
        default=2,
        description=(
            "Skip concern extraction for entities that appear in fewer than this "
            "many events (one-line side characters). Lowers cost on long "
            "manuscripts with large incidental casts."
        ),
    )

    # ------------------------------------------------------------------
    # Step 5d — belief proposition clustering (Affect Unification, Step 3)
    # ------------------------------------------------------------------
    enable_belief_clustering: bool = Field(
        default=True,
        description=(
            "If true, after concern extraction the pipeline runs an LLM "
            "clustering pass per (non-event) belief target to bind "
            "equivalent ``perceived_state`` strings to shared "
            "``Proposition`` records. The deterministic event-target "
            "backfill always runs regardless. Disable to leave non-event "
            "beliefs without a ``proposition_id`` (KL-based dramatic-irony "
            "scoring on those propositions then falls through to the "
            "audience prior). See ``/memories/repo/affect-unification-plan.md`` "
            "Step 3."
        ),
    )
    belief_cluster_min_beliefs: int = Field(
        default=2,
        description=(
            "Skip belief-clustering for targets with fewer than this many "
            "beliefs across all entities. Single-belief targets contribute "
            "no irony signal so the LLM call is wasted there."
        ),
    )

    # ------------------------------------------------------------------
    # Step 5e — audience-agent synthesis (Affect Unification, Step 4)
    # ------------------------------------------------------------------
    enable_audience_synthesis: bool = Field(
        default=True,
        description=(
            "If true, after concerns and belief clustering the pipeline "
            "synthesises the reserved ``ENT_AUDIENCE`` entity (deterministic, "
            "no LLM cost) by walking the syuzhet stream and emitting graded-"
            "confidence ``Belief`` snapshots at each event's fabula_time. "
            "This is the final substrate the unified affect scorers "
            "(suspense / surprise / dramatic irony / mystery) read against. "
            "Disable to leave the audience implicit (legacy behaviour); the "
            "unified scorers will then synthesise lazily on first call. See "
            "``/memories/repo/affect-unification-plan.md`` Step 4."
        ),
    )

    # ------------------------------------------------------------------
    # Step 3d — optional external research (off by default)
    # ------------------------------------------------------------------
    enable_research_agent: bool = Field(
        default=False,
        description=(
            "If true, after world-state assembly the pipeline calls the "
            "configured ``research_provider`` once per topic in "
            "``research_topics`` and a research-extraction agent distils "
            "each result into a ``WorldFact``. Facts are appended to "
            "``WorldStateV1.world_facts`` only — the agent is forbidden "
            "from mutating Entities, Events, RelationshipEdges or world "
            "traits. Off by default."
        ),
    )
    research_provider: Literal["none", "tavily"] = Field(
        default="none",
        description="Which ResearchProvider to use. 'none' = NullProvider (no-op).",
    )
    research_provider_model: str = Field(
        default="",
        description="Provider-specific search depth/model identifier (hashed into cache key).",
    )
    research_max_results_per_query: int = Field(
        default=5,
        description="Cap on snippets returned per provider call.",
    )
    research_topics: List[str] = Field(
        default_factory=list,
        description="Topics to look up at extraction time. May be empty.",
    )

    # ------------------------------------------------------------------
    # Tier 3 #11 — chunk-level checkpointing (off by default)
    # ------------------------------------------------------------------
    checkpoint_dir: Optional[str] = Field(
        default=None,
        description=(
            "If set, after each successfully-extracted chunk the "
            "pipeline persists ``ChunkTopology`` (plus the chunk text "
            "hash) to ``<checkpoint_dir>/chunk_<hash>_<index>.json``. "
            "On a subsequent run with the same chunks, those topologies "
            "are loaded from disk and the corresponding agent calls "
            "are skipped. Disabled when null."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _fill_from_settings(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for k, v in _ext_defaults().items():
                data.setdefault(k, v)
        return data


# =====================================================================
# Text Chunking
# =====================================================================

# Heading patterns that signal a new section (case-insensitive).
# Stave (Dickens — A Christmas Carol), Canto (Dante / Byron), and
# Scene (theatrical) added so that works which don't use Act/Chapter
# still get structural chunking instead of the paragraph fallback.
# Numeric/roman/word forms all accepted: "Chapter 3", "Act IV",
# "Stave One", "Canto the First" all match.
_NUMERAL_TOKEN = (
    r"(?:[IVXLCDM]+|\d+|"
    r"the\s+(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)|"
    r"first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
    r"eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|"
    r"seventeenth|eighteenth|nineteenth|twentieth|"
    r"one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|"
    r"seventeen|eighteen|nineteen|twenty)"
)
_HEADING_RE = re.compile(
    r"^\s*(?:"
    r"Act\s+" + _NUMERAL_TOKEN +
    r"|Scene\s+" + _NUMERAL_TOKEN +
    r"|Part\s+" + _NUMERAL_TOKEN +
    r"|Chapter\s+" + _NUMERAL_TOKEN +
    r"|Book\s+" + _NUMERAL_TOKEN +
    r"|Section\s+" + _NUMERAL_TOKEN +
    r"|Stave\s+" + _NUMERAL_TOKEN +
    r"|Canto\s+" + _NUMERAL_TOKEN +
    r"|Volume\s+" + _NUMERAL_TOKEN +
    r"|Episode\s+" + _NUMERAL_TOKEN +
    r")\b",
    re.IGNORECASE | re.MULTILINE,
)


# Hard cap for any single chunk. Even when a Stave/Chapter/Act-delimited
# chunk is found, it can be 5k+ characters and overwhelm a single physics
# pass — leading to anonymous events, missing dyads, and flashback
# fabula_time collapse. Subdivide oversized chunks on paragraph
# boundaries so each LLM call sees roughly one scene.
_MAX_CHUNK_CHARS = 1200


def _subdivide_chunk(chunk: str, max_chars: int = _MAX_CHUNK_CHARS) -> List[str]:
    """Split a single chunk on paragraph boundaries until each piece is
    at most ``max_chars`` characters. The first line of the original
    chunk (typically the heading like 'Stave One') is preserved as a
    prefix on every sub-chunk so each LLM call retains structural
    context.
    """
    if len(chunk) <= max_chars:
        return [chunk]

    # Capture an opening heading line (if any) to repeat on each piece.
    first_nl = chunk.find("\n")
    heading = ""
    body = chunk
    if first_nl != -1 and first_nl <= 80:
        candidate = chunk[:first_nl].strip()
        if _HEADING_RE.match(candidate):
            heading = candidate
            body = chunk[first_nl + 1:].lstrip()

    paragraphs = re.split(r"\n\s*\n", body)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    pieces: List[str] = []
    current = ""
    for para in paragraphs:
        candidate = para if not current else current + "\n\n" + para
        if len(candidate) <= max_chars or not current:
            current = candidate
        else:
            pieces.append(current)
            current = para
    if current:
        pieces.append(current)

    if heading:
        pieces = [f"{heading} (cont. {idx + 1}/{len(pieces)})\n\n{p}"
                  if idx > 0 else f"{heading}\n\n{p}"
                  for idx, p in enumerate(pieces)]
    return pieces


def chunk_text(
    text: str,
    strategy: str = "act_headings",
    min_chunk_chars: int = 800,
    max_chunk_chars: int = _MAX_CHUNK_CHARS,
) -> List[str]:
    """
    Split narrative text into chunks for Step 2 extraction.

    Parameters
    ----------
    text : str
        Full prose text.
    strategy : str
        ``"act_headings"`` — split on act/chapter/part headings, fall
        back to paragraphs if no headings are found.
        ``"paragraph"`` — split on double newlines.
    min_chunk_chars : int
        Minimum chunk size. Adjacent small paragraphs are merged
        until they reach this threshold (paragraph strategy only).

    Returns
    -------
    list[str]
        Non-empty text chunks in document order.
    """
    if strategy == "act_headings":
        # Re-attach heading lines to their bodies
        matches = list(_HEADING_RE.finditer(text))
        if matches:
            chunks: List[str] = []
            # Text before the first heading (if any)
            preamble = text[: matches[0].start()].strip()
            if preamble:
                chunks.append(preamble)
            for i, m in enumerate(matches):
                start = m.start()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                chunk = text[start:end].strip()
                if chunk:
                    chunks.append(chunk)
            if chunks:
                # Subdivide any oversized chunk so each LLM call sees
                # roughly one scene rather than a whole act.
                subdivided: List[str] = []
                for c in chunks:
                    subdivided.extend(_subdivide_chunk(c, max_chars=max_chunk_chars))
                if len(subdivided) != len(chunks):
                    logger.info(
                        "[Chunking] %d heading chunk(s) subdivided to %d "
                        "scene-sized pieces (max %d chars each).",
                        len(chunks), len(subdivided), max_chunk_chars,
                    )
                return subdivided
        # Fallback: no headings found — use paragraph strategy
        logger.info("[Chunking] No act/section headings found — falling back to paragraph split.")

    # Paragraph split
    paragraphs = re.split(r"\n\s*\n", text)
    raw = [p.strip() for p in paragraphs if p.strip()]

    # Merge adjacent small paragraphs up to min_chunk_chars
    if min_chunk_chars > 0 and raw:
        merged: List[str] = [raw[0]]
        for para in raw[1:]:
            if len(merged[-1]) < min_chunk_chars:
                merged[-1] += "\n\n" + para
            else:
                merged.append(para)
        # Final merge: if the last chunk is tiny, append to previous
        if len(merged) >= 2 and len(merged[-1]) < min_chunk_chars // 2:
            merged[-2] += "\n\n" + merged[-1]
            merged.pop()
        # Also enforce the hard cap on this branch.
        capped: List[str] = []
        for c in merged:
            capped.extend(_subdivide_chunk(c, max_chars=max_chunk_chars))
        return capped
    return raw


# =====================================================================
# Step 1 — Global Ontology Extraction (Three Separate Passes)
# =====================================================================

# --- Step 1a: Location extraction (no dependencies) ---

def _build_location_agent(config: ExtractionConfig) -> Agent[None, LocationRegister]:
    """Construct the Step 1a Location Agent."""
    return Agent(
        _resolve_model(config.model),
        output_type=NativeOutput(LocationRegister),
        system_prompt=_load_prompt("ontology_locations.md"),
        retries=config.output_retries,
    )


# --- Step 1b: Object extraction (depends on locations) ---

class _ObjectDeps(BaseModel):
    """Dependencies for Step 1b — objects need location IDs for location_id."""
    model_config = {"protected_namespaces": ()}
    location_register: LocationRegister


def _build_object_agent(config: ExtractionConfig) -> Agent[_ObjectDeps, ObjectRegister]:
    """Construct the Step 1b Object Agent."""
    agent: Agent[_ObjectDeps, ObjectRegister] = Agent(
        _resolve_model(config.model),
        deps_type=_ObjectDeps,
        output_type=NativeOutput(ObjectRegister),
        system_prompt=_load_prompt("ontology_objects.md"),
        retries=config.output_retries,
    )

    @agent.system_prompt
    def inject_locations_for_objects(ctx: RunContext[_ObjectDeps]) -> str:
        loc_ids = sorted(ctx.deps.location_register.locations.keys())
        loc_names = {
            lid: ctx.deps.location_register.locations[lid].name
            for lid in loc_ids
        }
        return (
            "=== LOCATION REGISTER (from Step 1a) ===\n"
            f"LOCATION IDs: {loc_ids}\n"
            f"LOCATION NAMES: {loc_names}\n"
            "\n"
            "Use ONLY these LOC_ IDs when assigning location_id to objects.\n"
            "Set location_id to null if the object is held by someone.\n"
            "owner_id should be the holder's CANONICAL NAME exactly as it "
            "appears in the text (e.g. 'Macbeth', 'Lady Macbeth', "
            "'Three Witches'), or null if the object is not held. "
            "Entity IDs do not exist yet at Step 1b \u2014 a post-pass "
            "resolver maps these names to ENT_ ids once Step 1c completes. "
            "Use simple, unambiguous names; avoid descriptions like "
            "'the king' when you know the name is 'Duncan'."
        )

    @agent.output_validator
    def validate_object_register(
        ctx: RunContext[_ObjectDeps], output: ObjectRegister,
    ) -> ObjectRegister:
        """Sanitise object owner_id and location_id.

        ``owner_id`` is declared as ``Optional[str]`` so the LLM can
        legally emit free-text names like ``"Marley's ghost"`` or
        ``"two men"``. Downstream code expects either ``None`` or an
        ``ENT_`` id, so anything that doesn't match the ENT_ prefix
        is coerced to ``None`` with a warning. ``location_id`` is
        cross-checked against the Step 1a register; unknown ids are
        nulled rather than left dangling.
        """
        loc_ids = set(ctx.deps.location_register.locations.keys())
        notes: List[str] = []
        for oid, obj in output.objects.items():
            if obj.location_id and obj.location_id not in loc_ids:
                notes.append(
                    f"[Auto-Fix] Object {oid}.location_id "
                    f"{obj.location_id!r} not in Step 1a register \u2014 "
                    f"setting to null."
                )
                obj.location_id = None
            # Note: owner_id may legitimately be a free-text name at
            # Step 1b (entities don't exist yet). ``_resolve_object_owner_ids``
            # runs after Step 1c and maps these names to canonical ENT_ IDs.
            # Coercing them to None here would discard the very signal that
            # resolver depends on, leaving every owned object unowned.
        if notes:
            for n in notes:
                logger.info("[Validator\u00b7Objects] %s", n)
        return output

    return agent


# --- Step 1c: Entity extraction (depends on locations + objects) ---

class _EntityDeps(BaseModel):
    """Dependencies for Step 1c — entities need location + object IDs."""
    model_config = {"protected_namespaces": ()}
    location_register: LocationRegister
    object_register: ObjectRegister


def _build_entity_agent(config: ExtractionConfig) -> Agent[_EntityDeps, EntityRegister]:
    """Construct the Step 1c Entity Agent."""
    agent: Agent[_EntityDeps, EntityRegister] = Agent(
        _resolve_model(config.model),
        deps_type=_EntityDeps,
        output_type=NativeOutput(EntityRegister),
        system_prompt=_load_prompt("ontology_entities.md"),
        retries=config.output_retries,
    )

    @agent.system_prompt
    def inject_registers_for_entities(ctx: RunContext[_EntityDeps]) -> str:
        loc_ids = sorted(ctx.deps.location_register.locations.keys())
        loc_names = {
            lid: ctx.deps.location_register.locations[lid].name
            for lid in loc_ids
        }
        obj_ids = sorted(ctx.deps.object_register.objects.keys())
        obj_names = {
            oid: ctx.deps.object_register.objects[oid].name
            for oid in obj_ids
        }
        return (
            "=== LOCATION REGISTER (from Step 1a) ===\n"
            f"LOCATION IDs: {loc_ids}\n"
            f"LOCATION NAMES: {loc_names}\n"
            "\n"
            "=== OBJECT REGISTER (from Step 1b) ===\n"
            f"OBJECT IDs: {obj_ids}\n"
            f"OBJECT NAMES: {obj_names}\n"
            "\n"
            "Use ONLY these LOC_ IDs when assigning location_id. If a "
            "character's home location is missing from the register "
            "(e.g. you'd like to write 'LOC_FRED_HOUSE' but no such "
            "LOC_ exists), pick the nearest existing location instead "
            "of inventing a new one \u2014 the validator will null any "
            "unknown LOC_ id and the entity will fall back to "
            "LOC_UNSPECIFIED.\n"
            "You may reference LOC_ and OBJ_ IDs in belief target_id fields.\n"
            "You may also reference ENT_ IDs you are creating in this pass."
        )

    @agent.output_validator
    def validate_entity_register(
        ctx: RunContext[_EntityDeps], output: EntityRegister,
    ) -> EntityRegister:
        """Cross-validate entity location_ids against the Step 1a register.

        The entity ontology routinely invents locations the location
        ontology never produced (LOC_FRED_HOUSE, LOC_CHARITY_OFFICE,
        LOC_GRAVEYARD), creating dangling references that downstream
        spatial validators flag as orphans. Try fuzzy-matching first,
        then null with a warning so the entity falls back to its
        default position handling.
        """
        loc_ids = set(ctx.deps.location_register.locations.keys())
        notes: List[str] = []
        for eid, ent in output.entities.items():
            if ent.location_id and ent.location_id not in loc_ids:
                resolved = _fuzzy_resolve_id(ent.location_id, loc_ids)
                if resolved:
                    notes.append(
                        f"[Auto-Fix] Entity {eid}.location_id "
                        f"{ent.location_id!r} \u2192 {resolved!r}"
                    )
                    ent.location_id = resolved
                else:
                    notes.append(
                        f"[Unknown-ID] Entity {eid}.location_id "
                        f"{ent.location_id!r} not in Step 1a register \u2014 "
                        f"keeping as-is (will be flagged at assembly)."
                    )
        if notes:
            for n in notes:
                logger.info("[Validator\u00b7Entities] %s", n)
        return output

    return agent


# --- Step 1d: World Traits extraction (no dependencies) ---

def _build_world_traits_agent(config: ExtractionConfig) -> Agent[None, WorldTraitsRegister]:
    """Construct the Step 1d World Traits Agent."""
    return Agent(
        _resolve_model(config.model),
        output_type=NativeOutput(WorldTraitsRegister),
        system_prompt=_load_prompt("ontology_world_traits.md"),
        retries=config.output_retries,
    )


def _resolve_object_owner_ids(
    objects: Dict[str, NarrativeObject],
    entities: Dict[str, Entity],
) -> Dict[str, NarrativeObject]:
    """
    Resolve owner_id strings from Step 1b (which may be names/descriptions)
    to canonical ENT_ IDs now that entities are available from Step 1c.
    """
    entity_name_map: Dict[str, str] = {}
    for eid, ent in entities.items():
        entity_name_map[ent.name.lower()] = eid
        entity_name_map[eid.lower()] = eid
        # Also index by individual name parts and common abbreviations
        # e.g. "King Duncan" → match "duncan", "king duncan"
        for part in ent.name.lower().split():
            if len(part) > 2 and part not in entity_name_map:
                entity_name_map[part] = eid
        # Strip parenthetical suffixes: "Macduff (Thane of Fife)" → "macduff"
        base = ent.name.split("(")[0].strip().lower()
        if base and base not in entity_name_map:
            entity_name_map[base] = eid
        # Map ID-like forms: "ENT_THREE_WITCHES" → also match "three witches"
        readable = eid.replace("ENT_", "").replace("_", " ").lower()
        if readable not in entity_name_map:
            entity_name_map[readable] = eid

    resolved: Dict[str, NarrativeObject] = {}
    for oid, obj in objects.items():
        if obj.owner_id and not obj.owner_id.startswith("ENT_"):
            owner_lower = obj.owner_id.lower()
            # Try exact match first
            matched = entity_name_map.get(owner_lower)
            # Try substring match if exact fails
            if not matched:
                for name_key, eid in entity_name_map.items():
                    if name_key in owner_lower or owner_lower in name_key:
                        matched = eid
                        break
            if matched:
                resolved[oid] = obj.model_copy(update={"owner_id": matched})
                logger.debug("[Step 1·Resolve] Object %s owner_id '%s' → '%s'", oid, obj.owner_id, matched)
            else:
                logger.warning("[Step 1·Resolve] Object %s owner_id '%s' could not be resolved to an entity.", oid, obj.owner_id)
                resolved[oid] = obj
        else:
            resolved[oid] = obj
    return resolved


# ---------------------------------------------------------------------
# User-context plumbing (item #9 of the audit).
#
# Cost-tracking metadata (user_id / project_id / version_id) needs to
# reach every ``agent.run[_sync]`` call so per-call cost is attributed
# to the right user / project / version row in the DB. Threading kwargs
# through every helper would be invasive, so we stash the dict in a
# ``ContextVar`` for the duration of a ``run_extraction_async`` call
# and read it via ``_user_kwargs()`` immediately before each agent run.
# ---------------------------------------------------------------------

_user_context_var: ContextVar[Optional[Dict[str, Optional[int]]]] = ContextVar(
    "shadow_loom_user_context", default=None,
)


def _user_kwargs() -> Dict[str, Optional[int]]:
    """Return the active user_context as kwargs for ``agent.run[_sync]``.

    Returns an empty dict when no context is set so callers can splat it
    unconditionally: ``agent.run_sync(msg, deps=..., **_user_kwargs())``.
    """
    ctx = _user_context_var.get()
    return dict(ctx) if ctx else {}


@contextlib.contextmanager
def _user_context_scope(user_context: Dict[str, Optional[int]]):
    """Set ``_user_context_var`` for the duration of the with-block.

    Captures the token returned by ``ContextVar.set`` and resets it on
    exit (including exceptions). Without this scope, back-to-back
    extractions on the same thread / async task would inherit stale
    user attribution metadata from the previous run.
    """
    token = _user_context_var.set(user_context)
    try:
        yield
    finally:
        _user_context_var.reset(token)



async def extract_ontology_async(
    text: str,
    config: ExtractionConfig | None = None,
    user_context: Optional[Dict[str, Optional[int]]] = None,
) -> GlobalRegister:
    """Step 1: Extract the global ontology (locations, objects, entities).

    Runs Step 1a (locations) first, then Steps 1b (objects) and
    1c (entities) in parallel via ``asyncio.gather``.  Entity
    extraction does not structurally depend on object IDs — objects
    reference entities via ``owner_id``, not the reverse — so
    running them concurrently is safe.  ``_resolve_object_owner_ids``
    is called after both complete.
    """
    config = config or ExtractionConfig()
    user_context = user_context or {}

    # --- Step 1a: Locations (must complete first — both 1b and 1c need it) ---
    location_agent = _build_location_agent(config)
    logger.info("[Step 1a] Extracting locations with %s …", config.model)
    try:
        loc_result = await location_agent.run(text, **user_context)
        loc_register = loc_result.output
    except Exception:
        logger.exception("[Step 1a] Location extraction failed — retrying once …")
        try:
            loc_result = await location_agent.run(text, **user_context)
            loc_register = loc_result.output
        except Exception as exc:
            raise RuntimeError(
                "Step 1a (location extraction) failed twice — cannot "
                "proceed without a location register. See logs above for "
                "the underlying LLM/API error."
            ) from exc
    logger.info("[Step 1a] Extracted %d locations.", len(loc_register.locations))

    # --- Steps 1b → 1c (chained) and 1d (parallel) ---
    # Step 1c (entities) consumes the Object Register in its prompt for
    # belief target_id grounding, so it must run AFTER Step 1b. The two
    # are chained inside one coroutine and that coroutine runs in
    # parallel with Step 1d (world traits, no deps).
    async def _extract_objects() -> ObjectRegister:
        object_agent = _build_object_agent(config)
        obj_deps = _ObjectDeps(location_register=loc_register)
        logger.info("[Step 1b] Extracting objects with %s …", config.model)
        try:
            obj_result = await object_agent.run(text, deps=obj_deps, **user_context)
            return obj_result.output
        except Exception:
            logger.exception("[Step 1b] Object extraction failed — retrying once …")
            try:
                obj_result = await object_agent.run(text, deps=obj_deps, **user_context)
                return obj_result.output
            except Exception as exc:
                raise RuntimeError(
                    "Step 1b (object extraction) failed twice — cannot "
                    "proceed without an object register. See logs above for "
                    "the underlying LLM/API error."
                ) from exc

    async def _extract_entities(obj_register: ObjectRegister) -> EntityRegister:
        entity_agent = _build_entity_agent(config)
        ent_deps = _EntityDeps(
            location_register=loc_register,
            object_register=obj_register,
        )
        logger.info("[Step 1c] Extracting entities with %s …", config.model)
        try:
            ent_result = await entity_agent.run(text, deps=ent_deps, **user_context)
            return ent_result.output
        except Exception:
            logger.exception("[Step 1c] Entity extraction failed — retrying once …")
            try:
                ent_result = await entity_agent.run(text, deps=ent_deps, **user_context)
                return ent_result.output
            except Exception as exc:
                raise RuntimeError(
                    "Step 1c (entity extraction) failed twice — cannot "
                    "proceed without an entity register. See logs above "
                    "for the underlying LLM/API error."
                ) from exc

    async def _extract_objects_then_entities() -> Tuple[ObjectRegister, EntityRegister]:
        obj_reg = await _extract_objects()
        ent_reg = await _extract_entities(obj_reg)
        return obj_reg, ent_reg

    async def _extract_world_traits() -> WorldTraitsRegister:
        world_traits_agent = _build_world_traits_agent(config)
        logger.info("[Step 1d] Extracting world traits with %s …", config.model)
        try:
            wt_result = await world_traits_agent.run(text, **user_context)
            return wt_result.output
        except Exception:
            logger.exception("[Step 1d] World traits extraction failed — retrying once …")
            try:
                wt_result = await world_traits_agent.run(text, **user_context)
                return wt_result.output
            except Exception as exc:
                raise RuntimeError(
                    "Step 1d (world-traits extraction) failed twice — "
                    "cannot proceed without a world-traits register. See "
                    "logs above for the underlying LLM/API error."
                ) from exc

    (obj_register, ent_register), wt_register = await asyncio.gather(
        _extract_objects_then_entities(), _extract_world_traits()
    )
    logger.info("[Step 1b] Extracted %d objects.", len(obj_register.objects))
    logger.info("[Step 1c] Extracted %d entities.", len(ent_register.entities))
    logger.info("[Step 1d] Extracted %d world traits.", len(wt_register.world_traits))

    # --- Resolve object owner_ids to ENT_ IDs (needs both registers) ---
    resolved_objects = _resolve_object_owner_ids(obj_register.objects, ent_register.entities)

    register = GlobalRegister(
        locations=loc_register.locations,
        objects=resolved_objects,
        entities=ent_register.entities,
        world_traits=wt_register.world_traits,
    )
    sanitisation_notes: List[str] = []
    register = _sanitize_register(register, sanitisation_notes)
    for note in sanitisation_notes:
        logger.info(note)
    logger.info(
        "[Step 1] Ontology extracted — %d locations, %d objects, %d entities, %d world traits.",
        len(register.locations), len(register.objects), len(register.entities),
        len(register.world_traits),
    )
    return register


# =====================================================================
# Step 2 — Semantic Scaffolding (Socratic QA)
# =====================================================================

class _SocraticDeps(BaseModel):
    """Dependencies for Step 2 (Socratic QA scaffolding)."""
    model_config = {"protected_namespaces": ()}
    global_register: GlobalRegister


def _build_socratic_agent(config: ExtractionConfig) -> Agent[_SocraticDeps, SocraticScaffold]:
    """Construct the Step 2 Socratic QA agent."""
    agent: Agent[_SocraticDeps, SocraticScaffold] = Agent(
        _resolve_model(config.model),
        deps_type=_SocraticDeps,
        output_type=NativeOutput(SocraticScaffold),
        system_prompt=_load_prompt("socratic_scaffolding.md"),
        retries=config.output_retries,
    )

    @agent.system_prompt
    def inject_register_for_socratic(ctx: RunContext[_SocraticDeps]) -> str:
        reg = ctx.deps.global_register
        entity_ids = sorted(reg.entities.keys())
        location_ids = sorted(reg.locations.keys())
        object_ids = sorted(reg.objects.keys())
        entity_names = {eid: reg.entities[eid].name for eid in entity_ids}
        location_names = {lid: reg.locations[lid].name for lid in location_ids}
        object_names = {oid: reg.objects[oid].name for oid in object_ids}
        world_trait_ids = sorted(reg.world_traits.keys())
        world_trait_names = {
            wid: reg.world_traits[wid].name for wid in world_trait_ids
        }
        return (
            "=== NARRATIVE REGISTER (from Step 1) ===\n"
            f"CHARACTERS: {entity_names}\n"
            f"LOCATIONS: {location_names}\n"
            f"OBJECTS: {object_names}\n"
            f"WORLD TRAITS: {world_trait_names}\n"
            "\n"
            "Reference these characters, locations, objects, and "
            "world traits by their canonical names or IDs in your "
            "answers. Use WORLD trait names when articulating ambient "
            "pressures (e.g. an oppressive regime, a prophetic curse, "
            "an offstage war) that motivate on-page behaviour."
        )

    return agent


# =====================================================================
# Step 2.5 — Proposition Catalogue (Phase A3)
#
# Single global pass over the full text that produces the canonical
# ``PROP_*`` registry (and per-entity ``ConcernSeed`` baselines)
# referenced by every per-chunk extractor. See
# ``/memories/repo/ingestion-plan-2026-05-07-propositions-concerns.md``
# Phase A3 + Annex.
# =====================================================================


class ConcernSeed(BaseModel):
    """Baseline standing-fear/desire entry produced by the Proposition
    Catalogue. Anchors a character's concern to a catalogued proposition.

    Drift over fabula time is owned by the per-chunk Affect agent's
    ``ConcernSnapshot`` output \u2014 the seed records the *initial* salience
    only.
    """
    model_config = {"protected_namespaces": ()}
    concern_id: str = Field(
        description="Unique CCN_ id (matches ^CCN_[A-Z0-9_]+$).",
    )
    entity_id: str = Field(
        description="ENT_ id of the concern-holder.",
    )
    proposition_id: str = Field(
        description="PROP_ id whose realisation realises (desire) or averts (fear) the concern.",
    )
    polarity: Literal["desire", "fear"] = Field(
        description="Whether the entity wants the proposition true or false.",
    )
    kind: Optional[str] = Field(
        default=None,
        description=(
            "Optional harm/benefit-kind label drawn from the affect "
            "vocabulary (betrayal, abandonment, irrelevance, death, ...)."
        ),
    )
    baseline_salience: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Initial per-entity weighting of the concern.",
    )
    evidence_strength: Literal["weak", "moderate", "strong"] = Field(
        default="moderate",
        description="Engine's certainty in the catalogue extraction of this seed.",
    )
    counter_concern_ids: List[str] = Field(
        default_factory=list,
        description="Other CCN_ ids forming an ambivalent pair with this seed.",
    )


class PropositionCatalogue(BaseModel):
    """Phase A3 output: global PROP_/CCN_ registry, plus concern seeds.

    The reconciler folds ``propositions`` into ``WorldStateV1.propositions``
    and materialises ``concern_seeds`` as ``Concern`` records on the
    referenced entities. Per-chunk extractors only ever *reference* these
    ids; they may not introduce new PROP_ ids and may introduce new
    CCN_ ids only via the Affect agent's ``new_concern_seeds`` channel.
    """
    propositions: List[Proposition] = Field(default_factory=list)
    concern_seeds: List[ConcernSeed] = Field(default_factory=list)


# Now that ``ConcernSeed`` exists, finalise the per-chunk topology
# wire format whose forward ref to it was deferred earlier.
ChunkTopology.model_rebuild()
ChunkAffectExtraction.model_rebuild()


class _PropCatalogueDeps(BaseModel):
    """Dependencies for the Phase A3 catalogue agent."""
    model_config = {"protected_namespaces": ()}
    global_register: GlobalRegister


def _build_proposition_catalogue_agent(
    config: ExtractionConfig,
) -> Agent[_PropCatalogueDeps, PropositionCatalogue]:
    """Construct the Phase A3 Proposition Catalogue agent.

    Sees the full source text and the Step-1 ontology register; emits
    PROP_ and CCN_ ids that downstream per-chunk agents reference as
    opaque labels. Owns no events, edges, beliefs, or snapshots.
    """
    agent: Agent[_PropCatalogueDeps, PropositionCatalogue] = Agent(
        _resolve_model(config.model),
        deps_type=_PropCatalogueDeps,
        output_type=NativeOutput(PropositionCatalogue),
        system_prompt=_load_prompt("proposition_catalogue.md"),
        retries=config.output_retries,
    )

    @agent.system_prompt
    def inject_register_for_catalogue(
        ctx: RunContext[_PropCatalogueDeps],
    ) -> str:
        reg = ctx.deps.global_register

        # Entity cards: name + status + headline traits (top 3 by value).
        ent_lines: List[str] = []
        for eid in sorted(reg.entities):
            ent = reg.entities[eid]
            traits = getattr(ent, "traits", {}) or {}
            top_traits = sorted(
                traits.items(),
                key=lambda kv: -float(getattr(kv[1], "value", 0.0)),
            )[:3]
            trait_str = ", ".join(
                f"{tn}={float(getattr(tv, 'value', 0.0)):.2f}"
                for tn, tv in top_traits
            ) or "—"
            ent_lines.append(
                f"  - {eid}: {ent.name} [{getattr(ent, 'status', 'unknown')}] "
                f"@{getattr(ent, 'location_id', '?')} traits=({trait_str})"
            )
        entities_block = "\n".join(ent_lines) if ent_lines else "  (none)"

        # Location cards: name + 1-line description.
        loc_lines: List[str] = []
        for lid in sorted(reg.locations):
            loc = reg.locations[lid]
            desc = (getattr(loc, "description", "") or "").strip().splitlines()
            desc_str = (desc[0][:120] if desc else "—")
            loc_lines.append(f"  - {lid}: {loc.name} — {desc_str}")
        locations_block = "\n".join(loc_lines) if loc_lines else "  (none)"

        # Object cards: name + owner/location + top affordance verbs.
        obj_lines: List[str] = []
        for oid in sorted(reg.objects):
            obj = reg.objects[oid]
            affs = getattr(obj, "affordances", []) or []
            verbs = ",".join(
                getattr(a, "action", str(a)) for a in affs[:3]
            ) or "—"
            holder = getattr(obj, "owner_id", None) or getattr(obj, "location_id", "?")
            obj_lines.append(
                f"  - {oid}: {obj.name} @{holder} affordances=({verbs})"
            )
        objects_block = "\n".join(obj_lines) if obj_lines else "  (none)"

        # World-trait cards: name + 1-line description.
        wt_lines: List[str] = []
        for wid in sorted(reg.world_traits):
            wt = reg.world_traits[wid]
            desc = (getattr(wt, "description", "") or "").strip().splitlines()
            desc_str = (desc[0][:120] if desc else "—")
            wt_lines.append(f"  - {wid}: {wt.name} — {desc_str}")
        world_traits_block = "\n".join(wt_lines) if wt_lines else "  (none)"

        return (
            "=== ONTOLOGY REGISTER (from Step 1) ===\n"
            "Use ONLY the ENT_/LOC_/OBJ_/WORLD_ ids below in `referent_ids` "
            "and `entity_id`. The compact cards give you the context you "
            "need to pick `referent_ids` that fully scope each proposition "
            "(e.g. a relation_holds proposition between two entities should "
            "list BOTH entity ids; a location-bound prophecy should list "
            "the location, not the speaker).\n"
            "\n"
            "ENTITIES:\n"
            f"{entities_block}\n"
            "\n"
            "LOCATIONS:\n"
            f"{locations_block}\n"
            "\n"
            "OBJECTS:\n"
            f"{objects_block}\n"
            "\n"
            "WORLD TRAITS:\n"
            f"{world_traits_block}\n"
            "\n"
            "You MAY mint new PROP_*/CCN_* ids \u2014 they are your namespace "
            "and downstream extractors will reference them by name. Do NOT "
            "mint EVT_ ids; per-chunk Physics / Social agents own that "
            "namespace."
        )

    @agent.output_validator
    def validate_catalogue_ids(
        ctx: RunContext[_PropCatalogueDeps],
        result: PropositionCatalogue,
    ) -> PropositionCatalogue:
        """Drop catalogue entries with invalid ontology references / id shapes.

        - PROP_ / CCN_ id shape is enforced by the model annotations on
          downstream consumers (and by ``PropositionId`` / ``ConcernId``
          aliases in ``models.py``); here we focus on referential integrity:
          referent ids must exist in the ontology, and concern seeds must
          point at a proposition the catalogue itself emits.
        - Duplicates collapse silently (later entry wins) so a slightly
          chatty extractor doesn't poison the world.
        """
        reg = ctx.deps.global_register
        valid_ontology = (
            set(reg.entities)
            | set(reg.locations)
            | set(reg.objects)
            | set(reg.world_traits)
        )
        prop_re = re.compile(r"^PROP_[A-Z0-9_]+$")
        ccn_re = re.compile(r"^CCN_[A-Z0-9_]+$")

        deduped_props: Dict[str, Proposition] = {}
        for p in result.propositions:
            if not prop_re.match(p.proposition_id):
                logger.warning(
                    "[PropCatalogue] Dropping proposition %r \u2014 id does not "
                    "match ^PROP_[A-Z0-9_]+$.", p.proposition_id,
                )
                continue
            kept_referents: List[str] = []
            for rid in p.referent_ids:
                if rid in valid_ontology:
                    kept_referents.append(rid)
                else:
                    logger.warning(
                        "[PropCatalogue] Proposition %s drops unknown referent_id %r.",
                        p.proposition_id, rid,
                    )
            if kept_referents != list(p.referent_ids):
                p = p.model_copy(update={"referent_ids": kept_referents})
            deduped_props[p.proposition_id] = p

        prop_ids = set(deduped_props)
        deduped_seeds: Dict[Tuple[str, str, str], ConcernSeed] = {}
        for s in result.concern_seeds:
            if not ccn_re.match(s.concern_id):
                logger.warning(
                    "[PropCatalogue] Dropping concern seed %r \u2014 id does not "
                    "match ^CCN_[A-Z0-9_]+$.", s.concern_id,
                )
                continue
            if s.entity_id not in reg.entities:
                logger.warning(
                    "[PropCatalogue] Dropping concern seed %s \u2014 unknown entity %r.",
                    s.concern_id, s.entity_id,
                )
                continue
            if s.proposition_id not in prop_ids:
                logger.warning(
                    "[PropCatalogue] Dropping concern seed %s \u2014 unknown "
                    "proposition_id %r.", s.concern_id, s.proposition_id,
                )
                continue
            deduped_seeds[(s.entity_id, s.proposition_id, s.polarity)] = s

        return PropositionCatalogue(
            propositions=list(deduped_props.values()),
            concern_seeds=list(deduped_seeds.values()),
        )

    return agent


async def extract_proposition_catalogue_async(
    text: str,
    register: GlobalRegister,
    config: ExtractionConfig,
) -> PropositionCatalogue:
    """Run Phase A3: build the global Proposition Catalogue.

    Single LLM call over the full source text. Failures degrade
    gracefully to an empty catalogue \u2014 every per-chunk agent then
    sees an empty PROP list and the legacy post-pass clustering path
    handles belief\u2192proposition binding.
    """
    agent = _build_proposition_catalogue_agent(config)
    deps = _PropCatalogueDeps(global_register=register)
    logger.info("[Step 2.5] Extracting proposition catalogue \u2026")
    try:
        result = await agent.run(text, deps=deps, **_user_kwargs())
        catalogue = result.output
    except Exception:
        logger.exception(
            "[Step 2.5] Proposition catalogue extraction FAILED \u2014 "
            "returning empty catalogue (per-chunk extractors will run "
            "with no PROP_ context; legacy post-pass clustering still "
            "fires).",
        )
        return PropositionCatalogue()
    logger.info(
        "[Step 2.5] Catalogue: %d propositions, %d concern seeds.",
        len(catalogue.propositions), len(catalogue.concern_seeds),
    )
    return catalogue


# =====================================================================
# Step 3 — Decomposed Topology Extraction (Physics + Social Agents)
# =====================================================================

class _PhysicsDeps(BaseModel):
    """Dependencies for Step 3a (Physics Agent: events + causal + spatial)."""
    model_config = {"protected_namespaces": ()}
    global_register: GlobalRegister
    scaffold: SocraticScaffold
    previous_event_ids: List[str] = Field(default_factory=list)
    # Subset of global_register entity IDs whose names actually appear
    # in the chunk text. Used by the system prompt to nudge the agent
    # toward extracting events for the on-page cast and away from
    # hallucinating events for offstage characters.
    on_page_entity_ids: List[str] = Field(default_factory=list)
    # Compact ``(PROP_id, description)`` pairs from the global
    # Proposition Catalogue (Phase A3). Empty list when the catalogue
    # has not been extracted yet (legacy worlds, ablation runs) — the
    # system prompt then renders an explicit "none yet" notice and
    # the agent leaves ``EventNode.resolves_proposition_ids`` empty.
    chunk_propositions: List[Tuple[str, str]] = Field(default_factory=list)


class _SocialDeps(BaseModel):
    """Dependencies for Step 3b (Social Agent: info + relationship)."""
    model_config = {"protected_namespaces": ()}
    global_register: GlobalRegister
    scaffold: SocraticScaffold
    chunk_event_ids: List[str] = Field(default_factory=list)
    # Full event objects from Physics — the Social agent uses these to
    # ground utterance/channel attribution in the actual on-page action
    # rather than re-deriving them from the prose.
    chunk_events: List[EventNode] = Field(default_factory=list)
    # Physics causal edges (esp. ``mutation_social``) — knowing which
    # events shift relationships helps Social emit relationship_edges
    # that match the causal topology rather than contradicting it.
    chunk_causal: List[CausalEdge] = Field(default_factory=list)
    previous_event_ids: List[str] = Field(default_factory=list)
    on_page_entity_ids: List[str] = Field(default_factory=list)
    # Standing channels already extracted from prior chunks. Threaded
    # through the per-chunk extraction loop so the LLM can
    # reuse a CHN_ id by reference instead of inventing a near-duplicate
    # for the same standing capability (a long-running letter
    # correspondence, an ongoing telepathic bond, a spy-master pipeline
    # that spans the whole novel). Empty in the parallel-async path
    # where chunks have no causal ordering — dedup at assembly handles
    # that case via shape-key collapse.
    previous_chunk_channels: Dict[str, "Channel"] = Field(default_factory=dict)
    # See ``_PhysicsDeps.chunk_propositions``. Surfaces PROP_ ids on the
    # social agent so utterances can carry ``asserts_proposition_id`` /
    # ``denies_proposition_id`` referencing canonical catalogue ids.
    chunk_propositions: List[Tuple[str, str]] = Field(default_factory=list)


class _ConsequencesDeps(BaseModel):
    """Dependencies for Step 3c (Consequences Agent: entity_updates).

    Receives the events and mutation/mutation_social edges already produced
    by the Physics Agent so that every EntityUpdate it emits is anchored
    to a concrete event and aligned with the causal mutations.

    Also receives the Social Agent's ``chunk_channels`` and
    ``chunk_utterance_events`` (when Step 3b has already run) so that
    any beliefs the agent emits can correctly populate
    ``acquired_via_event_id`` (the utterance) and
    ``acquired_via_channel_id`` (the standing capability the utterance
    rode over). Without this injection the prompt's belief-provenance
    fields are unfillable.
    """
    model_config = {"protected_namespaces": ()}
    global_register: GlobalRegister
    scaffold: SocraticScaffold
    chunk_events: List[EventNode] = Field(default_factory=list)
    chunk_causal: List[CausalEdge] = Field(default_factory=list)
    chunk_channels: Dict[str, "Channel"] = Field(default_factory=dict)
    chunk_utterance_events: List[EventNode] = Field(default_factory=list)
    previous_event_ids: List[str] = Field(default_factory=list)
    on_page_entity_ids: List[str] = Field(default_factory=list)
    # See ``_PhysicsDeps.chunk_propositions``. Surfaces PROP_ ids on the
    # consequences agent so beliefs whose ``target_id`` matches a
    # catalogue proposition's referent can carry ``proposition_id`` at
    # extraction time — removing the post-pass clustering round-trip
    # for the common case.
    chunk_propositions: List[Tuple[str, str]] = Field(default_factory=list)


class _AffectDeps(BaseModel):
    """Dependencies for Step 3d (Phase B4: Affect Agent).

    Carries the *full* catalogue (propositions and concern seeds), the
    chunk's events (Physics + Social merged) and entity_updates
    (Consequences output), so the affect agent can decide which
    propositions to snapshot, which truths to commit, and which
    concerns drift in this chunk.

    The reconciler folds the agent's outputs into
    ``Proposition.state_timeline`` / ``Proposition.truth_at_fabula``
    and ``Entity.concerns`` / ``Concern.state_timeline`` in Phase C.
    """
    model_config = {"protected_namespaces": ()}
    global_register: GlobalRegister
    chunk_events: List[EventNode] = Field(default_factory=list)
    chunk_entity_updates: List["EntityUpdate"] = Field(default_factory=list)
    # Full catalogue (propositions + concern seeds) — the affect agent
    # needs each proposition's kind / referents / baseline stakes /
    # baseline prior / description, and each seed's polarity / kind /
    # baseline salience, all of which the compact (id, desc) pairs on
    # the other deps drop. Empty when no catalogue stage ran.
    propositions: List[Proposition] = Field(default_factory=list)
    concern_seeds: List["ConcernSeed"] = Field(default_factory=list)


def _format_scaffold(scaffold: SocraticScaffold) -> str:
    """Format a SocraticScaffold as a readable text block for agent injection."""
    if not scaffold.qa_pairs:
        return "(No scaffolding QA available for this chunk.)"
    lines = []
    for qa in scaffold.qa_pairs:
        lines.append(f"  [{qa.category.upper()}] Q: {qa.question}")
        lines.append(f"           A: {qa.answer}")
    return "\n".join(lines)


def _format_proposition_catalogue(
    catalogue: List[Tuple[str, str]],
) -> str:
    """Render a compact ``(PROP_id, description)`` list for agent injection.

    The Phase A3 Proposition Catalogue Agent owns the canonical PROP_
    ids; per-chunk Physics / Social / Consequences agents reference
    them as opaque labels via ``deps.chunk_propositions``. When the
    catalogue is empty (legacy world, ablation run, catalogue stage
    skipped) this renders an explicit notice so the agent does not
    silently invent PROP_ ids.
    """
    if not catalogue:
        return (
            "  (no proposition catalogue available \u2014 leave "
            "proposition_id / asserts_proposition_id / "
            "denies_proposition_id / resolves_proposition_ids empty / "
            "null on every artefact in this chunk)"
        )
    return "\n".join(
        f"  - {prop_id}: {description}" for prop_id, description in catalogue
    )


# Verbs / cue-tokens that strongly imply on-page speech-acts. The Social
# retry quality gate uses this to avoid retrying on legitimate pure-action
# chunks (chases, silent set-pieces, scenic description) where zero
# channels and zero utterances is the *correct* answer.
#
# Quote-mark coverage (used for paired-quote detection only — a *single*
# stray quote mark is no longer enough to trigger the retry, since
# narrators routinely use quoted single words for titles, scare-quotes,
# and proper-noun glosses):
#   "  — straight double  (ASCII 0x22)
#   \u201c \u201d         — curly double (English)
#   \u2018 \u2019         — curly single (English)
#   \u201a \u201e \u201f  — German low / high
#   \u00ab \u00bb         — French / Russian guillemets
#   \u2039 \u203a         — single guillemets
#   \u300c \u300d         — Japanese corner brackets
_QUOTE_CHARS = '"\u201c\u201d\u2018\u2019\u201a\u201e\u201f\u00ab\u00bb\u2039\u203a\u300c\u300d'
# Open/close pairs we accept as "balanced dialogue":
_QUOTE_PAIRS = (
    ("\u201c", "\u201d"),  # English curly double
    ("\u2018", "\u2019"),  # English curly single
    ("\u00ab", "\u00bb"),  # French guillemets
    ("\u2039", "\u203a"),  # single guillemets
    ("\u300c", "\u300d"),  # Japanese corner brackets
    ("\u201e", "\u201c"),  # German low-high
)
_SPEECH_VERB_RE = re.compile(
    r'\b('
    r'said|says|told|tells|asked|asks|replied|replies|whispered|whispers|'
    r'shouted|shouts|cried|cries|murmured|murmurs|muttered|mutters|'
    r'declared|declares|announced|announces|warned|warns|promised|promises|'
    r'confessed|confesses|admitted|admits|wrote|writes|read|reads|'
    r'letter|letters|note|notes|message|messages|prophecy|prophesied|'
    r'order|orders|command|commands|rumour|rumor|gossip|spoke|speaks|'
    r'answered|answers|interrupted|interrupts|exclaimed|exclaims'
    r')\b',
    re.IGNORECASE,
)

# Em-dash dialogue convention (French, Russian, Spanish, James Joyce):
#   — Have you no shame? he asked.
# Detected as a line starting with em-dash or en-dash followed by a
# capitalised letter (Latin or Cyrillic). Restricted to start-of-line
# anchoring to avoid false positives on parenthetical em-dashes mid
# sentence.
_EMDASH_DIALOGUE_RE = re.compile(
    r'(?:^|\n)\s*[\u2014\u2013]\s+[A-Z\u00C0-\u024f\u0400-\u04ff]'
)


def _chunk_likely_contains_speech(chunk_text: str) -> bool:
    """Return True if the chunk shows linguistic evidence of dialogue or
    written/transmitted communication. Cheap heuristic used by the
    Social Agent's empty-result retry gate.

    Decision tree (any one is sufficient):
      1. Em-dash / en-dash line opener (Joyce / French / Russian dialogue style).
      2. A balanced pair of dialogue-class quote marks (≥2 straight
         double-quotes, OR matching curly / guillemet / corner-bracket
         pair). A single stray quote on its own no longer counts —
         narrators routinely quote single words for titles, scare-quotes,
         and proper-noun glosses, and that was generating spurious
         social-extraction retries on action chunks.
      3. A recognised speech / writing / transmission verb (covers
         epistolary, reported speech, and channel-prose without quote
         marks).
    """
    if _EMDASH_DIALOGUE_RE.search(chunk_text):
        return True
    # Balanced quotes
    if chunk_text.count('"') >= 2:
        return True
    for opener, closer in _QUOTE_PAIRS:
        if opener in chunk_text and closer in chunk_text:
            return True
    if _SPEECH_VERB_RE.search(chunk_text):
        return True
    return False


def _build_valid_id_set(reg: GlobalRegister, event_ids: List[str] | None = None) -> set[str]:
    """Build the complete set of valid IDs from the register + optional event IDs."""
    valid = (
        set(reg.entities.keys())
        | set(reg.locations.keys())
        | set(reg.objects.keys())
        | set(reg.world_traits.keys())
    )
    if event_ids:
        valid |= set(event_ids)
    return valid


# =====================================================================
# Fuzzy ID Resolution — fix LLM typos without expensive retries
# =====================================================================


def _normalize_id_candidate(candidate: str) -> str:
    """Strip whitespace and uppercase the prefix segment of an ID.

    LLMs occasionally emit IDs with stray whitespace (``"  ENT_X  "``)
    or mixed-case prefixes (``"Ent_x"``) — both forms are obviously
    broken but slip past exact-match before fuzzy lookup. Normalising
    here lets the simple ``candidate in valid_ids`` test recover most
    of these without burning a fuzzy-match round-trip.

    The body after the prefix is left untouched: entity / event id
    bodies are deliberately case-sensitive (``ENT_PETER`` vs
    ``ENT_PETER_PAN`` etc.) and re-casing them would produce more
    noise than signal.
    """
    if not isinstance(candidate, str):
        return candidate
    s = candidate.strip()
    for p in ("EVT_", "ENT_", "LOC_", "OBJ_", "WORLD_", "CHN_"):
        if s.upper().startswith(p):
            return p + s[len(p):]
    return s


def _fuzzy_resolve_id(candidate: str, valid_ids: set[str]) -> Optional[str]:
    """Attempt to resolve *candidate* to a valid ID via fuzzy matching.

    Matching strategy (in priority order):
    1. Exact match.
    2. Case-insensitive exact match (same prefix).
    3. Substring match on the base portion (``ENT_KING_DUNCAN`` ↔ ``ENT_DUNCAN``).

    Returns the matched valid ID or *None* if no match is found.
    Only considers IDs sharing the same prefix (``EVT_``, ``ENT_``, etc.)
    so prefix semantics are preserved and ``model_validator`` stays happy.
    """
    candidate = _normalize_id_candidate(candidate)
    if candidate in valid_ids:
        return candidate

    # Determine prefix
    prefix = ""
    for p in ("EVT_", "ENT_", "LOC_", "OBJ_", "WORLD_"):
        if candidate.startswith(p):
            prefix = p
            break
    if not prefix:
        return None  # can't fuzzy-match unprefixed IDs safely

    same_prefix = {vid for vid in valid_ids if vid.startswith(prefix)}
    if not same_prefix:
        return None

    # Case-insensitive exact match
    lower_map = {vid.lower(): vid for vid in same_prefix}
    if candidate.lower() in lower_map:
        return lower_map[candidate.lower()]

    # Substring match on the base (without prefix, underscores → spaces)
    cand_base = candidate[len(prefix):].replace("_", " ").lower().strip()
    if not cand_base:
        return None

    best: Optional[str] = None
    best_len = 0
    for vid in same_prefix:
        vid_base = vid[len(prefix):].replace("_", " ").lower().strip()
        if cand_base in vid_base or vid_base in cand_base:
            match_len = min(len(cand_base), len(vid_base))
            if match_len > best_len:
                best = vid
                best_len = match_len

    return best


def _fix_id(candidate: str, valid_ids: set[str], field_label: str, fixes: List[str]) -> Tuple[str, bool]:
    """Try to fuzzy-fix *candidate*. Returns (resolved_id, was_fixed).

    Appends a human-readable note to *fixes* when a correction is made.
    When *candidate* cannot be resolved (typo too far from any registered
    ID, or genuinely fabricated entity not in the ontology), append an
    explicit ``[Unknown-ID]`` warning so the caller's downstream logging
    surfaces fabrications like ``ENT_TINY_TIM`` that aren't in the
    Step-1 register. The candidate is returned unchanged so structural
    validators (e.g. the Consequences ``bad`` set) can still raise
    ModelRetry on it.
    """
    if candidate in valid_ids:
        return candidate, False
    resolved = _fuzzy_resolve_id(candidate, valid_ids)
    if resolved:
        fixes.append(f"[Auto-Fix] {field_label} '{candidate}' → '{resolved}'")
        return resolved, True
    # Only warn for ID-shaped strings (looks like LOC_/OBJ_/ENT_/EVT_/CHN_/WORLD_).
    # Plain strings or empties go to the caller's existing handling.
    if candidate and "_" in candidate and candidate.split("_", 1)[0] in {
        "LOC", "OBJ", "ENT", "EVT", "CHN", "WORLD",
    }:
        fixes.append(
            f"[Unknown-ID] {field_label} '{candidate}' is not in the "
            f"ontology register and could not be fuzzy-matched."
        )
    return candidate, False


# =====================================================================
# Numeric / semantic sanitisation helpers — used by every per-chunk
# output validator. Each helper returns the corrected value plus,
# where relevant, an "issue" string appended to a shared *notes* list
# so the caller can decide whether to log, drop, or pass through.
# =====================================================================


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp ``value`` into the inclusive range [lo, hi]."""
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


# Common spelling errors in ambient_state keys produced by extraction
# agents. Conservative list \u2014 only fix unambiguous typos.
_AMBIENT_TYPO_FIXES: Dict[str, str] = {
    "meloncholy": "melancholy",
    "melancholic": "melancholy",
    "concious": "conscious",
    "supernatrual": "supernatural",
    "atmoshpere": "atmosphere",
    "tention": "tension",
    "warmpth": "warmth",
    "danager": "danger",
    "vissibility": "visibility",
    "saftey": "safety",
    "concelment": "concealment",
}


def _normalise_ambient_key(key: str) -> str:
    """Normalise an ambient_state key.

    - Lowercases.
    - Replaces spaces / hyphens with underscores.
    - Strips leading/trailing whitespace and underscores.
    - Applies the conservative typo dictionary.
    """
    norm = key.strip().lower()
    norm = norm.replace("-", "_").replace(" ", "_")
    while "__" in norm:
        norm = norm.replace("__", "_")
    norm = norm.strip("_")
    return _AMBIENT_TYPO_FIXES.get(norm, norm)


def _sanitize_causal_edge(
    ce: CausalEdge, notes: List[str],
) -> Optional[CausalEdge]:
    """Apply numeric clamps and semantic fixes to a CausalEdge.

    Returns the (possibly mutated) edge or ``None`` if the edge is
    structurally incoherent enough that downstream physics would just
    silently drop it (in which case we drop it now and log).
    """
    updates: dict = {}

    # Drop self-loops — physics treats these as zero-effect cycles
    # and the AMWN structural diagram throws them away anyway.
    if ce.source_id == ce.target_id:
        notes.append(
            f"[Auto-Fix] Dropped self-loop CausalEdge '{ce.source_id}'→"
            f"'{ce.target_id}' ({ce.causality_type})"
        )
        return None

    # Clamp causal_force to [0, 10]
    if ce.causal_force is not None and not (0.0 <= ce.causal_force <= 10.0):
        clamped = _clamp(ce.causal_force, 0.0, 10.0)
        notes.append(
            f"[Auto-Fix] Clamped CausalEdge causal_force {ce.causal_force} → {clamped} "
            f"({ce.source_id}→{ce.target_id})"
        )
        updates["causal_force"] = clamped

    # Clamp trait_delta to [-1, 1]
    if ce.trait_delta is not None and not (-1.0 <= ce.trait_delta <= 1.0):
        clamped = _clamp(ce.trait_delta, -1.0, 1.0)
        notes.append(
            f"[Auto-Fix] Clamped CausalEdge trait_delta {ce.trait_delta} → {clamped} "
            f"({ce.source_id}→{ce.target_id})"
        )
        updates["trait_delta"] = clamped

    # Clamp propagation_delay to >= 0
    if ce.propagation_delay is not None and ce.propagation_delay < 0:
        notes.append(
            f"[Auto-Fix] Clamped negative propagation_delay {ce.propagation_delay} → 0 "
            f"({ce.source_id}→{ce.target_id})"
        )
        updates["propagation_delay"] = 0

    # Mutation edges should carry trait_target + trait_delta. The downstream
    # physics engine falls back to a generic trait-routing path when these
    # are missing, which silently degrades fidelity. Log so the issue is
    # visible without rejecting the edge entirely.
    if ce.causality_type in ("mutation", "mutation_social"):
        if not ce.trait_target:
            notes.append(
                f"[Quality] {ce.causality_type} edge {ce.source_id}→{ce.target_id} "
                "missing trait_target — physics will use generic routing."
            )
        if ce.trait_delta is None:
            notes.append(
                f"[Quality] {ce.causality_type} edge {ce.source_id}→{ce.target_id} "
                "missing trait_delta — physics will use a default (+1.0) magnitude."
            )

    # mutation_social: trait_target must be a relationship metric, and
    # rel_counterpart_id must differ from target_id. The schema-level
    # validator already requires rel_counterpart_id; we strengthen here.
    if ce.causality_type == "mutation_social":
        if ce.trait_target and ce.trait_target not in _RELATIONSHIP_METRICS:
            # Try to repair via the alias map before dropping.
            alias = _RELATIONSHIP_METRIC_ALIASES.get(
                str(ce.trait_target).lower().strip()
            )
            if alias:
                notes.append(
                    f"[Auto-Fix] mutation_social trait_target "
                    f"'{ce.trait_target}' → '{alias}' "
                    f"({ce.source_id}→{ce.target_id})"
                )
                updates["trait_target"] = alias
            else:
                notes.append(
                    f"[Auto-Fix] Dropped mutation_social edge {ce.source_id}→{ce.target_id}: "
                    f"trait_target '{ce.trait_target}' is not one of "
                    f"{sorted(_RELATIONSHIP_METRICS)}."
                )
                return None
        if ce.rel_counterpart_id and ce.rel_counterpart_id == ce.target_id:
            notes.append(
                f"[Auto-Fix] Dropped mutation_social edge {ce.source_id}→{ce.target_id}: "
                "rel_counterpart_id is the same as target_id (self-relationship)."
            )
            return None

    # Mechanism vocabulary check — fully informational. The physics engine
    # routes by mechanism via MECHANISM_TRAIT_MAP and falls back to a
    # penalty for unknown labels. We surface unknown-but-acceptable labels
    # so authors can spot persistent novel labels they may want canonised.
    if ce.mechanism:
        mech = ce.mechanism.strip()
        if mech != ce.mechanism:
            updates["mechanism"] = mech
        if (
            mech not in _CANONICAL_MECHANISMS
            and mech not in _TOLERATED_MECHANISMS
        ):
            notes.append(
                f"[Quality] CausalEdge mechanism '{mech}' is non-canonical "
                f"({ce.source_id}→{ce.target_id}) — physics will apply the "
                "mechanism-routing fallback (~20% impulse) for off-list traits."
            )

    if updates:
        try:
            return ce.model_copy(update=updates)
        except Exception:
            notes.append(
                f"[Auto-Fix] Dropped CausalEdge {ce.source_id}→{ce.target_id} "
                "after numeric clamp triggered a schema rejection."
            )
            return None
    return ce


def _sanitize_relationship_edge(
    re_edge: RelationshipEdge, notes: List[str],
) -> RelationshipEdge:
    """Clamp per-metric value/inertia ranges and coerce evidence_strength.

    Operates on the new ``metrics`` dict structure. Each axis has its
    own range:
      * affinity \u2208 [-1, 1]
      * fear \u2208 [0, 1]
      * power_dynamic \u2208 [-1, 1]
      * inertia \u2208 [0, 0.99] (1.0 would freeze the metric forever)
    """
    metric_ranges = {
        "affinity": (-1.0, 1.0),
        "fear": (0.0, 1.0),
        "power_dynamic": (-1.0, 1.0),
    }
    es_aliases = {
        "high": "strong", "low": "weak", "medium": "moderate",
        "med": "moderate", "uncertain": "weak", "certain": "strong",
    }

    new_metrics: dict = {}
    changed = False
    for name, m in re_edge.metrics.items():
        m_updates: dict = {}
        # Clamp the metric value to its native range.
        lo, hi = metric_ranges.get(name, (-1.0, 1.0))
        if not (lo <= m.value <= hi):
            clamped = _clamp(m.value, lo, hi)
            notes.append(
                f"[Auto-Fix] Clamped RelationshipEdge.metrics['{name}'].value "
                f"{m.value} \u2192 {clamped} "
                f"({re_edge.source_entity_id}\u2192{re_edge.target_entity_id})"
            )
            m_updates["value"] = clamped
        # Per-metric inertia clamp; cap at 0.99 so propagation never freezes.
        if not (0.0 <= m.inertia <= 1.0):
            clamped_in = _clamp(m.inertia, 0.0, 0.99)
            notes.append(
                f"[Auto-Fix] Clamped RelationshipEdge.metrics['{name}'].inertia "
                f"{m.inertia} \u2192 {clamped_in} "
                f"({re_edge.source_entity_id}\u2192{re_edge.target_entity_id})"
            )
            m_updates["inertia"] = clamped_in
        elif m.inertia >= 1.0:
            notes.append(
                f"[Auto-Fix] Capped RelationshipEdge.metrics['{name}'].inertia "
                f"at 0.99 ({re_edge.source_entity_id}\u2192"
                f"{re_edge.target_entity_id}) \u2014 1.0 would freeze the axis."
            )
            m_updates["inertia"] = 0.99
        # Coerce evidence_strength aliases.
        if m.evidence_strength not in ("weak", "moderate", "strong"):
            alias = es_aliases.get(str(m.evidence_strength).lower().strip())
            if alias:
                notes.append(
                    f"[Auto-Fix] RelationshipEdge.metrics['{name}']."
                    f"evidence_strength '{m.evidence_strength}' \u2192 "
                    f"'{alias}' ({re_edge.source_entity_id}\u2192"
                    f"{re_edge.target_entity_id})"
                )
                m_updates["evidence_strength"] = alias
            else:
                m_updates["evidence_strength"] = "moderate"
        if m_updates:
            new_metrics[name] = m.model_copy(update=m_updates)
            changed = True
        else:
            new_metrics[name] = m

    if changed:
        return re_edge.model_copy(update={"metrics": new_metrics})
    return re_edge


def _sanitize_register(register: "GlobalRegister", notes: List[str]) -> "GlobalRegister":
    """Clamp baseline numeric ranges on Step-1 ontology records.

    Pydantic ``field_validator``s already coerce ``evidence_strength``
    aliases on :class:`TraitVector`, :class:`AmbientVector`,
    :class:`Belief`, and :class:`InformationEdge` at construction time,
    so this pass focuses on numeric ranges that the LLM occasionally
    overshoots (``value``, ``inertia``, ``volatility``, ``confidence``)
    and caps inertia / volatility at ``0.99`` so propagation is never
    literally frozen by a bad ``1.0`` extraction.
    """
    # Entities: traits + beliefs
    for ent in register.entities.values():
        new_traits: Dict[str, TraitVector] = {}
        for tname, tv in ent.traits.items():
            new_value = _clamp(tv.value, 0.0, 1.0)
            new_inertia = _clamp(tv.inertia, 0.0, 1.0)
            if new_inertia >= 1.0:
                new_inertia = 0.99
                notes.append(
                    f"[Auto-Fix] Capped baseline trait '{ent.id}.{tname}' inertia at 0.99 "
                    f"\u2014 1.0 would freeze the trait for the whole story."
                )
            if new_value != tv.value or new_inertia != tv.inertia:
                if new_value != tv.value or new_inertia != tv.inertia:
                    notes.append(
                        f"[Auto-Fix] Clamped baseline trait '{ent.id}.{tname}' "
                        f"value/inertia {tv.value:.2f}/{tv.inertia:.2f} \u2192 "
                        f"{new_value:.2f}/{new_inertia:.2f}"
                    )
            new_traits[tname] = TraitVector(
                value=new_value,
                inertia=new_inertia,
                evidence_strength=tv.evidence_strength,
            )
        ent.traits = new_traits

        new_beliefs: List[Belief] = []
        for b in ent.beliefs:
            updates: dict = {}
            new_conf = _clamp(b.confidence, 0.0, 1.0)
            new_in = _clamp(b.inertia, 0.0, 1.0)
            if new_in >= 1.0:
                new_in = 0.99
            if new_conf != b.confidence:
                updates["confidence"] = new_conf
            if new_in != b.inertia:
                updates["inertia"] = new_in
            new_beliefs.append(b.model_copy(update=updates) if updates else b)
        ent.beliefs = new_beliefs

    # Locations: ambient_state
    for loc in register.locations.values():
        new_ambient = {}
        for aname, av in loc.ambient_state.items():
            # Normalise key: lowercase, snake_case, fix common typos.
            norm_name = _normalise_ambient_key(aname)
            if norm_name != aname:
                notes.append(
                    f"[Auto-Fix] Renamed ambient key '{loc.id}.{aname}' "
                    f"\u2192 '{norm_name}'"
                )
            new_value = _clamp(av.value, 0.0, 1.0)
            new_volatility = _clamp(av.volatility, 0.0, 1.0)
            if new_value != av.value or new_volatility != av.volatility:
                notes.append(
                    f"[Auto-Fix] Clamped ambient '{loc.id}.{norm_name}' "
                    f"value/volatility {av.value:.2f}/{av.volatility:.2f} \u2192 "
                    f"{new_value:.2f}/{new_volatility:.2f}"
                )
            new_ambient[norm_name] = AmbientVector(
                value=new_value,
                volatility=new_volatility,
                evidence_strength=av.evidence_strength,
            )
        loc.ambient_state = new_ambient

    # World traits: magnitude (TraitVector)
    for wt in register.world_traits.values():
        mag = wt.magnitude
        new_value = _clamp(mag.value, 0.0, 1.0)
        new_inertia = _clamp(mag.inertia, 0.0, 1.0)
        if new_inertia >= 1.0:
            new_inertia = 0.99
        if new_value != mag.value or new_inertia != mag.inertia:
            notes.append(
                f"[Auto-Fix] Clamped world-trait '{wt.id}.magnitude' "
                f"value/inertia {mag.value:.2f}/{mag.inertia:.2f} \u2192 "
                f"{new_value:.2f}/{new_inertia:.2f}"
            )
            wt.magnitude = TraitVector(
                value=new_value,
                inertia=new_inertia,
                evidence_strength=mag.evidence_strength,
            )

    return register


def _sanitize_entity_update(
    eu: "EntityUpdate", notes: List[str],
) -> Optional["EntityUpdate"]:
    """Clamp trait/inertia/confidence ranges and validate status enum.

    Drops the update entirely only when *every* field is no-op after
    sanitisation (the LLM produced a hollow record).
    """
    updates: dict = {}

    # status enum guard — schema enforces it, but LLMs sometimes return
    # title-case or synonyms. Try to coerce common aliases before giving up.
    if eu.new_status and eu.new_status not in _VALID_STATUSES:
        coerced = eu.new_status.lower().strip()
        alias = {
            "alive": "healthy", "well": "healthy",
            "wounded": "injured", "hurt": "injured",
            "sick": "ill", "diseased": "ill",
            "deceased": "dead", "killed": "dead",
            "ko": "unconscious", "knocked_out": "unconscious",
            "asleep": "unconscious",
        }.get(coerced)
        if alias:
            updates["new_status"] = alias
            notes.append(
                f"[Auto-Fix] EntityUpdate.new_status '{eu.new_status}' → '{alias}' "
                f"({eu.entity_id}@{eu.fabula_time})"
            )
        else:
            updates["new_status"] = None
            notes.append(
                f"[Auto-Fix] EntityUpdate.new_status '{eu.new_status}' is unknown — "
                f"dropped ({eu.entity_id}@{eu.fabula_time})."
            )

    # Clamp trait values + inertia to [0, 1]; physics asserts these.
    if eu.trait_updates:
        cleaned: Dict[str, TraitVector] = {}
        es_aliases = {
            "high": "strong", "low": "weak", "medium": "moderate",
            "med": "moderate", "uncertain": "weak", "certain": "strong",
        }
        for tname, tv in eu.trait_updates.items():
            new_value = _clamp(tv.value, 0.0, 1.0)
            new_inertia = _clamp(tv.inertia, 0.0, 1.0)
            # Avoid 1.0 inertia — it makes the trait literally unmovable.
            if new_inertia >= 1.0:
                new_inertia = 0.99
                notes.append(
                    f"[Auto-Fix] Capped EntityUpdate trait '{tname}' inertia at 0.99 "
                    f"({eu.entity_id}@{eu.fabula_time}) — 1.0 would freeze it forever."
                )
            if new_value != tv.value or new_inertia != tv.inertia:
                notes.append(
                    f"[Auto-Fix] Clamped EntityUpdate trait '{tname}' "
                    f"value/inertia {tv.value:.2f}/{tv.inertia:.2f} → "
                    f"{new_value:.2f}/{new_inertia:.2f} "
                    f"({eu.entity_id}@{eu.fabula_time})"
                )
            new_es = tv.evidence_strength
            if new_es not in ("weak", "moderate", "strong"):
                new_es = es_aliases.get(str(new_es).lower().strip(), "moderate")
                notes.append(
                    f"[Auto-Fix] EntityUpdate trait '{tname}' evidence_strength "
                    f"'{tv.evidence_strength}' → '{new_es}' "
                    f"({eu.entity_id}@{eu.fabula_time})"
                )
            cleaned[tname] = TraitVector(
                value=new_value, inertia=new_inertia, evidence_strength=new_es,
            )
        if cleaned != eu.trait_updates:
            updates["trait_updates"] = cleaned

    # Clamp belief confidence + inertia, ensure established_at_fabula is set.
    if eu.new_beliefs:
        cleaned_beliefs: List[Belief] = []
        for b in eu.new_beliefs:
            b_updates: dict = {}
            new_conf = _clamp(b.confidence, 0.0, 1.0)
            new_in = _clamp(b.inertia, 0.0, 1.0)
            if new_in >= 1.0:
                new_in = 0.99
            if new_conf != b.confidence:
                b_updates["confidence"] = new_conf
            if new_in != b.inertia:
                b_updates["inertia"] = new_in
            # Default missing established_at_fabula to the EntityUpdate's
            # own fabula_time so downstream time-slicing works.
            if not b.established_at_fabula and eu.fabula_time > 0:
                b_updates["established_at_fabula"] = eu.fabula_time
            # Coerce evidence_strength aliases (high/low/medium → strong/weak/moderate).
            if b.evidence_strength not in ("weak", "moderate", "strong"):
                alias = {
                    "high": "strong", "low": "weak", "medium": "moderate",
                    "med": "moderate", "uncertain": "weak", "certain": "strong",
                }.get(str(b.evidence_strength).lower().strip(), "moderate")
                b_updates["evidence_strength"] = alias
                notes.append(
                    f"[Auto-Fix] Belief.evidence_strength '{b.evidence_strength}' → "
                    f"'{alias}' ({eu.entity_id} re: {b.target_id})"
                )
            cleaned_beliefs.append(b.model_copy(update=b_updates) if b_updates else b)
        updates["new_beliefs"] = cleaned_beliefs

    # Drop entirely-empty updates: no traits, no beliefs, no
    # invalidations, no status, no location.
    candidate = eu.model_copy(update=updates) if updates else eu
    if (
        not candidate.trait_updates
        and not candidate.new_beliefs
        and not candidate.invalidated_belief_targets
        and candidate.new_status is None
        and candidate.new_location_id is None
    ):
        notes.append(
            f"[Auto-Fix] Dropped empty EntityUpdate for {eu.entity_id}@{eu.fabula_time} "
            "(no trait/belief/status/location change)."
        )
        return None
    return candidate


def _build_physics_agent(config: ExtractionConfig) -> Agent[_PhysicsDeps, PhysicsExtraction]:
    """Construct the Step 3a Physics Agent — events + causal + spatial edges."""
    agent: Agent[_PhysicsDeps, PhysicsExtraction] = Agent(
        _resolve_model(config.model),
        deps_type=_PhysicsDeps,
        output_type=NativeOutput(PhysicsExtraction),
        system_prompt=_load_prompt("physics_extraction.md"),
        retries=config.output_retries,
    )

    @agent.system_prompt
    def inject_register_for_physics(ctx: RunContext[_PhysicsDeps]) -> str:
        reg = ctx.deps.global_register
        entity_ids = sorted(reg.entities.keys())
        location_ids = sorted(reg.locations.keys())
        object_ids = sorted(reg.objects.keys())
        world_trait_ids = sorted(reg.world_traits.keys())
        entity_names = {eid: reg.entities[eid].name for eid in entity_ids}
        location_names = {lid: reg.locations[lid].name for lid in location_ids}
        object_names = {oid: reg.objects[oid].name for oid in object_ids}
        world_trait_names = {
            wid: reg.world_traits[wid].name for wid in world_trait_ids
        }
        scaffold_text = _format_scaffold(ctx.deps.scaffold)

        # Build compact entity baseline so the LLM knows starting trait values
        entity_baselines: List[str] = []
        for eid in entity_ids:
            ent = reg.entities[eid]
            traits_str = ", ".join(
                f"{k}={v.value:.1f}" for k, v in ent.traits.items()
            )
            entity_baselines.append(
                f"  {eid} ({ent.name}): status={ent.status}, "
                f"loc={ent.location_id}, traits=[{traits_str}]"
            )
        baselines_block = "\n".join(entity_baselines)

        return (
            "=== VALID ID REGISTER (from Step 1) ===\n"
            f"ENTITIES: {entity_names}\n"
            f"LOCATIONS: {location_names}\n"
            f"OBJECTS: {object_names}\n"
            f"WORLD TRAITS: {world_trait_names}\n"
            f"PREVIOUSLY EXTRACTED EVENT IDs: {ctx.deps.previous_event_ids}\n"
            "\n"
            "You MUST ONLY use ENT_, LOC_, OBJ_, WORLD_ IDs from the maps above.\n"
            "You MAY create new EVT_ IDs for events discovered in this chunk.\n"
            "Do NOT invent new ENT_, LOC_, OBJ_, or WORLD_ IDs. "
            "When the prose names a thing, look up its ID in the maps "
            "above by matching the name (e.g. 'the dagger' \u2192 "
            "whichever OBJ_ entry has name 'Bloody Daggers').\n"
            "\n"
            "=== ON-PAGE ENTITIES (substring-matched in this chunk's text) ===\n"
            f"{ctx.deps.on_page_entity_ids}\n"
            "\n"
            "These ENT_ ids are the *primary* cast for this chunk \u2014 the "
            "vast majority of events you emit should have at least one of "
            "these in actor_ids or target_ids. You MAY still reference "
            "offstage entities for memories, prophecies, gossip, "
            "absent-character utterances about them, or causal "
            "antecedents \u2014 the on-page list is advisory, not a hard "
            "filter \u2014 but if you find yourself emitting an event whose "
            "actor and target are BOTH offstage, double-check the chunk "
            "text supports it.\n"
            "\n"
            "=== ENTITY BASELINES (initial trait values \u2014 use for entity_updates) ===\n"
            f"{baselines_block}\n"
            "\n"
            "When emitting entity_updates, use these baselines as reference.\n"
            "The trait_updates values should be the NEW absolute value after the event, not the delta.\n"
            "\n"
            "=== SOCRATIC SCAFFOLD (semantic pre-analysis) ===\n"
            f"{scaffold_text}\n"
            "\n"
            "Use the scaffold above to inform your extraction — it identifies "
            "hidden motivations, implicit causal chains, and unobserved variables "
            "that you should capture as events and edges.\n"
            "\n"
            "=== PROPOSITION CATALOGUE (PROP_ ids referenceable on outcome events) ===\n"
            f"{_format_proposition_catalogue(ctx.deps.chunk_propositions)}\n"
            "\n"
            "When an `outcome` event commits the truth of one of the "
            "PROP_ ids above (e.g. EVT_DUNCAN_MURDER resolves "
            "PROP_DUNCAN_DEAD = true), list that PROP_ id in the "
            "event's `resolves_proposition_ids`. Do NOT invent new "
            "PROP_ ids — reference only the catalogue. Leave the "
            "field empty when no catalogued proposition is resolved."
        )

    @agent.output_validator
    def validate_physics_ids(ctx: RunContext[_PhysicsDeps], result: PhysicsExtraction) -> PhysicsExtraction:
        """Per-chunk validation — fix typos via fuzzy match, retry only for unfixable IDs."""
        reg = ctx.deps.global_register
        new_evt_ids = [e.id for e in result.events]
        valid = _build_valid_id_set(reg, ctx.deps.previous_event_ids + new_evt_ids)
        entity_ids = set(reg.entities.keys())
        location_ids = set(reg.locations.keys())
        fixes: List[str] = []
        bad: List[str] = []

        # --- Fix causal edge IDs ---
        fixed_causal: List[CausalEdge] = []
        for ce in result.causal_topology:
            updates: dict = {}
            src, src_fixed = _fix_id(ce.source_id, valid, "CausalEdge.source_id", fixes)
            tgt, tgt_fixed = _fix_id(ce.target_id, valid, "CausalEdge.target_id", fixes)
            if src != ce.source_id:
                updates["source_id"] = src
            if tgt != ce.target_id:
                updates["target_id"] = tgt
            if ce.rel_counterpart_id and ce.rel_counterpart_id not in valid:
                rc, rc_fixed = _fix_id(ce.rel_counterpart_id, valid, "CausalEdge.rel_counterpart_id", fixes)
                if rc != ce.rel_counterpart_id:
                    updates["rel_counterpart_id"] = rc
                if not rc_fixed and rc not in valid:
                    bad.append(f"CausalEdge rel_counterpart_id '{ce.rel_counterpart_id}' is not a valid ID.")
            if src not in valid and not updates.get("source_id"):
                bad.append(f"CausalEdge source_id '{ce.source_id}' is not a valid ID.")
            if tgt not in valid and not updates.get("target_id"):
                bad.append(f"CausalEdge target_id '{ce.target_id}' is not a valid ID.")
            try:
                fixed = ce.model_copy(update=updates) if updates else ce
            except Exception:
                # model_validator rejected the fix (prefix mismatch) — drop edge
                fixes.append(f"[Auto-Fix] Dropped causal edge {ce.source_id}→{ce.target_id} (validation error after fix)")
                continue
            sanitised = _sanitize_causal_edge(fixed, fixes)
            if sanitised is not None:
                fixed_causal.append(sanitised)

        # --- Fix spatial edge IDs ---
        object_ids_set = set(reg.objects.keys())
        fixed_spatial: List[SpatialEdge] = []
        for se in result.spatial_topology:
            updates = {}
            src, _ = _fix_id(se.source_id, location_ids, "SpatialEdge.source_id", fixes)
            tgt, _ = _fix_id(se.target_id, location_ids, "SpatialEdge.target_id", fixes)
            if src != se.source_id:
                updates["source_id"] = src
            if tgt != se.target_id:
                updates["target_id"] = tgt
            if src not in location_ids:
                bad.append(f"SpatialEdge source_id '{se.source_id}' is not a valid location.")
            elif tgt not in location_ids:
                bad.append(f"SpatialEdge target_id '{se.target_id}' is not a valid location.")
            elif src == tgt:
                fixes.append(
                    f"[Auto-Fix] Dropped self-loop SpatialEdge '{src}'→'{tgt}'"
                )
            else:
                # barrier_item_id must reference a real OBJ_ id when set;
                # fuzzy-fix typos, drop dangling pointers (the lock fact
                # survives — the engine treats unowned barriers as
                # "no key exists in the world", which is structurally
                # weaker than a barrier whose unlock affordance no
                # entity carries, but at least it's not pointing at a
                # non-existent object).
                if se.barrier_item_id:
                    bid, _ = _fix_id(
                        se.barrier_item_id, object_ids_set,
                        "SpatialEdge.barrier_item_id", fixes,
                    )
                    if bid in object_ids_set:
                        if bid != se.barrier_item_id:
                            updates["barrier_item_id"] = bid
                    else:
                        fixes.append(
                            f"[Auto-Fix] SpatialEdge {src}→{tgt} "
                            f"barrier_item_id '{se.barrier_item_id}' is not "
                            f"a valid OBJ_ id; clearing reference."
                        )
                        updates["barrier_item_id"] = None
                # Coherence: a passage flagged is_locked=True with no
                # barrier_item_id is structurally inert — nothing in the
                # world can ever unlock it, so the affordance check
                # silently falls through. Downgrade to is_locked=False
                # so the renderer doesn't describe the door as locked
                # while the simulator treats it as freely traversable.
                resolved_barrier = updates.get("barrier_item_id", se.barrier_item_id)
                if se.is_locked and not resolved_barrier:
                    fixes.append(
                        f"[Auto-Fix] SpatialEdge {src}→{tgt} is_locked=True "
                        f"with no barrier_item_id — clearing the lock "
                        f"(an unblockable barrier breaks affordance gating)."
                    )
                    updates["is_locked"] = False
                # Lifecycle sanity: destroyed_at_fabula must be strictly
                # after established_at_fabula, otherwise the passage is
                # destroyed before (or at the same tick as) it was built
                # — which makes the edge a no-op for the simulator and
                # is almost certainly an LLM transcription slip. Drop
                # the destruction tick rather than the whole edge so
                # the connectivity fact survives.
                est = se.established_at_fabula or 0
                dest = se.destroyed_at_fabula
                if dest is not None and dest <= est:
                    fixes.append(
                        f"[Auto-Fix] SpatialEdge {src}→{tgt} destroyed_at_fabula"
                        f"={dest} <= established_at_fabula={est}; clearing "
                        f"destruction tick (passage stays traversable)."
                    )
                    updates["destroyed_at_fabula"] = None
                fixed_spatial.append(se.model_copy(update=updates) if updates else se)

        # --- Fix entity_update IDs ---
        fixed_updates: List[EntityUpdate] = []
        for eu in result.entity_updates:
            updates = {}
            eid, _ = _fix_id(eu.entity_id, entity_ids, "EntityUpdate.entity_id", fixes)
            if eid != eu.entity_id:
                updates["entity_id"] = eid
            if eu.triggered_by and eu.triggered_by not in valid:
                trig, _ = _fix_id(eu.triggered_by, valid, "EntityUpdate.triggered_by", fixes)
                if trig != eu.triggered_by:
                    updates["triggered_by"] = trig
                if trig not in valid:
                    bad.append(f"EntityUpdate triggered_by '{eu.triggered_by}' is not a valid event.")
            if eu.new_location_id and eu.new_location_id not in location_ids:
                loc, _ = _fix_id(eu.new_location_id, location_ids, "EntityUpdate.new_location_id", fixes)
                if loc != eu.new_location_id:
                    updates["new_location_id"] = loc
                if loc not in location_ids:
                    bad.append(f"EntityUpdate new_location_id '{eu.new_location_id}' is not a valid location.")
            if eid not in entity_ids:
                bad.append(f"EntityUpdate entity_id '{eu.entity_id}' is not a valid entity.")
            else:
                cleaned_eu = eu.model_copy(update=updates) if updates else eu
                sanitised_eu = _sanitize_entity_update(cleaned_eu, fixes)
                if sanitised_eu is not None:
                    fixed_updates.append(sanitised_eu)

        if fixes:
            logger.info("[Validator·Physics] Auto-fixed %d ID(s): %s", len(fixes), "; ".join(fixes))

        if bad:
            raise ModelRetry(
                "The following IDs could not be auto-resolved. "
                "Fix them using ONLY IDs from the register:\n" + "\n".join(bad)
            )

        return PhysicsExtraction(
            events=result.events,
            causal_topology=fixed_causal,
            spatial_topology=fixed_spatial,
            entity_updates=fixed_updates,
        )

    @agent.output_validator
    def reject_physics_utterances(
        ctx: RunContext[_PhysicsDeps], result: PhysicsExtraction,
    ) -> PhysicsExtraction:
        """Strip ``event_type='utterance'`` events from physics output.

        Per the physics_extraction.md prompt, utterances belong to the
        Social Agent. The schema permits any ``EventLiteral`` so smaller
        models routinely violate this and the social pass then re-emits
        the same utterance, leaving duplicates that dedup may or may
        not catch by id coincidence. Drop them here so the social pass
        is the single canonical source of utterances.
        """
        utterance_events = [e for e in result.events if e.event_type == "utterance"]
        if not utterance_events:
            return result
        kept = [e for e in result.events if e.event_type != "utterance"]
        dropped_ids = [e.id for e in utterance_events]
        dropped_set = set(dropped_ids)
        # Drop any causal/entity-update edges referencing the dropped ids
        # so downstream programmatic validation does not surface broken
        # links for things we just removed.
        cleaned_causal = [
            ce for ce in result.causal_topology
            if ce.source_id not in dropped_set and ce.target_id not in dropped_set
        ]
        cleaned_updates = [
            eu for eu in result.entity_updates
            if eu.triggered_by not in dropped_set
        ]
        logger.info(
            "[Validator·Physics] Dropped %d utterance event(s) from physics "
            "output (utterances belong to the Social Agent): %s.",
            len(utterance_events), dropped_ids,
        )
        return PhysicsExtraction(
            events=kept,
            causal_topology=cleaned_causal,
            spatial_topology=result.spatial_topology,
            entity_updates=cleaned_updates,
        )

    return agent


def _build_social_agent(config: ExtractionConfig) -> Agent[_SocialDeps, SocialExtraction]:
    """Construct the Step 3b Social Agent — information + relationship edges."""
    agent: Agent[_SocialDeps, SocialExtraction] = Agent(
        _resolve_model(config.model),
        deps_type=_SocialDeps,
        output_type=NativeOutput(SocialExtraction),
        system_prompt=_load_prompt("social_extraction.md"),
        retries=config.output_retries,
    )

    @agent.system_prompt
    def inject_register_for_social(ctx: RunContext[_SocialDeps]) -> str:
        reg = ctx.deps.global_register
        entity_ids = sorted(reg.entities.keys())
        location_ids = sorted(reg.locations.keys())
        object_ids = sorted(reg.objects.keys())
        world_trait_ids = sorted(reg.world_traits.keys())
        entity_names = {eid: reg.entities[eid].name for eid in entity_ids}
        location_names = {lid: reg.locations[lid].name for lid in location_ids}
        object_names = {oid: reg.objects[oid].name for oid in object_ids}
        world_trait_names = {
            wid: reg.world_traits[wid].name for wid in world_trait_ids
        }
        all_evt_ids = ctx.deps.previous_event_ids + ctx.deps.chunk_event_ids
        scaffold_text = _format_scaffold(ctx.deps.scaffold)

        # Prior-chunk channel summary so the LLM can reuse standing
        # capabilities by id rather than reinventing them under a new
        # name. Each line carries everything needed to identify the
        # right channel: id, medium, participants, directionality, and
        # the fabula tick it was established at.
        prior_chn_lines: List[str] = []
        for cid, ch in ctx.deps.previous_chunk_channels.items():
            prior_chn_lines.append(
                f"  - {cid} (medium={ch.medium}, "
                f"participants={ch.participant_ids}, "
                f"directionality={ch.directionality}, "
                f"established_at_fabula={ch.established_at_fabula})"
            )
        prior_channels_block = (
            "\n".join(prior_chn_lines)
            if prior_chn_lines
            else "  (none — this is the first chunk to extract channels)"
        )

        # Compact event summary so the agent can attribute utterances /
        # channels / relationship_edges to specific Physics events
        # (e.g. setting an utterance's ``triggered_by``) without
        # re-deriving fabula_time or actors from the prose.
        evt_lines: List[str] = []
        for e in ctx.deps.chunk_events:
            evt_lines.append(
                f"  - {e.id} (fabula={e.fabula_time}, type={e.event_type}, "
                f"actors={e.actor_ids}, targets={e.target_ids}): {e.description}"
            )
        events_block = (
            "\n".join(evt_lines) if evt_lines else "  (no events extracted from this chunk)"
        )

        # Mutation_social hints — every such Physics edge implies a
        # relationship_edge update on the same axis. Surface them
        # explicitly so the Social agent's relationship_topology stays
        # consistent with the causal graph instead of disagreeing.
        mut_soc_lines: List[str] = []
        for ce in ctx.deps.chunk_causal:
            if ce.causality_type == "mutation_social":
                mut_soc_lines.append(
                    f"  - {ce.source_id} → {ce.target_id} "
                    f"(rel_counterpart={ce.rel_counterpart_id}, "
                    f"axis={ce.trait_target}, delta={ce.trait_delta})"
                )
        mut_soc_block = (
            "\n".join(mut_soc_lines)
            if mut_soc_lines
            else "  (none — emit relationship_edges from your own reading of the chunk)"
        )

        return (
            "=== VALID ID REGISTER ===\n"
            f"ENTITIES: {entity_names}\n"
            f"LOCATIONS: {location_names}\n"
            f"OBJECTS: {object_names}\n"
            f"WORLD TRAITS: {world_trait_names}\n"
            f"THIS CHUNK'S EVENT IDs: {ctx.deps.chunk_event_ids}\n"
            f"PREVIOUS CHUNKS' EVENT IDs: {ctx.deps.previous_event_ids}\n"
            f"ALL VALID EVENT IDs: {all_evt_ids}\n"
            "\n"
            "You MUST ONLY use the ENT_/LOC_/OBJ_/WORLD_/EVT_ ids "
            "listed above. Do NOT invent ENT_/LOC_/OBJ_/WORLD_/EVT_ "
            "ids. You MUST mint new CHN_* ids for any standing "
            "channels you extract and new EVT_UTT_* ids for utterance "
            "events (both are required by the schema and must be "
            "unique within this chunk).\n"
            "\n"
            "=== EVENTS EXTRACTED FROM THIS CHUNK (by Physics Agent) ===\n"
            f"{events_block}\n"
            "\n"
            "Anchor utterance events and channel ``established_at_fabula`` "
            "to these event IDs / fabula times rather than guessing.\n"
            "\n"
            "=== MUTATION_SOCIAL EDGES FROM PHYSICS (relationship hints) ===\n"
            f"{mut_soc_block}\n"
            "\n"
            "Every mutation_social edge above corresponds to a "
            "relationship_edge on the same (source, target, axis). "
            "Make sure your `relationship_topology` reflects these "
            "shifts; do not contradict the causal graph.\n"
            "\n"
            "=== STANDING CHANNELS ALREADY ESTABLISHED IN PRIOR CHUNKS ===\n"
            f"{prior_channels_block}\n"
            "\n"
            "When an utterance in THIS chunk travels over a standing "
            "capability that already appears above (e.g. an ongoing "
            "letter correspondence, a telephone line, a mind-bond, a "
            "spy pipeline), set the utterance's `via_channel_id` to "
            "the existing CHN_ id and DO NOT re-emit the channel in "
            "your `channels` dict. Only add a new entry to `channels` "
            "for genuinely-new standing capabilities established (or "
            "first observed) in this chunk.\n"
            "\n"
            "=== ON-PAGE ENTITIES (substring-matched in this chunk's text) ===\n"
            f"{ctx.deps.on_page_entity_ids}\n"
            "\n"
            "Channels and utterances should primarily involve these "
            "on-page entities. Offstage entities are valid as "
            "addressees of letters / messages / prophecies, but a "
            "channel with NO on-page participants is almost always a "
            "fabrication.\n"
            "\n"
            "=== SOCRATIC SCAFFOLD (semantic pre-analysis) ===\n"
            f"{scaffold_text}\n"
            "\n"
            "Use the scaffold above to identify implicit social dynamics, "
            "hidden information flows, and unspoken relationship shifts.\n"
            "\n"
            "=== PROPOSITION CATALOGUE (PROP_ ids referenceable on utterances) ===\n"
            f"{_format_proposition_catalogue(ctx.deps.chunk_propositions)}\n"
            "\n"
            "When a speaker affirms one of the PROP_ ids above on-page "
            "(confession, accusation, sworn testimony), set the "
            "utterance's `asserts_proposition_id` to that PROP_ id. "
            "When the speaker denies one (lie, alibi, dismissal), set "
            "`denies_proposition_id`. Do NOT invent new PROP_ ids; "
            "reference only the catalogue."
        )

    @agent.output_validator
    def validate_social_ids(ctx: RunContext[_SocialDeps], result: SocialExtraction) -> SocialExtraction:
        """Per-chunk validation — fix typos via fuzzy match, retry only for unfixable IDs."""
        reg = ctx.deps.global_register
        entity_ids = set(reg.entities.keys())
        node_ids = entity_ids | set(reg.objects.keys())
        fixes: List[str] = []
        bad: List[str] = []

        # --- Fix Channel participants ---
        fixed_channels: Dict[str, Channel] = {}
        for cid, ch in result.channels.items():
            updates: dict = {}
            new_pids: List[str] = []
            for pid in ch.participant_ids:
                p, _ = _fix_id(pid, node_ids, "Channel.participant_ids", fixes)
                if p in node_ids:
                    if p not in new_pids:
                        new_pids.append(p)
                else:
                    bad.append(f"Channel '{cid}' participant_id '{pid}' is not a valid entity/object.")
            if len(new_pids) < 2:
                fixes.append(
                    f"[Auto-Fix] Dropped Channel '{cid}' — fewer than 2 valid participants."
                )
                continue
            if new_pids != list(ch.participant_ids):
                updates["participant_ids"] = new_pids
            # Drop intelligibility entries for participants we removed
            if ch.intelligibility:
                pruned_intel = {k: v for k, v in ch.intelligibility.items() if k in new_pids}
                if pruned_intel != dict(ch.intelligibility):
                    updates["intelligibility"] = pruned_intel
            if ch.evidence_strength not in ("weak", "moderate", "strong"):
                alias = {
                    "high": "strong", "low": "weak", "medium": "moderate",
                    "med": "moderate", "uncertain": "weak", "certain": "strong",
                }.get(str(ch.evidence_strength).lower().strip(), "moderate")
                updates["evidence_strength"] = alias
                fixes.append(
                    f"[Auto-Fix] Channel.evidence_strength "
                    f"'{ch.evidence_strength}' → '{alias}' (on {cid})"
                )
            fixed_channels[cid] = ch.model_copy(update=updates) if updates else ch

        # --- Fix utterance EventNode ids (speaker, addressees, channel) ---
        valid_channel_ids = set(fixed_channels.keys()) | set(
            ctx.deps.previous_chunk_channels.keys()
        )
        all_evt_ids = set(ctx.deps.previous_event_ids) | set(ctx.deps.chunk_event_ids)
        fixed_utterances: List[EventNode] = []
        seen_utt_ids: set[str] = set()
        for ev in result.utterance_events:
            if ev.event_type != "utterance":
                fixes.append(
                    f"[Auto-Fix] Coerced utterance_events entry '{ev.id}' "
                    f"event_type from '{ev.event_type}' to 'utterance'."
                )
                ev = ev.model_copy(update={"event_type": "utterance"})
            updates: dict = {}
            # --- Enforce EVT_UTT_ prefix to avoid collisions with Physics ids ---
            new_id = ev.id
            if not new_id.startswith("EVT_UTT_"):
                if new_id.startswith("EVT_"):
                    new_id = "EVT_UTT_" + new_id[len("EVT_"):]
                else:
                    new_id = "EVT_UTT_" + re.sub(r"[^A-Z0-9_]+", "_", new_id.upper()).strip("_")
                fixes.append(
                    f"[Auto-Fix] Renamed utterance '{ev.id}' → '{new_id}' "
                    f"(EVT_UTT_ prefix is required)."
                )
            # Resolve collisions with Physics-extracted ids OR previous-chunk ids
            # OR an earlier utterance in this same batch.
            if new_id in all_evt_ids or new_id in seen_utt_ids:
                base = new_id
                suffix = 2
                while f"{base}_{suffix}" in all_evt_ids or f"{base}_{suffix}" in seen_utt_ids:
                    suffix += 1
                renamed = f"{base}_{suffix}"
                fixes.append(
                    f"[Auto-Fix] Renamed utterance '{new_id}' → '{renamed}' "
                    f"(id collided with an existing event)."
                )
                new_id = renamed
            seen_utt_ids.add(new_id)
            if new_id != ev.id:
                updates["id"] = new_id
            # speaker_id
            if ev.speaker_id:
                sp, _ = _fix_id(ev.speaker_id, node_ids, "EventNode.speaker_id", fixes)
                if sp not in node_ids:
                    bad.append(f"Utterance '{ev.id}' speaker_id '{ev.speaker_id}' is not a valid entity/object.")
                elif sp != ev.speaker_id:
                    updates["speaker_id"] = sp
            # addressee_ids
            new_addrs: List[str] = []
            for aid in ev.addressee_ids:
                a, _ = _fix_id(aid, node_ids, "EventNode.addressee_ids", fixes)
                if a in node_ids and a not in new_addrs:
                    new_addrs.append(a)
                elif a not in node_ids:
                    bad.append(f"Utterance '{ev.id}' addressee_ids entry '{aid}' is not a valid entity/object.")
            if new_addrs != list(ev.addressee_ids):
                updates["addressee_ids"] = new_addrs
            # via_channel_id
            if ev.via_channel_id and ev.via_channel_id not in valid_channel_ids:
                # Drop the broken channel reference rather than fail — the
                # utterance is still valid as an unmediated speech-act.
                fixes.append(
                    f"[Auto-Fix] Utterance '{ev.id}' via_channel_id "
                    f"'{ev.via_channel_id}' is not in the chunk's channels; "
                    f"clearing reference."
                )
                updates["via_channel_id"] = None
            elif ev.via_channel_id and ev.via_channel_id in fixed_channels:
                # Membership check: when an utterance rides a channel
                # extracted in THIS chunk, every speaker/addressee MUST
                # appear in that channel's participant_ids — otherwise
                # the channel mediation is structurally incoherent
                # (you can't broadcast over a phone line you're not on).
                # We can only enforce this for in-chunk channels because
                # prior-chunk channels' participant lists may have been
                # widened by intervening extractions.
                ch_pids = set(fixed_channels[ev.via_channel_id].participant_ids)
                # Resolve the post-fix speaker / addressees so the
                # check sees the same ids the validator just rewrote.
                resolved_speaker = updates.get("speaker_id", ev.speaker_id)
                resolved_addrs = updates.get("addressee_ids", list(ev.addressee_ids))
                participants_needed = {resolved_speaker} | set(resolved_addrs)
                participants_needed.discard(None)
                missing = sorted(participants_needed - ch_pids)
                if missing:
                    fixes.append(
                        f"[Auto-Fix] Utterance '{ev.id}' via_channel_id "
                        f"'{ev.via_channel_id}' missing participant(s) "
                        f"{missing}; clearing channel reference (the "
                        f"utterance survives as unmediated speech)."
                    )
                    updates["via_channel_id"] = None
            # actor_ids consistency: if speaker present, ensure it appears in actor_ids
            if ev.speaker_id and ev.speaker_id not in ev.actor_ids:
                merged_actors = list(ev.actor_ids)
                merged_actors.append(updates.get("speaker_id", ev.speaker_id))
                updates["actor_ids"] = merged_actors
            fixed_utterances.append(ev.model_copy(update=updates) if updates else ev)

        # --- Fix relationship edge IDs ---
        fixed_social: List[RelationshipEdge] = []
        for re_edge in result.social_topology:
            updates = {}
            src, _ = _fix_id(re_edge.source_entity_id, entity_ids, "RelationshipEdge.source_entity_id", fixes)
            tgt, _ = _fix_id(re_edge.target_entity_id, entity_ids, "RelationshipEdge.target_entity_id", fixes)
            if src != re_edge.source_entity_id:
                updates["source_entity_id"] = src
            if tgt != re_edge.target_entity_id:
                updates["target_entity_id"] = tgt
            if src not in entity_ids:
                bad.append(f"RelationshipEdge source_entity_id '{re_edge.source_entity_id}' is not a valid entity.")
            elif tgt not in entity_ids:
                bad.append(f"RelationshipEdge target_entity_id '{re_edge.target_entity_id}' is not a valid entity.")
            elif src == tgt:
                fixes.append(f"[Auto-Fix] Dropped self-referencing RelationshipEdge '{src}'→'{tgt}'")
            else:
                rebuilt = re_edge.model_copy(update=updates) if updates else re_edge
                sanitized = _sanitize_relationship_edge(rebuilt, fixes)
                # Drop dyads with no observed metrics — an edge whose
                # ``metrics`` dict is empty (or contains only
                # ``observed=False`` axes) is structurally valid but
                # contributes no signal: every aggregator returns 0.0
                # indistinguishable from a measured neutral, polluting
                # asymmetry / tension / propagation pipelines downstream.
                observed_axes = [
                    name for name, m in sanitized.metrics.items() if m.observed
                ]
                if not observed_axes:
                    fixes.append(
                        f"[Auto-Fix] Dropped RelationshipEdge "
                        f"{src}→{tgt} with no observed metrics "
                        f"(metrics={list(sanitized.metrics.keys())})."
                    )
                    continue
                fixed_social.append(sanitized)

        if fixes:
            logger.info("[Validator·Social] Auto-fixed %d issue(s): %s", len(fixes), "; ".join(fixes))

        if bad:
            raise ModelRetry(
                "The following IDs could not be auto-resolved. "
                "Fix them using ONLY IDs from the register:\n" + "\n".join(bad)
            )

        # --- Tier 3 #13: prune empty conversational channels -----------
        # A channel emitted in this chunk whose ``medium`` describes
        # face-to-face speech and which NO utterance references is
        # almost always a fabrication \u2014 conversational mediums by
        # definition need on-page utterances to exist. Standing
        # infrastructure mediums (telephone, mind_link,
        # classified_pipeline, mail_correspondence) can legitimately
        # be established in a chunk that contains only narration
        # about them, so we leave those alone.
        _ephemeral_mediums = {
            "speech", "verbal", "conversation", "face_to_face",
            "in_person", "talking", "spoken",
        }
        used_channel_ids: set[str] = set()
        for u in fixed_utterances:
            if u.via_channel_id:
                used_channel_ids.add(u.via_channel_id)
        prune_ids: List[str] = []
        for cid, ch in fixed_channels.items():
            if cid in used_channel_ids:
                continue
            medium_norm = (ch.medium or "").strip().lower().replace("-", "_")
            if medium_norm in _ephemeral_mediums:
                prune_ids.append(cid)
        for cid in prune_ids:
            logger.info(
                "[Validator\u00b7Social] Pruned conversational Channel %s "
                "(medium=%r, no in-chunk utterance references).",
                cid, fixed_channels[cid].medium,
            )
            del fixed_channels[cid]

        return SocialExtraction(
            channels=fixed_channels,
            utterance_events=fixed_utterances,
            social_topology=fixed_social,
        )

    return agent


def _build_consequences_agent(
    config: ExtractionConfig,
) -> Agent[_ConsequencesDeps, ConsequencesExtraction]:
    """Construct the Step 3c Consequences Agent — entity_updates only.

    The agent is given the events and mutation/mutation_social edges
    already produced by the Physics Agent, plus the entity baselines
    from the Global Register. Its sole job is to translate those into
    EntityUpdate records (trait deltas, new/invalidated beliefs, status
    changes, location changes).
    """
    agent: Agent[_ConsequencesDeps, ConsequencesExtraction] = Agent(
        _resolve_model(config.model),
        deps_type=_ConsequencesDeps,
        output_type=NativeOutput(ConsequencesExtraction),
        system_prompt=_load_prompt("consequences_extraction.md"),
        retries=config.output_retries,
    )

    @agent.system_prompt
    def inject_register_for_consequences(ctx: RunContext[_ConsequencesDeps]) -> str:
        reg = ctx.deps.global_register
        entity_ids = sorted(reg.entities.keys())
        location_ids = sorted(reg.locations.keys())
        object_ids = sorted(reg.objects.keys())
        world_trait_ids = sorted(reg.world_traits.keys())
        entity_names = {eid: reg.entities[eid].name for eid in entity_ids}
        location_names = {lid: reg.locations[lid].name for lid in location_ids}
        object_names = {oid: reg.objects[oid].name for oid in object_ids}
        world_trait_names = {
            wid: reg.world_traits[wid].name for wid in world_trait_ids
        }
        scaffold_text = _format_scaffold(ctx.deps.scaffold)

        # Compact entity baselines so the LLM knows starting trait values.
        entity_baselines: List[str] = []
        for eid in entity_ids:
            ent = reg.entities[eid]
            traits_str = ", ".join(
                f"{k}={v.value:.1f}/i={v.inertia:.2f}" for k, v in ent.traits.items()
            )
            entity_baselines.append(
                f"  {eid} ({ent.name}): status={ent.status}, "
                f"loc={ent.location_id}, traits=[{traits_str}]"
            )
        baselines_block = "\n".join(entity_baselines)

        # Compact event summary (the only events whose consequences matter).
        evt_lines: List[str] = []
        for e in ctx.deps.chunk_events:
            evt_lines.append(
                f"  - {e.id} (fabula={e.fabula_time}, type={e.event_type}, "
                f"actors={e.actor_ids}, targets={e.target_ids}): {e.description}"
            )
        events_block = "\n".join(evt_lines) if evt_lines else "  (no events in this chunk)"

        # Compact mutation hints from Physics — every mutation/mutation_social
        # edge implies an EntityUpdate is needed.
        mut_lines: List[str] = []
        for ce in ctx.deps.chunk_causal:
            if ce.causality_type in ("mutation", "mutation_social"):
                mut_lines.append(
                    f"  - {ce.source_id} → {ce.target_id} "
                    f"[{ce.causality_type}] trait={ce.trait_target} "
                    f"delta={ce.trait_delta} (force={ce.causal_force}, "
                    f"evidence={ce.evidence_strength})"
                )
        mutations_block = (
            "\n".join(mut_lines)
            if mut_lines
            else "  (no mutation edges — infer trait/belief deltas from events directly)"
        )

        chunk_evt_ids = [e.id for e in ctx.deps.chunk_events]

        # Compact channels + utterances summary so the agent can wire
        # belief provenance through ``acquired_via_event_id`` and
        # ``acquired_via_channel_id``.
        chn_lines: List[str] = []
        for cid, ch in ctx.deps.chunk_channels.items():
            chn_lines.append(
                f"  - {cid} (medium={ch.medium}, "
                f"participants={ch.participant_ids}, "
                f"directionality={ch.directionality})"
            )
        channels_block = "\n".join(chn_lines) if chn_lines else "  (no channels in this chunk)"

        utt_lines: List[str] = []
        for u in ctx.deps.chunk_utterance_events:
            utt_lines.append(
                f"  - {u.id} (fabula={u.fabula_time}, "
                f"speaker={u.speaker_id}, addressees={u.addressee_ids}, "
                f"via={u.via_channel_id}, truth={u.truth_value}): {u.description}"
            )
        utterances_block = (
            "\n".join(utt_lines) if utt_lines else "  (no utterance events in this chunk)"
        )

        return (
            "=== VALID ID REGISTER (from Step 1) ===\n"
            f"ENTITIES: {entity_names}\n"
            f"LOCATIONS: {location_names}\n"
            f"OBJECTS: {object_names}\n"
            f"WORLD TRAITS: {world_trait_names}\n"
            f"THIS CHUNK'S EVENT IDs: {chunk_evt_ids}\n"
            f"PREVIOUS CHUNKS' EVENT IDs: {ctx.deps.previous_event_ids}\n"
            "\n"
            "You MUST ONLY use IDs from the lists above. Do NOT invent any IDs.\n"
            "\n"
            "=== ENTITY BASELINES (initial trait values + inertia) ===\n"
            f"{baselines_block}\n"
            "\n"
            "Trait_updates values are the NEW absolute trait value after the "
            "event, NOT the delta. The engine computes deltas from baselines.\n"
            "\n"
            "=== EVENTS EXTRACTED FROM THIS CHUNK (by Physics Agent) ===\n"
            f"{events_block}\n"
            "\n"
            "=== MUTATION EDGES FROM PHYSICS (each implies an EntityUpdate) ===\n"
            f"{mutations_block}\n"
            "\n"
            "=== CHANNELS EXTRACTED FROM THIS CHUNK (by Social Agent) ===\n"
            f"{channels_block}\n"
            "\n"
            "=== UTTERANCE EVENTS EXTRACTED FROM THIS CHUNK (by Social Agent) ===\n"
            f"{utterances_block}\n"
            "\n"
            "When you emit a `new_beliefs` entry whose source is one of the "
            "utterance events above, set `acquired_via_event_id` to that "
            "utterance's id, and (when the utterance has a `via_channel_id`) "
            "set `acquired_via_channel_id` to the channel id. This lets "
            "counterfactual surgery prune downstream beliefs cleanly when "
            "the channel is severed or the utterance is rewritten.\n"
            "\n"
            "=== ON-PAGE ENTITIES (substring-matched in this chunk's text) ===\n"
            f"{ctx.deps.on_page_entity_ids}\n"
            "\n"
            "EntityUpdates should overwhelmingly be for these on-page "
            "entities. An EntityUpdate for an entity that does NOT "
            "appear in the chunk text and is NOT a target of any "
            "mutation edge is almost always a hallucination.\n"
            "\n"
            "=== SOCRATIC SCAFFOLD (semantic pre-analysis) ===\n"
            f"{scaffold_text}\n"
            "\n"
            "Use the scaffold's WHY/HOW answers to surface IMPLICIT trait "
            "shifts (guilt after killing, fear after threat, grief after loss) "
            "even when the prose does not name them.\n"
            "\n"
            "=== PROPOSITION CATALOGUE (PROP_ ids referenceable on beliefs) ===\n"
            f"{_format_proposition_catalogue(ctx.deps.chunk_propositions)}\n"
            "\n"
            "When you emit a `Belief` whose target maps to one of the "
            "PROP_ ids above (the belief is *about* that catalogued "
            "proposition — e.g. a belief that Duncan is dead targets "
            "PROP_DUNCAN_DEAD), set the belief's `proposition_id` to "
            "the matching PROP_ id. This wires the belief into the "
            "affect-unification layer at extraction time and avoids "
            "a downstream clustering call. Leave `proposition_id` "
            "null when no catalogued proposition matches."
        )

    @agent.output_validator
    def validate_consequences_ids(
        ctx: RunContext[_ConsequencesDeps],
        result: ConsequencesExtraction,
    ) -> ConsequencesExtraction:
        """Validate entity_update IDs — fix typos, retry only for unfixable IDs."""
        reg = ctx.deps.global_register
        chunk_evt_ids = [e.id for e in ctx.deps.chunk_events]
        valid = _build_valid_id_set(
            reg, ctx.deps.previous_event_ids + chunk_evt_ids,
        )
        entity_ids = set(reg.entities.keys())
        location_ids = set(reg.locations.keys())
        fixes: List[str] = []
        bad: List[str] = []

        fixed_updates: List[EntityUpdate] = []
        for eu in result.entity_updates:
            updates: dict = {}
            eid, _ = _fix_id(eu.entity_id, entity_ids, "EntityUpdate.entity_id", fixes)
            if eid != eu.entity_id:
                updates["entity_id"] = eid
            if eu.triggered_by and eu.triggered_by not in valid:
                trig, _ = _fix_id(eu.triggered_by, valid, "EntityUpdate.triggered_by", fixes)
                if trig != eu.triggered_by:
                    updates["triggered_by"] = trig
                if trig not in valid:
                    bad.append(
                        f"EntityUpdate triggered_by '{eu.triggered_by}' is not a valid event."
                    )
            if eu.new_location_id and eu.new_location_id not in location_ids:
                loc, _ = _fix_id(eu.new_location_id, location_ids, "EntityUpdate.new_location_id", fixes)
                if loc != eu.new_location_id:
                    updates["new_location_id"] = loc
                if loc not in location_ids:
                    bad.append(
                        f"EntityUpdate new_location_id '{eu.new_location_id}' is not a valid location."
                    )
            if eid not in entity_ids:
                bad.append(f"EntityUpdate entity_id '{eu.entity_id}' is not a valid entity.")
            else:
                cleaned_eu = eu.model_copy(update=updates) if updates else eu
                sanitised_eu = _sanitize_entity_update(cleaned_eu, fixes)
                if sanitised_eu is not None:
                    fixed_updates.append(sanitised_eu)

        # --- Mutation⇄EntityUpdate parity audit ---
        # Every Physics mutation/mutation_social edge that targets an entity
        # SHOULD have a matching EntityUpdate. Surface gaps so they are
        # visible in the log; do not retry (the agent already had every
        # mutation listed in its system prompt).
        eu_keys: set[Tuple[str, int, Optional[str]]] = {
            (eu.entity_id, eu.fabula_time, eu.triggered_by) for eu in fixed_updates
        }
        eu_entity_set: set[str] = {eu.entity_id for eu in fixed_updates}
        missing: List[str] = []
        for ce in ctx.deps.chunk_causal:
            if ce.causality_type not in ("mutation", "mutation_social"):
                continue
            target_ent = (
                ce.target_id if ce.causality_type == "mutation"
                else ce.target_id  # mutation_social: target_id is the perspective entity
            )
            if not target_ent.startswith("ENT_"):
                continue
            # Loose match: same entity touched by something in this chunk is OK.
            # We only flag when the entity has no update at all.
            if target_ent not in eu_entity_set:
                missing.append(
                    f"{ce.source_id} → {target_ent} "
                    f"({ce.causality_type}, trait={ce.trait_target})"
                )
        if missing:
            logger.info(
                "[Validator·Consequences] %d mutation edge(s) lack a "
                "corresponding EntityUpdate: %s",
                len(missing), "; ".join(missing[:5]),
            )

        # --- Dead-actor warning ---
        # If an EntityUpdate marks an entity dead, warn when subsequent
        # events in this chunk still list that entity as an actor.
        # Exception: undead / spectral / ghost entities (anything whose
        # ontology ``constants`` includes a supernatural marker) are
        # legitimately permitted to act after their death — Marley,
        # Banquo's ghost, the witches' apparitions, etc. Don't flag.
        SPECTRAL_MARKERS = {
            "undead", "spectral", "ghost", "ghostly", "spirit",
            "phantom", "revenant", "wraith", "apparition",
        }
        registry = ctx.deps.global_register.entities
        deaths: Dict[str, int] = {
            eu.entity_id: eu.fabula_time
            for eu in fixed_updates
            if eu.new_status == "dead"
        }
        for ev in ctx.deps.chunk_events:
            for actor in ev.actor_ids:
                if actor not in deaths or ev.fabula_time <= deaths[actor]:
                    continue
                ent = registry.get(actor)
                consts = {str(c).lower() for c in (getattr(ent, "constants", None) or [])} if ent else set()
                name_lower = (ent.name.lower() if ent and ent.name else "")
                if consts & SPECTRAL_MARKERS or "ghost" in name_lower or "spirit" in name_lower or "phantom" in name_lower:
                    # Spectral entity acting post-mortem is canonical.
                    continue
                logger.info(
                    "[Validator·Consequences] %s is marked dead at "
                    "fabula=%d but still acts in event %s at fabula=%d.",
                    actor, deaths[actor], ev.id, ev.fabula_time,
                )

        if fixes:
            logger.info(
                "[Validator·Consequences] Auto-fixed %d issue(s): %s",
                len(fixes), "; ".join(fixes[:8]),
            )
        if bad:
            raise ModelRetry(
                "The following IDs could not be auto-resolved. "
                "Fix them using ONLY IDs from the register:\n" + "\n".join(bad)
            )

        return ConsequencesExtraction(entity_updates=fixed_updates)

    return agent


# =====================================================================
# Step 3d — Per-chunk Affect Agent (Phase B4)
#
# Runs after Physics + Social + Consequences. Owns the diff between
# the global Phase A3 catalogue (PROP_ / CCN_ baselines) and the
# chunk's on-page events: which propositions' framing shifts, which
# truths commit, which concerns drift in salience or polarity. The
# reconciler folds the agent's outputs into Proposition.state_timeline /
# truth_at_fabula and Entity.concerns / Concern.state_timeline.
# =====================================================================


def _chunk_has_affect_signal(
    events: List[EventNode],
    entity_updates: List["EntityUpdate"],
    catalogue_prop_ids: Set[str],
    seeded_entity_ids: Set[str],
    *,
    referent_to_props: Dict[str, Set[str]] | None = None,
    concern_entity_to_props: Dict[str, Set[str]] | None = None,
) -> bool:
    """True iff this chunk plausibly affects a catalogue prop or seeded concern.

    Conservative gate so the affect LLM call is skipped on chunks
    whose extractor outputs reference no PROP_ id and no seeded
    entity. Runs in O(events + entity_updates + new_beliefs); the
    chunk-level cost is negligible compared to a saved per-chunk
    LLM round-trip.

    The optional ``referent_to_props`` map widens the gate so a chunk
    also fires when an event's actors / targets / participants
    overlap a catalogue proposition's ``referent_ids`` even if the
    event has no explicit ``asserts/denies/resolves`` link yet (the
    Affect agent is exactly where that link should be drawn).
    Similarly, ``concern_entity_to_props`` widens the gate when an
    event's participants include any entity carrying a proposition-
    tied concern, since salience/polarity drift on those concerns
    is the common case the narrow gate misses.
    """
    if not catalogue_prop_ids and not seeded_entity_ids:
        return False
    referent_to_props = referent_to_props or {}
    concern_entity_to_props = concern_entity_to_props or {}
    for e in events:
        if e.asserts_proposition_id and e.asserts_proposition_id in catalogue_prop_ids:
            return True
        if e.denies_proposition_id and e.denies_proposition_id in catalogue_prop_ids:
            return True
        for pid in e.resolves_proposition_ids:
            if pid in catalogue_prop_ids:
                return True
        # Widened: referent overlap on actors / targets / participants.
        evt_refs: Set[str] = set()
        evt_refs.update(getattr(e, "actor_ids", []) or [])
        evt_refs.update(getattr(e, "target_ids", []) or [])
        evt_refs.update(getattr(e, "participant_ids", []) or [])
        for rid in evt_refs:
            if rid in referent_to_props:
                return True
            if rid in concern_entity_to_props:
                return True
    for eu in entity_updates:
        if eu.entity_id in seeded_entity_ids:
            return True
        if eu.entity_id in concern_entity_to_props:
            return True
        for b in eu.new_beliefs:
            if b.proposition_id and b.proposition_id in catalogue_prop_ids:
                return True
            # Widened: a belief whose target_id is a catalogued referent
            # is almost certainly tied to that proposition; let Affect see it.
            if b.target_id in referent_to_props:
                return True
        # Widened: a belief invalidation about a catalogued referent
        # is a textbook resolution / framing-shift event.
        for tid in getattr(eu, "invalidated_belief_targets", []) or []:
            if tid in referent_to_props:
                return True
    return False


def _build_affect_agent(
    config: ExtractionConfig,
) -> Agent[_AffectDeps, ChunkAffectExtraction]:
    """Construct the Step 3d Affect Agent — proposition + concern drift.

    Sees the catalogue (full ``Proposition`` and ``ConcernSeed`` lists),
    this chunk's merged events (Physics + Social) and entity_updates
    (Consequences). Emits diff snapshots only — no events, no edges,
    no beliefs. The output_validator drops snapshots whose
    ``triggered_by`` cannot be resolved to a chunk event, snapshots
    referencing unknown PROP_ / CCN_ ids, and truth commits whose
    fabula_time disagrees with the triggering event.
    """
    agent: Agent[_AffectDeps, ChunkAffectExtraction] = Agent(
        _resolve_model(config.model),
        deps_type=_AffectDeps,
        output_type=NativeOutput(ChunkAffectExtraction),
        system_prompt=_load_prompt("affect_extraction.md"),
        retries=config.output_retries,
    )

    @agent.system_prompt
    def inject_register_for_affect(ctx: RunContext[_AffectDeps]) -> str:
        reg = ctx.deps.global_register
        entity_names = {eid: reg.entities[eid].name for eid in sorted(reg.entities)}

        # Catalogue propositions — render fully so the agent can decide
        # which to snapshot.
        if ctx.deps.propositions:
            prop_lines: List[str] = []
            for p in ctx.deps.propositions:
                prop_lines.append(
                    f"  - {p.proposition_id} [{p.kind}] referents={p.referent_ids} "
                    f"stakes={p.stakes:.2f} prior={p.audience_default_prior:.2f}: "
                    f"{p.description}"
                )
            propositions_block = "\n".join(prop_lines)
        else:
            propositions_block = "  (no propositions in catalogue)"

        # Concern seeds — keyed by (entity_id, proposition_id, polarity).
        if ctx.deps.concern_seeds:
            seed_lines: List[str] = []
            for s in ctx.deps.concern_seeds:
                seed_lines.append(
                    f"  - {s.concern_id} entity={s.entity_id} "
                    f"prop={s.proposition_id} polarity={s.polarity} "
                    f"kind={s.kind} salience={s.baseline_salience:.2f}"
                )
            seeds_block = "\n".join(seed_lines)
        else:
            seeds_block = "  (no concern seeds in catalogue)"

        # Chunk events the agent may anchor snapshots on.
        evt_lines: List[str] = []
        for e in ctx.deps.chunk_events:
            link_bits: List[str] = []
            if e.asserts_proposition_id:
                link_bits.append(f"asserts={e.asserts_proposition_id}")
            if e.denies_proposition_id:
                link_bits.append(f"denies={e.denies_proposition_id}")
            if e.resolves_proposition_ids:
                link_bits.append(f"resolves={e.resolves_proposition_ids}")
            link_str = (" " + ", ".join(link_bits)) if link_bits else ""
            evt_lines.append(
                f"  - {e.id} (fabula={e.fabula_time}, type={e.event_type}, "
                f"actors={e.actor_ids}, targets={e.target_ids}){link_str}: "
                f"{e.description}"
            )
        events_block = "\n".join(evt_lines) if evt_lines else "  (no events in this chunk)"

        # Entity updates with their PROP-tagged beliefs surfaced.
        eu_lines: List[str] = []
        for eu in ctx.deps.chunk_entity_updates:
            tagged_beliefs = [
                f"{b.target_id}->{b.proposition_id}"
                for b in eu.new_beliefs if b.proposition_id
            ]
            tagged_str = (
                f" tagged_beliefs={tagged_beliefs}" if tagged_beliefs else ""
            )
            eu_lines.append(
                f"  - {eu.entity_id} fabula={eu.fabula_time} "
                f"triggered_by={eu.triggered_by}{tagged_str}"
            )
        entity_updates_block = (
            "\n".join(eu_lines) if eu_lines else "  (no entity updates in this chunk)"
        )

        chunk_evt_ids = [e.id for e in ctx.deps.chunk_events]

        return (
            "=== VALID ID REGISTER (from Step 1) ===\n"
            f"ENTITIES: {entity_names}\n"
            f"THIS CHUNK'S EVENT IDs: {chunk_evt_ids}\n"
            "\n"
            "Every `triggered_by` you emit MUST be one of THIS CHUNK'S "
            "EVENT IDs. Snapshots that try to retro-attribute drift to a "
            "prior chunk's event are dropped during validation.\n"
            "\n"
            "=== PROPOSITION CATALOGUE (full) ===\n"
            f"{propositions_block}\n"
            "\n"
            "You MAY emit snapshots / truth commits ONLY against PROP_ ids "
            "in this list. Do NOT invent new PROP_ ids.\n"
            "\n"
            "=== CONCERN SEED CATALOGUE ===\n"
            f"{seeds_block}\n"
            "\n"
            "Concern snapshots target catalogue CCN_ ids. New CCN_ ids "
            "go through `new_concern_seeds` (sparingly).\n"
            "\n"
            "=== EVENTS IN THIS CHUNK (Physics + Social merged) ===\n"
            f"{events_block}\n"
            "\n"
            "=== ENTITY UPDATES IN THIS CHUNK (Consequences output) ===\n"
            f"{entity_updates_block}\n"
        )

    @agent.output_validator
    def validate_affect_ids(
        ctx: RunContext[_AffectDeps],
        result: ChunkAffectExtraction,
    ) -> ChunkAffectExtraction:
        """Drop affect entries that fail referential integrity gates.

        These are *the* hard contract for the affect agent (see
        ``affect_extraction.md`` rule list); enforcing them here saves
        the Phase C reconciler from a second filter pass.
        """
        chunk_evt_index: Dict[str, EventNode] = {
            e.id: e for e in ctx.deps.chunk_events
        }
        prop_ids = {p.proposition_id for p in ctx.deps.propositions}
        seed_ids = {s.concern_id for s in ctx.deps.concern_seeds}
        ent_ids = set(ctx.deps.global_register.entities)

        kept_prop_snaps: List[ChunkPropositionSnapshot] = []
        for s in result.proposition_snapshots:
            if s.proposition_id not in prop_ids:
                logger.warning(
                    "[Affect] Dropping proposition_snapshot %s — unknown PROP_ id.",
                    s.proposition_id,
                )
                continue
            evt = chunk_evt_index.get(s.triggered_by)
            if evt is None:
                logger.warning(
                    "[Affect] Dropping proposition_snapshot for %s — "
                    "triggered_by %r not in this chunk's events.",
                    s.proposition_id, s.triggered_by,
                )
                continue
            kept_prop_snaps.append(s)

        kept_truth: List[PropositionTruthCommit] = []
        for c in result.proposition_truth_commits:
            if c.proposition_id not in prop_ids:
                logger.warning(
                    "[Affect] Dropping truth_commit %s — unknown PROP_ id.",
                    c.proposition_id,
                )
                continue
            evt = chunk_evt_index.get(c.triggered_by)
            if evt is None:
                logger.warning(
                    "[Affect] Dropping truth_commit %s — triggered_by %r "
                    "not in this chunk's events.",
                    c.proposition_id, c.triggered_by,
                )
                continue
            if evt.fabula_time != c.fabula_time:
                logger.warning(
                    "[Affect] Realigning truth_commit %s fabula_time "
                    "%d → %d to match triggering event %s.",
                    c.proposition_id, c.fabula_time, evt.fabula_time, evt.id,
                )
                c = c.model_copy(update={"fabula_time": evt.fabula_time})
            kept_truth.append(c)

        kept_concern_snaps: List[ChunkConcernSnapshot] = []
        for s in result.concern_snapshots:
            if s.concern_id not in seed_ids:
                logger.warning(
                    "[Affect] Dropping concern_snapshot %s — unknown CCN_ id "
                    "(not in catalogue; new concerns must go through new_concern_seeds).",
                    s.concern_id,
                )
                continue
            if s.triggered_by not in chunk_evt_index:
                logger.warning(
                    "[Affect] Dropping concern_snapshot for %s — "
                    "triggered_by %r not in this chunk's events.",
                    s.concern_id, s.triggered_by,
                )
                continue
            kept_concern_snaps.append(s)

        ccn_re = re.compile(r"^CCN_[A-Z0-9_]+$")
        kept_new_seeds: List["ConcernSeed"] = []
        for ns in result.new_concern_seeds:
            if not ccn_re.match(ns.concern_id):
                logger.warning(
                    "[Affect] Dropping new_concern_seed %r — id does not "
                    "match ^CCN_[A-Z0-9_]+$.", ns.concern_id,
                )
                continue
            if ns.concern_id in seed_ids:
                logger.warning(
                    "[Affect] Dropping new_concern_seed %s — collides with "
                    "an existing catalogue concern id.", ns.concern_id,
                )
                continue
            if ns.entity_id not in ent_ids:
                logger.warning(
                    "[Affect] Dropping new_concern_seed %s — unknown entity %r.",
                    ns.concern_id, ns.entity_id,
                )
                continue
            if ns.proposition_id not in prop_ids:
                logger.warning(
                    "[Affect] Dropping new_concern_seed %s — unknown "
                    "proposition_id %r (must reference an existing catalogue "
                    "proposition; affect agent may not introduce new propositions).",
                    ns.concern_id, ns.proposition_id,
                )
                continue
            kept_new_seeds.append(ns)

        return ChunkAffectExtraction(
            proposition_snapshots=kept_prop_snaps,
            proposition_truth_commits=kept_truth,
            concern_snapshots=kept_concern_snaps,
            new_concern_seeds=kept_new_seeds,
        )

    return agent


# =====================================================================
# Per-chunk quality gates — soft retries that fire when the LLM
# under-extracts on a structural axis the prompts already require but
# the schema can't enforce. Each helper returns True iff a retry is
# warranted; the caller is responsible for actually running it.
# =====================================================================


def _physics_causal_density_low(physics: "PhysicsExtraction") -> bool:
    """True iff the chunk has events but suspiciously few causal edges.

    Rule of thumb: every non-trivial event should participate in at
    least one causal edge (rule #12 of the physics prompt). If a chunk
    yields ``N >= 2`` events but ``len(causal_topology) < N``, the
    extractor very likely missed mutation / chain_reaction /
    affordance_gate edges. We retry with an explicit nudge.
    """
    n_evt = len(physics.events)
    if n_evt < 2:
        return False
    return len(physics.causal_topology) < n_evt


def _physics_anonymous_events(physics: "PhysicsExtraction") -> List[str]:
    """Return the IDs of events that have *both* empty ``actor_ids`` and
    empty ``target_ids`` and are not utterances.

    Anonymous events are extraction failures: every meaningful narrative
    event should have at least one named participant on at least one
    side. Without one, the propagator can't anchor the event to the
    social/physical graph and the affective scorer treats it as a noop.
    The Christmas Carol log produced 13 anonymous events out of 30 by
    extracting whole-vision sequences ("Cratchits mourn Tim", "thieves
    steal Scrooge") with empty actor/target lists even though the
    chunk text named the actors explicitly.

    Utterances are excluded because they are extracted by the Social
    Agent and use ``speaker_id`` / ``addressee_ids`` instead of the
    physics ``actor_ids`` / ``target_ids`` axes.
    """
    bad: List[str] = []
    for e in physics.events:
        if e.event_type == "utterance":
            continue
        if not (e.actor_ids or []) and not (e.target_ids or []):
            bad.append(e.id)
    return bad


def _physics_missing_mutation_social(
    physics: "PhysicsExtraction",
    social: "SocialExtraction",
) -> List[str]:
    """Return a list of axes for which Social observed a non-zero value
    but Physics emitted no matching ``mutation_social`` edge.

    Per the per-axis coverage rule in ``physics_extraction.md``: every
    ``RelationshipEdge`` axis with ``observed=True`` and ``value != 0``
    commits Physics to at least one ``mutation_social`` edge for that
    axis. Without it the affective gauges (danger, conflict,
    power-dynamic) read flat. We collect violations as ``"axis"``
    strings (deduplicated) so the retry message can be specific.
    """
    if not social.social_topology or not physics.events:
        return []
    observed_axes: set[str] = set()
    for rel in social.social_topology:
        for axis_name, m in rel.metrics.items():
            if getattr(m, "observed", True) and float(m.value) != 0.0:
                observed_axes.add(axis_name)
    if not observed_axes:
        return []
    covered: set[str] = set()
    for ce in physics.causal_topology:
        if ce.causality_type == "mutation_social" and ce.trait_target:
            covered.add(str(ce.trait_target))
    missing = sorted(observed_axes - covered)
    return missing


def _physics_missing_mutation_social_per_dyad(
    physics: "PhysicsExtraction",
    social: "SocialExtraction",
) -> List[Tuple[str, str, str]]:
    """Return per-dyad-per-axis gaps in ``mutation_social`` coverage.

    Stronger than :func:`_physics_missing_mutation_social`: instead of
    only checking that *some* mutation_social edge exists per axis
    globally, this checks that **each individual dyad with an
    ``observed=True`` non-zero axis** has at least one mutation_social
    edge wired to *that exact (dyad, axis)* triple. Without this
    finer-grained check, a fixture can satisfy the global axis count
    while leaving individual relationship traces flat — the symptom
    we observed on the bundled Star Wars project, where six of nine
    dyads carried observed axes that no mutation_social edge ever
    touched. The downstream timeline reconstructor cannot move a
    metric without a mutation, so the affective gauges' sub-curves
    sit at the authored value across the entire fabula axis.

    Returns a list of ``(target_id, counterpart_id, axis)`` triples
    in deterministic order, one per missing dyad-axis combination.
    """
    if not social.social_topology or not physics.events:
        return []

    # Build the set of (target, counterpart, axis) triples Physics
    # actually covered. ``mutation_social`` semantics: target_id is
    # the perspective entity (whose view of the relationship mutates),
    # rel_counterpart_id is the other half of the dyad.
    #
    # ``RelationshipEdge`` is *directed* — affinity, fear and
    # power_dynamic on edge ``A→B`` describe how A feels/stands toward
    # B, and may differ from the reverse ``B→A`` edge. Indexing both
    # directions as covered (an earlier version of this function did
    # so on the assumption that affinity is undirected) silently
    # accepted single-direction mutation_social coverage as covering
    # both halves of the dyad — exactly the bug this finer-grained
    # check is meant to surface. We now key strictly on the directed
    # (target, counterpart) pair so a missing reverse-direction
    # mutation is still flagged when only the forward direction was
    # extracted.
    covered: set[Tuple[str, str, str]] = set()
    for ce in physics.causal_topology:
        if ce.causality_type != "mutation_social":
            continue
        if not ce.trait_target or not ce.target_id:
            continue
        counterpart = getattr(ce, "rel_counterpart_id", None) or ""
        axis = str(ce.trait_target).lower()
        if axis == "power":
            axis = "power_dynamic"
        covered.add((ce.target_id, counterpart, axis))

    missing: list[Tuple[str, str, str]] = []
    seen: set[Tuple[str, str, str]] = set()
    for rel in social.social_topology:
        src = rel.source_entity_id
        tgt = rel.target_entity_id
        for axis_name, m in rel.metrics.items():
            if not getattr(m, "observed", True):
                continue
            if float(m.value) == 0.0:
                continue
            key = (src, tgt, axis_name)
            if key in seen:
                continue
            if key in covered:
                continue
            seen.add(key)
            missing.append(key)
    missing.sort()
    return missing


def _merge_physics_retry(
    base: "PhysicsExtraction",
    retry: "PhysicsExtraction",
) -> "PhysicsExtraction":
    """Merge a retry physics output into the base, preserving the base's
    event/causal/spatial set and *adding* new mutation_social edges and
    new events from the retry.

    Why merge instead of replace: the previous behaviour overwrote the
    base output entirely, which silently
      (a) lost previously-covered mutation_social triples (a retry
          aimed at adding 2 missing dyads could erase 5 already-good
          ones), and
      (b) crushed flashback fabula_times — if the base correctly placed
          a memory event at fabula=-700 (negative time = before story
          start) and the retry re-flattened it to fabula=600 (because
          the retry prompt didn't re-emphasise the chunk's flashback
          structure), the merged world lost the chronology.

    Strategy:
      * Keep ALL of the base's events, causal edges, spatial edges, and
        entity_updates verbatim — including their fabula_times.
      * From the retry, add only:
          - events whose ``id`` is not already in the base
          - causal edges whose ``(source_id, target_id, causality_type,
            trait_target, rel_counterpart_id)`` key is not already
            present in the base — this is what allows newly-extracted
            mutation_social edges to land while preventing duplicates.
          - spatial edges whose ``(source_id, target_id)`` is new
          - entity_updates whose ``(entity_id, fabula_time)`` is new
    """
    base_event_ids = {e.id for e in base.events}
    new_events = list(base.events) + [
        e for e in retry.events if e.id not in base_event_ids
    ]

    def _ce_key(ce: "CausalEdge") -> tuple:
        return (
            ce.source_id, ce.target_id, ce.causality_type,
            getattr(ce, "trait_target", None) or "",
            getattr(ce, "rel_counterpart_id", None) or "",
        )

    base_causal_keys = {_ce_key(c) for c in base.causal_topology}
    new_causal = list(base.causal_topology) + [
        c for c in retry.causal_topology if _ce_key(c) not in base_causal_keys
    ]

    base_spatial_keys = {(s.source_id, s.target_id) for s in base.spatial_topology}
    new_spatial = list(base.spatial_topology) + [
        s for s in retry.spatial_topology
        if (s.source_id, s.target_id) not in base_spatial_keys
    ]

    base_eu_keys = {(eu.entity_id, eu.fabula_time) for eu in base.entity_updates}
    new_eus = list(base.entity_updates) + [
        eu for eu in retry.entity_updates
        if (eu.entity_id, eu.fabula_time) not in base_eu_keys
    ]

    return PhysicsExtraction(
        events=new_events,
        causal_topology=new_causal,
        spatial_topology=new_spatial,
        entity_updates=new_eus,
    )


def _dedupe_scene_events(
    physics: "PhysicsExtraction",
    fabula_window: int = 50,
) -> "PhysicsExtraction":
    """Collapse multiple events that describe the same on-page beat.

    Multi-pass extraction (first pass + axis_retry + dyad_retry on the
    same chunk) routinely produces 2\u20133 distinct EVT_ ids that all
    describe one scene moment \u2014 e.g. EVT_SCROOGE_REFUSES_DINNER /
    EVT_REFUSE_FRED_INVITATION / EVT_SCROOGE_REFUSES_FRED. Once
    retries merge instead of replacing, all of them survive and clog
    the causal graph.

    Strategy: group non-utterance events by
    ``(frozenset(actor_ids), frozenset(target_ids), event_type,
    fabula_time // fabula_window)`` and keep the *first* event in
    each group as the canonical id. All subsequent events in that
    group are dropped, and any causal edge / entity_update referring
    to a dropped id is rewritten to the canonical id (then
    re-deduplicated by key).

    Utterance events are left untouched \u2014 dialogue lines look
    similar but each one is a distinct speech act.
    """
    if not physics.events:
        return physics

    canonical_of: Dict[str, str] = {}
    seen_keys: Dict[tuple, str] = {}
    kept_events: List["EventNode"] = []
    for e in physics.events:
        if e.event_type == "utterance":
            kept_events.append(e)
            canonical_of[e.id] = e.id
            continue
        key = (
            frozenset(e.actor_ids or []),
            frozenset(e.target_ids or []),
            e.event_type,
            (e.fabula_time or 0) // fabula_window,
        )
        # Skip key-based dedup if both actor and target are empty \u2014
        # those are anonymous events the anon retry will handle.
        if not key[0] and not key[1]:
            kept_events.append(e)
            canonical_of[e.id] = e.id
            continue
        if key in seen_keys:
            canonical_of[e.id] = seen_keys[key]
        else:
            seen_keys[key] = e.id
            kept_events.append(e)
            canonical_of[e.id] = e.id

    dropped = {
        eid for eid, canon in canonical_of.items() if canon != eid
    }
    if not dropped:
        return physics

    logger.info(
        "[Scene-Dedup] Collapsing %d duplicate event(s) onto canonical "
        "ids: %s",
        len(dropped),
        {eid: canonical_of[eid] for eid in list(dropped)[:5]},
    )

    def _rewrite(eid: str) -> str:
        return canonical_of.get(eid, eid)

    rewritten_causal: List["CausalEdge"] = []
    seen_causal: set[tuple] = set()
    for ce in physics.causal_topology:
        new_ce = ce.model_copy(update={
            "source_id": _rewrite(ce.source_id),
            "target_id": _rewrite(ce.target_id),
        })
        # Drop self-loops created by collapsing.
        if new_ce.source_id == new_ce.target_id:
            continue
        key = (
            new_ce.source_id, new_ce.target_id, new_ce.causality_type,
            getattr(new_ce, "trait_target", None) or "",
            getattr(new_ce, "rel_counterpart_id", None) or "",
        )
        if key in seen_causal:
            continue
        seen_causal.add(key)
        rewritten_causal.append(new_ce)

    rewritten_eus: List["EntityUpdate"] = []
    seen_eu: set[tuple] = set()
    for eu in physics.entity_updates:
        new_trig = _rewrite(eu.triggered_by) if eu.triggered_by else eu.triggered_by
        new_eu = eu.model_copy(update={"triggered_by": new_trig})
        key = (new_eu.entity_id, new_eu.fabula_time)
        if key in seen_eu:
            continue
        seen_eu.add(key)
        rewritten_eus.append(new_eu)

    return PhysicsExtraction(
        events=kept_events,
        causal_topology=rewritten_causal,
        spatial_topology=physics.spatial_topology,
        entity_updates=rewritten_eus,
    )


def _merge_scaffold_retry(
    base: "SocraticScaffold",
    retry: "SocraticScaffold",
) -> "SocraticScaffold":
    """Merge a retry scaffold into the base, deduplicating QA pairs.

    Same rationale as ``_merge_physics_retry``: a retry that aimed to
    add a missing category (Why / How) shouldn't quietly drop the
    Who / What / Where / When pairs already produced. Dedup keys on
    (category, normalised question text).
    """
    seen = {(qa.category, qa.question.strip().lower()) for qa in base.qa_pairs}
    merged = list(base.qa_pairs)
    for qa in retry.qa_pairs:
        key = (qa.category, qa.question.strip().lower())
        if key not in seen:
            seen.add(key)
            merged.append(qa)
    return SocraticScaffold(qa_pairs=merged)


def _merge_social_retry(
    base: "SocialExtraction",
    retry: "SocialExtraction",
) -> "SocialExtraction":
    """Merge a retry social extraction into the base.

    Why merge: the same "lose previously-covered material on retry"
    pathology that bit the physics retries applies to every social
    sub-retry — info-recovery (zero channels/utterances), channel
    inference (under-extracted standing capabilities), and
    relationship recovery (zero edges across multi-entity events).
    A naive replace can drop legitimate edges from the first pass.

    Dedup rules:
      * channels — keyed by ``CHN_`` id; base wins on collision.
      * utterance_events — keyed by event ``id``; base wins.
      * social_topology — keyed by
        (source_entity_id, target_entity_id, frozenset(metrics keys));
        base wins. Directed pairs are kept independent so an
        asymmetric A→B / B→A pair survives.
    """
    merged_channels = dict(base.channels)
    for cid, ch in retry.channels.items():
        if cid not in merged_channels:
            merged_channels[cid] = ch

    base_utt_ids = {e.id for e in base.utterance_events}
    merged_utterances = list(base.utterance_events) + [
        e for e in retry.utterance_events if e.id not in base_utt_ids
    ]

    def _re_key(re: "RelationshipEdge") -> tuple:
        # Include the metrics-axis set so a retry that adds a new axis
        # for an already-seen dyad doesn't silently get dropped.
        try:
            metrics_keys = tuple(sorted(re.metrics.keys()))
        except AttributeError:
            metrics_keys = ()
        return (re.source_entity_id, re.target_entity_id, metrics_keys)

    base_re_keys = {_re_key(r) for r in base.social_topology}
    merged_topology = list(base.social_topology) + [
        r for r in retry.social_topology if _re_key(r) not in base_re_keys
    ]

    return SocialExtraction(
        channels=merged_channels,
        utterance_events=merged_utterances,
        social_topology=merged_topology,
    )


def _merge_anonymous_utterance_retry(
    base: "SocialExtraction",
    retry: "SocialExtraction",
) -> "SocialExtraction":
    """Variant of :func:`_merge_social_retry` for the anonymous-utterance
    retry. When the retry re-emits a base utterance with the SAME id
    and now-populated speaker_id / addressee_ids, the retry's version
    replaces the base one. Channels and social_topology follow the
    standard merge rules.
    """
    retry_by_id = {u.id: u for u in retry.utterance_events}
    new_utts: List["EventNode"] = []
    seen_ids: set[str] = set()
    for u in base.utterance_events:
        retry_u = retry_by_id.get(u.id)
        base_anon = not u.speaker_id or not (u.addressee_ids or [])
        if retry_u is not None and base_anon and (
            retry_u.speaker_id and (retry_u.addressee_ids or [])
        ):
            new_utts.append(
                u.model_copy(update={
                    "speaker_id": retry_u.speaker_id,
                    "addressee_ids": retry_u.addressee_ids,
                })
            )
        else:
            new_utts.append(u)
        seen_ids.add(u.id)
    for r in retry.utterance_events:
        if r.id not in seen_ids:
            new_utts.append(r)
            seen_ids.add(r.id)

    base_with_new = base.model_copy(update={"utterance_events": new_utts})
    return _merge_social_retry(base_with_new, retry)


def _merge_anonymous_retry(
    base: "PhysicsExtraction",
    retry: "PhysicsExtraction",
) -> "PhysicsExtraction":
    """Variant of :func:`_merge_physics_retry` for the anonymous-events
    retry. The retry's *purpose* is to replace base events that have
    empty actor_ids/target_ids with corrected versions, so for events
    whose ``id`` is shared, the *retry's* actor/target lists win iff
    the retry actually fills in at least one of the participant lists.
    Other base properties (causal/spatial/entity_updates) follow
    standard merge rules.
    """
    new_events: List["EventNode"] = []
    seen_ids: set[str] = set()
    retry_by_id = {r.id: r for r in retry.events}
    for e in base.events:
        retry_e = retry_by_id.get(e.id)
        base_anon = (
            e.event_type != "utterance"
            and not (e.actor_ids or [])
            and not (e.target_ids or [])
        )
        if retry_e is not None and base_anon and (
            (retry_e.actor_ids or []) or (retry_e.target_ids or [])
        ):
            new_events.append(
                e.model_copy(update={
                    "actor_ids": retry_e.actor_ids,
                    "target_ids": retry_e.target_ids,
                })
            )
        else:
            new_events.append(e)
        seen_ids.add(e.id)
    for r in retry.events:
        if r.id not in seen_ids:
            new_events.append(r)
            seen_ids.add(r.id)

    base_with_new_events = base.model_copy(update={"events": new_events})
    return _merge_physics_retry(base_with_new_events, retry)


def _merge_consequences_retry(
    base: "ConsequencesExtraction",
    retry: "ConsequencesExtraction",
) -> "ConsequencesExtraction":
    """Merge a retry consequences extraction into the base.

    Dedup keys on (entity_id, fabula_time) — at the same fabula_time
    only one EntityUpdate per entity is meaningful (downstream
    snapshot reduction collapses them anyway). Base wins on collision
    so a parity retry that re-emits an EntityUpdate already produced
    can't blow away the original (which may have richer
    new_beliefs / new_status / new_location_id fields).
    """
    seen = {(eu.entity_id, eu.fabula_time) for eu in base.entity_updates}
    merged = list(base.entity_updates)
    for eu in retry.entity_updates:
        key = (eu.entity_id, eu.fabula_time)
        if key not in seen:
            seen.add(key)
            merged.append(eu)
    return ConsequencesExtraction(entity_updates=merged)



def _consequences_mutation_parity_broken(
    physics: "PhysicsExtraction",
    consequences: "ConsequencesExtraction",
) -> List[str]:
    """Return entity ids that have an inbound mutation edge from Physics
    but no matching ``EntityUpdate`` in Consequences.

    Each ``mutation`` edge with an ``ENT_`` target should produce at
    least one ``EntityUpdate`` for that entity (the parity contract
    spelt out at the top of ``consequences_extraction.md``). When the
    parity is broken, the snapshot the UI shows for that entity will
    sit at the pre-story baseline and the trait shift the edge declared
    is never anchored on the timeline.
    """
    targets: set[str] = set()
    for ce in physics.causal_topology:
        if ce.causality_type == "mutation" and ce.target_id.startswith("ENT_"):
            targets.add(ce.target_id)
    if not targets:
        return []
    covered: set[str] = {eu.entity_id for eu in consequences.entity_updates}
    return sorted(targets - covered)


def _social_channel_underextracted(
    social: "SocialExtraction",
    *,
    min_repeated_dyad: int = 2,
) -> bool:
    """True when the chunk's utterance pattern strongly implies a
    standing channel that the LLM failed to extract.

    Heuristic: at least one ``(speaker_id, frozenset(addressee_ids))``
    pair appears in ``min_repeated_dyad`` or more utterance events
    AND none of those utterances carries a ``via_channel_id`` AND
    no Channel was emitted that already covers that pair.

    The cue is weakly-but-consistently informative: two letters from
    the same hand to the same recipient, two telephone calls between
    the same two people, two wireless broadcasts to the same
    audience — each is much more naturally modelled as a single
    persistent capability than as N independent unmediated
    speech-acts. Without a Channel, downstream cycle-detection,
    mediation tracking, and counterfactual surgery (severing a line
    of communication) all silently lose teeth.

    Returns True only when ALL three conditions hold simultaneously,
    so a chunk legitimately full of face-to-face conversation
    (different participants each time, or already-mediated
    correspondence) does not trigger a false-positive retry.
    """
    if not social.utterance_events:
        return False
    # Skip if every relevant utterance is already wired to some channel
    # (either this chunk's channels or a prior chunk's channel that
    # the validator preserved on ``via_channel_id``).
    pair_counts: Dict[Tuple[str, frozenset], int] = {}
    pair_unmediated: Dict[Tuple[str, frozenset], int] = {}
    for u in social.utterance_events:
        if not u.speaker_id or not u.addressee_ids:
            continue
        key = (u.speaker_id, frozenset(u.addressee_ids))
        pair_counts[key] = pair_counts.get(key, 0) + 1
        if not u.via_channel_id:
            pair_unmediated[key] = pair_unmediated.get(key, 0) + 1
    if not pair_counts:
        return False
    # A pair is already 'covered' if any channel in this chunk has a
    # superset of its participants — in that case we don't need to
    # retry just because via_channel_id wasn't filled in.
    covered_pairs: set[Tuple[str, frozenset]] = set()
    for ch in social.channels.values():
        ch_pids = set(ch.participant_ids)
        for key in pair_counts:
            speaker, addrs = key
            if {speaker} <= ch_pids and set(addrs) <= ch_pids:
                covered_pairs.add(key)
    for key, count in pair_counts.items():
        if (
            count >= min_repeated_dyad
            and pair_unmediated.get(key, 0) >= min_repeated_dyad
            and key not in covered_pairs
        ):
            return True
    return False


# --- Tier 1: per-chunk social/physics quality helpers (2026-05-06) ---

def _social_mirror_suspicious_dyads(
    social: "SocialExtraction",
    *,
    eps: float = 0.05,
) -> List[Tuple[str, str, List[str]]]:
    """Return per-chunk dyad pairs whose forward + reverse RelationshipEdges
    carry near-identical metric values across all shared axes.

    Mirrors the post-assembly :func:`_warn_suspicious_mirror_dyads`
    detection but runs at the social-agent stage so we can RETRY
    rather than just log. ``power_dynamic`` is signed so opposite
    signs are expected; flagged only when forward + reverse are
    *equal* (not opposite).

    Tolerance ``eps`` is looser than the assembly check (1e-6) because
    LLMs round to 1-2 decimal places, so two real readings that
    happen to converge within 0.05 are statistically indistinguishable
    from a lazy mirror.
    """
    by_pair: Dict[Tuple[str, str], "RelationshipEdge"] = {
        (e.source_entity_id, e.target_entity_id): e
        for e in social.social_topology
    }
    seen: set[frozenset[str]] = set()
    suspect: List[Tuple[str, str, List[str]]] = []
    for (src, tgt), fwd in by_pair.items():
        key = frozenset({src, tgt})
        if key in seen:
            continue
        rev = by_pair.get((tgt, src))
        if rev is None:
            continue
        seen.add(key)
        shared = set(fwd.metrics.keys()) & set(rev.metrics.keys())
        if not shared:
            continue
        mirrored: List[str] = []
        for axis in shared:
            f_val = float(fwd.metrics[axis].value)
            r_val = float(rev.metrics[axis].value)
            if abs(f_val - r_val) < eps and abs(f_val) > eps:
                mirrored.append(axis)
        if mirrored and len(mirrored) == len(shared):
            suspect.append((src, tgt, mirrored))
    return suspect


def _social_anonymous_utterances(social: "SocialExtraction") -> List[str]:
    """Return EVT_ ids of utterance events missing speaker_id OR addressee_ids.

    A speech-act with no named speaker or no named addressees can't be
    routed onto a Channel and breaks the social cycle detector.
    """
    bad: List[str] = []
    for u in social.utterance_events:
        if not u.speaker_id or not (u.addressee_ids or []):
            bad.append(u.id)
    return bad


def _utterance_parity_orphans(
    physics: "PhysicsExtraction",
    social: "SocialExtraction",
) -> Tuple[List[str], List[str]]:
    """Return (social_only_ids, physics_only_ids) for utterance events.

    The Social Agent owns ``utterance_events``; Physics also collects
    its own utterance EventNodes from raw event extraction. After
    Step 3b they should agree by id. Orphans in either direction
    indicate one side missed the speech-act.
    """
    physics_utt_ids = {
        e.id for e in physics.events if e.event_type == "utterance"
    }
    social_utt_ids = {u.id for u in social.utterance_events}
    social_only = sorted(social_utt_ids - physics_utt_ids)
    physics_only = sorted(physics_utt_ids - social_utt_ids)
    return social_only, physics_only


def _social_mutation_coverage(
    physics: "PhysicsExtraction",
    social: "SocialExtraction",
) -> Tuple[int, int]:
    """Return (covered, total) per-(dyad, axis) mutation_social coverage.

    For every axis present on every RelationshipEdge in
    ``social.social_topology``, count whether at least one
    mutation_social CausalEdge in ``physics.causal_topology`` targets
    that (entity, counterpart, axis) triple. Returns a fraction so
    the caller can threshold (e.g. < 0.6 → retry).
    """
    triples: set[Tuple[str, str, str]] = set()
    for re in social.social_topology:
        for axis in re.metrics.keys():
            triples.add((re.source_entity_id, re.target_entity_id, axis))
    if not triples:
        return (0, 0)
    covered: set[Tuple[str, str, str]] = set()
    for ce in physics.causal_topology:
        if ce.causality_type != "mutation_social":
            continue
        cp = getattr(ce, "rel_counterpart_id", None) or ""
        ax = getattr(ce, "trait_target", None) or ""
        if not cp or not ax:
            continue
        key = (ce.target_id, cp, ax)
        if key in triples:
            covered.add(key)
    return (len(covered), len(triples))


# --- Tier 2: prompt/process improvements (2026-05-06) ---

def _on_page_entity_ids(
    chunk: str,
    register: "GlobalRegister",
) -> List[str]:
    """Return ENT_ ids whose ``name`` (or any whitespace-separated
    sub-token of length \u2265 4) appears as a substring of ``chunk``.

    Pure substring match \u2014 over-permissive on purpose so we don't
    miss canonical references the LLM might use ("Scrooge" matches
    "old Scrooge", "Mr. Scrooge", etc.). The result is advisory only:
    the system prompt still surfaces the full register so the LLM
    can reference offstage entities when narratively appropriate
    (memories, prophecies, gossip).
    """
    text = chunk.lower()
    found: List[str] = []
    for eid, ent in register.entities.items():
        name = (ent.name or "").strip()
        if not name:
            continue
        # Try the full name first.
        if name.lower() in text:
            found.append(eid)
            continue
        # Then surname / first name tokens of length >= 4 (avoids
        # false positives on short tokens like "of", "the", "a").
        for tok in name.split():
            tok_clean = tok.strip(".,;:!?\"'()[]").lower()
            if len(tok_clean) >= 4 and tok_clean in text:
                found.append(eid)
                break
    return sorted(set(found))


def _scaffold_mentioned_entities(
    scaffold: "SocraticScaffold",
    register: "GlobalRegister",
) -> Set[str]:
    """ENT_ ids referenced (by name) in any QA pair's question or answer."""
    text_blob = " ".join(
        f"{qa.question} {qa.answer}" for qa in scaffold.qa_pairs
    ).lower()
    found: Set[str] = set()
    for eid, ent in register.entities.items():
        name = (ent.name or "").strip()
        if name and name.lower() in text_blob:
            found.add(eid)
            continue
        for tok in (name or "").split():
            tok_clean = tok.strip(".,;:!?\"'()[]").lower()
            if len(tok_clean) >= 4 and tok_clean in text_blob:
                found.add(eid)
                break
    return found


def _physics_event_entities(physics: "PhysicsExtraction") -> Set[str]:
    """ENT_ ids referenced as actor or target in any physics event."""
    found: Set[str] = set()
    for e in physics.events:
        for x in (e.actor_ids or []) + (e.target_ids or []):
            if x.startswith("ENT_"):
                found.add(x)
    return found


def _social_participating_entities(social: "SocialExtraction") -> Set[str]:
    """ENT_ ids referenced anywhere in social output (channels,
    utterance events, relationship edges).

    Used by the optional second-drift pass: a scaffold-flagged entity
    that didn't appear in any *physics* event may still be covered by
    a Channel speaker, an utterance, or a relationship edge that
    Social produced. Counting that coverage avoids a wasted physics
    retry for entities the chunk *did* legitimately address through
    the social topology.
    """
    found: Set[str] = set()
    for ch in getattr(social, "channels", []) or []:
        sp = getattr(ch, "speaker_id", None)
        if sp and sp.startswith("ENT_"):
            found.add(sp)
        for pid in getattr(ch, "participant_ids", []) or []:
            if pid.startswith("ENT_"):
                found.add(pid)
    for ue in getattr(social, "utterance_events", []) or []:
        for x in (getattr(ue, "actor_ids", None) or []) + (
            getattr(ue, "target_ids", None) or []
        ):
            if isinstance(x, str) and x.startswith("ENT_"):
                found.add(x)
    for re_ in getattr(social, "social_topology", []) or []:
        for attr in ("source_entity_id", "target_entity_id"):
            v = getattr(re_, attr, None)
            if isinstance(v, str) and v.startswith("ENT_"):
                found.add(v)
    return found


def _scaffold_drift_ratio(
    scaffold: "SocraticScaffold",
    physics: "PhysicsExtraction",
    register: "GlobalRegister",
) -> Tuple[float, Set[str]]:
    """Return (coverage_ratio, missed_ent_ids).

    ``coverage_ratio`` = |scaffold_entities \u2229 physics_event_entities|
    \u00f7 |scaffold_entities|. ``missed_ent_ids`` is the set of
    scaffold-mentioned entities Physics produced no event for. A low
    ratio indicates Physics ignored the scaffold's analysis.
    """
    scaffold_ents = _scaffold_mentioned_entities(scaffold, register)
    if not scaffold_ents:
        return (1.0, set())
    physics_ents = _physics_event_entities(physics)
    overlap = scaffold_ents & physics_ents
    missed = scaffold_ents - physics_ents
    return (len(overlap) / len(scaffold_ents), missed)


def _physics_fabula_monotonicity_violations(
    physics: "PhysicsExtraction",
) -> List[Tuple[str, str]]:
    """Return pairs of EVT_ ids inside one chunk where syuzhet order
    contradicts fabula order for events sharing the same primary
    actor and a non-negative fabula_time (i.e. excluding flashbacks
    which legitimately go backward).

    Heuristic: group by ``frozenset(actor_ids)`` and walk events in
    syuzhet order; flag any pair where ``fabula_time[i+1] <
    fabula_time[i]`` and BOTH fabula_times are >= 0. Negative-time
    events are flashbacks and are expected to be retrograde relative
    to the present, so we exclude them from this check.
    """
    by_actors: Dict[frozenset, List["EventNode"]] = {}
    for e in physics.events:
        actors = frozenset(e.actor_ids or [])
        if not actors:
            continue
        by_actors.setdefault(actors, []).append(e)
    violations: List[Tuple[str, str]] = []
    for actors, evs in by_actors.items():
        evs.sort(key=lambda x: x.syuzhet_index)
        for prev, curr in zip(evs, evs[1:]):
            if (
                (prev.fabula_time or 0) >= 0
                and (curr.fabula_time or 0) >= 0
                and curr.fabula_time < prev.fabula_time
            ):
                violations.append((prev.id, curr.id))
    return violations


# --- Tier 3 #11: chunk-level checkpointing helpers ---
#
# Audit fixes #4 + #5: the on-disk envelope now records
#   1. a config/prompt fingerprint so a checkpoint produced under one
#      model / prompt revision / extraction config is never silently
#      reused by another, and
#   2. the per-chunk stage_flags so a chunk that previously failed
#      (physics / social / consequences) is *not* served back as
#      "clean" on resume — the chunk is re-extracted instead.
# Older flat-format checkpoints (pre-fix) are detected and ignored.
_CHECKPOINT_VERSION = 2


def _extraction_fingerprint(config: "ExtractionConfig") -> str:
    """Hash the extraction-config fields that materially affect output.

    Includes the model id, prompt-related flags, chunking parameters,
    and the SHA of every prompt file in ``_PROMPTS_DIR`` so any prompt
    edit invalidates prior checkpoints. Cheap to recompute (handful of
    file reads) and stable across runs.
    """
    parts: List[str] = []
    fields = [
        "model", "chunk_strategy", "min_chunk_chars",
        "chunk_overlap_chars", "fabula_time_spacing",
        "estimated_events_per_chunk", "enable_consequences_agent",
    ]
    for f in fields:
        parts.append(f"{f}={getattr(config, f, None)!r}")
    # Optional fields added by later fixes — tolerate absence on older
    # ExtractionConfig instances.
    for f in ("max_chunk_chars", "enable_chunk_carry_over",
              "scaffold_drift_retry", "fabula_monotonicity_retry"):
        if hasattr(config, f):
            parts.append(f"{f}={getattr(config, f)!r}")
    try:
        prompt_files = sorted(_PROMPTS_DIR.glob("*.md"))
        if not prompt_files:
            # Empty dir: still record an explicit sentinel so the
            # fingerprint is intentional rather than silently identical
            # to "prompts dir missing".
            parts.append("prompt:_empty_dir")
        for pf in prompt_files:
            try:
                h = hashlib.sha256(pf.read_bytes()).hexdigest()[:8]
            except Exception:
                h = "??"
            parts.append(f"prompt:{pf.name}={h}")
    except Exception:
        # Prompts dir missing in some test/install layouts — fall back
        # to a placeholder so the fingerprint still varies across other
        # config changes.
        parts.append("prompt:_unavailable")
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()[:16]


def _chunk_checkpoint_path(checkpoint_dir: str, chunk_text: str, idx: int) -> Path:
    h = hashlib.sha256(chunk_text.encode("utf-8", errors="replace")).hexdigest()[:12]
    return Path(checkpoint_dir) / f"chunk_{idx:04d}_{h}.json"


def _load_chunk_checkpoint(
    checkpoint_dir: Optional[str], chunk_text: str, idx: int,
    fingerprint: str,
) -> Optional[Tuple["ChunkTopology", Dict[str, int]]]:
    """Load a chunk checkpoint if it matches the current fingerprint and
    its stage_flags are all clean. Returns ``None`` to force re-extraction
    when either guard fails. The topology + stage_flags tuple lets the
    orchestrator preserve the >50%-failure-threshold accounting on resume.
    """
    if not checkpoint_dir:
        return None
    p = _chunk_checkpoint_path(checkpoint_dir, chunk_text, idx)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("[Checkpoint] Could not parse %s — ignoring.", p)
        return None
    # Reject pre-envelope checkpoints (no version field) outright.
    if not isinstance(data, dict) or data.get("version") != _CHECKPOINT_VERSION:
        logger.info(
            "[Checkpoint] %s is from an older format (no v%d envelope) "
            "— ignoring and re-extracting.",
            p, _CHECKPOINT_VERSION,
        )
        return None
    if data.get("fingerprint") != fingerprint:
        logger.info(
            "[Checkpoint] %s fingerprint mismatch "
            "(disk=%s, current=%s) — ignoring and re-extracting.",
            p, data.get("fingerprint"), fingerprint,
        )
        return None
    stage_flags = data.get("stage_flags") or {}
    if any(int(v or 0) for v in stage_flags.values()):
        logger.info(
            "[Checkpoint] %s recorded stage failures %s — ignoring "
            "and re-extracting to give the chunk a clean attempt.",
            p, {k: v for k, v in stage_flags.items() if v},
        )
        return None
    try:
        topo = ChunkTopology.model_validate(data["topology"])
    except Exception:
        logger.warning(
            "[Checkpoint] %s topology body invalid — ignoring.", p,
        )
        return None
    return topo, {
        "physics": int(stage_flags.get("physics", 0)),
        "social": int(stage_flags.get("social", 0)),
        "consequences": int(stage_flags.get("consequences", 0)),
    }


def _save_chunk_checkpoint(
    checkpoint_dir: Optional[str],
    chunk_text: str,
    idx: int,
    topo: "ChunkTopology",
    fingerprint: str,
    stage_flags: Dict[str, int],
) -> None:
    """Persist a checkpoint envelope. Chunks with any failed stage are
    intentionally NOT written so a resume cannot mistake a degraded
    extraction for a clean one — see audit fix #5."""
    if not checkpoint_dir:
        return
    if any(int(v or 0) for v in stage_flags.values()):
        logger.debug(
            "[Checkpoint] Skipping save for chunk %d (stage failures: %s).",
            idx, {k: v for k, v in stage_flags.items() if v},
        )
        return
    try:
        p = _chunk_checkpoint_path(checkpoint_dir, chunk_text, idx)
        p.parent.mkdir(parents=True, exist_ok=True)
        envelope = {
            "version": _CHECKPOINT_VERSION,
            "fingerprint": fingerprint,
            "stage_flags": dict(stage_flags),
            "topology": json.loads(topo.model_dump_json()),
        }
        p.write_text(
            json.dumps(envelope, indent=2),
            encoding="utf-8",
        )
        logger.debug("[Checkpoint] Wrote %s", p)
    except Exception:
        logger.exception(
            "[Checkpoint] Could not write checkpoint for chunk %d.", idx,
        )



def _pipeline_checkpoint_path(
    checkpoint_dir: str, kind: str, text: str, fingerprint: str,
) -> Path:
    """Path for a high-level (register / topologies) checkpoint.

    Keyed on ``sha256(text)[:16] + fingerprint`` so any change to the
    source narrative OR to the extraction config invalidates it.
    """
    text_hash = hashlib.sha256(
        text.encode("utf-8", errors="replace"),
    ).hexdigest()[:16]
    return (
        Path(checkpoint_dir)
        / f"{kind}_{text_hash}_{fingerprint}.json"
    )


def _load_register_checkpoint(
    checkpoint_dir: Optional[str], text: str, fingerprint: str,
) -> Optional["GlobalRegister"]:
    """Audit fix #3 (final pass): high-level register checkpoint.

    Returns the previously-saved ``GlobalRegister`` for this exact
    (text, config) pair, or ``None`` if no envelope exists, the
    envelope is from a different version, or the body fails to
    validate. The caller falls back to running Step 1 normally on a
    miss.
    """
    if not checkpoint_dir:
        return None
    p = _pipeline_checkpoint_path(checkpoint_dir, "register", text, fingerprint)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("[Checkpoint] Could not parse %s — ignoring.", p)
        return None
    if (
        not isinstance(data, dict)
        or data.get("version") != _CHECKPOINT_VERSION
        or data.get("fingerprint") != fingerprint
    ):
        logger.info(
            "[Checkpoint] %s envelope mismatch — ignoring and "
            "re-extracting Step 1.", p,
        )
        return None
    try:
        reg = GlobalRegister.model_validate(data["register"])
    except Exception:
        logger.warning(
            "[Checkpoint] %s register body invalid — ignoring.", p,
        )
        return None
    logger.info("[Checkpoint] Loaded register from %s.", p)
    return reg


def _save_register_checkpoint(
    checkpoint_dir: Optional[str], text: str, fingerprint: str,
    register: "GlobalRegister",
) -> None:
    if not checkpoint_dir:
        return
    try:
        p = _pipeline_checkpoint_path(
            checkpoint_dir, "register", text, fingerprint,
        )
        p.parent.mkdir(parents=True, exist_ok=True)
        envelope = {
            "version": _CHECKPOINT_VERSION,
            "fingerprint": fingerprint,
            "register": json.loads(register.model_dump_json()),
        }
        p.write_text(json.dumps(envelope, indent=2), encoding="utf-8")
        logger.debug("[Checkpoint] Wrote register checkpoint %s", p)
    except Exception:
        logger.exception(
            "[Checkpoint] Could not write register checkpoint.",
        )


def _load_topologies_checkpoint(
    checkpoint_dir: Optional[str], text: str, fingerprint: str,
) -> Optional[Tuple[List["ChunkTopology"], Optional["PropositionCatalogue"]]]:
    """High-level topology-list checkpoint.

    Returns ``(topologies, catalogue)`` for this exact (text, config)
    pair, or ``None`` if missing / mismatched / invalid. ``catalogue``
    is ``None`` for legacy checkpoints written before Phase A3.
    Skips the entire Step 2 (chunking + per-chunk extraction +
    reconciliation) on a hit.
    """
    if not checkpoint_dir:
        return None
    p = _pipeline_checkpoint_path(
        checkpoint_dir, "topologies", text, fingerprint,
    )
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("[Checkpoint] Could not parse %s — ignoring.", p)
        return None
    if (
        not isinstance(data, dict)
        or data.get("version") != _CHECKPOINT_VERSION
        or data.get("fingerprint") != fingerprint
    ):
        logger.info(
            "[Checkpoint] %s envelope mismatch — ignoring and "
            "re-extracting Step 2.", p,
        )
        return None
    body = data.get("topologies")
    if not isinstance(body, list):
        return None
    try:
        topos = [ChunkTopology.model_validate(t) for t in body]
    except Exception:
        logger.warning(
            "[Checkpoint] %s topology list invalid — ignoring.", p,
        )
        return None
    catalogue: Optional[PropositionCatalogue] = None
    cat_body = data.get("catalogue")
    if cat_body is not None:
        try:
            catalogue = PropositionCatalogue.model_validate(cat_body)
        except Exception:
            logger.warning(
                "[Checkpoint] %s catalogue body invalid — dropping.", p,
            )
    logger.info(
        "[Checkpoint] Loaded %d chunk topologies from %s%s.",
        len(topos), p,
        f" (+ catalogue: {len(catalogue.propositions)} props, "
        f"{len(catalogue.concern_seeds)} seeds)" if catalogue else "",
    )
    return topos, catalogue


def _save_topologies_checkpoint(
    checkpoint_dir: Optional[str], text: str, fingerprint: str,
    topologies: List["ChunkTopology"],
    catalogue: Optional["PropositionCatalogue"] = None,
) -> None:
    if not checkpoint_dir:
        return
    try:
        p = _pipeline_checkpoint_path(
            checkpoint_dir, "topologies", text, fingerprint,
        )
        p.parent.mkdir(parents=True, exist_ok=True)
        envelope: Dict[str, Any] = {
            "version": _CHECKPOINT_VERSION,
            "fingerprint": fingerprint,
            "topologies": [
                json.loads(t.model_dump_json()) for t in topologies
            ],
        }
        if catalogue is not None:
            envelope["catalogue"] = json.loads(catalogue.model_dump_json())
        p.write_text(json.dumps(envelope, indent=2), encoding="utf-8")
        logger.debug(
            "[Checkpoint] Wrote topologies checkpoint %s (%d chunks%s).",
            p, len(topologies),
            " + catalogue" if catalogue is not None else "",
        )
    except Exception:
        logger.exception(
            "[Checkpoint] Could not write topologies checkpoint.",
        )


def _check_chunk_failure_threshold(
    failure_counts: Dict[str, int],
    total_chunks: int,
    sample_errors: Optional[Dict[str, str]] = None,
) -> None:
    """Escalate to RuntimeError when any sub-stage failed on >50% of chunks.

    A handful of failed chunks is acceptable noise (one bad LLM round-trip,
    a transient connection drop), but if more than half the chunks failed
    a given sub-stage the resulting graph is unreliable and we'd rather
    raise loudly than silently persist a half-extracted world.

    ``sample_errors`` (audit fix #2, second pass): an optional mapping
    ``{stage: "<short repr of first captured exception>"}`` that the
    caller threads through so the raised ``RuntimeError`` can surface
    the underlying cause (model not found, auth failure, timeout, …)
    instead of just the aggregate counts. Without this the user sees
    only "extraction failed sub-stages: physics=1/1" and has to dig
    through logs to find why.
    """
    sample_errors = sample_errors or {}

    def _format_with_cause(details: str) -> str:
        if not sample_errors:
            return details
        cause_lines = [
            f"  - {stage}: {sample_errors[stage]}"
            for stage in sample_errors
            if stage in {s.split("=")[0] for s in details.split(", ")}
        ]
        if not cause_lines:
            return details
        return details + "\nFirst captured per-stage error:\n" + "\n".join(cause_lines)

    if total_chunks <= 0:
        return
    if total_chunks == 1:
        # Single-chunk runs cannot have ">half" failures by ratio, but
        # if the only chunk failed any sub-stage the resulting topology
        # is silently empty. Escalate any failure on the only chunk.
        breached = {s: c for s, c in failure_counts.items() if c >= 1}
        if breached:
            details = ", ".join(f"{s}={c}/1" for s, c in breached.items())
            raise RuntimeError(
                f"Single-chunk extraction failed sub-stages: {details}. "
                f"Refusing to return an empty topology.\n"
                + _format_with_cause(details)
            )
        return
    threshold = total_chunks / 2
    breached = {
        stage: count for stage, count in failure_counts.items()
        if count > threshold
    }
    if breached:
        details = ", ".join(
            f"{stage}={count}/{total_chunks}" for stage, count in breached.items()
        )
        raise RuntimeError(
            f"Chunk extraction failure threshold breached (>50%): {details}. "
            f"Refusing to return a partial topology — inspect upstream "
            f"agent / model errors.\n"
            + _format_with_cause(details)
        )
    if any(failure_counts.values()):
        details = ", ".join(
            f"{stage}={count}" for stage, count in failure_counts.items() if count
        )
        logger.warning(
            "[Pipeline] Chunk sub-stage failures (under threshold): %s of %d chunks.",
            details, total_chunks,
        )


# =====================================================================
# Parallel Chunk Extraction (Async)
# =====================================================================


class _ChunkParams(BaseModel):
    """Pre-allocated parameters for a single chunk in parallel extraction.

    ``syuzhet_offset`` is deterministic from chunk position because syuzhet
    *is* narration order. ``fabula_time`` is intentionally NOT pre-allocated:
    chunks must be free to encode flashbacks, prologues, and other
    non-monotone story-world chronologies (see
    `docs/academic-foundations.md` §1.1).

    ``previous_event_ids`` and ``previous_chunk_channels`` carry forward
    output from the syuzhet-prior chunk(s). These are populated only in
    serial mode (``ExtractionConfig.enable_chunk_carry_over=True``) — the
    default parallel dispatch leaves them empty because chunks are
    extracted concurrently and have no causal ordering at dispatch time
    (audit fix #2).
    """
    model_config = {"arbitrary_types_allowed": True}
    chunk_index: int
    total_chunks: int
    syuzhet_offset: int
    prev_chunk_tail: str
    previous_event_ids: List[str] = Field(default_factory=list)
    previous_chunk_channels: Dict[str, Channel] = Field(default_factory=dict)
    # Compact ``(PROP_id, description)`` pairs from the global Phase A3
    # catalogue. Same shape consumed by ``_format_proposition_catalogue``
    # in the per-chunk system_prompt injectors. Empty when the
    # catalogue stage is skipped or returned no propositions.
    chunk_propositions: List[Tuple[str, str]] = Field(default_factory=list)
    # Phase B4: full Phase A3 catalogue (propositions + concern seeds)
    # threaded onto every chunk so the Affect agent can see baseline
    # framing / salience values without a second registry round-trip.
    # Optional so the field stays absent when the catalogue stage is
    # disabled or fails (in which case the affect agent is also skipped).
    chunk_catalogue: Optional["PropositionCatalogue"] = None


def _pre_allocate_chunk_params(
    chunks: List[str],
    config: ExtractionConfig,
) -> List[_ChunkParams]:
    """Pre-compute per-chunk extraction parameters for parallel dispatch.

    ``syuzhet_offset`` is allocated deterministically from chunk position
    because syuzhet position equals narration position. ``fabula_time``
    is intentionally NOT pre-allocated: forcing chunk order onto fabula
    order would make flashbacks structurally impossible. Chunks are
    expected to use absolute story-world chronology, and the global
    ``run_extraction_async`` pipeline relies on the validator to flag
    any temporal contradictions.
    """
    params: List[_ChunkParams] = []
    est = config.estimated_events_per_chunk
    for i, chunk in enumerate(chunks):
        tail = ""
        if i > 0 and config.chunk_overlap_chars > 0:
            tail = chunks[i - 1][-config.chunk_overlap_chars:]
        params.append(_ChunkParams(
            chunk_index=i,
            total_chunks=len(chunks),
            syuzhet_offset=i * est,
            prev_chunk_tail=tail,
        ))
    return params


async def _extract_single_chunk_async(
    chunk: str,
    params: _ChunkParams,
    register: GlobalRegister,
    config: ExtractionConfig,
    socratic_agent: Agent,
    physics_agent: Agent,
    social_agent: Agent,
    consequences_agent: Optional[Agent] = None,
    affect_agent: Optional[Agent] = None,
) -> Tuple[ChunkTopology, Dict[str, int], Dict[str, str]]:
    """Process one chunk through the per-chunk agent pipeline (async).

    Runs Socratic scaffolding → Physics → Social → Consequences for a
    single chunk. Social and Consequences are sequential because
    Consequences depends on Social's ``utterance_events`` and
    ``channels`` to wire ``Belief.acquired_via_event_id`` /
    ``acquired_via_channel_id`` provenance correctly.
    ``previous_event_ids`` is empty (advisory context only; the
    ``GlobalRegister`` provides structural ID validation).

    Returns ``(topology, failure_flags, stage_errors)`` where
    ``failure_flags`` is a ``{stage: 0|1}`` dict (used for the
    >50%-of-chunks escalation rule) and ``stage_errors`` is a
    ``{stage: short repr}`` dict carrying the first captured exception
    per stage (audit fix #2, second pass) so the orchestrator can
    surface the underlying cause when it raises.
    """
    i = params.chunk_index
    n = params.total_chunks
    failure_flags: Dict[str, int] = {"physics": 0, "social": 0, "consequences": 0}
    stage_errors: Dict[str, str] = {}

    fingerprint = _extraction_fingerprint(config)

    # Tier 3 #11 + audit fix #4/#5: try checkpoint before any agent calls.
    cached = _load_chunk_checkpoint(
        config.checkpoint_dir, chunk, i, fingerprint,
    )
    if cached is not None:
        cached_topo, cached_flags = cached
        logger.info(
            "[Checkpoint\u00b7Async] Chunk %d/%d loaded from disk \u2014 "
            "skipping extraction (%d events, %d causal, %d social).",
            i + 1, n,
            len(cached_topo.events),
            len(cached_topo.causal_topology),
            len(cached_topo.social_topology),
        )
        return cached_topo, cached_flags, stage_errors

    # Prepend trailing context from previous chunk for coreference
    overlap_ctx = ""
    if params.prev_chunk_tail and config.chunk_overlap_chars > 0:
        overlap_ctx = (
            f"[CONTEXT FROM PREVIOUS CHUNK — do NOT re-extract events from this]\n"
            f"{params.prev_chunk_tail}\n"
            f"[END CONTEXT]\n\n"
        )
    chunk_with_ctx = f"{overlap_ctx}{chunk}"

    # --- Step 2: Socratic QA Scaffolding ---
    logger.info("[Step 2·Async] Processing chunk %d/%d (%d chars) — scaffolding …", i + 1, n, len(chunk))
    socratic_msg = f"Chunk {i + 1} of {n}:\n\n{chunk_with_ctx}"
    socratic_deps = _SocraticDeps(global_register=register)
    try:
        scaffold_result = await socratic_agent.run(socratic_msg, deps=socratic_deps, **_user_kwargs())
        scaffold = scaffold_result.output
    except Exception:
        logger.exception("[Step 2·Async] Chunk %d scaffolding FAILED — using empty scaffold.", i + 1)
        scaffold = SocraticScaffold()

    logger.info("[Step 2·Async] Chunk %d: %d QA pairs generated.", i + 1, len(scaffold.qa_pairs))

    # Tier 2 #6: pre-compute on-page entity roster.
    on_page = _on_page_entity_ids(chunk, register)

    # --- Step 3a: Physics Agent ---
    logger.info("[Step 3a·Async] Processing chunk %d/%d — physics …", i + 1, n)
    physics_msg = (
        f"Chunk {i + 1} of {n} "
        f"(syuzhet_index offset: {params.syuzhet_offset}, "
        f"fabula_time_spacing: {config.fabula_time_spacing}).\n\n"
        f"Use ABSOLUTE story-world chronology for fabula_time. Earlier "
        f"story-time = smaller fabula_time, later story-time = larger. "
        f"The narration MAY jump in either direction: flashbacks / "
        f"prologues use SMALLER fabula_time than surrounding chunks; "
        f"flash-forwards / prophecies / glimpses of the future use "
        f"LARGER fabula_time. Chunk position in the syuzhet does NOT "
        f"determine fabula order.\n\n"
        f"{chunk_with_ctx}"
    )
    physics_deps = _PhysicsDeps(
        global_register=register,
        scaffold=scaffold,
        # Audit fix #2: previous_event_ids is populated only when the
        # caller dispatches chunks serially with carry-over enabled;
        # parallel mode leaves it empty because chunks have no causal
        # ordering at dispatch time.
        previous_event_ids=list(params.previous_event_ids),
        on_page_entity_ids=on_page,
        chunk_propositions=list(params.chunk_propositions),
    )
    try:
        physics_result = await physics_agent.run(physics_msg, deps=physics_deps, **_user_kwargs())
        physics = physics_result.output
    except Exception as exc:
        logger.exception("[Step 3a·Async] Chunk %d FAILED — returning empty physics.", i + 1)
        physics = PhysicsExtraction()
        failure_flags["physics"] = 1
        stage_errors.setdefault("physics", f"{type(exc).__name__}: {exc}")

    # Retry once if zero events from a substantive chunk
    if not physics.events and len(chunk) > 500:
        logger.info("[Step 3a·Async] Chunk %d: 0 events from %d chars — retrying …", i + 1, len(chunk))
        retry_msg = (
            "IMPORTANT: The previous extraction returned zero events. "
            "Re-read the chunk carefully — every narrative chunk contains "
            "at least one event (choice, outcome, or revelation). "
            "Look for decisions, consequences, emotional shifts, and "
            "information reveals.\n\n" + physics_msg
        )
        try:
            physics_result = await physics_agent.run(retry_msg, deps=physics_deps, **_user_kwargs())
            retry_physics = physics_result.output
            physics = _merge_physics_retry(physics, retry_physics)
        except Exception:
            logger.exception("[Step 3a·Async] Chunk %d retry FAILED.", i + 1)

    # Causal-density retry — see sync path for rationale.
    if _physics_causal_density_low(physics):
        n_evt = len(physics.events)
        n_causal = len(physics.causal_topology)
        logger.info(
            "[Step 3a·Async] Chunk %d: low causal density (%d edges across "
            "%d events) — retrying with emphasis …",
            i + 1, n_causal, n_evt,
        )
        density_msg = (
            "IMPORTANT: The previous extraction produced "
            f"{n_evt} events but only {n_causal} causal edges. "
            "Every event MUST participate in at least one causal "
            "edge — re-extract with explicit attention to:\n"
            "  - chain_reaction edges between consecutive events,\n"
            "  - mutation edges for every event that changes a "
            "character's traits / status / location,\n"
            "  - mutation_social edges for every event that "
            "shifts a relationship axis (affinity / fear / "
            "power_dynamic), with the rel_counterpart_id and "
            "trait_target both set,\n"
            "  - affordance_gate edges for state-prerequisites,\n"
            "  - ambient_propagation for background drift.\n"
            "Aim for AT LEAST one outgoing causal edge per event "
            "and emit ALL implied mutations.\n\n" + physics_msg
        )
        try:
            density_result = await physics_agent.run(
                density_msg, deps=physics_deps, **_user_kwargs(),
            )
            density_physics = density_result.output
            merged_density = _merge_physics_retry(physics, density_physics)
            if len(merged_density.causal_topology) > n_causal:
                physics = merged_density
                logger.info(
                    "[Step 3a·Async] Chunk %d: density retry recovered "
                    "%d→%d causal edges (merged).",
                    i + 1, n_causal, len(physics.causal_topology),
                )
        except Exception:
            logger.exception(
                "[Step 3a·Async] Chunk %d causal density retry FAILED.", i + 1,
            )

    # Anonymous-events retry (async). See sync path for rationale.
    anon_ids = _physics_anonymous_events(physics)
    if anon_ids:
        sample = anon_ids[:6]
        logger.info(
            "[Step 3a·Async] Chunk %d: %d anonymous event(s) (no actor + "
            "no target): %s\u2026 \u2014 retrying with participant emphasis \u2026",
            i + 1, len(anon_ids), sample,
        )
        anon_msg = (
            "IMPORTANT: The previous extraction produced "
            f"{len(anon_ids)} non-utterance event(s) with EMPTY "
            f"actor_ids AND EMPTY target_ids: {sample}. Every "
            "non-utterance event MUST name at least one ENT_ in "
            "actor_ids OR at least one ENT_/OBJ_ in target_ids. "
            "Re-emit those events with the on-page participants "
            "filled in, keeping the SAME id and fabula_time so the "
            "merge step replaces the anonymous version.\n\n" + physics_msg
        )
        try:
            anon_result = await physics_agent.run(
                anon_msg, deps=physics_deps, **_user_kwargs(),
            )
            anon_physics = anon_result.output
            merged_anon = _merge_anonymous_retry(physics, anon_physics)
            new_anon = _physics_anonymous_events(merged_anon)
            if len(new_anon) < len(anon_ids):
                physics = merged_anon
                logger.info(
                    "[Step 3a·Async] Chunk %d: anon retry covered "
                    "%d/%d anonymous events.",
                    i + 1,
                    len(anon_ids) - len(new_anon),
                    len(anon_ids),
                )
        except Exception:
            logger.exception(
                "[Step 3a·Async] Chunk %d anon retry FAILED.", i + 1,
            )

    # Collapse duplicate events from multi-pass extraction.
    physics = _dedupe_scene_events(physics)

    # Tier 2 #7: scaffold-drift diagnostic + retry (audit fix #8).
    drift_ratio, missed = _scaffold_drift_ratio(scaffold, physics, register)
    if drift_ratio < 0.5 and missed:
        logger.warning(
            "[Scaffold-Drift\u00b7Async] Chunk %d: physics produced events "
            "for only %.0f%% of scaffold-mentioned entities; missed %s",
            i + 1, 100.0 * drift_ratio, sorted(missed)[:8],
        )
        if config.scaffold_drift_retry:
            missed_list = sorted(missed)[:20]
            drift_msg = (
                "IMPORTANT: The Socratic scaffold for this chunk "
                "flagged the following on-page entities, but your "
                "previous extraction emitted no events involving "
                f"them: {missed_list}. Re-read the chunk and emit at "
                "least one event per missed entity (a choice, "
                "observation, utterance, or state-change), keeping "
                "ALL previously-extracted events and edges with their "
                "original ids and fabula_times UNCHANGED.\n\n"
                + physics_msg
            )
            try:
                drift_result = await physics_agent.run(
                    drift_msg, deps=physics_deps, **_user_kwargs(),
                )
                drift_physics = drift_result.output
                merged_drift = _merge_physics_retry(physics, drift_physics)
                new_ratio, new_missed = _scaffold_drift_ratio(
                    scaffold, merged_drift, register,
                )
                if len(new_missed) < len(missed):
                    physics = merged_drift
                    logger.info(
                        "[Scaffold-Drift\u00b7Async] Chunk %d: drift retry "
                        "covered %d/%d missed entities (merged).",
                        i + 1,
                        len(missed) - len(new_missed),
                        len(missed),
                    )
            except Exception:
                logger.exception(
                    "[Scaffold-Drift\u00b7Async] Chunk %d drift retry FAILED.",
                    i + 1,
                )

    # Tier 2 #9: fabula-time monotonicity check + retry (audit fix #8).
    violations = _physics_fabula_monotonicity_violations(physics)
    if violations:
        logger.warning(
            "[Fabula-Monotonicity\u00b7Async] Chunk %d: %d retrograde event "
            "pair(s) where syuzhet order contradicts fabula order "
            "(non-flashback): %s",
            i + 1, len(violations), violations[:5],
        )
        if config.fabula_monotonicity_retry:
            mono_msg = (
                "IMPORTANT: Your previous extraction has events where "
                "syuzhet order moves FORWARD but fabula_time moves "
                f"BACKWARD without being marked as flashbacks: "
                f"{violations[:10]}. For each pair, choose ONE: "
                "(a) re-emit the later event with a fabula_time >= "
                "the earlier one (correct ordering), or (b) re-emit "
                "the BACKWARDS event with a NEGATIVE fabula_time to "
                "mark it as a deliberate flashback. Keep all event "
                "ids unchanged and keep ALL other events and edges "
                "as-is.\n\n" + physics_msg
            )
            try:
                mono_result = await physics_agent.run(
                    mono_msg, deps=physics_deps, **_user_kwargs(),
                )
                mono_physics = mono_result.output
                # Replace events that were re-emitted with corrected
                # fabula_time (same id), keep everything else from base.
                # Audit fix #3 (second pass): also re-anchor any causal
                # edge whose own fabula_time mirrored an event we just
                # shifted, so chain-reaction timing constraints stay
                # consistent with the corrected event timeline.
                mono_by_id = {e.id: e for e in mono_physics.events}
                fixed_events = []
                shifted_event_times: Dict[str, int] = {}
                for e in physics.events:
                    cand = mono_by_id.get(e.id)
                    if cand is not None and cand.fabula_time != e.fabula_time:
                        fixed_events.append(cand)
                        shifted_event_times[e.id] = cand.fabula_time
                    else:
                        fixed_events.append(e)
                fixed_causal: List[CausalEdge] = []
                for ce in physics.causal_topology:
                    src_t = shifted_event_times.get(ce.source_id)
                    # Only re-anchor when the edge's fabula_time matched
                    # the event's old time exactly — otherwise it was
                    # set independently and we leave it alone.
                    if src_t is not None and ce.fabula_time == 0:
                        fixed_causal.append(ce)
                    elif src_t is not None:
                        fixed_causal.append(ce.model_copy(update={
                            "fabula_time": src_t,
                        }))
                    else:
                        fixed_causal.append(ce)
                physics = physics.model_copy(update={
                    "events": fixed_events,
                    "causal_topology": fixed_causal,
                })
                new_violations = _physics_fabula_monotonicity_violations(physics)
                if len(new_violations) < len(violations):
                    logger.info(
                        "[Fabula-Monotonicity\u00b7Async] Chunk %d: retry "
                        "resolved %d/%d violation(s); shifted %d event(s) "
                        "and re-anchored matching causal edges.",
                        i + 1,
                        len(violations) - len(new_violations),
                        len(violations),
                        len(shifted_event_times),
                    )
            except Exception:
                logger.exception(
                    "[Fabula-Monotonicity\u00b7Async] Chunk %d retry FAILED.",
                    i + 1,
                )

    logger.info(
        "[Step 3a·Async] Chunk %d: %d events, %d causal, %d spatial edges.",
        i + 1, len(physics.events), len(physics.causal_topology),
        len(physics.spatial_topology),
    )

    # Build event summary for Social Agent
    chunk_evt_ids = [e.id for e in physics.events]
    event_summary_lines = []
    for e in physics.events:
        event_summary_lines.append(
            f"  - {e.id} (fabula={e.fabula_time}, syuzhet={e.syuzhet_index}, "
            f"type={e.event_type}, actors={e.actor_ids}, targets={e.target_ids}): "
            f"{e.description}"
        )
    event_summary = "\n".join(event_summary_lines)

    # --- Step 3b: Social Agent (must complete before Consequences) ---
    # --- Step 3c: Consequences Agent (optional, runs after Social so it
    # can wire belief provenance through utterance / channel ids) ---
    social = SocialExtraction()
    entity_updates_final = physics.entity_updates  # legacy fallback

    # Hoisted to outer scope so the mirror / anonymous-utterance retries
    # below (which reuse social_msg + social_deps) can see them after
    # ``_run_social`` returns. Previously these were defined inside
    # ``_run_social`` and the retries hit a NameError when triggered.
    social_msg = (
        f"Chunk {i + 1} of {n}.\n\n"
        f"EVENTS EXTRACTED FROM THIS CHUNK:\n{event_summary}\n\n"
        f"ORIGINAL TEXT:\n{chunk_with_ctx}"
    )
    social_deps = _SocialDeps(
        global_register=register,
        scaffold=scaffold,
        chunk_event_ids=chunk_evt_ids,
        chunk_events=list(physics.events),
        chunk_causal=list(physics.causal_topology),
        previous_event_ids=list(params.previous_event_ids),
        previous_chunk_channels=dict(params.previous_chunk_channels),
        on_page_entity_ids=on_page,
        chunk_propositions=list(params.chunk_propositions),
    )

    async def _run_social() -> SocialExtraction:
        # The social pass runs even when physics yielded zero events,
        # provided the chunk shows linguistic evidence of dialogue or
        # written communication — see comment in the sync pipeline.
        if not physics.events and not _chunk_likely_contains_speech(chunk):
            logger.info(
                "[Step 3b·Async] Chunk %d: skipping social pass (no events, no speech cues).", i + 1,
            )
            return SocialExtraction()
        logger.info("[Step 3b·Async] Processing chunk %d/%d — social …", i + 1, n)
        try:
            social_result = await social_agent.run(social_msg, deps=social_deps, **_user_kwargs())
            local_social = social_result.output
        except Exception as exc:
            logger.exception(
                "[Step 3b·Async] Chunk %d FAILED — returning empty social.", i + 1,
            )
            failure_flags["social"] = 1
            stage_errors.setdefault("social", f"{type(exc).__name__}: {exc}")
            return SocialExtraction()

        # Retry if zero channels AND zero utterance events (quality gate)
        # AND the chunk shows linguistic evidence of dialogue. Pure-action
        # chunks (chases, silent set-pieces) legitimately produce neither.
        if (
            len(physics.events) >= 2
            and not local_social.channels
            and not local_social.utterance_events
            and _chunk_likely_contains_speech(chunk)
        ):
            logger.info(
                "[Step 3b·Async] Chunk %d: 0 channels and 0 utterance events — retrying with emphasis …",
                i + 1,
            )
            retry_social_msg = (
                "IMPORTANT: The previous extraction returned zero Channel "
                "entries AND zero utterance events. Most narrative chunks "
                "contain conversations, prophecies, letters, confessions, "
                "orders, announcements, or rumours — each one MUST produce "
                "either an EventNode(event_type='utterance') or a Channel. "
                "Re-read the text and extract ALL information flows.\n\n"
                + social_msg
            )
            try:
                retry_result = await social_agent.run(retry_social_msg, deps=social_deps, **_user_kwargs())
                retry_social = retry_result.output
                if retry_social.channels or retry_social.utterance_events:
                    local_social = _merge_social_retry(local_social, retry_social)
                    logger.info(
                        "[Step 3b·Async] Chunk %d: retry recovered %d channels, %d utterances (merged).",
                        i + 1,
                        len(local_social.channels),
                        len(local_social.utterance_events),
                    )
            except Exception:
                logger.exception("[Step 3b·Async] Chunk %d info retry FAILED.", i + 1)

        # Channel under-extraction quality gate (async port of the
        # sync pipeline's check). Triggers when ≥2 utterances share a
        # speaker→addressee dyad with no `via_channel_id` and no
        # Channel covers the pair.
        if _social_channel_underextracted(local_social):
            logger.info(
                "[Step 3b·Async] Chunk %d: utterances cluster on a "
                "speaker→addressee dyad with no Channel — retrying "
                "with channel-inference emphasis …",
                i + 1,
            )
            chn_retry_msg = (
                "IMPORTANT: Your previous extraction emitted "
                "MULTIPLE utterance events between the SAME speaker "
                "and addressee(s) but no Channel that covers them, "
                "and none of those utterances had a `via_channel_id`. "
                "Repeated communication between the same parties is "
                "almost always carried by a STANDING capability "
                "(letter correspondence, telephone line, telepathic "
                "bond, courier route, broadcast frequency). Re-emit "
                "the extraction with: (a) at least one Channel for "
                "each repeated dyad whose medium the text supports, "
                "and (b) `via_channel_id` set on every utterance "
                "that rides over one of those channels. Keep all "
                "previously-extracted utterances and relationship "
                "edges.\n\n" + social_msg
            )
            try:
                chn_retry_result = await social_agent.run(
                    chn_retry_msg, deps=social_deps, **_user_kwargs(),
                )
                chn_retry = chn_retry_result.output
                merged_chn = _merge_social_retry(local_social, chn_retry)
                if (
                    len(merged_chn.channels) > len(local_social.channels)
                    and not _social_channel_underextracted(merged_chn)
                ):
                    local_social = merged_chn
                    logger.info(
                        "[Step 3b·Async] Chunk %d: channel-inference "
                        "retry recovered %d channels (merged).",
                        i + 1, len(local_social.channels),
                    )
            except Exception:
                logger.exception(
                    "[Step 3b·Async] Chunk %d channel-inference retry FAILED.",
                    i + 1,
                )

        # Symmetric retry on empty social_topology when the chunk's
        # events involve multiple distinct entities — ports the sync
        # extract_topology_async behaviour so async runs don't quietly drop
        # social-edge recall (audit item #6).
        multi_entity_events = [
            e for e in physics.events
            if len(set(e.actor_ids) | set(e.target_ids)) >= 2
        ]
        if multi_entity_events and not local_social.social_topology:
            logger.info(
                "[Step 3b·Async] Chunk %d: 0 social edges across %d "
                "multi-entity event(s) — retrying with emphasis …",
                i + 1, len(multi_entity_events),
            )
            retry_rel_msg = (
                "IMPORTANT: The previous extraction returned zero "
                "RelationshipEdge entries despite the chunk containing "
                "events with multiple distinct participants. For each "
                "such event, infer the *minimum* relationship axes the "
                "text supports — even one observed axis per dyad is "
                "valuable. Use ``observed=True`` for axes the text "
                "speaks to, and omit unobserved axes entirely (do not "
                "fabricate neutral zeros).\n\n" + social_msg
            )
            try:
                rel_retry_result = await social_agent.run(
                    retry_rel_msg, deps=social_deps, **_user_kwargs(),
                )
                rel_retry = rel_retry_result.output
                if rel_retry.social_topology:
                    local_social = _merge_social_retry(local_social, rel_retry)
                    logger.info(
                        "[Step 3b·Async] Chunk %d: rel retry recovered %d social edges (merged).",
                        i + 1, len(local_social.social_topology),
                    )
            except Exception:
                logger.exception("[Step 3b·Async] Chunk %d social retry FAILED.", i + 1)
        return local_social

    async def _run_consequences(local_social: SocialExtraction) -> Optional[ConsequencesExtraction]:
        if consequences_agent is None or not physics.events:
            return None
        logger.info("[Step 3c·Async] Processing chunk %d/%d — consequences …", i + 1, n)
        consequences_msg = (
            f"Chunk {i + 1} of {n}.\n\n"
            f"EVENTS EXTRACTED FROM THIS CHUNK:\n{event_summary}\n\n"
            f"ORIGINAL TEXT:\n{chunk_with_ctx}"
        )
        consequences_deps = _ConsequencesDeps(
            global_register=register,
            scaffold=scaffold,
            chunk_events=list(physics.events) + list(local_social.utterance_events),
            chunk_causal=physics.causal_topology,
            chunk_channels=local_social.channels,
            chunk_utterance_events=local_social.utterance_events,
            previous_event_ids=list(params.previous_event_ids),
            on_page_entity_ids=on_page,
            chunk_propositions=list(params.chunk_propositions),
        )
        try:
            consequences_result = await consequences_agent.run(
                consequences_msg, deps=consequences_deps, **_user_kwargs(),
            )
            consequences = consequences_result.output
        except Exception as exc:
            logger.exception(
                "[Step 3c·Async] Chunk %d FAILED — falling back to physics.entity_updates.",
                i + 1,
            )
            failure_flags["consequences"] = 1
            stage_errors.setdefault("consequences", f"{type(exc).__name__}: {exc}")
            return None

        # Mutation-parity retry — see sync path for rationale.
        missing_entities = _consequences_mutation_parity_broken(
            physics, consequences,
        )
        if missing_entities:
            logger.info(
                "[Step 3c·Async] Chunk %d: mutation parity broken for "
                "%d entit(y/ies) %s — retrying consequences …",
                i + 1, len(missing_entities), missing_entities,
            )
            parity_msg = (
                "IMPORTANT: The Physics Agent emitted mutation "
                "causal edges that target the following entities, "
                "but the previous extraction returned NO "
                f"EntityUpdate for them: {missing_entities}. For "
                "each one, emit at least one EntityUpdate "
                "anchored on the triggering event's fabula_time, "
                "with the new absolute trait values implied by "
                "the mutation edge's trait_target / trait_delta "
                "(and any implicit belief / status / location "
                "changes the event causes). Keep all "
                "EntityUpdates from your previous extraction.\n\n"
                + consequences_msg
            )
            try:
                parity_result = await consequences_agent.run(
                    parity_msg, deps=consequences_deps, **_user_kwargs(),
                )
                parity_consequences = parity_result.output
                new_missing = _consequences_mutation_parity_broken(
                    physics, parity_consequences,
                )
                if len(new_missing) < len(missing_entities):
                    consequences = _merge_consequences_retry(
                        consequences, parity_consequences,
                    )
                    logger.info(
                        "[Step 3c·Async] Chunk %d: parity retry covered "
                        "%d/%d missing entities (merged).",
                        i + 1,
                        len(missing_entities) - len(new_missing),
                        len(missing_entities),
                    )
            except Exception:
                logger.exception(
                    "[Step 3c·Async] Chunk %d parity retry FAILED.", i + 1,
                )
        return consequences

    social = await _run_social()

    # Audit fix #3: capture the event-id set BEFORE the post-Social
    # Physics retries (axis / dyad / coverage / parity / synthetic-stub
    # injection) so we can detect whether any added events.
    # ``_merge_physics_retry`` *does* admit new events (not just causal
    # edges), and the social pass that already ran was computed against
    # the pre-retry set — leaving the new events without channels,
    # utterances, or relationship readings until we re-sync below.
    pre_retry_event_ids: Set[str] = {e.id for e in physics.events}

    # Per-axis mutation_social coverage — async equivalent of the sync
    # path's axis retry. Refines ``physics`` in-place before the
    # consequences pass so any newly-added mutation_social edges are
    # visible to consequences (and to the assembler downstream).
    missing_axes = _physics_missing_mutation_social(physics, social)
    if missing_axes:
        logger.info(
            "[Step 3a·Async] Chunk %d: missing mutation_social axes %s "
            "— retrying physics with axis-specific emphasis …",
            i + 1, missing_axes,
        )
        axis_msg = (
            "IMPORTANT: The Social Agent observed non-zero "
            f"relationship reading(s) on the following axes "
            f"but the Physics Agent emitted NO matching "
            f"mutation_social causal edge for them: "
            f"{missing_axes}. For each axis, find the on-page "
            "event that produced the reading and emit a "
            "mutation_social edge with source_id=<that event>, "
            "target_id=<perspective entity>, "
            "rel_counterpart_id=<other entity>, "
            "trait_target=<axis>, and a signed trait_delta. "
            "Keep ALL events and causal edges from your "
            "previous extraction WITH THEIR ORIGINAL fabula_time "
            "VALUES UNCHANGED — in particular, do not collapse "
            "flashback events that previously had negative "
            "fabula_time into the present timeline.\n\n" + physics_msg
        )
        try:
            axis_result = await physics_agent.run(
                axis_msg, deps=physics_deps, **_user_kwargs(),
            )
            axis_physics = axis_result.output
            merged_axis = _merge_physics_retry(physics, axis_physics)
            new_missing_after_merge = _physics_missing_mutation_social(
                merged_axis, social,
            )
            if len(new_missing_after_merge) < len(missing_axes):
                physics = merged_axis
                logger.info(
                    "[Step 3a·Async] Chunk %d: axis retry covered "
                    "%d/%d missing axes (merged).",
                    i + 1,
                    len(missing_axes) - len(new_missing_after_merge),
                    len(missing_axes),
                )
        except Exception:
            logger.exception(
                "[Step 3a·Async] Chunk %d axis retry FAILED.", i + 1,
            )

    # Per-DYAD-per-axis coverage (async). Mirrors the sync path; see
    # the longer rationale in ``extract_topology_async``. The retry is the
    # last opportunity to anchor a specific dyad-axis combination on
    # an on-page event before consequences runs.
    missing_dyads = _physics_missing_mutation_social_per_dyad(
        physics, social,
    )
    if missing_dyads:
        summary_lines = [
            f"  - dyad ({tgt} ↔ {cp}) is missing axis '{ax}'"
            for tgt, cp, ax in missing_dyads[:25]
        ]
        more = (
            f"\n  …and {len(missing_dyads) - 25} more"
            if len(missing_dyads) > 25 else ""
        )
        logger.info(
            "[Step 3a·Async] Chunk %d: %d dyad×axis pair(s) lack a "
            "mutation_social edge — retrying physics with "
            "per-dyad emphasis …",
            i + 1, len(missing_dyads),
        )
        dyad_msg = (
            "IMPORTANT: For each of the following observed "
            "relationship axes, the Physics Agent emitted no "
            "mutation_social edge wired to that *specific* "
            "dyad-axis combination. Without one, the timeline "
            "reconstructor cannot evolve the metric and the "
            "corresponding sub-curve in the affective dashboard "
            "renders as a flat line.\n\n"
            "Missing dyad×axis triples:\n"
            + "\n".join(summary_lines) + more + "\n\n"
            "For each missing triple, find (or invent if the "
            "narrative implies one) the on-page event that "
            "produced or shifted the reading, and emit a "
            "mutation_social CausalEdge:\n"
            "  source_id=<EVT_ id>, "
            "causality_type='mutation_social', "
            "target_id=<perspective entity>, "
            "rel_counterpart_id=<other entity in the dyad>, "
            "trait_target=<axis>, "
            "trait_delta=<signed magnitude>.\n"
            "Keep ALL events and causal edges from your "
            "previous extraction WITH THEIR ORIGINAL fabula_time "
            "VALUES UNCHANGED — in particular, do not collapse "
            "flashback events that previously had negative "
            "fabula_time into the present timeline. The merge step "
            "keys on event ``id``, so reusing the same id with a "
            "new fabula_time has no effect.\n\n" + physics_msg
        )
        try:
            dyad_result = await physics_agent.run(
                dyad_msg, deps=physics_deps, **_user_kwargs(),
            )
            dyad_physics = dyad_result.output
            merged_physics = _merge_physics_retry(physics, dyad_physics)
            new_missing_dyads = (
                _physics_missing_mutation_social_per_dyad(
                    merged_physics, social,
                )
            )
            if len(new_missing_dyads) < len(missing_dyads):
                physics = merged_physics
                logger.info(
                    "[Step 3a·Async] Chunk %d: dyad retry covered "
                    "%d/%d missing dyad×axis triples (merged).",
                    i + 1,
                    len(missing_dyads) - len(new_missing_dyads),
                    len(missing_dyads),
                )
        except Exception:
            logger.exception(
                "[Step 3a·Async] Chunk %d dyad retry FAILED.", i + 1,
            )

    # --- Tier 1 #1: Mirror-suspicious dyad retry (async) -----------
    mirror_suspect = _social_mirror_suspicious_dyads(social)
    if mirror_suspect:
        sample = ", ".join(
            f"{a}\u2194{b} ({'/'.join(ax)})"
            for a, b, ax in mirror_suspect[:5]
        )
        logger.info(
            "[Step 3b\u00b7Async] Chunk %d: %d dyad(s) with near-identical "
            "bidirectional metrics (%s) \u2014 retrying \u2026",
            i + 1, len(mirror_suspect), sample,
        )
        mirror_msg = (
            "IMPORTANT: The previous extraction emitted "
            f"{len(mirror_suspect)} dyad(s) where the forward and "
            "reverse RelationshipEdges carry IDENTICAL metric values "
            f"across all shared axes: {sample}. Re-emit each of these "
            "dyads with two distinct, asymmetric readings grounded in "
            "what each character separately experiences. Keep all "
            "other channels, utterances, and edges UNCHANGED.\n\n"
            + social_msg
        )
        try:
            mirror_result = await social_agent.run(
                mirror_msg, deps=social_deps, **_user_kwargs(),
            )
            mirror_retry = mirror_result.output
            merged_mirror = _merge_social_retry(social, mirror_retry)
            new_suspect = _social_mirror_suspicious_dyads(merged_mirror)
            if len(new_suspect) < len(mirror_suspect):
                social = merged_mirror
                logger.info(
                    "[Step 3b\u00b7Async] Chunk %d: mirror retry resolved "
                    "%d/%d (merged).",
                    i + 1,
                    len(mirror_suspect) - len(new_suspect),
                    len(mirror_suspect),
                )
        except Exception:
            logger.exception(
                "[Step 3b\u00b7Async] Chunk %d mirror retry FAILED.", i + 1,
            )

    # --- Tier 1 #5: Anonymous-utterance retry (async) ---------------
    anon_utts = _social_anonymous_utterances(social)
    if anon_utts:
        logger.info(
            "[Step 3b\u00b7Async] Chunk %d: %d utterance(s) missing "
            "speaker_id or addressee_ids: %s\u2026 \u2014 retrying \u2026",
            i + 1, len(anon_utts), anon_utts[:5],
        )
        anon_utt_msg = (
            "IMPORTANT: The previous extraction produced "
            f"{len(anon_utts)} utterance event(s) with EMPTY "
            f"speaker_id OR EMPTY addressee_ids: {anon_utts[:10]}. "
            "Every utterance MUST name exactly one ENT_ in speaker_id "
            "and at least one ENT_ in addressee_ids. Re-emit those "
            "utterances with the on-page speaker and listener(s) "
            "filled in, keeping the SAME id and fabula_time so the "
            "merge step replaces the anonymous version. Keep all "
            "other channels, utterances, and edges UNCHANGED.\n\n"
            + social_msg
        )
        try:
            anon_utt_result = await social_agent.run(
                anon_utt_msg, deps=social_deps, **_user_kwargs(),
            )
            anon_utt_retry = anon_utt_result.output
            merged_anon_utt = _merge_anonymous_utterance_retry(
                social, anon_utt_retry,
            )
            new_anon_utt = _social_anonymous_utterances(merged_anon_utt)
            if len(new_anon_utt) < len(anon_utts):
                social = merged_anon_utt
                logger.info(
                    "[Step 3b\u00b7Async] Chunk %d: anon-utterance retry "
                    "resolved %d/%d (merged).",
                    i + 1,
                    len(anon_utts) - len(new_anon_utt),
                    len(anon_utts),
                )
        except Exception:
            logger.exception(
                "[Step 3b\u00b7Async] Chunk %d anon-utterance retry FAILED.",
                i + 1,
            )

    # --- Tier 1 #2: Utterance \u2194 physics parity (async) ----------
    social_only, physics_only = _utterance_parity_orphans(physics, social)
    if social_only:
        logger.info(
            "[Step 3a/b\u00b7Async] Chunk %d: %d social utterance(s) have "
            "no matching physics event \u2014 synthesizing stubs: %s\u2026",
            i + 1, len(social_only), social_only[:5],
        )
        physics_known = {e.id for e in physics.events}
        stubs_added = 0
        for u in social.utterance_events:
            if u.id in physics_known:
                continue
            physics.events.append(u.model_copy())
            stubs_added += 1
        if stubs_added:
            logger.info(
                "[Step 3a/b\u00b7Async] Chunk %d: added %d stub utterance "
                "events to physics from social.",
                i + 1, stubs_added,
            )
    if physics_only:
        logger.info(
            "[Step 3a/b\u00b7Async] Chunk %d: %d physics utterance(s) have "
            "no matching social channel/utterance.",
            i + 1, len(physics_only),
        )

    # --- Tier 1 #4: mutation_social coverage gauge (async) ----------
    covered, total = _social_mutation_coverage(physics, social)
    if total >= 4 and covered / total < 0.6:
        logger.info(
            "[Step 3a\u00b7Async] Chunk %d: mutation_social coverage "
            "%d/%d (%.0f%%) below threshold \u2014 retrying physics \u2026",
            i + 1, covered, total, 100.0 * covered / total,
        )
        cov_msg = (
            "IMPORTANT: The Social Agent observed "
            f"{total} dyad\u00d7axis readings but the Physics Agent only "
            f"emitted mutation_social edges for {covered} of them "
            f"({100.0 * covered / total:.0f}%%). Every observed "
            "relationship axis needs at least one mutation_social "
            "CausalEdge wired to the specific (target_id, "
            "rel_counterpart_id, trait_target) triple. Re-emit the "
            "extraction with the missing mutation_social edges added. "
            "Keep all other events, edges, and fabula_times "
            "UNCHANGED.\n\n" + physics_msg
        )
        try:
            cov_result = await physics_agent.run(
                cov_msg, deps=physics_deps, **_user_kwargs(),
            )
            cov_physics = cov_result.output
            merged_cov = _merge_physics_retry(physics, cov_physics)
            new_covered, _ = _social_mutation_coverage(merged_cov, social)
            if new_covered > covered:
                physics = merged_cov
                logger.info(
                    "[Step 3a\u00b7Async] Chunk %d: coverage retry raised "
                    "mutation_social coverage %d\u2192%d / %d (merged).",
                    i + 1, covered, new_covered, total,
                )
        except Exception:
            logger.exception(
                "[Step 3a\u00b7Async] Chunk %d coverage retry FAILED.", i + 1,
            )

    # Audit fix #3: re-sync Social on any events the post-Social
    # Physics retries newly introduced. Without this, Consequences (and
    # the chunk's final social_topology) would be blind to channels /
    # utterances / relationship readings the new events should have
    # produced.
    post_retry_event_ids: Set[str] = {e.id for e in physics.events}
    new_event_ids = post_retry_event_ids - pre_retry_event_ids
    if new_event_ids:
        new_events_listing = "\n".join(
            f"  - {e.id} (fabula={e.fabula_time}, type={e.event_type}, "
            f"actors={e.actor_ids}, targets={e.target_ids}): {e.description}"
            for e in physics.events if e.id in new_event_ids
        )
        logger.info(
            "[Step 3b\u00b7Async] Chunk %d: %d new event(s) added by "
            "post-Social Physics retries \u2014 re-syncing Social to "
            "cover them.",
            i + 1, len(new_event_ids),
        )
        # Rebuild the event summary against the *current* (post-retry)
        # physics event set so the agent's view of the chunk matches
        # what's now in the topology.
        refreshed_event_summary = "\n".join(
            f"  - {e.id} (fabula={e.fabula_time}, syuzhet={e.syuzhet_index}, "
            f"type={e.event_type}, actors={e.actor_ids}, targets={e.target_ids}): "
            f"{e.description}"
            for e in physics.events
        )
        refreshed_social_msg = (
            f"Chunk {i + 1} of {n}.\n\n"
            f"EVENTS EXTRACTED FROM THIS CHUNK:\n{refreshed_event_summary}\n\n"
            f"ORIGINAL TEXT:\n{chunk_with_ctx}"
        )
        resync_msg = (
            # Audit fix #7 (second pass): keep the *full* primary
            # social-prompt scaffold (event summary + chunk text) and
            # only PREFIX a re-sync directive. The earlier
            # implementation built a stripped-down message that omitted
            # the consolidated event summary, which left the agent
            # without the structural context it had on the first pass
            # and produced lower-quality re-sync output.
            "IMPORTANT: Physics retried after your previous social "
            f"extraction and added {len(new_event_ids)} NEW event(s) "
            "not seen on your first pass. Emit any channels, utterance "
            "events, or relationship_edge updates these new events "
            "imply. Keep ALL of your previously-extracted social "
            "material UNCHANGED.\n\n"
            "NEW events to cover:\n" + new_events_listing + "\n\n"
            "----- ORIGINAL SOCIAL PROMPT (with the full updated "
            "event set) -----\n" + refreshed_social_msg
        )
        # Refresh deps so the agent sees the updated event/causal set.
        resync_deps = social_deps.model_copy(update={
            "chunk_event_ids": [e.id for e in physics.events],
            "chunk_events": list(physics.events),
            "chunk_causal": list(physics.causal_topology),
        })
        try:
            resync_result = await social_agent.run(
                resync_msg, deps=resync_deps, **_user_kwargs(),
            )
            resync_social = resync_result.output
            if (
                resync_social.channels
                or resync_social.utterance_events
                or resync_social.social_topology
            ):
                social = _merge_social_retry(social, resync_social)
                logger.info(
                    "[Step 3b\u00b7Async] Chunk %d: re-sync added %d ch / "
                    "%d utt / %d rel (merged).",
                    i + 1,
                    len(resync_social.channels),
                    len(resync_social.utterance_events),
                    len(resync_social.social_topology),
                )
        except Exception:
            logger.exception(
                "[Step 3b\u00b7Async] Chunk %d social re-sync FAILED.", i + 1,
            )

    # Audit fix (final pass) — second drift check, this time crediting
    # social participants (channel speakers, utterance actors,
    # relationship-edge endpoints) against scaffold-flagged entities.
    # Gated on ``config.second_drift_pass`` AND the first drift retry
    # being enabled (otherwise we'd be doing the *first* retry under a
    # different name). Off by default; costs one extra Physics
    # round-trip per affected chunk.
    if config.second_drift_pass and config.scaffold_drift_retry:
        scaffold_ents = _scaffold_mentioned_entities(scaffold, register)
        if scaffold_ents:
            covered = (
                _physics_event_entities(physics)
                | _social_participating_entities(social)
            )
            still_missed = scaffold_ents - covered
            ratio2 = (
                len(scaffold_ents & covered) / len(scaffold_ents)
            )
            if ratio2 < 0.5 and still_missed:
                missed2_list = sorted(still_missed)[:20]
                logger.warning(
                    "[Scaffold-Drift\u00b7Async\u00b72nd] Chunk %d: "
                    "after social, still %.0f%% scaffold coverage; "
                    "missed %s",
                    i + 1, 100.0 * ratio2, missed2_list[:8],
                )
                drift2_msg = (
                    "IMPORTANT (second drift pass): Even after the "
                    "Social agent ran, the following on-page entities "
                    "from the scaffold still have NO event, channel, "
                    "utterance, or relationship-edge mention: "
                    f"{missed2_list}. Re-read the chunk and emit at "
                    "least one event per missed entity (a choice, "
                    "observation, utterance, or state-change), keeping "
                    "ALL previously-extracted events and edges with "
                    "their original ids and fabula_times "
                    "UNCHANGED.\n\n" + physics_msg
                )
                try:
                    drift2_result = await physics_agent.run(
                        drift2_msg, deps=physics_deps, **_user_kwargs(),
                    )
                    drift2_physics = drift2_result.output
                    merged2 = _merge_physics_retry(physics, drift2_physics)
                    new_covered2 = (
                        _physics_event_entities(merged2)
                        | _social_participating_entities(social)
                    )
                    new_missed2 = scaffold_ents - new_covered2
                    if len(new_missed2) < len(still_missed):
                        physics = merged2
                        logger.info(
                            "[Scaffold-Drift\u00b7Async\u00b72nd] "
                            "Chunk %d: 2nd drift retry covered %d/%d "
                            "still-missed entities (merged).",
                            i + 1,
                            len(still_missed) - len(new_missed2),
                            len(still_missed),
                        )
                except Exception:
                    logger.exception(
                        "[Scaffold-Drift\u00b7Async\u00b72nd] Chunk %d "
                        "second drift retry FAILED.", i + 1,
                    )

    consequences_out = await _run_consequences(social)
    if consequences_out is not None:
        entity_updates_final = consequences_out.entity_updates

    # Merge utterance events from the Social Agent into the chunk's event list.
    merged_events = _merge_utterances_into_events(
        physics.events, social.utterance_events,
        chunk_label=f"Step 3b·Async chunk {i + 1}",
    )

    # ----- Step 3d (Phase B4): Affect Agent — proposition + concern drift.
    # Skip cleanly when the agent isn't built (catalogue empty / disabled),
    # when the catalogue isn't on params (legacy callers), or when
    # ``_chunk_has_affect_signal`` says the chunk touches no PROP_ id and
    # no seeded entity.
    affect_props: List[ChunkPropositionSnapshot] = []
    affect_truth: List[PropositionTruthCommit] = []
    affect_concerns: List[ChunkConcernSnapshot] = []
    affect_new_seeds: List["ConcernSeed"] = []
    if affect_agent is not None and params.chunk_catalogue is not None:
        catalogue = params.chunk_catalogue
        catalogue_prop_ids = {p.proposition_id for p in catalogue.propositions}
        seeded_entity_ids = {s.entity_id for s in catalogue.concern_seeds}
        # Build the widened-gate lookup maps once per chunk.
        referent_to_props: Dict[str, Set[str]] = {}
        for prop in catalogue.propositions:
            for rid in prop.referent_ids:
                referent_to_props.setdefault(rid, set()).add(prop.proposition_id)
        concern_entity_to_props: Dict[str, Set[str]] = {}
        for seed in catalogue.concern_seeds:
            concern_entity_to_props.setdefault(seed.entity_id, set()).add(
                seed.proposition_id,
            )
        if _chunk_has_affect_signal(
            merged_events, entity_updates_final,
            catalogue_prop_ids, seeded_entity_ids,
            referent_to_props=referent_to_props,
            concern_entity_to_props=concern_entity_to_props,
        ):
            affect_deps = _AffectDeps(
                global_register=register,
                chunk_events=list(merged_events),
                chunk_entity_updates=list(entity_updates_final),
                propositions=list(catalogue.propositions),
                concern_seeds=list(catalogue.concern_seeds),
            )
            affect_msg = (
                f"Chunk {i + 1} of {n}.\n\n"
                f"Emit proposition_snapshots, proposition_truth_commits, "
                f"concern_snapshots and (sparingly) new_concern_seeds for "
                f"the events listed in the system prompt. Skip cleanly with "
                f"empty lists if nothing in this chunk shifts the "
                f"catalogue.\n\nORIGINAL TEXT:\n{chunk_with_ctx}"
            )
            try:
                affect_result = await affect_agent.run(
                    affect_msg, deps=affect_deps, **_user_kwargs(),
                )
                affect_out = affect_result.output
                affect_props = list(affect_out.proposition_snapshots)
                affect_truth = list(affect_out.proposition_truth_commits)
                affect_concerns = list(affect_out.concern_snapshots)
                affect_new_seeds = list(affect_out.new_concern_seeds)
                logger.info(
                    "[Step 3d·Async] Chunk %d affect: %d prop_snaps, "
                    "%d truth_commits, %d concern_snaps, %d new_seeds.",
                    i + 1, len(affect_props), len(affect_truth),
                    len(affect_concerns), len(affect_new_seeds),
                )
            except Exception as exc:
                logger.exception(
                    "[Step 3d·Async] Chunk %d affect FAILED — "
                    "proceeding without per-chunk affect deltas.", i + 1,
                )
                failure_flags["affect"] = 1
                stage_errors.setdefault(
                    "affect", f"{type(exc).__name__}: {exc}",
                )
        else:
            logger.debug(
                "[Step 3d·Async] Chunk %d: no affect signal — skipping.",
                i + 1,
            )

    topo = ChunkTopology(
        events=merged_events,
        causal_topology=physics.causal_topology,
        channels=social.channels,
        social_topology=social.social_topology,
        spatial_topology=physics.spatial_topology,
        entity_updates=entity_updates_final,
        proposition_snapshots=affect_props,
        proposition_truth_commits=affect_truth,
        concern_snapshots=affect_concerns,
        new_concern_seeds=affect_new_seeds,
    )
    logger.info(
        "[Step 3·Async] Chunk %d: %d events (%d utterances), %d causal, %d social, %d spatial, %d channels.",
        i + 1,
        len(topo.events),
        len(social.utterance_events),
        len(topo.causal_topology),
        len(topo.social_topology),
        len(topo.spatial_topology),
        len(topo.channels),
    )
    # Tier 3 #11 + audit fix #4/#5: persist topology so a re-run skips
    # the agents — but only when no stage failed (else resume could
    # serve a degraded extraction as clean).
    _save_chunk_checkpoint(
        config.checkpoint_dir, chunk, i, topo, fingerprint, failure_flags,
    )
    return topo, failure_flags, stage_errors


def _reconcile_chunk_topologies(
    topologies: List[ChunkTopology],
    config: ExtractionConfig,
) -> List[ChunkTopology]:
    """Post-merge reconciliation for parallel-extracted chunk topologies.

    1. Detects and renames duplicate ``EVT_`` IDs across chunks
       (appends ``_cN`` suffix where N is the chunk index).
    2. Re-numbers ``syuzhet_index`` globally in chunk order — syuzhet IS
       narration order, so the chunk-position assignment is canonical.
       Utterance events carry their own ``syuzhet_index`` and are
       re-numbered through the same standard event-remap pass; there
       is no longer a separate ``discovered_at_syuzhet`` field on
       Channel that needs special handling.

    Note: there is intentionally NO inter-chunk ``fabula_time`` shift.
    Forcing chunk order onto fabula order would erase flashbacks (per
    the fabula/syuzhet design — see `docs/academic-foundations.md`
    §1.1). Temporal contradictions inside the merged graph are caught
    later by ``_validate_time_ordering`` and the auditor.
    """
    # --- Pass 1: Detect and resolve duplicate event IDs across chunks ---
    global_evt_ids: Dict[str, int] = {}  # evt_id → first chunk index
    chunk_renames: List[Dict[str, str]] = [{} for _ in topologies]

    for ci, topo in enumerate(topologies):
        for evt in topo.events:
            if evt.id in global_evt_ids:
                # Collision — rename in the later chunk
                new_id = f"{evt.id}_c{ci}"
                # Ensure the rename itself doesn't collide
                suffix = ci
                while new_id in global_evt_ids:
                    suffix += len(topologies)
                    new_id = f"{evt.id}_c{suffix}"
                chunk_renames[ci][evt.id] = new_id
                global_evt_ids[new_id] = ci
                logger.info(
                    "[Reconcile] Duplicate EVT ID %s in chunk %d — renamed to %s.",
                    evt.id, ci, new_id,
                )
            else:
                global_evt_ids[evt.id] = ci

    # Apply renames to events and all edge references
    reconciled: List[ChunkTopology] = []
    for ci, topo in enumerate(topologies):
        rmap = chunk_renames[ci]
        if rmap:
            topo = _apply_event_renames(topo, rmap)
        reconciled.append(topo)

    # --- Pass 1b (audit fix #7): cross-chunk semantic dedup ---
    #
    # Adjacent chunks frequently re-extract the same boundary event
    # under different ids (the chunker prepends a prev-chunk overlap
    # tail; the LLM occasionally re-extracts events from that tail
    # despite the "do not re-extract" instruction). Within-chunk
    # ``_dedupe_scene_events`` already handles intra-chunk duplicates;
    # this pass extends the same key (actor set, target set, event_type,
    # rounded fabula_time) across chunks. The earliest-syuzhet
    # occurrence is canonical; later occurrences are dropped and all
    # of their causal / entity_update / belief / utterance refs are
    # rewritten to the canonical id.
    #
    # SAFETY GUARDS (audit fix #1, second pass): the dedup is restricted
    # to ADJACENT chunks (|chunk_i - chunk_j| <= 1) and to events whose
    # description text overlaps significantly. Without these guards,
    # repeated confrontations between the same characters across the
    # novel collapse onto a single beat. Utterances are excluded
    # entirely (each speech act is distinct).
    fabula_window = 50
    _DESC_TOKEN_OVERLAP_MIN = 0.45  # Jaccard threshold on lowercased word tokens
    seen_groups: Dict[tuple, List[Tuple[int, str, str]]] = {}
    cross_chunk_renames: Dict[str, str] = {}
    dropped_events: Set[str] = set()

    def _evt_group_key(e) -> Optional[tuple]:
        if e.event_type == "utterance":
            return None
        if not e.actor_ids and not e.target_ids:
            return None
        return (
            frozenset(e.actor_ids or []),
            frozenset(e.target_ids or []),
            e.event_type,
            (e.fabula_time or 0) // fabula_window,
        )

    def _desc_tokens(d: str) -> Set[str]:
        return {t for t in re.findall(r"[A-Za-z']+", (d or "").lower()) if len(t) > 2}

    for ci, topo in enumerate(reconciled):
        for e in topo.events:
            key = _evt_group_key(e)
            if key is None:
                continue
            entries = seen_groups.setdefault(key, [])
            tokens = _desc_tokens(e.description)
            best_match: Optional[Tuple[int, str, str]] = None
            for prev_ci, prev_id, prev_desc in entries:
                # Restrict to adjacent (or same) chunks — beats that
                # repeat across distant chunks are almost always
                # legitimately separate scene events.
                if abs(ci - prev_ci) > 1:
                    continue
                prev_tokens = _desc_tokens(prev_desc)
                if not tokens or not prev_tokens:
                    continue
                jacc = len(tokens & prev_tokens) / max(1, len(tokens | prev_tokens))
                if jacc >= _DESC_TOKEN_OVERLAP_MIN:
                    best_match = (prev_ci, prev_id, prev_desc)
                    break
            if best_match is not None and best_match[1] != e.id:
                cross_chunk_renames[e.id] = best_match[1]
                dropped_events.add(e.id)
            else:
                entries.append((ci, e.id, e.description))

    if cross_chunk_renames:
        logger.info(
            "[Reconcile] Cross-chunk dedup: collapsing %d boundary-"
            "duplicate event(s) onto their first-occurrence id.",
            len(cross_chunk_renames),
        )
        # Rewrite refs first (so dropped ids vanish from edges /
        # entity_updates / beliefs), then drop the duplicate event
        # nodes themselves.
        rewritten: List[ChunkTopology] = []
        for topo in reconciled:
            renamed = _apply_event_renames(topo, cross_chunk_renames)
            kept = [e for e in renamed.events if e.id not in dropped_events]
            rewritten.append(renamed.model_copy(update={"events": kept}))
        reconciled = rewritten

    # --- Pass 2: Re-number syuzhet_index globally in chunk order ---
    syuzhet_counter = 0
    for topo in reconciled:
        sorted_events = sorted(topo.events, key=lambda e: e.syuzhet_index)
        for evt in sorted_events:
            evt.syuzhet_index = syuzhet_counter
            syuzhet_counter += 1

    # --- Pass 3: Heal "LLM restarted fabula numbering" pathology ---
    #
    # Fabula order is intentionally free across chunks so that flashbacks
    # (chunk reaches into the past) and flash-forwards (chunk jumps to
    # the future) remain expressible. We MUST NOT blindly stack chunk N
    # after chunk N-1 — that would erase those structures.
    #
    # However, when a chunk's per-chunk extractor ignores the
    # "continue from prev_max" hint and emits small sequential integers
    # (1, 2, 3 …) instead of values aligned to ``fabula_time_spacing``,
    # those values silently collide with prior chunks once they reach
    # ``_normalize_fabula_times`` — distinct events collapse onto the
    # same normalised tick. We detect that pathology *only* when every
    # signal points to it and never when a flashback (deliberate use of
    # an earlier absolute fabula_time) is plausible:
    #
    #   1. The chunk has at least two events and its full positive
    #      fabula_time range fits inside a single ``fabula_time_spacing``
    #      slot (i.e. ``chunk_max < spacing``). A deliberate flashback
    #      uses *absolute* story-time values aligned to the global
    #      spacing — even a tightly-clustered flashback scene at
    #      fabula 100, 200, 300 with spacing 1000 keeps each event in
    #      its own slot when it eventually normalises, so this gate
    #      stays closed.
    #   2. The chunk's events are clustered as small consecutive
    #      integers (max ≤ events × 4) — the canonical "LLM gave up on
    #      the spacing hint and just counted" signature.
    #   3. Every positive fabula_time in chunk N is strictly below
    #      ``prev_max``. A flash-forward (chunk N already past
    #      prior chunks) trips this and is left alone, as is any
    #      mixed chunk with even a single event ≥ prev_max.
    #   4. The chunk has zero causal_topology edges referencing any
    #      event in a prior chunk. A deliberate flashback that
    #      revisits or causally connects to an earlier event almost
    #      always carries a ``chain_reaction`` / ``mutation`` edge
    #      linking back to the event being remembered — its absence
    #      reinforces the diagnosis of structural disconnection.
    #
    # When all four signals fire the chunk is shifted forward by
    # ``prior_max + spacing - chunk_min`` — preserving the chunk's
    # *internal* spacing (and therefore any intra-chunk ordering)
    # while placing the whole block after prior content with a
    # one-spacing gap.
    if config.fabula_time_spacing > 0 and len(reconciled) > 1:
        prior_max = 0
        prior_event_ids: set[str] = set()
        spacing = config.fabula_time_spacing
        for ci, topo in enumerate(reconciled):
            chunk_event_ids = {e.id for e in topo.events}
            chunk_fabs = [e.fabula_time for e in topo.events if e.fabula_time > 0]
            if ci == 0 or not chunk_fabs:
                if chunk_fabs:
                    prior_max = max(prior_max, max(chunk_fabs))
                prior_event_ids |= chunk_event_ids
                continue
            chunk_max = max(chunk_fabs)
            chunk_min = min(chunk_fabs)
            cross_chunk_edges = any(
                (ce.source_id in prior_event_ids) or (ce.target_id in prior_event_ids)
                for ce in topo.causal_topology
            )
            looks_restarted = (
                len(chunk_fabs) >= 2
                and chunk_max < spacing
                and chunk_max <= len(chunk_fabs) * 4
            )
            if (
                looks_restarted
                and chunk_max < prior_max
                and not cross_chunk_edges
            ):
                # Shift so chunk N starts at prior_max + spacing —
                # leaving a one-spacing gap to keep events distinguishable
                # without overstating a temporal jump.
                shift = prior_max + spacing - chunk_min
                _shift_fabula_times(topo, shift)
                logger.warning(
                    "[Reconcile] Chunk %d looked restarted (range %d-%d, "
                    "%d events, prev_max=%d, no causal links into prior "
                    "chunks) — shifted forward by %d to avoid fabula "
                    "collision.",
                    ci, chunk_min, chunk_max, len(chunk_fabs), prior_max, shift,
                )
                chunk_fabs = [e.fabula_time for e in topo.events if e.fabula_time > 0]
            if chunk_fabs:
                prior_max = max(prior_max, max(chunk_fabs))
            prior_event_ids |= chunk_event_ids

    return reconciled


def _apply_event_renames(topo: ChunkTopology, rmap: Dict[str, str]) -> ChunkTopology:
    """Apply event ID renames to all fields in a ChunkTopology.

    Channel objects do not carry event-id references and are therefore
    untouched. Utterance events live in ``topo.events`` and are renamed
    through the standard event loop.
    """
    def _r(eid: str) -> str:
        return rmap.get(eid, eid)

    def _r_list(ids: List[str]) -> List[str]:
        # ``target_ids`` may legitimately contain a renamed EVT_ ref
        # (e.g. an utterance whose target is the prior choice it
        # responds to). Preserve list order and identity for non-EVT ids.
        return [rmap.get(i, i) for i in ids]

    new_events = [
        e.model_copy(update={
            "id": _r(e.id),
            "target_ids": _r_list(e.target_ids),
            "actor_ids": _r_list(e.actor_ids),
        })
        for e in topo.events
    ]
    new_causal = [
        ce.model_copy(update={
            "source_id": _r(ce.source_id),
            "target_id": _r(ce.target_id),
        }) for ce in topo.causal_topology
    ]
    new_entity_updates = [
        eu.model_copy(update={
            "triggered_by": _r(eu.triggered_by) if eu.triggered_by else None,
            "new_beliefs": [
                b.model_copy(update={
                    "acquired_via_event_id": _r(b.acquired_via_event_id)
                    if b.acquired_via_event_id else None,
                })
                for b in eu.new_beliefs
            ],
        }) for eu in topo.entity_updates
    ]
    # Phase B4: rename triggered_by on every affect snapshot / truth
    # commit too, so cross-chunk EVT_ id collisions resolved by Pass 1
    # of ``_reconcile_chunk_topologies`` keep the affect → event link
    # intact for the Phase C reconciler.
    new_prop_snaps = [
        ps.model_copy(update={"triggered_by": _r(ps.triggered_by)})
        for ps in topo.proposition_snapshots
    ]
    new_truth_commits = [
        tc.model_copy(update={"triggered_by": _r(tc.triggered_by)})
        for tc in topo.proposition_truth_commits
    ]
    new_concern_snaps = [
        cs.model_copy(update={"triggered_by": _r(cs.triggered_by)})
        for cs in topo.concern_snapshots
    ]
    # ``model_copy(update=...)`` rather than rebuilding the model
    # explicitly so any ChunkTopology field NOT touched by rename
    # (new_entities, new_objects, new_locations, new_world_traits,
    # new_propositions, new_concerns, proposition_truth_commits,
    # belief_confidence_updates, deletions, supersession_updates,
    # new_concern_seeds, etc.) is preserved verbatim. Rebuilding the
    # model with a hand-listed subset silently dropped those fields,
    # which manifested as snapshot-era / merge-era data going missing
    # whenever cross-chunk EVT_ id reconciliation fired.
    return topo.model_copy(update={
        "events": new_events,
        "causal_topology": new_causal,
        "entity_updates": new_entity_updates,
        "proposition_snapshots": new_prop_snaps,
        "proposition_truth_commits": new_truth_commits,
        "concern_snapshots": new_concern_snaps,
    })


def _shift_fabula_times(topo: ChunkTopology, shift: int) -> None:
    """Shift all fabula_time values in a ChunkTopology by *shift* (in-place).

    Values <= 0 are treated as the "pre-story baseline" sentinel and
    left untouched, so beliefs and edges that were established before
    the narrative begins are not pushed into story-time.
    """
    for evt in topo.events:
        if evt.fabula_time > 0:
            evt.fabula_time += shift
    for ce in topo.causal_topology:
        if ce.fabula_time > 0:
            ce.fabula_time += shift
    for ch in topo.channels.values():
        if ch.established_at_fabula > 0:
            ch.established_at_fabula += shift
        if ch.terminated_at_fabula is not None and ch.terminated_at_fabula > 0:
            ch.terminated_at_fabula += shift
    for se in topo.social_topology:
        # last_updated_fabula is now per-axis; shift each observed
        # metric independently to preserve relative ordering.
        for m in se.metrics.values():
            if m.last_updated_fabula > 0:
                m.last_updated_fabula += shift
    for sp in topo.spatial_topology:
        if sp.established_at_fabula > 0:
            sp.established_at_fabula += shift
        if sp.destroyed_at_fabula is not None and sp.destroyed_at_fabula > 0:
            sp.destroyed_at_fabula += shift
    for eu in topo.entity_updates:
        if eu.fabula_time > 0:
            eu.fabula_time += shift
        for belief in eu.new_beliefs:
            if belief.established_at_fabula > 0:
                belief.established_at_fabula += shift


async def extract_topology_async(
    chunks: List[str],
    register: GlobalRegister,
    config: ExtractionConfig | None = None,
    catalogue: Optional[PropositionCatalogue] = None,
) -> List[ChunkTopology]:
    """Extract per-chunk topology — chunks are processed in parallel.

    Chunks are dispatched concurrently (limited by
    ``config.max_concurrent_chunks``) with pre-allocated syuzhet and
    fabula_time ranges.  After all chunks complete, a reconciliation
    pass re-numbers ``syuzhet_index`` globally, ensures inter-chunk
    ``fabula_time`` ordering, and resolves any duplicate event IDs.
    """
    config = config or ExtractionConfig()
    socratic_agent = _build_socratic_agent(config)
    physics_agent = _build_physics_agent(config)
    social_agent = _build_social_agent(config)
    consequences_agent = (
        _build_consequences_agent(config)
        if config.enable_consequences_agent else None
    )
    # Phase B4: build the affect agent only when both the global
    # catalogue stage actually produced something AND the user hasn't
    # disabled the per-chunk affect pass. ``_extract_single_chunk_async``
    # additionally gates per-chunk on ``_chunk_has_affect_signal``.
    affect_agent = (
        _build_affect_agent(config)
        if (
            config.enable_affect_agent
            and catalogue is not None
            and (catalogue.propositions or catalogue.concern_seeds)
        )
        else None
    )

    params_list = _pre_allocate_chunk_params(chunks, config)
    # Phase A3: thread the global Proposition Catalogue id list onto
    # every chunk's params so the per-chunk system_prompt injectors
    # render the catalogue block. Empty list (no catalogue stage / no
    # propositions found) leaves the existing default-empty behaviour.
    if catalogue is not None and catalogue.propositions:
        catalogue_pairs: List[Tuple[str, str]] = [
            (p.proposition_id, p.description) for p in catalogue.propositions
        ]
        params_list = [
            p.model_copy(update={
                "chunk_propositions": catalogue_pairs,
                "chunk_catalogue": catalogue,
            })
            for p in params_list
        ]
    elif catalogue is not None:
        # Concern-seeds-only edge case: still surface the catalogue so
        # the affect agent can drift seeded concerns even when the
        # catalogue declared no propositions.
        params_list = [
            p.model_copy(update={"chunk_catalogue": catalogue})
            for p in params_list
        ]
    semaphore = asyncio.Semaphore(config.max_concurrent_chunks)
    chunk_timeout = getattr(config, "per_chunk_timeout_seconds", 0)
    # Cap how many consecutive timeouts to tolerate in serial mode
    # before bailing out — without this, a wedged model would let the
    # pipeline burn through the whole text producing only empty
    # topologies (audit fix #5, second pass).
    _CARRY_TIMEOUT_CONSECUTIVE_LIMIT = 3

    async def _guarded_extract(
        chunk: str, params: _ChunkParams,
    ) -> Tuple[ChunkTopology, Dict[str, int], Dict[str, str]]:
        async with semaphore:
            coro = _extract_single_chunk_async(
                chunk, params, register, config,
                socratic_agent, physics_agent, social_agent,
                consequences_agent, affect_agent,
            )
            if chunk_timeout and chunk_timeout > 0:
                try:
                    return await asyncio.wait_for(coro, timeout=chunk_timeout)
                except asyncio.TimeoutError:
                    logger.warning(
                        "[Pipeline\u00b7Async] Chunk %d/%d exceeded per-chunk "
                        "timeout (%.0fs) \u2014 returning empty topology so "
                        "the rest of the run can proceed.",
                        params.chunk_index + 1, params.total_chunks, chunk_timeout,
                    )
                    timeout_msg = (
                        f"asyncio.TimeoutError: per-chunk pipeline did "
                        f"not complete within {chunk_timeout:.0f}s"
                    )
                    return (
                        ChunkTopology(),
                        {"physics": 1, "social": 1, "consequences": 1},
                        {
                            "physics": timeout_msg,
                            "social": timeout_msg,
                            "consequences": timeout_msg,
                        },
                    )
            return await coro

    if config.enable_chunk_carry_over:
        # Audit fix #2: serial dispatch with carry-over. Chunk N's deps
        # see the prior chunk(s)' event ids and accumulated standing
        # channels so the LLM can re-use existing CHN_ ids and resolve
        # back-references to prior events instead of inventing fresh
        # ids that look like duplicates after assembly. Sacrifices the
        # max_concurrent_chunks speedup; intended for runs where
        # cross-chunk continuity outweighs wall time.
        logger.info(
            "[Pipeline\u00b7Async] Chunk carry-over ENABLED \u2014 "
            "dispatching %d chunk(s) serially.", len(chunks),
        )
        chunk_results: List[Tuple[ChunkTopology, Dict[str, int], Dict[str, str]]] = []
        accumulated_event_ids: List[str] = []
        accumulated_channels: Dict[str, Channel] = {}
        consecutive_timeouts = 0
        # Cap how many prior event ids we forward so the prompt size
        # stays bounded on long runs. The most recent ones carry the
        # most coreference value.
        _CARRY_EVENT_ID_TAIL = 60
        for chunk, params in zip(chunks, params_list):
            params = params.model_copy(update={
                "previous_event_ids": list(accumulated_event_ids[-_CARRY_EVENT_ID_TAIL:]),
                "previous_chunk_channels": dict(accumulated_channels),
            })
            result = await _guarded_extract(chunk, params)
            chunk_results.append(result)
            topo, flags, _errs = result
            # Detect "all stages failed" as a probable timeout / wedged
            # model and short-circuit before the whole text is consumed
            # by empty extractions.
            if all(flags.values()):
                consecutive_timeouts += 1
                if consecutive_timeouts >= _CARRY_TIMEOUT_CONSECUTIVE_LIMIT:
                    logger.error(
                        "[Pipeline\u00b7Async] %d consecutive fully-failed "
                        "chunk(s) in serial carry-over mode \u2014 aborting "
                        "the chunk loop early; the failure-threshold check "
                        "below will surface the cause.",
                        consecutive_timeouts,
                    )
                    break
            else:
                consecutive_timeouts = 0
            accumulated_event_ids.extend(e.id for e in topo.events)
            for cid, ch in topo.channels.items():
                accumulated_channels.setdefault(cid, ch)
    else:
        # Audit fix #4 (second pass): use return_exceptions so an
        # unexpected un-caught failure in one chunk doesn't take down
        # the whole gather. Convert any raised exception to the same
        # "all stages failed" sentinel the timeout path uses so the
        # threshold check sees a uniform failure signal.
        gathered = await asyncio.gather(*[
            _guarded_extract(chunk, params)
            for chunk, params in zip(chunks, params_list)
        ], return_exceptions=True)
        chunk_results = []
        for r in gathered:
            if isinstance(r, BaseException):
                err = f"{type(r).__name__}: {r}"
                logger.exception(
                    "[Pipeline\u00b7Async] Chunk task raised: %s", err,
                )
                chunk_results.append((
                    ChunkTopology(),
                    {"physics": 1, "social": 1, "consequences": 1},
                    {"physics": err, "social": err, "consequences": err},
                ))
            else:
                chunk_results.append(r)
    topologies_list = [r[0] for r in chunk_results]
    failure_counts: Dict[str, int] = {"physics": 0, "social": 0, "consequences": 0}
    sample_errors: Dict[str, str] = {}
    for _, flags, errs in chunk_results:
        for stage, flag in flags.items():
            failure_counts[stage] += flag
        for stage, msg in errs.items():
            sample_errors.setdefault(stage, msg)
    _check_chunk_failure_threshold(
        failure_counts, len(chunks), sample_errors=sample_errors,
    )

    # Post-merge reconciliation
    topologies_list = _reconcile_chunk_topologies(topologies_list, config)

    logger.info(
        "[Step 3·Async] All %d chunks extracted and reconciled.", len(topologies_list),
    )
    return topologies_list


# =====================================================================
# Step 3d — Optional Research Extraction (segregated)
# =====================================================================

def _build_research_agent(
    config: ExtractionConfig,
) -> Agent[None, "WorldFact"]:
    """Construct the Step 3d Research-Extraction Agent.

    The agent's output is a single ``WorldFact``. It is forbidden from
    emitting Entities, Events, or any topology — Pydantic's NativeOutput
    on ``WorldFact`` enforces that structurally; the prompt reinforces it.
    """
    from shadow_loom.research import WorldFact  # local to avoid top-level cycle risk

    agent: Agent[None, WorldFact] = Agent(
        _resolve_model(config.model),
        output_type=NativeOutput(WorldFact),
        system_prompt=_load_prompt("research_extraction.md"),
        retries=config.output_retries,
    )
    return agent


_FACT_ID_RE = re.compile(r"^FACT_(\d+)$")


def _next_fact_index(facts: List[Any]) -> int:
    """Return the next collision-free numeric suffix for a FACT_ id.

    Uses ``max(existing numeric suffix) + 1`` rather than ``len + 1``
    so sparse / hand-edited fact lists don't generate ids that collide
    with surviving entries (e.g. FACT_001, FACT_003 \u2192 next-by-length
    would re-emit FACT_003).
    """
    highest = 0
    for f in facts:
        fid = getattr(f, "id", None)
        if not isinstance(fid, str):
            continue
        m = _FACT_ID_RE.match(fid)
        if m:
            try:
                highest = max(highest, int(m.group(1)))
            except ValueError:
                continue
    return highest + 1


def _run_research_step(
    world_state: WorldStateV1,
    config: ExtractionConfig,
) -> WorldStateV1:
    """Step 3d (sync): for each configured topic, query provider + distil to WorldFact.

    Pure additive: only ``world_state.world_facts`` is mutated; entities,
    events and edges are untouched. Failures (provider error, agent
    refusal, empty results) are logged and skipped — research is best-
    effort and never fails extraction.
    """
    if not config.enable_research_agent:
        return world_state
    if not config.research_topics:
        logger.info("[Pipeline·Research] enabled but research_topics is empty — skipping.")
        return world_state

    from shadow_loom.research import (
        ResearchSnippet,
        build_provider,
    )

    try:
        # ``build_provider`` accepts a small fixed kwarg set
        # (``api_key``, ``search_depth``) plus provider-specific extras
        # forwarded into the provider constructor. Historically we
        # passed ``provider_model`` / ``max_results`` here, but the
        # Tavily constructor accepts neither and the call would raise
        # silently into the broad ``except`` below — making research
        # "work" only on the no-op NullProvider. ``provider_model`` is
        # used as the Tavily search-depth ("basic" or "advanced");
        # ``max_results`` is per-search and is forwarded into the
        # ``provider.search`` call below instead of construction.
        provider_kwargs: dict = {}
        depth = (config.research_provider_model or "").strip().lower()
        if depth in {"basic", "advanced"}:
            provider_kwargs["search_depth"] = depth
        provider = build_provider(
            config.research_provider,
            **provider_kwargs,
        )
    except Exception:
        logger.exception("[Pipeline·Research] failed to build provider — skipping.")
        return world_state

    agent = _build_research_agent(config)
    # Compute the next FACT id from the highest existing numeric suffix
    # rather than ``len(world_facts) + 1``. The latter collides when
    # facts have been hand-edited / partially deleted upstream and the
    # surviving id sequence is sparse (e.g. FACT_001, FACT_003 → next
    # by length is FACT_003 again).
    next_idx = _next_fact_index(world_state.world_facts)

    for topic in config.research_topics:
        try:
            snippets: List[ResearchSnippet] = list(
                provider.search(
                    topic,
                    max_results=config.research_max_results_per_query,
                )
            )
        except Exception:
            logger.exception("[Pipeline·Research] provider.search failed for topic=%r", topic)
            continue

        if not snippets:
            logger.info("[Pipeline·Research] no snippets for topic=%r — skipping.", topic)
            continue

        # Build the user message — minimal, structured.
        snippet_block = "\n\n".join(
            f"[{i+1}] {s.title}\nURL: {s.url}\n{s.content}"
            for i, s in enumerate(snippets)
        )
        user_msg = (
            f"Topic: {topic}\n\n"
            f"Snippets ({len(snippets)}):\n\n{snippet_block}\n\n"
            "Distil the above into a single WorldFact per the system prompt."
        )

        try:
            result = agent.run_sync(user_msg, **_user_kwargs())
            fact: WorldFact = result.output
        except Exception:
            logger.exception("[Pipeline·Research] agent failed for topic=%r — skipping.", topic)
            continue

        # Stamp pipeline-controlled fields the agent doesn't get to choose.
        fact_id = f"FACT_{next_idx:03d}"
        next_idx += 1
        stamped = fact.model_copy(update={
            "id": fact_id,
            "topic": topic,
            "provider": config.research_provider,
            "raw_snippets": snippets,
        })
        world_state.world_facts.append(stamped)
        logger.info(
            "[Pipeline·Research] +WorldFact %s topic=%r confidence=%s",
            fact_id, topic, stamped.confidence,
        )

    return world_state


async def _run_research_step_async(
    world_state: WorldStateV1,
    config: ExtractionConfig,
) -> WorldStateV1:
    """Async variant of ``_run_research_step``.

    Topics are processed sequentially (research is naturally low-volume
    and provider rate limits are typically the bottleneck, not local
    concurrency).
    """
    if not config.enable_research_agent:
        return world_state
    if not config.research_topics:
        logger.info("[Pipeline·Research·Async] enabled but no topics — skipping.")
        return world_state

    from shadow_loom.research import (
        ResearchSnippet,
        build_provider,
    )

    try:
        # See ``_run_research_step`` for why we don't pass
        # provider_model / max_results into ``build_provider``.
        provider_kwargs: dict = {}
        depth = (config.research_provider_model or "").strip().lower()
        if depth in {"basic", "advanced"}:
            provider_kwargs["search_depth"] = depth
        provider = build_provider(
            config.research_provider,
            **provider_kwargs,
        )
    except Exception:
        logger.exception("[Pipeline·Research·Async] failed to build provider — skipping.")
        return world_state

    agent = _build_research_agent(config)
    next_idx = _next_fact_index(world_state.world_facts)

    for topic in config.research_topics:
        try:
            # Provider.search is sync (Tavily client is sync); run in thread.
            snippets: List[ResearchSnippet] = list(
                await asyncio.to_thread(
                    provider.search,
                    topic,
                    config.research_max_results_per_query,
                )
            )
        except Exception:
            logger.exception("[Pipeline·Research·Async] provider.search failed for topic=%r", topic)
            continue

        if not snippets:
            continue

        snippet_block = "\n\n".join(
            f"[{i+1}] {s.title}\nURL: {s.url}\n{s.content}"
            for i, s in enumerate(snippets)
        )
        user_msg = (
            f"Topic: {topic}\n\n"
            f"Snippets ({len(snippets)}):\n\n{snippet_block}\n\n"
            "Distil the above into a single WorldFact per the system prompt."
        )

        try:
            result = await agent.run(user_msg, **_user_kwargs())
            fact: WorldFact = result.output
        except Exception:
            logger.exception("[Pipeline·Research·Async] agent failed for topic=%r — skipping.", topic)
            continue

        fact_id = f"FACT_{next_idx:03d}"
        next_idx += 1
        stamped = fact.model_copy(update={
            "id": fact_id,
            "topic": topic,
            "provider": config.research_provider,
            "raw_snippets": snippets,
        })
        world_state.world_facts.append(stamped)
        logger.info(
            "[Pipeline·Research·Async] +WorldFact %s topic=%r confidence=%s",
            fact_id, topic, stamped.confidence,
        )

    return world_state


# =====================================================================
# Fabula-Time Normalization
# =====================================================================


def _normalize_fabula_times(ws: WorldStateV1, spacing: int = 1000) -> WorldStateV1:
    """
    Re-space ``fabula_time`` values using *spacing* when the LLM ignores
    the requested 100-scale and returns small sequential integers (1, 2, 3 …).

    Builds a monotonic mapping over the union of every fabula_time value
    found anywhere in the world-state — events, every edge type, beliefs,
    and timeline snapshots — then applies it uniformly. This guarantees
    edges and beliefs stay synchronised with their referenced events
    after rescaling.

    The 0 value is preserved as the "pre-story baseline" sentinel: it
    is never remapped, and any belief / edge with fabula_time == 0 stays
    at 0 to keep its pre-story semantics.

    Returns the original world-state unchanged when event times are
    already well-spaced (median gap ≥ spacing / 2).

    KNOWN LIMITATION (cross-chunk fabula collision): the mapping is
    keyed on raw fabula_time alone. When chunk 2's LLM ignores the
    "max so far" hint and restarts numbering at 1, chunk 1's
    ``fabula_time=1`` and chunk 2's ``fabula_time=1`` would
    deduplicate to the same key and collapse onto the same normalised
    tick. The chunk-restart healer in
    :func:`_reconcile_chunk_topologies` (Pass 3, added May 2026)
    detects this pattern *before* normalisation runs and shifts the
    affected chunk forward — preserving its internal spacing and
    leaving genuine flashbacks (chunks linked causally to prior
    events) and flash-forwards (chunks already past prior_max)
    untouched. A stray restart that slips past the heuristic (e.g.
    a multi-restart cascade across many chunks, or a chunk that
    coincidentally meets all flashback signals) can still merge
    unrelated events; re-ingesting under the strengthened prompts
    remains the recommended remedy for fixtures predating that
    healer.
    """
    if not ws.events:
        return ws

    # --- Decide whether normalisation is needed (event spacing only) ---
    event_times = sorted({e.fabula_time for e in ws.events if e.fabula_time > 0})
    if len(event_times) < 2:
        return ws
    diffs = [event_times[i + 1] - event_times[i] for i in range(len(event_times) - 1)]
    median_diff = sorted(diffs)[len(diffs) // 2]
    if median_diff >= spacing // 2:
        return ws  # already well-spaced

    # --- Collect ALL fabula_time values across the world-state ---
    all_times: set[int] = set()
    for e in ws.events:
        all_times.add(e.fabula_time)
    for ce in ws.causal_topology:
        all_times.add(ce.fabula_time)
    for ch in ws.channels.values():
        all_times.add(ch.established_at_fabula)
        if ch.terminated_at_fabula is not None:
            all_times.add(ch.terminated_at_fabula)
    for se in ws.spatial_topology:
        all_times.add(se.established_at_fabula)
        if se.destroyed_at_fabula is not None:
            all_times.add(se.destroyed_at_fabula)
    for re_edge in ws.social_topology:
        for m in re_edge.metrics.values():
            all_times.add(m.last_updated_fabula)
    for ent in ws.entities.values():
        for b in ent.beliefs:
            all_times.add(b.established_at_fabula)
        for snap in ent.state_timeline:
            all_times.add(snap.fabula_time)
    for wt in ws.world_traits.values():
        for snap in wt.state_timeline:
            all_times.add(snap.fabula_time)

    # 0 is the pre-story sentinel — keep it pinned at 0.
    nonzero_sorted = sorted(t for t in all_times if t > 0)
    if not nonzero_sorted:
        return ws

    # Build old → new mapping. 0 always stays 0.
    time_map: dict[int, int] = {0: 0}
    for i, t in enumerate(nonzero_sorted):
        time_map[t] = (i + 1) * spacing

    def _map(t: int | None) -> int | None:
        if t is None:
            return None
        return time_map.get(t, t)

    new_events = [
        e.model_copy(update={"fabula_time": _map(e.fabula_time) or e.fabula_time}) for e in ws.events
    ]
    new_causal = [
        ce.model_copy(update={"fabula_time": _map(ce.fabula_time) or ce.fabula_time})
        for ce in ws.causal_topology
    ]
    new_channels = {
        cid: ch.model_copy(update={
            "established_at_fabula": _map(ch.established_at_fabula) or ch.established_at_fabula,
            "terminated_at_fabula": _map(ch.terminated_at_fabula),
        })
        for cid, ch in ws.channels.items()
    }
    new_social = []
    for re_edge in ws.social_topology:
        # Remap each per-axis last_updated_fabula independently.
        new_metrics = {
            name: m.model_copy(update={
                "last_updated_fabula": _map(m.last_updated_fabula) or m.last_updated_fabula,
            })
            for name, m in re_edge.metrics.items()
        }
        new_social.append(re_edge.model_copy(update={"metrics": new_metrics}))
    new_spatial = [
        se.model_copy(update={
            "established_at_fabula": _map(se.established_at_fabula) or se.established_at_fabula,
            "destroyed_at_fabula": _map(se.destroyed_at_fabula),
        })
        for se in ws.spatial_topology
    ]

    # Remap belief established_at_fabula (0 = pre-story, stays 0)
    new_entities: dict[str, Entity] = {}
    for eid, ent in ws.entities.items():
        new_beliefs = [
            b.model_copy(update={
                "established_at_fabula": _map(b.established_at_fabula) or b.established_at_fabula,
            })
            for b in ent.beliefs
        ]
        new_timeline = [
            snap.model_copy(update={"fabula_time": _map(snap.fabula_time) or snap.fabula_time})
            for snap in ent.state_timeline
        ]
        new_entities[eid] = ent.model_copy(update={
            "beliefs": new_beliefs,
            "state_timeline": new_timeline,
        })

    logger.info(
        "[Normalize] Rescaled %d unique fabula_time values (median event gap %d → %d).",
        len(nonzero_sorted), median_diff, spacing,
    )

    # Remap world trait snapshot fabula_times
    new_world_traits = {}
    for wid, wt in ws.world_traits.items():
        new_wt_timeline = [
            snap.model_copy(update={"fabula_time": _map(snap.fabula_time) or snap.fabula_time})
            for snap in wt.state_timeline
        ]
        new_world_traits[wid] = wt.model_copy(update={"state_timeline": new_wt_timeline})

    return WorldStateV1(
        locations=ws.locations,
        objects=ws.objects,
        entities=new_entities,
        events=new_events,
        world_traits=new_world_traits,
        causal_topology=new_causal,
        spatial_topology=new_spatial,
        channels=new_channels,
        social_topology=new_social,
    )


# =====================================================================
# Step 3 — Assembly + Validation
# =====================================================================


def _merge_utterances_into_events(
    physics_events: List[EventNode],
    utterance_events: List[EventNode],
    *,
    chunk_label: str,
) -> List[EventNode]:
    """Append Social-agent utterance events onto the Physics event list.

    Drops any utterance whose id collides with a Physics event id (the
    validator ought to have renamed it via the EVT_UTT_ prefix rule, but
    we belt-and-brace here so that downstream code never sees a
    duplicate id). Returns a new list; ``physics_events`` is not
    mutated.
    """
    merged: List[EventNode] = list(physics_events)
    if not utterance_events:
        return merged
    existing_ids = {e.id for e in merged}
    for uev in utterance_events:
        if uev.id in existing_ids:
            logger.info(
                "[%s] dropping utterance '%s' — id collides with a Physics event.",
                chunk_label, uev.id,
            )
            continue
        merged.append(uev)
        existing_ids.add(uev.id)
    return merged


def _deduplicate_social(edges: List[RelationshipEdge]) -> List[RelationshipEdge]:
    """Merge multiple edges for the same (source, target) pair into one.

    Under the per-metric ``RelationshipEdge`` schema, two extractions of
    the same dyad may carry *different* observed axes (e.g. one chunk
    only mutated ``fear``, another only ``power_dynamic``). Naively
    keeping the most recent whole edge would discard the older axis.
    Instead we merge per-axis, picking the metric with the larger
    ``last_updated_fabula`` for each axis independently.
    """
    # Branch-aware key: same (src, tgt) on factual vs shadow must NOT
    # collapse — Pearl Rung-2/3 forks legitimately produce parallel
    # edges with the same endpoints on the shadow branch.
    best: dict[tuple[str, str, str], RelationshipEdge] = {}
    for e in edges:
        key = (
            e.source_entity_id,
            e.target_entity_id,
            getattr(e, "world_id", "factual"),
        )
        if key not in best:
            best[key] = e
            continue
        merged_metrics = dict(best[key].metrics)
        for name, m in e.metrics.items():
            existing = merged_metrics.get(name)
            # ``>=`` (rather than ``>``) so a later-appended metric at
            # the same fabula tick wins. Pipeline appends the sandbox
            # bridge's ``RelationshipEdge``s *after* the prose
            # extractor's edges, so this lets the deterministic
            # physics value override the LLM extractor's reading on
            # ties without needing to fudge ``last_updated_fabula``
            # into the future (which would break time-slicing).
            if existing is None or m.last_updated_fabula >= existing.last_updated_fabula:
                merged_metrics[name] = m
        best[key] = best[key].model_copy(update={"metrics": merged_metrics})
    return list(best.values())


def _warn_suspicious_mirror_dyads(edges: List[RelationshipEdge]) -> None:
    """Log a warning for dyads whose two directions carry identical
    metric values across the board.

    A real, observed dyad almost always has asymmetric metrics: A's
    affinity for B differs from B's for A; subordinate fears superior
    more than the reverse; ``power_dynamic`` is signed and should
    flip across the two directions. When the forward and reverse
    edges report the same numbers on every shared metric (with
    `power_dynamic` checked sign-aware: forward + reverse should sum
    to ~0), it almost always means an extractor or fixture author
    duplicated one perspective rather than reading both sides.

    This is a *warning only* — we do not mutate the data, because
    fixtures may legitimately encode symmetric dyads in rare cases
    (twin sisters, mirrored placeholder edges). The warning gives
    operators a hook to upgrade the extraction prompt or hand-edit
    the fixture.
    """
    by_pair: dict[tuple[str, str], RelationshipEdge] = {
        (e.source_entity_id, e.target_entity_id): e for e in edges
    }
    seen: set[frozenset[str]] = set()
    suspect: list[tuple[str, str, list[str]]] = []
    for (src, tgt), fwd in by_pair.items():
        key = frozenset({src, tgt})
        if key in seen:
            continue
        rev = by_pair.get((tgt, src))
        if rev is None:
            continue
        seen.add(key)
        shared = set(fwd.metrics.keys()) & set(rev.metrics.keys())
        if not shared:
            continue
        mirrored_axes: list[str] = []
        for axis in shared:
            f_val = float(fwd.metrics[axis].value)
            r_val = float(rev.metrics[axis].value)
            if axis == "power_dynamic":
                # Opposite signs expected — flag if values are equal
                # (same sign, same magnitude).
                if abs(f_val - r_val) < 1e-6 and abs(f_val) > 1e-6:
                    mirrored_axes.append(axis)
            else:
                if abs(f_val - r_val) < 1e-6 and abs(f_val) > 1e-6:
                    mirrored_axes.append(axis)
        if mirrored_axes and len(mirrored_axes) == len(shared):
            suspect.append((src, tgt, mirrored_axes))
    if suspect:
        logger.warning(
            "[Asymmetry] %d dyad(s) have identical bidirectional "
            "metrics — likely lazy mirroring rather than real "
            "two-sided extraction: %s",
            len(suspect),
            ", ".join(f"{a}↔{b} ({'/'.join(ax)})" for a, b, ax in suspect[:8])
            + ("…" if len(suspect) > 8 else ""),
        )


def _deduplicate_spatial(edges: List[SpatialEdge]) -> List[SpatialEdge]:
    """Merge spatial edges per (source, target), preserving lifecycle state.

    Two extractor passes can describe the same passage with different
    lifecycle facts: one chunk may report the door as initially
    traversable (``established_at_fabula=0``), a later chunk may report
    it as locked from a particular tick (``is_locked=True``,
    ``barrier_item_id``), and a still-later chunk may report it as
    destroyed (``destroyed_at_fabula``). The previous implementation
    keyed only on ``(source, target)`` and let the latest-established
    edge win, silently dropping the lock and destruction facts.

    Strategy: keep the *earliest-established* edge per pair (so the
    passage's birth tick is preserved) and merge in any subsequent
    edge's lock and destruction facts.
    """
    # Branch-aware key: factual and shadow spatial edges over the same
    # passage must remain distinct so a counterfactual lock/destruction
    # never bleeds into the factual world (and vice versa).
    by_pair: dict[tuple[str, str, str], List[SpatialEdge]] = {}
    for e in edges:
        wid = getattr(e, "world_id", "factual")
        by_pair.setdefault((e.source_id, e.target_id, wid), []).append(e)
    merged: List[SpatialEdge] = []
    for pair, group in by_pair.items():
        # Sort by established_at_fabula so the earliest is canonical.
        group.sort(key=lambda e: e.established_at_fabula)
        canonical = group[0]
        update: dict = {}
        # Lock semantics: any pass reporting locked=True wins (a
        # passage explicitly described as locked at any point should
        # not be silently treated as freely traversable).
        for e in group[1:]:
            if e.is_locked and not canonical.is_locked:
                update["is_locked"] = True
                if e.barrier_item_id and not canonical.barrier_item_id:
                    update["barrier_item_id"] = e.barrier_item_id
            elif e.barrier_item_id and not canonical.barrier_item_id and not update.get("barrier_item_id"):
                update["barrier_item_id"] = e.barrier_item_id
        # Destruction: take the earliest non-null destroyed_at_fabula
        # across the group (if multiple chunks report destruction the
        # earliest tick is the one that fires).
        destroyed_ticks = [
            e.destroyed_at_fabula for e in group
            if e.destroyed_at_fabula is not None
        ]
        if destroyed_ticks:
            earliest = min(destroyed_ticks)
            if canonical.destroyed_at_fabula is None or earliest < canonical.destroyed_at_fabula:
                update["destroyed_at_fabula"] = earliest
        merged.append(canonical.model_copy(update=update) if update else canonical)
    return merged


def _deduplicate_causal(
    edges: List[CausalEdge],
    *,
    fabula_tolerance: int = 1,
) -> List[CausalEdge]:
    """Deduplicate causal edges by full semantic identity.

    Two ``mutation`` edges from the same event onto the same target
    entity but with *different ``trait_target``* (e.g. EVT_MURDER
    → ENT_MACBETH on ``guilt`` vs on ``ambition``) are NOT
    duplicates — they describe different state changes and must both
    survive. The same applies to ``mutation_social`` edges that
    differ on ``rel_counterpart_id`` (Macbeth's bond toward Banquo
    vs toward Lady Macbeth) and to edges that differ on
    ``mechanism`` (a kinetic vs psychological consequence of the
    same trigger).

    Keeps the edge with the highest ``causal_force`` when duplicates
    are found (the stronger signal wins).

    A second pass collapses neighbouring duplicates whose
    ``fabula_time`` differs by at most ``fabula_tolerance`` ticks
    (default ``1``). This absorbs LLM tick-jitter on re-extraction
    of the same causal arc, which previously survived as separate
    edges because the exact ``fabula_time`` differed by a single
    tick. Set ``fabula_tolerance=0`` to restore strict dedup.
    """
    def _key(e: CausalEdge) -> tuple:
        # Branch-aware: the same causal arc on factual vs shadow must
        # remain a distinct edge so Pearl Rung-2/3 forks survive merge.
        return (
            e.source_id,
            e.target_id,
            e.causality_type,
            e.trait_target,
            e.rel_counterpart_id,
            e.mechanism,
            e.fabula_time,
            getattr(e, "world_id", "factual"),
        )

    best: dict[tuple, CausalEdge] = {}
    for e in edges:
        key = _key(e)
        if key not in best or e.causal_force > best[key].causal_force:
            best[key] = e
    deduped = list(best.values())
    if fabula_tolerance <= 0:
        return deduped

    # Second pass: collapse neighbouring (source, target, type,
    # trait_target, rel_counterpart_id, mechanism) edges whose
    # fabula_time is within tolerance, keeping the higher force.
    deduped.sort(key=lambda e: (
        e.source_id, e.target_id, e.causality_type,
        e.trait_target or "", e.rel_counterpart_id or "", e.mechanism,
        getattr(e, "world_id", "factual"),
        e.fabula_time,
    ))
    collapsed: List[CausalEdge] = []
    for e in deduped:
        if collapsed:
            prev = collapsed[-1]
            if (
                prev.source_id == e.source_id
                and prev.target_id == e.target_id
                and prev.causality_type == e.causality_type
                and prev.trait_target == e.trait_target
                and prev.rel_counterpart_id == e.rel_counterpart_id
                and prev.mechanism == e.mechanism
                and getattr(prev, "world_id", "factual")
                    == getattr(e, "world_id", "factual")
                and abs(e.fabula_time - prev.fabula_time) <= fabula_tolerance
            ):
                if e.causal_force > prev.causal_force:
                    collapsed[-1] = e
                continue
        collapsed.append(e)
    return collapsed


def _deduplicate_channels_with_map(
    channel_dicts: List[Dict[str, Channel]],
    fabula_tolerance: int = 2,
) -> Tuple[Dict[str, Channel], Dict[str, str]]:
    """Merge per-chunk Channel dicts and also return an old→canonical id map.

    Keyed on ``(medium, sorted(participant_ids), directionality,
    established_at_fabula // max(1, fabula_tolerance))`` rather than the
    LLM-generated ``CHN_`` id, because two chunks may each invent their
    own id for the same standing capability. Directionality is part of
    the key so a duplex channel and a broadcast channel (e.g. a public
    proclamation vs a private chat) over the same participants are
    NOT collapsed.

    ``fabula_tolerance`` (default 2) bucketises ``established_at_fabula``
    so two chunks that report the same channel with slightly jittered
    establishment ticks (a common LLM artefact) still collapse. The
    earliest tick within a merged group becomes the canonical
    ``established_at_fabula``. Set to 0 to disable bucketing.

    When two chunks describe the same channel, ``intelligibility``
    maps are *merged* per-participant (later wins on collisions);
    if both are non-empty the merged result preserves keys from
    both chunks. ``terminated_at_fabula`` collapses to the earliest
    non-null tick (the channel actually goes dead at the first
    reported termination).

    The returned ``forwarding_map`` lets callers rewrite every
    ``EventNode.via_channel_id`` and ``Belief.acquired_via_channel_id``
    that pointed at a now-collapsed id, so dedup never silently orphans
    those references (which used to be nulled by ``_auto_repair``).
    """
    bucket_size = max(1, fabula_tolerance)
    best: dict[tuple, Channel] = {}
    # Track every id ever seen for each shape-key so the forwarding map
    # covers every collapsed alias, not just the most recent.
    aliases: dict[tuple, list[str]] = {}
    for chunk_channels in channel_dicts:
        for ch in chunk_channels.values():
            # Branch-aware bucket: a shadow-branch channel with the
            # same medium/participants/directionality as a factual
            # channel must NOT collapse — they describe parallel
            # capabilities in distinct AMWN branches.
            key = (
                ch.medium,
                tuple(sorted(ch.participant_ids)),
                ch.directionality,
                ch.established_at_fabula // bucket_size,
                getattr(ch, "world_id", "factual"),
            )
            aliases.setdefault(key, []).append(ch.id)
            existing = best.get(key)
            if existing is None:
                best[key] = ch
                continue
            # Merge intelligibility maps (union of keys; later value
            # wins on key collision so the more recent extraction's
            # decode probability survives).
            merged_intel = dict(existing.intelligibility)
            merged_intel.update(ch.intelligibility)
            # Earliest non-null termination wins.
            term_candidates = [
                t for t in (existing.terminated_at_fabula, ch.terminated_at_fabula)
                if t is not None
            ]
            merged_term: Optional[int] = min(term_candidates) if term_candidates else None
            # Earliest non-null syuzhet discovery wins (the channel
            # becomes audience-knowledge at the first reveal beat;
            # later re-mentions don't push the discovery later).
            disc_candidates = [
                d for d in (
                    getattr(existing, "discovered_at_syuzhet", None),
                    getattr(ch, "discovered_at_syuzhet", None),
                )
                if d is not None
            ]
            merged_disc: Optional[int] = min(disc_candidates) if disc_candidates else None
            # Earliest establishment tick wins (channel exists from the
            # earliest report onward; later jittered re-reports were
            # the LLM observing the same standing capability later).
            merged_estab = min(existing.established_at_fabula, ch.established_at_fabula)
            best[key] = ch.model_copy(update={
                "intelligibility": merged_intel,
                "terminated_at_fabula": merged_term,
                "discovered_at_syuzhet": merged_disc,
                "established_at_fabula": merged_estab,
            })
    deduped = {ch.id: ch for ch in best.values()}
    forwarding: Dict[str, str] = {}
    for key, ids in aliases.items():
        canonical = best[key].id
        for old in ids:
            if old != canonical:
                forwarding[old] = canonical
    return deduped, forwarding


def _deduplicate_channels(channel_dicts: List[Dict[str, Channel]]) -> Dict[str, Channel]:
    """Backwards-compatible shim: returns just the deduped dict.

    Prefer :func:`_deduplicate_channels_with_map` at call sites that
    can apply the forwarding map to ``via_channel_id`` /
    ``acquired_via_channel_id`` references.
    """
    deduped, _ = _deduplicate_channels_with_map(channel_dicts)
    return deduped


def _apply_channel_forwarding(
    forwarding: Dict[str, str],
    *,
    events: List[EventNode],
    entity_updates: Optional[List["EntityUpdate"]] = None,
) -> None:
    """Rewrite ``via_channel_id`` and ``Belief.acquired_via_channel_id`` in place.

    No-op when ``forwarding`` is empty. ``events`` and any
    ``entity_updates[*].new_beliefs`` lists are mutated; their
    container objects are replaced via ``model_copy`` so we don't rely
    on Pydantic's mutability semantics for nested models.
    """
    if not forwarding:
        return
    for i, evt in enumerate(events):
        if evt.via_channel_id and evt.via_channel_id in forwarding:
            events[i] = evt.model_copy(update={
                "via_channel_id": forwarding[evt.via_channel_id],
            })
    if entity_updates:
        for j, eu in enumerate(entity_updates):
            new_beliefs = eu.new_beliefs
            replaced_any = False
            rebuilt: List[Belief] = []
            for b in new_beliefs:
                if (
                    b.acquired_via_channel_id
                    and b.acquired_via_channel_id in forwarding
                ):
                    rebuilt.append(b.model_copy(update={
                        "acquired_via_channel_id": forwarding[b.acquired_via_channel_id],
                    }))
                    replaced_any = True
                else:
                    rebuilt.append(b)
            if replaced_any:
                entity_updates[j] = eu.model_copy(update={"new_beliefs": rebuilt})


# Public aliases for reuse outside the ingestion pipeline
deduplicate_social = _deduplicate_social
deduplicate_spatial = _deduplicate_spatial
deduplicate_causal = _deduplicate_causal
deduplicate_channels = _deduplicate_channels


def _snapshot_sort_key(s) -> tuple:
    """Stable, deterministic sort key for snapshot lists.

    Primary: ``fabula_time``. Secondary keys break ties when two extraction
    runs (or correction patches) produce snapshots with identical fabula
    times — without them, sort order depends on insertion order, which is
    non-deterministic under async chunk processing. Works for both
    :class:`EntityStateSnapshot` and :class:`WorldTraitSnapshot`.
    """
    return (
        getattr(s, "fabula_time", 0),
        getattr(s, "triggered_by", None) or "",
        getattr(s, "status", None) or "",
        getattr(s, "location_id", None) or "",
        len(getattr(s, "traits", None) or {}),
        len(getattr(s, "beliefs_added", None) or []),
    )


def _coalesce_snapshots(
    snaps: List[EntityStateSnapshot],
) -> List[EntityStateSnapshot]:
    """Merge same-fabula_time snapshots into a single deterministic snap.

    When two chunks emit an EntityUpdate for the same (entity,
    fabula_time) pair, the resulting EntityStateSnapshot list contains
    both records and replay-order becomes extraction-order dependent
    (audit item #10). This coalesces them per-tick using a stable rule
    set and returns the list time-sorted. Order within the same tick
    is preserved: the *first* snapshot at a tick keeps its position
    after merging.
    """
    if not snaps:
        return snaps
    by_tick: Dict[int, List[EntityStateSnapshot]] = {}
    order: List[int] = []
    for s in snaps:
        if s.fabula_time not in by_tick:
            order.append(s.fabula_time)
            by_tick[s.fabula_time] = []
        by_tick[s.fabula_time].append(s)

    coalesced: List[EntityStateSnapshot] = []
    for tick in sorted(set(order)):
        group = by_tick[tick]
        if len(group) == 1:
            coalesced.append(group[0])
            continue
        # All snapshots in a group MUST share a world_id; if a chunk
        # mixed factual + shadow snapshots at the same tick that is
        # itself a bug we want surfaced loudly rather than silently
        # retagged. Default to the group's first world_id.
        group_world_ids = {getattr(s, "world_id", "factual") for s in group}
        if len(group_world_ids) > 1:
            logger.warning(
                "_coalesce_snapshots: tick %d has mixed world_ids %s; "
                "keeping snapshots un-merged to preserve branch tagging.",
                tick, group_world_ids,
            )
            coalesced.extend(group)
            continue
        merged_world_id = next(iter(group_world_ids))
        # Merge fields:
        merged_traits: dict = {}
        # Audit fix #9: when two snapshots at the same tick disagree
        # on the *same* trait key, prefer the snapshot with more
        # corroborating evidence (longer beliefs_added list, then
        # longest invalidated list, then richest concrete-state
        # signal). The legacy "later wins" rule silently lost
        # whichever value the deterministic chunk-iteration order
        # happened to put first; we now log every conflict so the
        # ingestion-warnings panel surfaces them.
        def _evidence_score(item: Tuple[int, EntityStateSnapshot]) -> tuple:
            idx, s = item
            return (
                len(s.beliefs_added or []),
                len(s.beliefs_invalidated or []),
                # Snapshots that name a triggering event / status /
                # location are richer than bare trait deltas.
                1 if s.triggered_by else 0,
                1 if s.status else 0,
                1 if s.location_id else 0,
                # Final tie-break: original input position (stable,
                # semantically meaningful — earlier extraction order
                # wins ties so output is deterministic across runs).
                -idx,
            )
        ranked_group = [
            s for _, s in sorted(
                enumerate(group), key=_evidence_score, reverse=True,
            )
        ]
        merged_beliefs_added: List[Belief] = []
        seen_belief_keys: set = set()
        merged_invalidated: List[str] = []
        seen_invalid: set = set()
        first_triggered_by: Optional[str] = None
        first_status: Optional[str] = None
        first_location_id: Optional[str] = None
        # Track which snapshot supplied each trait so conflicts on the
        # same key can be reported with provenance.
        trait_provenance: Dict[str, str] = {}
        for s in ranked_group:
            for k, v in (s.traits or {}).items():
                if k in merged_traits and merged_traits[k] != v:
                    logger.warning(
                        "[Coalesce-Conflict] Tick %d: trait %r set to "
                        "different values (%r from %s vs %r from %s) \u2014 "
                        "keeping the higher-evidence snapshot's value.",
                        tick, k,
                        merged_traits[k], trait_provenance.get(k, "?"),
                        v, s.triggered_by or "?",
                    )
                    continue  # higher-evidence value already in place
                merged_traits[k] = v
                trait_provenance[k] = s.triggered_by or "?"
            for b in s.beliefs_added or []:
                # Dedup beliefs by (target_id, perceived_state,
                # acquired_via_event_id, acquired_via_channel_id) so
                # the *same* belief emitted twice is collapsed but two
                # acquisitions of the same proposition through
                # different provenance (e.g. directly witnessed AND
                # later told) are both kept.
                key = (
                    b.target_id,
                    b.perceived_state,
                    getattr(b, "acquired_via_event_id", None),
                    getattr(b, "acquired_via_channel_id", None),
                )
                if key in seen_belief_keys:
                    continue
                seen_belief_keys.add(key)
                merged_beliefs_added.append(b)
            for tgt in s.beliefs_invalidated or []:
                if tgt in seen_invalid:
                    continue
                seen_invalid.add(tgt)
                merged_invalidated.append(tgt)
            if first_triggered_by is None and s.triggered_by:
                first_triggered_by = s.triggered_by
            elif (
                s.triggered_by
                and first_triggered_by
                and s.triggered_by != first_triggered_by
            ):
                logger.warning(
                    "_coalesce_snapshots: tick %d has conflicting "
                    "triggered_by values (%s vs %s); keeping first.",
                    tick, first_triggered_by, s.triggered_by,
                )
            if first_status is None and s.status:
                first_status = s.status
            if first_location_id is None and s.location_id:
                first_location_id = s.location_id

        coalesced.append(EntityStateSnapshot(
            world_id=merged_world_id,
            fabula_time=tick,
            triggered_by=first_triggered_by,
            traits=merged_traits,
            beliefs_added=merged_beliefs_added,
            beliefs_invalidated=merged_invalidated,
            status=first_status,
            location_id=first_location_id,
        ))
    return coalesced


def assemble_world_state(
    register: GlobalRegister,
    topologies: List[ChunkTopology],
) -> WorldStateV1:
    """
    Merge the Step 1 register and Step 2 chunk topologies into a
    single ``WorldStateV1``.

    Social and spatial edges are deduplicated per (source, target) pair,
    keeping the most recently updated edge.
    """
    events: List[EventNode] = []
    causal_topology: List[CausalEdge] = []
    channel_dicts: List[Dict[str, Channel]] = []
    social_topology: List[RelationshipEdge] = []
    spatial_topology: List[SpatialEdge] = []

    for topo in topologies:
        events.extend(topo.events)
        causal_topology.extend(topo.causal_topology)
        channel_dicts.append(topo.channels)
        social_topology.extend(topo.social_topology)
        spatial_topology.extend(topo.spatial_topology)

    # --- Channel dedup with forwarding map ---
    # Done up-front so we can rewrite stale via_channel_id /
    # acquired_via_channel_id references on events and entity_updates
    # before they get baked into the world state. Otherwise dedup would
    # silently orphan those references and ``_auto_repair`` would null
    # them out (lossy).
    raw_channel_count = sum(len(c) for c in channel_dicts)
    channels, channel_forwarding = _deduplicate_channels_with_map(channel_dicts)
    if channel_forwarding:
        # Rewrite events first (utterance.via_channel_id), then
        # mutate each topology's entity_updates so their beliefs pick
        # up the new channel ids before being folded into snapshots.
        _apply_channel_forwarding(channel_forwarding, events=events)
        for topo in topologies:
            _apply_channel_forwarding(
                channel_forwarding,
                events=[],  # events already covered globally
                entity_updates=topo.entity_updates,
            )
        logger.info(
            "[Step 3] Channel dedup forwarding: %d alias(es) rewritten.",
            len(channel_forwarding),
        )

    # Collect entity updates from all chunks into state_timeline
    all_entity_updates: Dict[str, List[EntityStateSnapshot]] = {}
    for topo in topologies:
        for eu in topo.entity_updates:
            snap = EntityStateSnapshot(
                fabula_time=eu.fabula_time,
                triggered_by=eu.triggered_by,
                traits=eu.trait_updates,
                beliefs_added=eu.new_beliefs,
                beliefs_invalidated=eu.invalidated_belief_targets,
                status=eu.new_status,
                location_id=eu.new_location_id,
            )
            all_entity_updates.setdefault(eu.entity_id, []).append(snap)

    # Coalesce same-(entity, fabula_time) snapshots into a single
    # deterministic snapshot. Without this, two updates emitted by
    # different chunks for the same tick get appended verbatim and
    # replay order becomes extraction-order dependent \u2014 producing
    # unstable reconstructed state and double-applied belief mutations
    # (audit item #10). Merge rules per field:
    #   - traits: later (later in input order) wins per key
    #   - beliefs_added: union, deduped by target_id
    #   - beliefs_invalidated: union
    #   - triggered_by / status / location_id: first non-null wins
    for eid, snaps in all_entity_updates.items():
        all_entity_updates[eid] = _coalesce_snapshots(snaps)

    # Sort events chronologically. Add stable secondary keys so two
    # extraction runs over the same input produce byte-identical AMWN
    # ordering even when several events share a fabula tick (a common
    # case at chapter boundaries where multiple things happen "now").
    # Without these tie-breakers ordering depends on chunk-extraction
    # insertion order, which under async parallelism is itself
    # non-deterministic.
    events.sort(key=lambda e: (e.fabula_time, e.id))
    causal_topology.sort(
        key=lambda c: (c.fabula_time, c.source_id, c.target_id, c.causality_type)
    )

    # Deduplicate relationship, spatial, causal across chunks (channels
    # were deduped earlier so the forwarding map could rewrite events).
    # Reverse-direction mirroring of one-sided dyads now happens in
    # ``WorldStateV1``'s post-init validator so every consumer (ingestion,
    # example_worlds fixtures, snapshot reconstructions, test fixtures)
    # sees the same mirrored shape — no explicit call here.
    social_before = len(social_topology)
    social_topology = _deduplicate_social(social_topology)
    # Asymmetry sanity check: flag dyads where both directions exist
    # AND every shared metric carries an *identical* value (sign-aware
    # for power_dynamic, which should be opposite-signed). Identical
    # bidirectional values are almost always a lazy mirror of one
    # extraction rather than two real readings — they suppress the
    # asymmetric structure the affective dashboard depends on. We log
    # rather than mutate so authored fixtures stay authoritative.
    _warn_suspicious_mirror_dyads(social_topology)
    spatial_before = len(spatial_topology)
    spatial_topology = _deduplicate_spatial(spatial_topology)
    causal_before = len(causal_topology)
    causal_topology = _deduplicate_causal(causal_topology)
    deduped_parts = []
    if social_before != len(social_topology):
        deduped_parts.append(f"social {social_before}→{len(social_topology)}")
    if spatial_before != len(spatial_topology):
        deduped_parts.append(f"spatial {spatial_before}→{len(spatial_topology)}")
    if causal_before != len(causal_topology):
        deduped_parts.append(f"causal {causal_before}→{len(causal_topology)}")
    if raw_channel_count != len(channels):
        deduped_parts.append(f"channels {raw_channel_count}→{len(channels)}")
    if deduped_parts:
        logger.info("[Step 3] Deduplicated edges: %s.", ", ".join(deduped_parts))

    ws = WorldStateV1(
        locations=register.locations,
        objects=register.objects,
        entities={
            eid: (
                ent.model_copy(update={"state_timeline": sorted(all_entity_updates[eid], key=_snapshot_sort_key)})
                if eid in all_entity_updates
                else ent
            )
            for eid, ent in register.entities.items()
        },
        world_traits=register.world_traits,
        events=events,
        causal_topology=causal_topology,
        spatial_topology=spatial_topology,
        channels=channels,
        social_topology=social_topology,
    )
    utterance_count = sum(1 for e in events if e.event_type == "utterance")
    logger.info(
        "[Step 3] Assembled WorldStateV1 — %d events (%d utterances), %d causal, %d social, "
        "%d spatial, %d channels, %d world traits.",
        len(ws.events), utterance_count, len(ws.causal_topology), len(ws.social_topology),
        len(ws.spatial_topology), len(ws.channels),
        len(ws.world_traits),
    )

    # --- Tier 1 #3: trajectory population check ---------------------
    # An entity that participates in many events but ends up with an
    # empty state_timeline almost always means the consequences pass
    # silently no-opped. Surface as a warning so operators don't
    # discover flat affective curves only at dashboard render time.
    event_participation: Dict[str, int] = {}
    for e in events:
        for eid in (e.actor_ids or []) + (e.target_ids or []):
            event_participation[eid] = event_participation.get(eid, 0) + 1
    flat_entities: List[Tuple[str, int]] = []
    for eid, ent in ws.entities.items():
        n_events = event_participation.get(eid, 0)
        if n_events >= 3 and not ent.state_timeline:
            flat_entities.append((eid, n_events))
    if flat_entities:
        flat_entities.sort(key=lambda p: -p[1])
        sample = ", ".join(f"{eid}({n})" for eid, n in flat_entities[:5])
        logger.warning(
            "[Pipeline-Skip] %d entit(y/ies) participate in 3+ events "
            "but have an empty state_timeline \u2014 consequences pass "
            "may have no-opped. Top: %s",
            len(flat_entities), sample,
        )

    # --- Tier 2 #8: belief-consistency check ------------------------
    # For each EntityUpdate's invalidated_belief_targets, verify the
    # entity actually held a belief with that target_id at or before
    # the snapshot's fabula_time. Invalidations of never-held beliefs
    # are no-ops at best and indicators of confused extraction at
    # worst (the LLM is mixing up which entity held the belief, or
    # confabulating beliefs the text never established).
    belief_inconsistencies: List[str] = []
    for eid, ent in ws.entities.items():
        # Build cumulative belief target history at each tick.
        held: Set[str] = {b.target_id for b in (ent.beliefs or [])}
        for snap in ent.state_timeline:
            for added in (snap.beliefs_added or []):
                held.add(added.target_id)
            for invalidated in (snap.beliefs_invalidated or []):
                if invalidated not in held:
                    belief_inconsistencies.append(
                        f"{eid}@fabula={snap.fabula_time}: invalidates "
                        f"belief target {invalidated!r} never held"
                    )
                else:
                    held.discard(invalidated)
    if belief_inconsistencies:
        logger.warning(
            "[Belief-Consistency] %d invalidation(s) target beliefs "
            "the entity never held. Sample: %s",
            len(belief_inconsistencies),
            belief_inconsistencies[:5],
        )

    return ws


# =====================================================================
# Phase C — Affect Reconciler
#
# Folds the Phase A3 ``PropositionCatalogue`` and the per-chunk Phase B4
# affect outputs (``ChunkTopology.proposition_snapshots`` /
# ``proposition_truth_commits`` / ``concern_snapshots`` /
# ``new_concern_seeds``) into the assembled ``WorldStateV1``:
#
#   * Catalogue propositions land on ``WorldStateV1.propositions``,
#     stable-merged with any pre-existing ones (catalogue framing
#     wins; ``truth_at_fabula`` keeps prior commits).
#   * ``EventNode.resolves_proposition_ids`` /
#     ``asserts_proposition_id`` (with ``truth_value``) /
#     ``denies_proposition_id`` are swept into
#     ``Proposition.truth_at_fabula`` so the per-chunk extractors that
#     wrote the link at extraction time become world-level truths
#     without a second LLM call.
#   * ``ChunkPropositionSnapshot`` / ``ChunkConcernSnapshot`` payloads
#     are stripped of their wire-format discriminator
#     (``proposition_id`` / ``concern_id``) and routed onto the matching
#     proposition / concern's ``state_timeline``, then run through
#     :func:`_coalesce_timeline` for deterministic replay.
#   * Catalogue ``concern_seeds`` and per-chunk ``new_concern_seeds``
#     are materialised as ``Concern`` records on the named entity,
#     deduped on ``(entity_id, proposition_id, polarity)``.
#
# The function is intentionally *additive*: legacy worlds without a
# catalogue or affect outputs round-trip unchanged. Idempotent given
# the same inputs (re-running collapses duplicates).
# =====================================================================


def _truth_value_to_bool(tv: Optional[str]) -> Optional[bool]:
    """Map an utterance ``truth_value`` to the ``truth_at_fabula`` bool.

    'true' / 'false' commit; 'unknown' / 'performative' / None do not
    contribute to the proposition's ground-truth ledger (an unreliable
    or non-truth-apt utterance is captured by the per-agent belief
    layer, not by world truth).
    """
    if tv == "true":
        return True
    if tv == "false":
        return False
    return None


def reconcile_affect(
    world: WorldStateV1,
    catalogue: Optional["PropositionCatalogue"],
    topologies: List[ChunkTopology],
) -> WorldStateV1:
    """Phase C: fold catalogue + per-chunk affect outputs into the world.

    Mutates ``world`` in place (and returns it) for symmetry with the
    other post-assembly passes (``_normalize_fabula_times``,
    ``_auto_repair``, ``extract_world_trait_timelines_async``).
    """
    # Lazy import — affect_unification depends on shadow_loom.models
    # which is a leaf, so no cycle, but we keep the import local to
    # match the rest of the post-assembly call sites in this file.
    from shadow_loom.affect_unification import _coalesce_timeline

    # ----------------------------------------------------------------
    # 1. Merge catalogue propositions into ``world.propositions``.
    # ----------------------------------------------------------------
    prop_index: Dict[str, Proposition] = {
        p.proposition_id: p for p in world.propositions
    }
    if catalogue is not None:
        for cat_prop in catalogue.propositions:
            existing = prop_index.get(cat_prop.proposition_id)
            if existing is None:
                prop_index[cat_prop.proposition_id] = cat_prop.model_copy(deep=True)
                continue
            # Catalogue wins on framing fields (description, kind,
            # referents, audience prior, stakes); existing keeps its
            # accumulated state_timeline / truth_at_fabula. We do
            # NOT clobber state_timeline / truth_at_fabula because a
            # legacy ``synthesise_propositions`` pass may have written
            # them earlier.
            prop_index[cat_prop.proposition_id] = existing.model_copy(update={
                "kind": cat_prop.kind,
                "referent_ids": list(cat_prop.referent_ids),
                "description": cat_prop.description,
                "audience_default_prior": cat_prop.audience_default_prior,
                "stakes": cat_prop.stakes,
            })

    # ----------------------------------------------------------------
    # 2. Sweep events for inline PROP links and commit truth_at_fabula.
    #    These came from the per-chunk Physics / Social agents writing
    #    ``resolves_proposition_ids`` / ``asserts_proposition_id`` /
    #    ``denies_proposition_id`` at extraction time.
    # ----------------------------------------------------------------
    truth_writes: Dict[Tuple[str, int], bool] = {}
    for evt in world.events:
        for pid in evt.resolves_proposition_ids:
            if pid in prop_index:
                truth_writes[(pid, evt.fabula_time)] = True
        if evt.asserts_proposition_id and evt.asserts_proposition_id in prop_index:
            tv = _truth_value_to_bool(evt.truth_value)
            # Default for assertions without an explicit truth_value is
            # True (the speaker stands behind the claim); deny stays
            # False.
            if tv is None and evt.truth_value not in ("unknown", "performative"):
                tv = True
            if tv is not None:
                truth_writes[(evt.asserts_proposition_id, evt.fabula_time)] = tv
        if evt.denies_proposition_id and evt.denies_proposition_id in prop_index:
            truth_writes[(evt.denies_proposition_id, evt.fabula_time)] = False

    for (pid, fab), val in truth_writes.items():
        prop = prop_index[pid]
        # Last write at a tick wins, but we warn on flips so the
        # auditor can flag intentional Schr\u00f6dinger reveals separately
        # from extraction noise.
        existing_truth = prop.truth_at_fabula.get(fab)
        if existing_truth is not None and existing_truth != val:
            logger.info(
                "[Phase C] Truth flip on %s@fabula=%d: %s \u2192 %s "
                "(intentional Schr\u00f6dinger reveal or conflicting extractors).",
                pid, fab, existing_truth, val,
            )
        new_truth = dict(prop.truth_at_fabula)
        new_truth[fab] = val
        prop_index[pid] = prop.model_copy(update={"truth_at_fabula": new_truth})

    # ----------------------------------------------------------------
    # 3. Per-chunk affect outputs: proposition framing snapshots +
    #    concern drift snapshots + truth commits + new concern seeds.
    # ----------------------------------------------------------------
    chunk_truth_commits: Dict[Tuple[str, int], bool] = {}
    prop_snap_buckets: Dict[str, List[PropositionSnapshot]] = {}
    concern_snap_buckets: Dict[str, List[ConcernSnapshot]] = {}
    new_seed_records: List["ConcernSeed"] = []

    for topo in topologies:
        for ps in topo.proposition_snapshots:
            if ps.proposition_id not in prop_index:
                # Affect agent's output_validator should have already
                # dropped these, but legacy / replayed checkpoints may
                # still carry them. Drop silently with a debug log.
                logger.debug(
                    "[Phase C] Dropping prop_snapshot for unknown PROP_ id %s.",
                    ps.proposition_id,
                )
                continue
            prop_snap_buckets.setdefault(ps.proposition_id, []).append(
                PropositionSnapshot(
                    fabula_time=ps.fabula_time,
                    triggered_by=ps.triggered_by,
                    stakes=ps.stakes,
                    audience_default_prior=ps.audience_default_prior,
                    description=ps.description,
                )
            )
        for tc in topo.proposition_truth_commits:
            if tc.proposition_id not in prop_index:
                continue
            chunk_truth_commits[(tc.proposition_id, tc.fabula_time)] = tc.truth
        for cs in topo.concern_snapshots:
            concern_snap_buckets.setdefault(cs.concern_id, []).append(
                ConcernSnapshot(
                    fabula_time=cs.fabula_time,
                    triggered_by=cs.triggered_by,
                    salience=cs.salience,
                    polarity=cs.polarity,
                    activation_fabula_window=cs.activation_fabula_window,
                    counter_concern_ids=cs.counter_concern_ids,
                    kind=cs.kind,
                )
            )
        for ns in topo.new_concern_seeds:
            new_seed_records.append(ns)

    # Apply chunk-level truth commits (from explicit affect agent
    # output) over the inline-event ones — affect agent has full
    # context and may correct an incorrect inline default.
    for (pid, fab), val in chunk_truth_commits.items():
        prop = prop_index[pid]
        new_truth = dict(prop.truth_at_fabula)
        new_truth[fab] = val
        prop_index[pid] = prop.model_copy(update={"truth_at_fabula": new_truth})

    # ----------------------------------------------------------------
    # 4. Fold proposition snapshots onto each proposition's timeline
    #    and coalesce.
    # ----------------------------------------------------------------
    _PROP_DIFF_FIELDS = ("stakes", "audience_default_prior", "description")
    for pid, snaps in prop_snap_buckets.items():
        prop = prop_index[pid]
        merged = list(prop.state_timeline) + snaps
        merged = _coalesce_timeline(merged, _PROP_DIFF_FIELDS)
        prop_index[pid] = prop.model_copy(update={"state_timeline": merged})

    # Final propositions list \u2014 sorted by id for deterministic output.
    world.propositions = [prop_index[pid] for pid in sorted(prop_index)]

    # ----------------------------------------------------------------
    # 5. Concerns: materialise catalogue seeds + chunk new_concern_seeds
    #    onto entities, then fold concern snapshots onto each concern's
    #    timeline.
    # ----------------------------------------------------------------
    seed_records: List["ConcernSeed"] = []
    if catalogue is not None:
        seed_records.extend(catalogue.concern_seeds)
    seed_records.extend(new_seed_records)

    if seed_records:
        # Per-entity concern dedup index keyed on
        # (proposition_id, polarity) so re-runs collapse duplicates.
        for seed in seed_records:
            ent = world.entities.get(seed.entity_id)
            if ent is None:
                logger.debug(
                    "[Phase C] Dropping concern seed %s \u2014 entity %s missing.",
                    seed.concern_id, seed.entity_id,
                )
                continue
            existing_keys = {
                (c.proposition_id, c.polarity) for c in ent.concerns
            }
            if (seed.proposition_id, seed.polarity) in existing_keys:
                continue
            new_concern = Concern(
                concern_id=seed.concern_id,
                proposition_id=seed.proposition_id,
                polarity=seed.polarity,
                kind=seed.kind,
                salience=seed.baseline_salience,
                counter_concern_ids=list(seed.counter_concern_ids),
            )
            ent.concerns = list(ent.concerns) + [new_concern]

    # Apply concern snapshots (now that the concern records exist).
    _CONCERN_DIFF_FIELDS = (
        "salience", "polarity", "activation_fabula_window",
        "counter_concern_ids", "kind",
    )
    if concern_snap_buckets:
        # Build (entity_id, concern_id) \u2192 Concern lookup.
        concern_index: Dict[str, Tuple[str, Concern]] = {}
        for eid, ent in world.entities.items():
            for c in ent.concerns:
                concern_index[c.concern_id] = (eid, c)
        for ccn_id, snaps in concern_snap_buckets.items():
            entry = concern_index.get(ccn_id)
            if entry is None:
                logger.debug(
                    "[Phase C] Dropping %d concern_snapshot(s) for unknown CCN_ id %s.",
                    len(snaps), ccn_id,
                )
                continue
            eid, concern = entry
            merged = list(concern.state_timeline) + snaps
            merged = _coalesce_timeline(merged, _CONCERN_DIFF_FIELDS)
            updated = concern.model_copy(update={"state_timeline": merged})
            ent = world.entities[eid]
            ent.concerns = [
                updated if c.concern_id == ccn_id else c
                for c in ent.concerns
            ]

    return world


# =====================================================================
# Auto-Repair — programmatically fix broken links
# =====================================================================

def _auto_repair(ws: WorldStateV1) -> Tuple[WorldStateV1, List[str]]:
    """
    Programmatically repair a WorldStateV1 by removing broken edges
    and duplicate events. Returns (repaired_ws, list_of_repairs).

    This is inspired by GraphRAG's entity-summarization merging step
    but applied at the validation layer — strip provably broken
    references rather than forcing LLM re-extraction.
    """
    repairs: List[str] = []

    valid_ids = (
        set(ws.locations.keys())
        | set(ws.objects.keys())
        | set(ws.entities.keys())
        | set(ws.world_traits.keys())
        | {e.id for e in ws.events}
    )
    entity_ids = set(ws.entities.keys())
    location_ids = set(ws.locations.keys())
    node_ids = entity_ids | set(ws.objects.keys())

    # --- Deduplicate events (keep first occurrence) ---
    seen_evt: set[str] = set()
    deduped_events: List[EventNode] = []
    for evt in ws.events:
        if evt.id in seen_evt:
            repairs.append(f"Removed duplicate event '{evt.id}'.")
        else:
            seen_evt.add(evt.id)
            deduped_events.append(evt)

    # --- Fix broken event actor_ids / target_ids references ---
    object_ids = set(ws.objects.keys())
    world_trait_ids = set(ws.world_traits.keys())
    event_id_set = {e.id for e in deduped_events}
    clean_events: List[EventNode] = []
    for evt in deduped_events:
        updates: dict = {}
        # Match the validator's per-event-type allowlist (see
        # _programmatic_validation): utterance actors may include OBJ_
        # (a dossier, a telescreen broadcast etc.), and utterance
        # targets additionally include EVT_/WORLD_/LOC_. Stripping all
        # non-entity actors here would silently destroy valid
        # speech-act provenance.
        is_utterance = evt.event_type == "utterance"
        actor_allowed = (entity_ids | object_ids) if is_utterance else entity_ids
        target_allowed = (
            entity_ids | object_ids | event_id_set | world_trait_ids | location_ids
            if is_utterance
            else entity_ids | object_ids
        )
        bad_actors = [a for a in evt.actor_ids if a not in actor_allowed]
        if bad_actors:
            repairs.append(f"Removed invalid actor_ids {bad_actors} from event '{evt.id}'.")
            updates["actor_ids"] = [a for a in evt.actor_ids if a in actor_allowed]
        bad_targets = [t for t in evt.target_ids if t not in target_allowed]
        if bad_targets:
            repairs.append(f"Removed invalid target_ids {bad_targets} from event '{evt.id}'.")
            updates["target_ids"] = [t for t in evt.target_ids if t in target_allowed]
        clean_events.append(evt.model_copy(update=updates) if updates else evt)

    # --- Fix broken entity location_ids ---
    # Prefer NULL over a positive but-likely-wrong assignment: silently
    # planting an entity in "the first location" can create plausible-
    # looking but completely fabricated geography. Downstream readers
    # already handle ``location_id is None`` (entity off-page / unknown).
    new_entities_map: dict[str, Entity] = {}
    needs_rewrite = False
    for eid, ent in ws.entities.items():
        if ent.location_id is not None and ent.location_id not in location_ids:
            repairs.append(
                f"Cleared entity '{eid}' invalid location_id "
                f"'{ent.location_id}' (set to None — no safe fallback)."
            )
            new_entities_map[eid] = ent.model_copy(update={"location_id": None})
            needs_rewrite = True
        else:
            new_entities_map[eid] = ent
    if needs_rewrite:
        ws = ws.model_copy(update={"entities": new_entities_map})

    # --- Strip broken causal edges ---
    clean_causal: List[CausalEdge] = []
    for ce in ws.causal_topology:
        if ce.source_id not in valid_ids:
            repairs.append(f"Removed causal edge: source '{ce.source_id}' not in node set.")
        elif ce.target_id not in valid_ids:
            repairs.append(f"Removed causal edge: target '{ce.target_id}' not in node set.")
        elif ce.rel_counterpart_id and ce.rel_counterpart_id not in valid_ids:
            repairs.append(f"Removed causal edge: rel_counterpart_id '{ce.rel_counterpart_id}' not in node set.")
        else:
            clean_causal.append(ce)

    # --- Strip broken social edges ---
    clean_social: List[RelationshipEdge] = []
    for re_edge in ws.social_topology:
        if re_edge.source_entity_id not in entity_ids or re_edge.target_entity_id not in entity_ids:
            repairs.append(
                f"Removed relationship edge: '{re_edge.source_entity_id}' → '{re_edge.target_entity_id}'."
            )
        else:
            clean_social.append(re_edge)

    # --- Strip broken spatial edges ---
    clean_spatial: List[SpatialEdge] = []
    for se in ws.spatial_topology:
        if se.source_id not in location_ids or se.target_id not in location_ids:
            repairs.append(f"Removed spatial edge: '{se.source_id}' → '{se.target_id}'.")
        else:
            clean_spatial.append(se)

    # --- Strip broken channels ---
    clean_channels: Dict[str, Channel] = {}
    for cid, ch in ws.channels.items():
        valid_pids = [p for p in ch.participant_ids if p in node_ids]
        bad_pids = [p for p in ch.participant_ids if p not in node_ids]
        for bp in bad_pids:
            repairs.append(f"Removed channel '{cid}' participant '{bp}' (not in entities/objects).")
        if len(valid_pids) < 2:
            repairs.append(f"Removed channel '{cid}' (fewer than 2 valid participants).")
            continue
        if valid_pids != list(ch.participant_ids):
            pruned_intel = {k: v for k, v in ch.intelligibility.items() if k in valid_pids}
            clean_channels[cid] = ch.model_copy(update={
                "participant_ids": valid_pids,
                "intelligibility": pruned_intel,
            })
        else:
            clean_channels[cid] = ch

    # --- Strip utterance events whose via_channel_id no longer resolves ---
    valid_channel_ids = set(clean_channels.keys())
    repaired_events: List[EventNode] = []
    for evt in clean_events:
        if (
            evt.event_type == "utterance"
            and evt.via_channel_id
            and evt.via_channel_id not in valid_channel_ids
        ):
            repairs.append(
                f"Cleared dangling via_channel_id '{evt.via_channel_id}' on utterance '{evt.id}'."
            )
            repaired_events.append(evt.model_copy(update={"via_channel_id": None}))
        else:
            repaired_events.append(evt)
    clean_events = repaired_events

    # --- Fuzzy-fix entity state_timeline.triggered_by references ---
    # Common failure mode: a consequences-extractor pass coined an EVT_ ID
    # spelt slightly differently from the physics-extractor pass (e.g.
    # ``EVT_LARS_MASSACRE`` vs ``EVT_LAR_MASSACRE``). Without this pass
    # the dangling reference would be flagged as a hard error and the
    # whole world-state would be sent through the LLM correction loop —
    # historically a much more destructive operation than just renaming
    # one ID. We try a fuzzy resolve first; if that fails we null out
    # the reference (a warning, not an error).
    event_ids_now = {e.id for e in clean_events}
    new_entities_map: Dict[str, Entity] = {}
    entities_changed = False
    for eid, ent in ws.entities.items():
        new_timeline: List[EntityStateSnapshot] = []
        timeline_changed = False
        for snap in ent.state_timeline:
            if snap.triggered_by and snap.triggered_by not in event_ids_now:
                resolved = _fuzzy_resolve_id(snap.triggered_by, event_ids_now)
                if resolved:
                    repairs.append(
                        f"Repaired entity '{eid}' state_timeline triggered_by "
                        f"'{snap.triggered_by}' → '{resolved}'."
                    )
                    new_timeline.append(snap.model_copy(update={"triggered_by": resolved}))
                    timeline_changed = True
                    continue
                else:
                    repairs.append(
                        f"Cleared dangling triggered_by '{snap.triggered_by}' on "
                        f"entity '{eid}' state_timeline (no fuzzy match)."
                    )
                    new_timeline.append(snap.model_copy(update={"triggered_by": None}))
                    timeline_changed = True
                    continue
            new_timeline.append(snap)
        if timeline_changed:
            new_entities_map[eid] = ent.model_copy(update={"state_timeline": new_timeline})
            entities_changed = True
        else:
            new_entities_map[eid] = ent
    if entities_changed:
        ws = ws.model_copy(update={"entities": new_entities_map})

    # --- Fuzzy-fix world_trait state_timeline.triggered_by references ---
    new_world_traits_map: Dict[str, GlobalTrait] = {}
    world_traits_changed = False
    for wid, wt in ws.world_traits.items():
        new_wt_timeline: List[WorldTraitSnapshot] = []
        wt_timeline_changed = False
        for snap in wt.state_timeline:
            trig = getattr(snap, "triggered_by", None)
            if trig and trig not in event_ids_now:
                resolved = _fuzzy_resolve_id(trig, event_ids_now)
                if resolved:
                    repairs.append(
                        f"Repaired world_trait '{wid}' state_timeline triggered_by "
                        f"'{trig}' → '{resolved}'."
                    )
                    new_wt_timeline.append(snap.model_copy(update={"triggered_by": resolved}))
                    wt_timeline_changed = True
                    continue
                else:
                    repairs.append(
                        f"Cleared dangling triggered_by '{trig}' on world_trait "
                        f"'{wid}' state_timeline (no fuzzy match)."
                    )
                    new_wt_timeline.append(snap.model_copy(update={"triggered_by": None}))
                    wt_timeline_changed = True
                    continue
            new_wt_timeline.append(snap)
        if wt_timeline_changed:
            new_world_traits_map[wid] = wt.model_copy(update={"state_timeline": new_wt_timeline})
            world_traits_changed = True
        else:
            new_world_traits_map[wid] = wt
    if world_traits_changed:
        ws = ws.model_copy(update={"world_traits": new_world_traits_map})

    # --- Self-referencing social edges ---
    # The validator flags these as "contradiction" warnings; under any
    # reasonable reading they're extraction noise. Drop them here so
    # the LLM correction loop is never invoked for self-loops.
    cleaned_social: List[RelationshipEdge] = []
    for re_edge in clean_social:
        if re_edge.source_entity_id == re_edge.target_entity_id:
            repairs.append(
                f"Removed self-referencing social edge: "
                f"'{re_edge.source_entity_id}' \u2192 '{re_edge.target_entity_id}'."
            )
        else:
            cleaned_social.append(re_edge)
    clean_social = cleaned_social

    # --- Channel terminate-before-establish ---
    # Drop the impossible termination tick rather than the whole channel:
    # the channel itself is usually correctly extracted, only the
    # terminated_at_fabula is a hallucinated date.
    fixed_channels: Dict[str, Channel] = {}
    for cid, ch in clean_channels.items():
        if (
            ch.terminated_at_fabula is not None
            and ch.terminated_at_fabula < ch.established_at_fabula
        ):
            repairs.append(
                f"Cleared invalid terminated_at_fabula={ch.terminated_at_fabula} "
                f"on channel '{cid}' (predates established_at_fabula="
                f"{ch.established_at_fabula})."
            )
            fixed_channels[cid] = ch.model_copy(update={"terminated_at_fabula": None})
        else:
            fixed_channels[cid] = ch
    clean_channels = fixed_channels

    # --- Belief provenance: rewrite/null dangling acquired_via_event_id /
    # --- acquired_via_channel_id refs on entity beliefs and snapshots.
    valid_event_ids = {e.id for e in clean_events}
    valid_channel_ids_set = set(clean_channels.keys())

    def _repair_belief(b: Belief, owner: str, *, in_snapshot_at: Optional[int] = None) -> Belief:
        update: dict = {}
        if b.acquired_via_event_id and b.acquired_via_event_id not in valid_event_ids:
            resolved = _fuzzy_resolve_id(b.acquired_via_event_id, valid_event_ids)
            label = f"belief about '{b.target_id}' on '{owner}'"
            if in_snapshot_at is not None:
                label += f" (snapshot at fabula={in_snapshot_at})"
            if resolved:
                repairs.append(
                    f"Repaired {label} acquired_via_event_id "
                    f"'{b.acquired_via_event_id}' \u2192 '{resolved}'."
                )
                update["acquired_via_event_id"] = resolved
            else:
                repairs.append(
                    f"Cleared dangling acquired_via_event_id "
                    f"'{b.acquired_via_event_id}' on {label} (no fuzzy match)."
                )
                update["acquired_via_event_id"] = None
        if (
            b.acquired_via_channel_id
            and b.acquired_via_channel_id not in valid_channel_ids_set
        ):
            repairs.append(
                f"Cleared dangling acquired_via_channel_id "
                f"'{b.acquired_via_channel_id}' on belief about '{b.target_id}' "
                f"on '{owner}'."
            )
            update["acquired_via_channel_id"] = None
        return b.model_copy(update=update) if update else b

    repaired_entities: Dict[str, Entity] = {}
    entities_belief_changed = False
    for eid, ent in ws.entities.items():
        ent_update: dict = {}
        # Standing beliefs
        if ent.beliefs:
            new_beliefs = [_repair_belief(b, eid) for b in ent.beliefs]
            if new_beliefs != list(ent.beliefs):
                ent_update["beliefs"] = new_beliefs
        # Snapshot-embedded beliefs
        if ent.state_timeline:
            new_tl: List[EntityStateSnapshot] = []
            tl_changed = False
            for snap in ent.state_timeline:
                if snap.beliefs_added:
                    repaired = [
                        _repair_belief(b, eid, in_snapshot_at=snap.fabula_time)
                        for b in snap.beliefs_added
                    ]
                    if repaired != list(snap.beliefs_added):
                        new_tl.append(snap.model_copy(update={"beliefs_added": repaired}))
                        tl_changed = True
                        continue
                new_tl.append(snap)
            if tl_changed:
                ent_update["state_timeline"] = new_tl
        if ent_update:
            entities_belief_changed = True
            repaired_entities[eid] = ent.model_copy(update=ent_update)
        else:
            repaired_entities[eid] = ent
    if entities_belief_changed:
        ws = ws.model_copy(update={"entities": repaired_entities})

    # --- Utterance missing-field auto-fixes ---
    # When an utterance lacks ``speaker_id`` but has a single ``actor_ids``
    # entry, take the actor as the speaker (the physics extractor often
    # writes the speaker into actor_ids). When ``addressee_ids`` is empty
    # but ``target_ids`` contains entity/object ids, take those as
    # addressees. These are deterministic shifts that the LLM correction
    # loop was previously being invoked for.
    repaired_utterances: List[EventNode] = []
    valid_addressees = entity_ids | set(ws.objects.keys())
    for evt in clean_events:
        if evt.event_type != "utterance":
            repaired_utterances.append(evt)
            continue
        update: dict = {}
        if not evt.speaker_id and len(evt.actor_ids) == 1 and evt.actor_ids[0] in valid_addressees:
            update["speaker_id"] = evt.actor_ids[0]
            repairs.append(
                f"Promoted single actor '{evt.actor_ids[0]}' to speaker_id "
                f"on utterance '{evt.id}'."
            )
        if not evt.addressee_ids and evt.target_ids:
            cand = [t for t in evt.target_ids if t in valid_addressees]
            if cand:
                update["addressee_ids"] = cand
                repairs.append(
                    f"Promoted target_ids \u2192 addressee_ids on utterance "
                    f"'{evt.id}': {cand}."
                )
        repaired_utterances.append(evt.model_copy(update=update) if update else evt)
    clean_events = repaired_utterances

    if repairs:
        logger.info("[Auto-Repair] Applied %d repairs.", len(repairs))
        ws = WorldStateV1(
            locations=ws.locations,
            objects=ws.objects,
            entities=ws.entities,
            events=clean_events,
            world_traits=ws.world_traits,
            causal_topology=clean_causal,
            spatial_topology=clean_spatial,
            channels=clean_channels,
            social_topology=clean_social,
        )

    return ws, repairs


def _programmatic_validation(ws: WorldStateV1) -> List[ValidationIssue]:
    """Fast structural checks that don't require an LLM."""
    issues: List[ValidationIssue] = []

    # Cross-register ID-namespace collision check. The valid_ids set
    # below is built by union, which silently absorbs collisions; e.g.
    # a stray ``OBJ_DAGGER`` mistakenly registered under
    # ``ws.entities`` and a real ``OBJ_DAGGER`` in ``ws.objects`` both
    # collapse to a single membership token. Downstream lookups would
    # then resolve the ID to whichever register the consumer happened
    # to query first \u2014 a classic source of \"phantom entity\" bugs
    # during counterfactual surgery.
    register_views: List[Tuple[str, set]] = [
        ("locations", set(ws.locations.keys())),
        ("objects", set(ws.objects.keys())),
        ("entities", set(ws.entities.keys())),
        ("world_traits", set(ws.world_traits.keys())),
        ("channels", set(ws.channels.keys())),
    ]
    for i in range(len(register_views)):
        for j in range(i + 1, len(register_views)):
            name_a, ids_a = register_views[i]
            name_b, ids_b = register_views[j]
            overlap = ids_a & ids_b
            for dup in sorted(overlap):
                issues.append(ValidationIssue(
                    severity="error", category="duplicate",
                    detail=(
                        f"ID '{dup}' is registered in both "
                        f"ws.{name_a} and ws.{name_b}; downstream "
                        f"lookups will resolve ambiguously."
                    ),
                ))

    # Build the valid ID set
    valid_ids = (
        set(ws.locations.keys())
        | set(ws.objects.keys())
        | set(ws.entities.keys())
        | set(ws.world_traits.keys())
        | {e.id for e in ws.events}
    )

    # Check causal edges
    event_ids = {e.id for e in ws.events}
    for ce in ws.causal_topology:
        if ce.source_id not in valid_ids:
            issues.append(ValidationIssue(
                severity="error", category="broken_link",
                detail=f"CausalEdge.source_id '{ce.source_id}' not in node set.",
            ))
        if ce.target_id not in valid_ids:
            issues.append(ValidationIssue(
                severity="error", category="broken_link",
                detail=f"CausalEdge.target_id '{ce.target_id}' not in node set.",
            ))

    # Check relationship edges
    entity_ids = set(ws.entities.keys())
    for re_edge in ws.social_topology:
        if re_edge.source_entity_id not in entity_ids:
            issues.append(ValidationIssue(
                severity="error", category="broken_link",
                detail=f"RelationshipEdge.source_entity_id '{re_edge.source_entity_id}' not in entities.",
            ))
        if re_edge.target_entity_id not in entity_ids:
            issues.append(ValidationIssue(
                severity="error", category="broken_link",
                detail=f"RelationshipEdge.target_entity_id '{re_edge.target_entity_id}' not in entities.",
            ))

    # Check spatial edges
    location_ids = set(ws.locations.keys())
    for se in ws.spatial_topology:
        if se.source_id not in location_ids:
            issues.append(ValidationIssue(
                severity="error", category="broken_link",
                detail=f"SpatialEdge.source_id '{se.source_id}' not in locations.",
            ))
        if se.target_id not in location_ids:
            issues.append(ValidationIssue(
                severity="error", category="broken_link",
                detail=f"SpatialEdge.target_id '{se.target_id}' not in locations.",
            ))

    # Check channels
    node_ids = set(ws.entities.keys()) | set(ws.objects.keys())
    for cid, ch in ws.channels.items():
        for pid in ch.participant_ids:
            if pid not in node_ids:
                issues.append(ValidationIssue(
                    severity="error", category="broken_link",
                    detail=f"Channel '{cid}' participant_id '{pid}' not in entities/objects.",
                ))
        if len(ch.participant_ids) < 2:
            issues.append(ValidationIssue(
                severity="error", category="broken_link",
                detail=f"Channel '{cid}' has fewer than 2 participants.",
            ))
        if len(set(ch.participant_ids)) < len(ch.participant_ids):
            dupes = [p for p in ch.participant_ids if ch.participant_ids.count(p) > 1]
            issues.append(ValidationIssue(
                severity="warning", category="duplicate",
                detail=(
                    f"Channel '{cid}' has duplicate participant_ids "
                    f"{sorted(set(dupes))}; downstream consumers will see a "
                    f"phantom n-way channel."
                ),
            ))
        for k in ch.intelligibility.keys():
            if k not in ch.participant_ids:
                issues.append(ValidationIssue(
                    severity="warning", category="broken_link",
                    detail=(
                        f"Channel '{cid}' intelligibility entry for '{k}' is not "
                        f"a participant of the channel."
                    ),
                ))

    # Check utterance event references
    valid_channel_ids = set(ws.channels.keys())
    for evt in ws.events:
        if evt.event_type != "utterance":
            continue
        # Required fields for utterances. The Pydantic model marks
        # these Optional so that non-utterance events can omit them,
        # but for ``event_type='utterance'`` an absent speaker or
        # empty addressees breaks downstream consumers (belief
        # propagation, channel intelligibility routing, the social
        # propagator that mirrors speaker_id into actor_ids).
        if not evt.speaker_id:
            issues.append(ValidationIssue(
                severity="error", category="missing_field",
                detail=(
                    f"Utterance '{evt.id}' is missing required field "
                    f"'speaker_id'."
                ),
            ))
        if not evt.addressee_ids:
            issues.append(ValidationIssue(
                severity="error", category="missing_field",
                detail=(
                    f"Utterance '{evt.id}' is missing required field "
                    f"'addressee_ids' (must contain at least one ENT_ id)."
                ),
            ))
        if evt.via_channel_id and evt.via_channel_id not in valid_channel_ids:
            issues.append(ValidationIssue(
                severity="error", category="broken_link",
                detail=(
                    f"Utterance '{evt.id}' via_channel_id '{evt.via_channel_id}' "
                    f"not in channels."
                ),
            ))
        # If utterance routes through a channel, both speaker and every
        # addressee must actually be participants in that channel —
        # otherwise the propagator will silently drop the message at
        # intelligibility-check time.
        if evt.via_channel_id and evt.via_channel_id in ws.channels:
            chan_participants = set(ws.channels[evt.via_channel_id].participant_ids)
            if evt.speaker_id and evt.speaker_id not in chan_participants:
                issues.append(ValidationIssue(
                    severity="error", category="broken_link",
                    detail=(
                        f"Utterance '{evt.id}' speaker_id '{evt.speaker_id}' "
                        f"is not a participant of via_channel_id '{evt.via_channel_id}'."
                    ),
                ))
            for aid in evt.addressee_ids:
                if aid not in chan_participants:
                    issues.append(ValidationIssue(
                        severity="error", category="broken_link",
                        detail=(
                            f"Utterance '{evt.id}' addressee '{aid}' is not a "
                            f"participant of via_channel_id '{evt.via_channel_id}'."
                        ),
                    ))
        if evt.speaker_id and evt.speaker_id not in node_ids:
            issues.append(ValidationIssue(
                severity="error", category="broken_link",
                detail=f"Utterance '{evt.id}' speaker_id '{evt.speaker_id}' not in entities/objects.",
            ))
        for aid in evt.addressee_ids:
            if aid not in node_ids:
                issues.append(ValidationIssue(
                    severity="error", category="broken_link",
                    detail=f"Utterance '{evt.id}' addressee_ids entry '{aid}' not in entities/objects.",
                ))

    # Check belief provenance: acquired_via_event_id must point at a real
    # event, acquired_via_channel_id at a real channel. Counterfactual
    # surgery uses these to prune downstream beliefs when an event /
    # channel is removed; dangling refs would silently break that.
    valid_event_ids = {e.id for e in ws.events}
    for eid, ent in ws.entities.items():
        for b in ent.beliefs:
            if (
                b.acquired_via_event_id
                and b.acquired_via_event_id not in valid_event_ids
            ):
                issues.append(ValidationIssue(
                    severity="warning", category="broken_link",
                    detail=(
                        f"Entity '{eid}' belief about '{b.target_id}' has "
                        f"acquired_via_event_id='{b.acquired_via_event_id}' "
                        f"that is not in events."
                    ),
                ))
            if (
                b.acquired_via_channel_id
                and b.acquired_via_channel_id not in valid_channel_ids
            ):
                issues.append(ValidationIssue(
                    severity="warning", category="broken_link",
                    detail=(
                        f"Entity '{eid}' belief about '{b.target_id}' has "
                        f"acquired_via_channel_id='{b.acquired_via_channel_id}' "
                        f"that is not in channels."
                    ),
                ))
        # State-timeline snapshots carry the same belief shape.
        for snap in ent.state_timeline:
            for b in snap.beliefs_added:
                if (
                    b.acquired_via_event_id
                    and b.acquired_via_event_id not in valid_event_ids
                ):
                    issues.append(ValidationIssue(
                        severity="warning", category="broken_link",
                        detail=(
                            f"Entity '{eid}' snapshot belief at "
                            f"fabula={snap.fabula_time} has dangling "
                            f"acquired_via_event_id='{b.acquired_via_event_id}'."
                        ),
                    ))
                if (
                    b.acquired_via_channel_id
                    and b.acquired_via_channel_id not in valid_channel_ids
                ):
                    issues.append(ValidationIssue(
                        severity="warning", category="broken_link",
                        detail=(
                            f"Entity '{eid}' snapshot belief at "
                            f"fabula={snap.fabula_time} has dangling "
                            f"acquired_via_channel_id='{b.acquired_via_channel_id}'."
                        ),
                    ))

    # Belief provenance — semantic coherence checks (warnings).
    #
    # The structural checks above only verify the referenced IDs exist.
    # These checks verify the *meaning* of the provenance edge:
    #   * The referenced event should normally be an utterance or a
    #     revelation-class event — beliefs acquired via random
    #     unrelated events are usually extraction noise.
    #   * When both channel and event provenance are set, they must be
    #     consistent: the utterance event's via_channel_id should match
    #     the belief's acquired_via_channel_id.
    #   * High-confidence beliefs acquired through low-intelligibility
    #     channels are epistemically suspect — flag for human review.
    event_index = {e.id: e for e in ws.events}
    revelation_event_types = {"utterance", "revelation", "discovery", "observation"}
    intel_warn_threshold = 0.3
    for eid, ent in ws.entities.items():
        for b in ent.beliefs:
            ev_id = b.acquired_via_event_id
            ch_id = b.acquired_via_channel_id
            if ev_id and ev_id in event_index:
                src_evt = event_index[ev_id]
                if src_evt.event_type not in revelation_event_types:
                    issues.append(ValidationIssue(
                        severity="warning", category="semantic_provenance",
                        detail=(
                            f"Entity '{eid}' belief about '{b.target_id}' "
                            f"is acquired_via_event_id='{ev_id}' whose "
                            f"event_type='{src_evt.event_type}' is not a "
                            f"revelation-class event "
                            f"({sorted(revelation_event_types)}). Likely "
                            f"extraction noise."
                        ),
                    ))
                # Cross-field coherence: if both channel and utterance
                # are set, the utterance must travel via that channel.
                if (
                    ch_id
                    and src_evt.event_type == "utterance"
                    and src_evt.via_channel_id
                    and src_evt.via_channel_id != ch_id
                ):
                    issues.append(ValidationIssue(
                        severity="warning", category="semantic_provenance",
                        detail=(
                            f"Entity '{eid}' belief about '{b.target_id}' "
                            f"declares acquired_via_channel_id='{ch_id}' "
                            f"but the source utterance '{ev_id}' was "
                            f"transmitted via_channel_id="
                            f"'{src_evt.via_channel_id}'. Provenance is "
                            f"internally inconsistent."
                        ),
                    ))
            # High-confidence belief through low-intelligibility channel.
            if ch_id and ch_id in ws.channels:
                ch = ws.channels[ch_id]
                intel = float(ch.intelligibility.get(eid, 1.0))
                conf = float(getattr(b, "confidence", 1.0) or 1.0)
                if intel < intel_warn_threshold and conf >= 0.8:
                    issues.append(ValidationIssue(
                        severity="warning", category="semantic_provenance",
                        detail=(
                            f"Entity '{eid}' holds confident belief "
                            f"(conf={conf:.2f}) about '{b.target_id}' "
                            f"acquired through channel '{ch_id}' where "
                            f"its intelligibility is only {intel:.2f}. "
                            f"Low-intelligibility channels should not "
                            f"yield high-confidence beliefs."
                        ),
                    ))

    # Check event actor_ids and target_ids.
    #
    # For non-utterance events: actor_ids must be ENT_, target_ids must
    # be ENT_/OBJ_ (mirrors the physics_extraction prompt).
    #
    # For utterance events: per social_extraction.md, the speaker (and
    # therefore actor_ids[0]) MAY be ENT_ or OBJ_ (e.g. a dossier, a
    # telescreen broadcast); target_ids MAY additionally include EVT_,
    # WORLD_, and LOC_ ids — utterances are *about* topics, and topics
    # are commonly past events, world facts, or places. Restricting
    # utterance target_ids to ENT_/OBJ_ would block the prompt's own
    # documented "X tells Y about EVT_Z" pattern.
    object_ids = set(ws.objects.keys())
    world_trait_ids = set(ws.world_traits.keys())
    event_id_set = {e.id for e in ws.events}
    for evt in ws.events:
        is_utterance = evt.event_type == "utterance"
        actor_allowed = (entity_ids | object_ids) if is_utterance else entity_ids
        target_allowed = (
            entity_ids | object_ids | event_id_set | world_trait_ids | location_ids
            if is_utterance
            else entity_ids | object_ids
        )
        actor_label = "entity/object" if is_utterance else "entity"
        target_label = (
            "entity/object/event/world_trait/location"
            if is_utterance
            else "entity/object"
        )
        for aid in evt.actor_ids:
            if aid not in actor_allowed:
                issues.append(ValidationIssue(
                    severity="error", category="hallucinated_id",
                    detail=f"EventNode '{evt.id}' actor_ids entry '{aid}' is not a valid {actor_label}.",
                ))
        for tid in evt.target_ids:
            if tid not in target_allowed:
                issues.append(ValidationIssue(
                    severity="error", category="hallucinated_id",
                    detail=f"EventNode '{evt.id}' target_ids entry '{tid}' is not a valid {target_label}.",
                ))

    # Check entity location_id references
    for eid, ent in ws.entities.items():
        if ent.location_id not in location_ids:
            issues.append(ValidationIssue(
                severity="error", category="broken_link",
                detail=f"Entity '{eid}' location_id '{ent.location_id}' not in locations.",
            ))

    # Check object location_id and owner_id references
    for oid, obj in ws.objects.items():
        if obj.location_id and obj.location_id not in location_ids:
            issues.append(ValidationIssue(
                severity="warning", category="broken_link",
                detail=f"Object '{oid}' location_id '{obj.location_id}' not in locations.",
            ))
        if obj.owner_id and obj.owner_id not in entity_ids:
            issues.append(ValidationIssue(
                severity="warning", category="broken_link",
                detail=f"Object '{oid}' owner_id '{obj.owner_id}' not in entities.",
            ))

    # Check entity belief target_id references
    all_valid_belief_targets = (
        set(ws.locations.keys()) | set(ws.objects.keys())
        | set(ws.entities.keys()) | set(ws.world_traits.keys())
        | {e.id for e in ws.events}
    )
    for eid, ent in ws.entities.items():
        for belief in ent.beliefs:
            if belief.target_id not in all_valid_belief_targets:
                issues.append(ValidationIssue(
                    severity="warning", category="broken_link",
                    detail=f"Entity '{eid}' belief target_id '{belief.target_id}' not in locations/objects/entities/events.",
                ))
        # Same shape applies to beliefs sitting on snapshots; they
        # become an entity's live ``beliefs`` after world-state replay,
        # so a dangling target_id here is just a delayed broken_link.
        for snap in ent.state_timeline:
            for belief in snap.beliefs_added:
                if belief.target_id not in all_valid_belief_targets:
                    issues.append(ValidationIssue(
                        severity="warning", category="broken_link",
                        detail=(
                            f"Entity '{eid}' snapshot belief at "
                            f"fabula={snap.fabula_time} target_id "
                            f"'{belief.target_id}' not in "
                            f"locations/objects/entities/events/world_traits."
                        ),
                    ))

    # Check entity state_timeline references
    for eid, ent in ws.entities.items():
        prev_ft = -1
        for snap in ent.state_timeline:
            if snap.triggered_by and snap.triggered_by not in event_ids:
                issues.append(ValidationIssue(
                    severity="warning", category="broken_link",
                    detail=f"Entity '{eid}' state_timeline triggered_by '{snap.triggered_by}' not in events.",
                ))
            if snap.location_id and snap.location_id not in location_ids:
                issues.append(ValidationIssue(
                    severity="warning", category="broken_link",
                    detail=f"Entity '{eid}' state_timeline location_id '{snap.location_id}' not in locations.",
                ))
            if snap.fabula_time < prev_ft:
                issues.append(ValidationIssue(
                    severity="warning", category="temporal",
                    detail=f"Entity '{eid}' state_timeline not monotonic: fabula_time {snap.fabula_time} follows {prev_ft}.",
                ))
            prev_ft = snap.fabula_time

    # Check world_trait state_timeline references (mirror of the entity
    # block above). Without this, EVT renames / drops in the patch path
    # silently leave dangling triggered_by ids on world traits, which
    # the LLM auditor is unlikely to surface and downstream surgery
    # cannot undo.
    for wid, wt in ws.world_traits.items():
        prev_ft = -1
        for snap in wt.state_timeline:
            if snap.triggered_by and snap.triggered_by not in event_ids:
                issues.append(ValidationIssue(
                    severity="warning", category="broken_link",
                    detail=(
                        f"World trait '{wid}' state_timeline triggered_by "
                        f"'{snap.triggered_by}' not in events."
                    ),
                ))
            if snap.fabula_time < prev_ft:
                issues.append(ValidationIssue(
                    severity="warning", category="temporal",
                    detail=(
                        f"World trait '{wid}' state_timeline not monotonic: "
                        f"fabula_time {snap.fabula_time} follows {prev_ft}."
                    ),
                ))
            prev_ft = snap.fabula_time

    # Check for duplicate event IDs
    seen_evt_ids: set[str] = set()
    for evt in ws.events:
        if evt.id in seen_evt_ids:
            issues.append(ValidationIssue(
                severity="error", category="duplicate",
                detail=f"Duplicate event ID: '{evt.id}'.",
            ))
        seen_evt_ids.add(evt.id)

    # --- Orphan event check ---
    if ws.events:
        referenced_events: set[str] = set()
        for ce in ws.causal_topology:
            if ce.source_id.startswith("EVT_"):
                referenced_events.add(ce.source_id)
            if ce.target_id.startswith("EVT_"):
                referenced_events.add(ce.target_id)
        orphan_events = [e for e in ws.events if e.id not in referenced_events]
        if orphan_events:
            orphan_ids = [e.id for e in orphan_events[:10]]
            issues.append(ValidationIssue(
                severity="warning", category="orphan",
                detail=(
                    f"{len(orphan_events)} event(s) not referenced by any causal edge: "
                    f"{orphan_ids}{'…' if len(orphan_events) > 10 else ''}. "
                    f"Consider adding causal connections."
                ),
            ))

    # --- Information density check ---
    # Scale the expected information signal by the *narratively
    # information-bearing* event types only. Action-heavy chunks
    # (heists, battles, chases — Reservoir Dogs, Apocalypse Now) can
    # legitimately be all action/outcome events with no utterances or
    # channels, and the previous unconditional rule was flagging those
    # as "missing information" and feeding them into the LLM correction
    # loop, which then invented spurious channels.
    utterance_count = sum(1 for e in ws.events if e.event_type == "utterance")
    info_signal = len(ws.channels) + utterance_count
    info_bearing = sum(
        1 for e in ws.events
        if e.event_type in ("choice", "revelation", "utterance")
    )
    if info_bearing >= 3 and info_signal == 0:
        issues.append(ValidationIssue(
            severity="warning", category="missing_information",
            detail=(
                f"Zero channels and zero utterance events extracted across "
                f"{info_bearing} information-bearing event(s) "
                f"(choice/revelation/utterance). Most narratives with that "
                f"many decisions or revelations contain conversations, "
                f"letters, or proclamations that should produce a Channel "
                f"or an utterance EventNode."
            ),
        ))
    elif info_bearing >= 5 and info_signal < info_bearing // 5:
        issues.append(ValidationIssue(
            severity="warning", category="missing_information",
            detail=(
                f"Low information density: {len(ws.channels)} channels + "
                f"{utterance_count} utterances for {info_bearing} "
                f"information-bearing events (ratio "
                f"{info_signal / info_bearing:.2f}). Expected at least 1 "
                f"information signal per 5 information-bearing events."
            ),
        ))

    # --- Self-referencing relationship edges ---
    for re_edge in ws.social_topology:
        if re_edge.source_entity_id == re_edge.target_entity_id:
            issues.append(ValidationIssue(
                severity="warning", category="contradiction",
                detail=(
                    f"Self-referencing RelationshipEdge: "
                    f"'{re_edge.source_entity_id}' → '{re_edge.target_entity_id}'. "
                    f"Relationships should be between different entities."
                ),
            ))

    # --- Mutation edge coverage check ---
    # Events that change entity state should have mutation causal edges
    if ws.events and ws.causal_topology:
        mutation_target_events: set[str] = set()
        for ce in ws.causal_topology:
            if ce.causality_type in ("mutation", "mutation_social"):
                mutation_target_events.add(ce.source_id)
        # Significant events: choices and outcomes typically cause state changes
        sig_events = [
            e for e in ws.events
            if e.event_type in ("choice", "outcome") and e.target_ids
        ]
        unmutated = [e for e in sig_events if e.id not in mutation_target_events]
        if len(unmutated) > len(sig_events) // 2 and len(unmutated) >= 3:
            sample_ids = [e.id for e in unmutated[:5]]
            issues.append(ValidationIssue(
                severity="warning", category="missing_mutation",
                detail=(
                    f"{len(unmutated)}/{len(sig_events)} significant events "
                    f"(choices/outcomes with targets) lack mutation causal edges: "
                    f"{sample_ids}{'…' if len(unmutated) > 5 else ''}. "
                    f"Events that affect characters should produce mutation edges."
                ),
            ))

    # --- State timeline coverage check ---
    # Entities involved in events (as targets) should have state_timeline entries
    if ws.events:
        targeted_entities: set[str] = set()
        for evt in ws.events:
            for tid in evt.target_ids:
                if tid in entity_ids:
                    targeted_entities.add(tid)
        entities_with_timeline = {
            eid for eid in entity_ids
            if ws.entities[eid].state_timeline
        }
        missing_timeline = targeted_entities - entities_with_timeline
        if missing_timeline and len(missing_timeline) >= 2:
            issues.append(ValidationIssue(
                severity="warning", category="missing_state_timeline",
                detail=(
                    f"{len(missing_timeline)} entities targeted by events lack "
                    f"state_timeline entries: {sorted(missing_timeline)[:5]}. "
                    f"Entity state changes should be tracked in state_timeline."
                ),
            ))

    # --- Mutation-parity check ---
    #
    # Surfaces the single largest gap the 2026-05-01 plot-models audit found:
    # 75 mutation/mutation_social edges declared a (trait_target, trait_delta)
    # but the target entity had no corresponding state_timeline snapshot at
    # the edge's fabula_time. Without the snapshot the physics engine has
    # nothing to anchor downstream propagation and abduction reads to, and
    # the mutation effectively vanishes after one tick.
    #
    # We allow ±1 fabula tick of slack so authors can co-locate a snapshot
    # at a near-by event boundary (the engine's interpolation handles tiny
    # offsets cleanly).
    if ws.events and ws.causal_topology:
        snapshots_by_entity: Dict[str, set[int]] = {}
        for eid, ent in ws.entities.items():
            snapshots_by_entity[eid] = {snap.fabula_time for snap in ent.state_timeline}
        unmatched: List[str] = []
        for ce in ws.causal_topology:
            if ce.causality_type not in ("mutation", "mutation_social"):
                continue
            if ce.trait_target is None or ce.trait_delta is None:
                continue
            if ce.target_id not in entity_ids:
                continue
            snaps = snapshots_by_entity.get(ce.target_id, set())
            # ±1 tick slack to tolerate authoring co-location.
            if not any(abs(t - ce.fabula_time) <= 1 for t in snaps):
                unmatched.append(
                    f"{ce.source_id}\u2192{ce.target_id} "
                    f"({ce.trait_target}, fabula={ce.fabula_time})"
                )
        if unmatched:
            sample = unmatched[:5]
            issues.append(ValidationIssue(
                severity="warning", category="mutation_parity",
                detail=(
                    f"{len(unmatched)} mutation edge(s) declare a "
                    f"trait_target+trait_delta but the target entity has no "
                    f"state_timeline snapshot at the edge's fabula_time "
                    f"(\u00b11 tick slack): {sample}"
                    f"{'\u2026' if len(unmatched) > 5 else ''}. The mutation "
                    f"is recorded on the edge but never anchored on the "
                    f"entity, so propagation and abduction will under-read it."
                ),
            ))

    # --- Phase E: Proposition / Concern / Belief\u2192PROP validation ---
    #
    # Programmatic checks for the affect-unification namespaces. Errors
    # name explicit categories so the correction LLM can map them onto
    # the new patch ops (add_propositions, update_proposition_snapshots,
    # commit_proposition_truth, add_concerns, update_concern_snapshots,
    # set_belief_proposition_ids).
    prop_ids = {p.proposition_id for p in ws.propositions}

    # E1.a Proposition.referent_ids must resolve in the ontology.
    # Channels are valid referents (a proposition can name a CHN_* node, e.g.
    # "the locket-letter is a covert channel between Macbeth and his wife").
    valid_referent_targets = (
        location_ids | object_ids | entity_ids | world_trait_ids
        | event_id_set | set(ws.channels.keys())
    )
    for prop in ws.propositions:
        if not prop.proposition_id.startswith("PROP_"):
            issues.append(ValidationIssue(
                severity="error", category="bad_proposition_id",
                detail=(
                    f"Proposition id '{prop.proposition_id}' does not "
                    f"match the PROP_ namespace."
                ),
            ))
        for ref in prop.referent_ids:
            if ref not in valid_referent_targets:
                issues.append(ValidationIssue(
                    severity="warning", category="broken_link",
                    detail=(
                        f"Proposition '{prop.proposition_id}' referent_id "
                        f"'{ref}' is not in locations/objects/entities/"
                        f"world_traits/events."
                    ),
                ))
        # E1.f truth_at_fabula keys within event-fabula range.
        if prop.truth_at_fabula and ws.events:
            evt_fabulas = [e.fabula_time for e in ws.events]
            evt_min, evt_max = min(evt_fabulas), max(evt_fabulas)
            for fab in prop.truth_at_fabula.keys():
                if fab < evt_min or fab > evt_max:
                    issues.append(ValidationIssue(
                        severity="warning", category="temporal",
                        detail=(
                            f"Proposition '{prop.proposition_id}' "
                            f"truth_at_fabula key {fab} is outside the "
                            f"event fabula range [{evt_min}, {evt_max}]."
                        ),
                    ))

    # E1.b PropositionSnapshot.triggered_by must resolve to an EVT_ id;
    # snapshot fabula_time within event-fabula range.
    if ws.events:
        evt_max_fabula = max(e.fabula_time for e in ws.events)
    else:
        evt_max_fabula = None
    for prop in ws.propositions:
        for snap in prop.state_timeline:
            if snap.triggered_by and snap.triggered_by not in event_id_set:
                issues.append(ValidationIssue(
                    severity="warning", category="snapshot_no_event",
                    detail=(
                        f"Proposition '{prop.proposition_id}' snapshot at "
                        f"fabula={snap.fabula_time} triggered_by="
                        f"'{snap.triggered_by}' does not resolve to an event."
                    ),
                ))
            if (
                evt_max_fabula is not None
                and snap.fabula_time > evt_max_fabula
            ):
                issues.append(ValidationIssue(
                    severity="warning", category="temporal",
                    detail=(
                        f"Proposition '{prop.proposition_id}' snapshot "
                        f"fabula_time={snap.fabula_time} exceeds max event "
                        f"fabula={evt_max_fabula}."
                    ),
                ))

    # E1.c Concern.proposition_id must exist in propositions; concern_id
    # must match the CCN_ namespace; ConcernSnapshot.triggered_by must
    # resolve; counter_concern_ids must be symmetric (warn if not).
    all_concerns: Dict[str, Tuple[str, Concern]] = {}
    for eid, ent in ws.entities.items():
        for c in ent.concerns:
            if not c.concern_id.startswith("CCN_"):
                issues.append(ValidationIssue(
                    severity="error", category="bad_concern_id",
                    detail=(
                        f"Concern id '{c.concern_id}' on entity '{eid}' "
                        f"does not match the CCN_ namespace."
                    ),
                ))
            if c.concern_id in all_concerns:
                issues.append(ValidationIssue(
                    severity="error", category="duplicate",
                    detail=(
                        f"Concern id '{c.concern_id}' appears on both "
                        f"'{all_concerns[c.concern_id][0]}' and '{eid}'."
                    ),
                ))
            else:
                all_concerns[c.concern_id] = (eid, c)
            if c.proposition_id not in prop_ids:
                issues.append(ValidationIssue(
                    severity="error", category="unknown_proposition_id",
                    detail=(
                        f"Concern '{c.concern_id}' on entity '{eid}' "
                        f"references proposition '{c.proposition_id}' "
                        f"which is not in WorldStateV1.propositions."
                    ),
                ))
            for snap in c.state_timeline:
                if snap.triggered_by and snap.triggered_by not in event_id_set:
                    issues.append(ValidationIssue(
                        severity="warning", category="snapshot_no_event",
                        detail=(
                            f"Concern '{c.concern_id}' snapshot at "
                            f"fabula={snap.fabula_time} triggered_by="
                            f"'{snap.triggered_by}' does not resolve to an event."
                        ),
                    ))
                if (
                    evt_max_fabula is not None
                    and snap.fabula_time > evt_max_fabula
                ):
                    issues.append(ValidationIssue(
                        severity="warning", category="temporal",
                        detail=(
                            f"Concern '{c.concern_id}' snapshot "
                            f"fabula_time={snap.fabula_time} exceeds max "
                            f"event fabula={evt_max_fabula}."
                        ),
                    ))
    # counter_concern_ids symmetry check (warning).
    for ccn_id, (eid, c) in all_concerns.items():
        for partner in c.counter_concern_ids:
            entry = all_concerns.get(partner)
            if entry is None:
                issues.append(ValidationIssue(
                    severity="warning", category="broken_link",
                    detail=(
                        f"Concern '{ccn_id}' lists counter_concern_id "
                        f"'{partner}' which does not exist."
                    ),
                ))
                continue
            if ccn_id not in entry[1].counter_concern_ids:
                issues.append(ValidationIssue(
                    severity="warning", category="asymmetric_counter_concern",
                    detail=(
                        f"Concern '{ccn_id}' lists '{partner}' as a "
                        f"counter_concern but '{partner}' does not "
                        f"reciprocate."
                    ),
                ))

    # E1.d Belief.proposition_id, when set, must exist in propositions.
    for eid, ent in ws.entities.items():
        for b in ent.beliefs:
            if b.proposition_id and b.proposition_id not in prop_ids:
                issues.append(ValidationIssue(
                    severity="error", category="unknown_proposition_id",
                    detail=(
                        f"Entity '{eid}' belief about '{b.target_id}' "
                        f"references proposition '{b.proposition_id}' "
                        f"which is not in WorldStateV1.propositions."
                    ),
                ))

    # E1.e EventNode prop-link fields must resolve in propositions.
    for evt in ws.events:
        for pid in evt.resolves_proposition_ids:
            if pid not in prop_ids:
                issues.append(ValidationIssue(
                    severity="warning", category="unknown_proposition_id",
                    detail=(
                        f"Event '{evt.id}' resolves_proposition_ids entry "
                        f"'{pid}' is not in WorldStateV1.propositions."
                    ),
                ))
        if (
            evt.asserts_proposition_id
            and evt.asserts_proposition_id not in prop_ids
        ):
            issues.append(ValidationIssue(
                severity="warning", category="unknown_proposition_id",
                detail=(
                    f"Event '{evt.id}' asserts_proposition_id "
                    f"'{evt.asserts_proposition_id}' is not in "
                    f"WorldStateV1.propositions."
                ),
            ))
        if (
            evt.denies_proposition_id
            and evt.denies_proposition_id not in prop_ids
        ):
            issues.append(ValidationIssue(
                severity="warning", category="unknown_proposition_id",
                detail=(
                    f"Event '{evt.id}' denies_proposition_id "
                    f"'{evt.denies_proposition_id}' is not in "
                    f"WorldStateV1.propositions."
                ),
            ))

    # E1.g mutation_social edges should have at least one concern
    # touching them on either source or target. A mutation_social
    # edge represents a social-fact change (alliance forms, betrayal,
    # marriage, public exposure) — the affect substrate cares about
    # these only insofar as some entity has a stake in them. An edge
    # that no concern references is structurally valid but produces
    # no felt suspense / irony / surprise on the affect ledger,
    # because both `_audience_concern_for_event` and the per-entity
    # appraisal computations will route around it. Warn (not error)
    # so legacy worlds without populated concern catalogues still
    # pass; the LLM correction agent can surface this as a hint to
    # add the missing concerns.
    if ws.causal_topology and ws.entities:
        # Build a lookup: entity_id → set of proposition referent
        # entity ids the entity has concerns about. A mutation_social
        # edge is "concern-covered" when its source or target entity
        # appears in this lookup.
        concern_touches: Dict[str, Set[str]] = {}
        prop_referents: Dict[str, Set[str]] = {
            p.proposition_id: set(p.referent_ids) for p in ws.propositions
        }
        for ent_id, ent in ws.entities.items():
            touched: Set[str] = set()
            for c in ent.concerns:
                touched |= prop_referents.get(c.proposition_id, set())
            if touched:
                concern_touches[ent_id] = touched
        events_by_id_local = {e.id: e for e in ws.events}
        for ce in ws.causal_topology:
            if ce.mechanism != "mutation_social":
                continue
            # Resolve the entity participants of this edge. Source
            # and target may be event ids or entity ids; for events
            # we expand to actor + target sets.
            participants: Set[str] = set()
            for eid in (ce.source_id, ce.target_id):
                if eid in ws.entities:
                    participants.add(eid)
                elif eid in events_by_id_local:
                    evt = events_by_id_local[eid]
                    participants |= set(evt.actor_ids)
                    participants |= set(evt.target_ids)
            if not participants:
                continue  # orphan edge — caught by other validators
            covered = any(p in concern_touches for p in participants)
            if not covered:
                issues.append(ValidationIssue(
                    severity="warning",
                    category="uncovered_mutation_social",
                    detail=(
                        f"mutation_social edge {ce.source_id} → "
                        f"{ce.target_id} has no concern from any of its "
                        f"participants ({sorted(participants)}). The "
                        f"affect substrate will route around it — add a "
                        f"concern referencing this edge's social stake "
                        f"so suspense/irony register the change."
                    ),
                ))

    # --- Time validation ---
    issues.extend(_validate_time_ordering(ws))

    # --- Dead-actor validation ---
    issues.extend(_validate_dead_actors(ws))

    return issues


def _validate_dead_actors(ws: WorldStateV1) -> List[ValidationIssue]:
    """Check that entities marked dead do not act after their death event.

    Emits **warnings** (not errors) because narratives commonly use fake
    deaths, ghost scenes, flashback POV, and murder-suicides. The LLM
    validator can promote these to errors when it knows the story context.
    """
    issues: List[ValidationIssue] = []

    # Find entities with status == "dead"
    dead_entities = {eid for eid, ent in ws.entities.items() if ent.status == "dead"}
    if not dead_entities:
        return issues

    # For each dead entity, find the earliest death-like event.
    # A death event can be ANY event_type — suicides are often "choice",
    # learning someone died is "revelation", and killings are "outcome".
    # We match:
    #   (a) entity is in target_ids, OR
    #   (b) entity is in actor_ids with no targets (self-caused death)
    death_times: dict[str, int] = {}
    death_events: dict[str, str] = {}
    for evt in ws.events:
        # Case (a): entity is a target
        for tid in evt.target_ids:
            if tid in dead_entities:
                if tid not in death_times or evt.fabula_time < death_times[tid]:
                    death_times[tid] = evt.fabula_time
                    death_events[tid] = evt.id
        # Case (b): entity is an actor with no targets (suicide / self-death)
        if not evt.target_ids:
            for aid in evt.actor_ids:
                if aid in dead_entities:
                    if aid not in death_times or evt.fabula_time < death_times[aid]:
                        death_times[aid] = evt.fabula_time
                        death_events[aid] = evt.id

    # Check for actor references after death
    for evt in ws.events:
        for aid in evt.actor_ids:
            if aid in death_times:
                death_t = death_times[aid]
                death_evt = death_events[aid]
                # Skip the death event itself (the entity is the actor of their own death)
                if evt.id == death_evt:
                    continue
                if evt.fabula_time > death_t:
                    issues.append(ValidationIssue(
                        severity="warning",
                        category="contradiction",
                        detail=(
                            f"Dead entity '{aid}' acts in '{evt.id}' "
                            f"(fabula={evt.fabula_time}) after death in "
                            f"'{death_evt}' (fabula={death_t}). "
                            f"Could be a fake death, ghost, or flashback."
                        ),
                    ))

    return issues


def _validate_time_ordering(ws: WorldStateV1) -> List[ValidationIssue]:
    """Validate fabula_time and syuzhet_index correctness."""
    issues: List[ValidationIssue] = []

    if not ws.events:
        return issues

    # 1. Check syuzhet_index is contiguous (no gaps, no duplicates)
    syuzhet_indices = sorted(e.syuzhet_index for e in ws.events)
    syuzhet_set = set(syuzhet_indices)
    if len(syuzhet_set) != len(ws.events):
        dupes = [s for s in syuzhet_set if syuzhet_indices.count(s) > 1]
        issues.append(ValidationIssue(
            severity="error", category="temporal",
            detail=f"Duplicate syuzhet_index values: {dupes}.",
        ))
    if syuzhet_indices:
        expected = list(range(syuzhet_indices[0], syuzhet_indices[0] + len(syuzhet_indices)))
        if syuzhet_indices != expected:
            gaps = set(expected) - syuzhet_set
            if gaps:
                issues.append(ValidationIssue(
                    severity="warning", category="temporal",
                    detail=f"Non-contiguous syuzhet_index — gaps at: {sorted(gaps)[:10]}.",
                ))

    # 2. Check fabula_time has reasonable spacing (warn if contiguous 1,2,3…)
    fabula_times = sorted(set(e.fabula_time for e in ws.events))
    if len(fabula_times) >= 3:
        diffs = [fabula_times[i + 1] - fabula_times[i] for i in range(len(fabula_times) - 1)]
        median_diff = sorted(diffs)[len(diffs) // 2]
        if median_diff <= 1:
            issues.append(ValidationIssue(
                severity="warning", category="temporal",
                detail=(
                    f"fabula_time values use contiguous small integers "
                    f"(median gap={median_diff}). Expected spacing ~100 to leave "
                    f"room for flashbacks and interstitial events."
                ),
            ))

    # 3. Check causal edges: for chain_reaction (event→event), cause must
    #    precede or coincide with effect in fabula_time.
    evt_fabula = {e.id: e.fabula_time for e in ws.events}
    for ce in ws.causal_topology:
        if ce.causality_type != "chain_reaction":
            continue  # temporal ordering only meaningful for event→event
        src_t = evt_fabula.get(ce.source_id)
        tgt_t = evt_fabula.get(ce.target_id)
        if src_t is not None and tgt_t is not None and src_t + ce.propagation_delay > tgt_t:
            issues.append(ValidationIssue(
                severity="error", category="temporal",
                detail=(
                    f"Causal edge '{ce.source_id}' (fabula={src_t}, delay={ce.propagation_delay}) → "
                    f"'{ce.target_id}' (fabula={tgt_t}): effect manifests before cause + delay."
                ),
            ))

    # 4. Check channels: terminated cannot precede established
    for cid, ch in ws.channels.items():
        if ch.terminated_at_fabula is not None and ch.terminated_at_fabula < ch.established_at_fabula:
            issues.append(ValidationIssue(
                severity="error", category="temporal",
                detail=(
                    f"Channel '{cid}' (participants={ch.participant_ids}): "
                    f"terminated_at_fabula ({ch.terminated_at_fabula}) < "
                    f"established_at_fabula ({ch.established_at_fabula})."
                ),
            ))

    return issues


def _build_validation_agent(config: ExtractionConfig) -> Agent[None, ValidationReport]:
    """Construct the Step 3 LLM validation agent.

    Uses :class:`PromptedOutput` rather than :class:`NativeOutput` because
    Ollama's OpenAI-compat ``response_format=json_schema`` path can return
    ``400 invalid message content type: <nil>`` for some models (e.g.
    ``qwen3.6:35b``). Prompted output injects the schema into the system
    prompt and parses JSON from plain text, which works reliably across
    both the local Ollama backend and OpenAI-compat providers like
    OpenRouter.
    """
    return Agent(
        _resolve_model(config.model),
        output_type=PromptedOutput(ValidationReport),
        system_prompt=_load_prompt("validation.md"),
        retries=config.output_retries,
    )


def _build_correction_agent(config: ExtractionConfig) -> Agent[None, WorldStateV1]:
    """Construct the legacy whole-WorldState correction agent.

    Retained for backwards compatibility but no longer used by the
    pipeline; ``_build_correction_patch_agent`` is now the preferred
    entry point because it asks the LLM for a *diff* instead of a full
    re-emission, eliminating the catastrophic-shrinkage failure mode
    where the model returned a JSON document missing whole topologies.
    """
    return Agent(
        _resolve_model(config.model),
        output_type=NativeOutput(WorldStateV1),
        system_prompt=_load_prompt("correction.md"),
        retries=config.output_retries,
    )


# =====================================================================
# Patch-based correction (preferred path) — see correction.md
# =====================================================================


class _EdgeRef(BaseModel):
    """Identifies a single causal/social/spatial edge by its endpoints."""
    source_id: str
    target_id: str


class _BeliefPropAssignment(BaseModel):
    """Identifies a belief by ``(entity, target)`` and assigns a PROP_ id.

    Used by ``WorldStatePatch.set_belief_proposition_ids`` so the
    correction agent can backfill ``Belief.proposition_id`` on a
    belief whose target matches a catalogued proposition without
    having to re-emit the entire belief record.
    """
    entity_id: str = Field(description="ENT_ id of the believer.")
    target_id: str = Field(
        description=(
            "The belief's existing ``target_id`` (used to locate the "
            "belief on the entity)."
        ),
    )
    proposition_id: str = Field(
        pattern=r"^PROP_[A-Z0-9_]+$",
        description="PROP_ id to write into ``Belief.proposition_id``.",
    )


class WorldStatePatch(BaseModel):
    """A *diff* to apply to an existing WorldStateV1.

    The correction agent emits one of these instead of a full
    WorldStateV1 so it can only describe *changes*, never accidentally
    drop unrelated parts of the world (the failure mode that destroyed
    every causal/social/spatial edge in the Star Wars fixture when the
    LLM was asked to re-emit a complete WorldStateV1 within a token
    budget).

    All fields default to "no change". The patch is applied in this
    order:

      1. ``event_renames``        — rewrite EVT_ IDs everywhere they appear
      2. ``drop_event_ids``       — remove events and dangling references
      3. ``update_event_fields``  — partial field updates on surviving events
      4. ``update_entity_location`` — fix dangling entity.location_id
      5. ``add_state_timeline_entries`` — extend entity.state_timeline
      6. ``drop_*`` for edges/channels — remove specific edges by endpoint
      7. ``add_*`` for edges/channels — append new edges/channels
      8. Final pass: ``_auto_repair`` prunes any newly-dangling refs
    """
    model_config = {"protected_namespaces": ()}

    event_renames: Dict[str, str] = Field(
        default_factory=dict,
        description="EVT_ ID renames: {old_id: new_id}. Applied to every reference in the world state.",
    )
    drop_event_ids: List[str] = Field(
        default_factory=list,
        description="Event IDs to remove entirely. Edges referencing these will be auto-pruned.",
    )
    update_event_fields: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-event field overrides: {event_id: {field: new_value}}. Use to add a missing speaker_id, addressee_ids, via_channel_id, etc.",
    )
    update_entity_location: Dict[str, str] = Field(
        default_factory=dict,
        description="Entity location_id overrides: {entity_id: new_location_id}.",
    )
    add_state_timeline_entries: Dict[str, List[EntityStateSnapshot]] = Field(
        default_factory=dict,
        description="Append snapshots to entity.state_timeline: {entity_id: [snapshot, ...]}.",
    )
    drop_causal_edges: List[_EdgeRef] = Field(
        default_factory=list,
        description="Causal edges to drop, identified by (source_id, target_id).",
    )
    add_causal_edges: List[CausalEdge] = Field(
        default_factory=list,
        description="New causal edges to append.",
    )
    drop_social_edges: List[_EdgeRef] = Field(
        default_factory=list,
        description="Social edges to drop.",
    )
    add_social_edges: List[RelationshipEdge] = Field(
        default_factory=list,
        description="New social edges to append.",
    )
    drop_spatial_edges: List[_EdgeRef] = Field(
        default_factory=list,
        description="Spatial edges to drop.",
    )
    add_spatial_edges: List[SpatialEdge] = Field(
        default_factory=list,
        description="New spatial edges to append.",
    )
    drop_channel_ids: List[str] = Field(
        default_factory=list,
        description="Channel IDs to drop entirely.",
    )
    add_channels: Dict[str, Channel] = Field(
        default_factory=dict,
        description="New channels keyed by channel_id.",
    )
    channel_renames: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Channel CHN_ ID renames: {old_id: new_id}. Forwards every "
            "via_channel_id / acquired_via_channel_id reference to the "
            "new id, so a simple typo fix does not have to be expressed "
            "as drop+add (which would null the provenance of every "
            "belief / utterance pointing at the old channel)."
        ),
    )
    update_channel_intelligibility: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description=(
            "Edit per-participant decode probabilities on existing "
            "channels: {channel_id: {participant_id: 0.0..1.0}}. Each "
            "value is clamped to [0, 1] and merged into the channel's "
            "existing ``intelligibility`` map (existing entries for "
            "other participants are preserved). Use this for narrative "
            "tweaks (\"the line is now noisy for ENT_BOB\") without "
            "rebuilding the channel."
        ),
    )
    # ----- Phase E: proposition / concern / belief-proposition ops -----
    add_propositions: Dict[str, Proposition] = Field(
        default_factory=dict,
        description=(
            "New propositions keyed by PROP_ id. Use when a belief / "
            "concern / event reference an uncatalogued PROP_ id and "
            "the right fix is to introduce the proposition rather than "
            "drop the reference."
        ),
    )
    update_proposition_snapshots: Dict[str, List[PropositionSnapshot]] = Field(
        default_factory=dict,
        description=(
            "Append snapshots into Proposition.state_timeline: "
            "{prop_id: [PropositionSnapshot, ...]}."
        ),
    )
    commit_proposition_truth: Dict[str, Dict[int, bool]] = Field(
        default_factory=dict,
        description=(
            "Write into Proposition.truth_at_fabula: "
            "{prop_id: {fabula_time: bool}}. Last write wins; existing "
            "commits at other fabula_times are preserved."
        ),
    )
    add_concerns: Dict[str, List[Concern]] = Field(
        default_factory=dict,
        description=(
            "Append concerns onto entities: {entity_id: [Concern, ...]}. "
            "Each Concern.proposition_id MUST exist in propositions "
            "after the patch is applied (catalogue or add_propositions)."
        ),
    )
    update_concern_snapshots: Dict[str, Dict[str, List[ConcernSnapshot]]] = Field(
        default_factory=dict,
        description=(
            "Append snapshots into a concern's state_timeline: "
            "{entity_id: {concern_id: [ConcernSnapshot, ...]}}."
        ),
    )
    set_belief_proposition_ids: List[_BeliefPropAssignment] = Field(
        default_factory=list,
        description=(
            "Backfill Belief.proposition_id on existing beliefs matched "
            "by (entity_id, target_id)."
        ),
    )
    notes: str = Field(
        default="",
        description="Free-text rationale for the maintainer log; not applied to the world state.",
    )


def _apply_world_state_patch(
    ws: WorldStateV1, patch: WorldStatePatch,
) -> Tuple[WorldStateV1, List[str]]:
    """Apply *patch* to *ws* and return the new world state + a change log."""
    changes: List[str] = []
    renames = dict(patch.event_renames)
    drop_evts = set(patch.drop_event_ids)
    chan_renames = dict(patch.channel_renames)

    def _r(evt_id: Optional[str]) -> Optional[str]:
        if evt_id is None:
            return None
        return renames.get(evt_id, evt_id)

    def _rc(chan_id: Optional[str]) -> Optional[str]:
        if chan_id is None:
            return None
        return chan_renames.get(chan_id, chan_id)

    # 1. Apply renames + drops to events (with field updates).
    field_updates = patch.update_event_fields
    new_events: List[EventNode] = []
    for evt in ws.events:
        new_id = renames.get(evt.id, evt.id)
        if new_id in drop_evts or evt.id in drop_evts:
            changes.append(f"Dropped event '{evt.id}'.")
            continue
        update: dict = {}
        if new_id != evt.id:
            update["id"] = new_id
            changes.append(f"Renamed event '{evt.id}' → '{new_id}'.")
        # Carry over field-level overrides keyed by either old or new id.
        overrides = field_updates.get(evt.id) or field_updates.get(new_id)
        if overrides:
            update.update(overrides)
            changes.append(f"Updated event '{new_id}' fields: {sorted(overrides.keys())}.")
        new_events.append(evt.model_copy(update=update) if update else evt)

    surviving_event_ids = {e.id for e in new_events}

    # 2. Rewrite event references in causal edges; drop those targeting removed events.
    drop_causal_pairs = {(e.source_id, e.target_id) for e in patch.drop_causal_edges}
    new_causal: List[CausalEdge] = []
    for ce in ws.causal_topology:
        new_src = _r(ce.source_id) if ce.source_id.startswith("EVT_") else ce.source_id
        new_tgt = _r(ce.target_id) if ce.target_id.startswith("EVT_") else ce.target_id
        if (ce.source_id, ce.target_id) in drop_causal_pairs or (new_src, new_tgt) in drop_causal_pairs:
            changes.append(f"Dropped causal edge {ce.source_id}→{ce.target_id}.")
            continue
        update: dict = {}
        if new_src != ce.source_id:
            update["source_id"] = new_src
        if new_tgt != ce.target_id:
            update["target_id"] = new_tgt
        if ce.rel_counterpart_id and ce.rel_counterpart_id.startswith("EVT_"):
            new_rc = _r(ce.rel_counterpart_id)
            if new_rc != ce.rel_counterpart_id:
                update["rel_counterpart_id"] = new_rc
        new_causal.append(ce.model_copy(update=update) if update else ce)

    # 3. Append new causal edges (with renames pre-applied).
    for ce in patch.add_causal_edges:
        update: dict = {}
        if ce.source_id.startswith("EVT_") and ce.source_id in renames:
            update["source_id"] = renames[ce.source_id]
        if ce.target_id.startswith("EVT_") and ce.target_id in renames:
            update["target_id"] = renames[ce.target_id]
        new_causal.append(ce.model_copy(update=update) if update else ce)
        changes.append(f"Added causal edge {ce.source_id}→{ce.target_id}.")

    # 4. Social edges — drop / add (no event renames apply).
    drop_social_pairs = {(e.source_id, e.target_id) for e in patch.drop_social_edges}
    new_social: List[RelationshipEdge] = []
    for re_edge in ws.social_topology:
        if (re_edge.source_entity_id, re_edge.target_entity_id) in drop_social_pairs:
            changes.append(f"Dropped social edge {re_edge.source_entity_id}→{re_edge.target_entity_id}.")
            continue
        new_social.append(re_edge)
    for re_edge in patch.add_social_edges:
        new_social.append(re_edge)
        changes.append(f"Added social edge {re_edge.source_entity_id}→{re_edge.target_entity_id}.")

    # 5. Spatial edges — drop / add.
    drop_spatial_pairs = {(e.source_id, e.target_id) for e in patch.drop_spatial_edges}
    new_spatial: List[SpatialEdge] = []
    for se in ws.spatial_topology:
        if (se.source_id, se.target_id) in drop_spatial_pairs:
            changes.append(f"Dropped spatial edge {se.source_id}→{se.target_id}.")
            continue
        new_spatial.append(se)
    for se in patch.add_spatial_edges:
        new_spatial.append(se)
        changes.append(f"Added spatial edge {se.source_id}→{se.target_id}.")

    # 6. Channels — rename, then drop, then add. Renames forward
    #    references from old → new id; drops still null references.
    drop_chan_ids = set(patch.drop_channel_ids)
    new_channels: Dict[str, Channel] = {}
    for cid, ch in ws.channels.items():
        new_cid = chan_renames.get(cid, cid)
        if new_cid in drop_chan_ids or cid in drop_chan_ids:
            changes.append(f"Dropped channel '{cid}'.")
            continue
        if new_cid != cid:
            changes.append(f"Renamed channel '{cid}' → '{new_cid}'.")
            new_channels[new_cid] = ch.model_copy(update={"id": new_cid})
        else:
            new_channels[cid] = ch
    for cid, ch in patch.add_channels.items():
        new_channels[cid] = ch
        changes.append(f"Added channel '{cid}'.")

    # 6b. Per-participant intelligibility edits on channels that
    # survived the rename/drop/add pass. Clamps to [0, 1] and merges
    # into the existing map so an edit for ENT_BOB does not erase the
    # value for ENT_ALICE.
    for cid, intel_updates in (patch.update_channel_intelligibility or {}).items():
        target_cid = chan_renames.get(cid, cid)
        ch = new_channels.get(target_cid)
        if ch is None:
            changes.append(
                f"Skipped intelligibility update for missing channel '{cid}'."
            )
            continue
        merged: Dict[str, float] = dict(ch.intelligibility or {})
        for pid, raw in (intel_updates or {}).items():
            try:
                val = float(raw)
            except (TypeError, ValueError):
                continue
            merged[pid] = max(0.0, min(1.0, val))
        new_channels[target_cid] = ch.model_copy(
            update={"intelligibility": merged}
        )
        changes.append(
            f"Updated intelligibility on channel '{target_cid}' "
            f"for {sorted((intel_updates or {}).keys())}."
        )

    # 7. Entity location overrides + state_timeline appends.
    new_entities: Dict[str, Entity] = {}
    for eid, ent in ws.entities.items():
        update: dict = {}
        new_loc = patch.update_entity_location.get(eid)
        if new_loc and new_loc != ent.location_id:
            update["location_id"] = new_loc
            changes.append(f"Updated entity '{eid}' location_id → '{new_loc}'.")
        extras = patch.add_state_timeline_entries.get(eid)
        if extras:
            # Rewrite triggered_by through renames before appending.
            normalised_extras: List[EntityStateSnapshot] = []
            for snap in extras:
                snap_update: dict = {}
                if snap.triggered_by and snap.triggered_by in renames:
                    snap_update["triggered_by"] = renames[snap.triggered_by]
                normalised_extras.append(
                    snap.model_copy(update=snap_update) if snap_update else snap
                )
            merged = list(ent.state_timeline) + normalised_extras
            merged.sort(key=_snapshot_sort_key)
            update["state_timeline"] = merged
            changes.append(
                f"Appended {len(normalised_extras)} state_timeline snapshot(s) to entity '{eid}'."
            )
        new_entities[eid] = ent.model_copy(update=update) if update else ent

    # 8. Rewrite event-id and channel-id references throughout entity
    #    timelines and beliefs so renames/drops propagate transparently.
    #    Without this step the patch path silently leaves dangling
    #    Belief.acquired_via_event_id / acquired_via_channel_id refs,
    #    which the validator only flags at warning severity \u2014 and
    #    counterfactual surgery relies on these refs to roll back
    #    beliefs when their source event/channel is removed (so dangling
    #    provenance silently leaks invalidated beliefs into do-surgery).
    needs_rewrite = (
        bool(renames) or bool(drop_evts)
        or bool(drop_chan_ids) or bool(chan_renames)
    )
    if needs_rewrite:
        def _rewrite_belief(b: Belief) -> Tuple[Belief, bool]:
            update: dict = {}
            if b.acquired_via_event_id:
                new_evt = _r(b.acquired_via_event_id)
                if new_evt in drop_evts:
                    update["acquired_via_event_id"] = None
                elif new_evt != b.acquired_via_event_id:
                    update["acquired_via_event_id"] = new_evt
            if b.acquired_via_channel_id:
                new_chan = _rc(b.acquired_via_channel_id)
                if new_chan in drop_chan_ids:
                    update["acquired_via_channel_id"] = None
                elif new_chan != b.acquired_via_channel_id:
                    update["acquired_via_channel_id"] = new_chan
            return (b.model_copy(update=update), True) if update else (b, False)

        rewritten: Dict[str, Entity] = {}
        for eid, ent in new_entities.items():
            ent_update: dict = {}
            # --- Standing beliefs on the entity itself ---
            new_beliefs: List[Belief] = []
            beliefs_changed = False
            for b in ent.beliefs:
                rb, changed = _rewrite_belief(b)
                beliefs_changed = beliefs_changed or changed
                new_beliefs.append(rb)
            if beliefs_changed:
                ent_update["beliefs"] = new_beliefs

            # --- state_timeline snapshots: triggered_by + nested beliefs ---
            tl_changed = False
            new_tl: List[EntityStateSnapshot] = []
            for snap in ent.state_timeline:
                snap_update: dict = {}
                if snap.triggered_by:
                    new_trig = _r(snap.triggered_by)
                    if new_trig in drop_evts:
                        snap_update["triggered_by"] = None
                        tl_changed = True
                    elif new_trig != snap.triggered_by:
                        snap_update["triggered_by"] = new_trig
                        tl_changed = True
                # beliefs_added inside the snapshot
                if snap.beliefs_added:
                    snap_beliefs: List[Belief] = []
                    snap_beliefs_changed = False
                    for b in snap.beliefs_added:
                        rb, changed = _rewrite_belief(b)
                        snap_beliefs_changed = snap_beliefs_changed or changed
                        snap_beliefs.append(rb)
                    if snap_beliefs_changed:
                        snap_update["beliefs_added"] = snap_beliefs
                        tl_changed = True
                new_tl.append(snap.model_copy(update=snap_update) if snap_update else snap)
            if tl_changed:
                ent_update["state_timeline"] = new_tl

            rewritten[eid] = ent.model_copy(update=ent_update) if ent_update else ent
        new_entities = rewritten

        # --- Utterance via_channel_id (forward renames, null drops) ---
        if drop_chan_ids or chan_renames:
            updated_events: List[EventNode] = []
            for evt in new_events:
                if evt.event_type == "utterance" and evt.via_channel_id:
                    new_chan = _rc(evt.via_channel_id)
                    if new_chan in drop_chan_ids:
                        updated_events.append(evt.model_copy(update={"via_channel_id": None}))
                        changes.append(
                            f"Cleared dangling via_channel_id on utterance '{evt.id}' "
                            f"(channel was dropped by patch)."
                        )
                        continue
                    if new_chan != evt.via_channel_id:
                        updated_events.append(evt.model_copy(update={"via_channel_id": new_chan}))
                        continue
                updated_events.append(evt)
            new_events = updated_events

        # --- World-trait state_timeline triggered_by ---
        # The patch contract advertises that EVT renames/drops cascade
        # everywhere. Without this block the world-trait timeline
        # silently retains stale ids; the validator only catches it
        # at warning severity so a renamed event can leak unfixed
        # provenance into counterfactual surgery.
        if renames or drop_evts:
            new_world_traits: Dict[str, GlobalTrait] = {}
            wt_changed_any = False
            for wid, wt in ws.world_traits.items():
                wt_tl_changed = False
                new_wt_tl: List[Any] = []
                for snap in wt.state_timeline:
                    if snap.triggered_by:
                        new_trig = _r(snap.triggered_by)
                        if new_trig in drop_evts:
                            new_wt_tl.append(snap.model_copy(update={"triggered_by": None}))
                            wt_tl_changed = True
                            continue
                        if new_trig != snap.triggered_by:
                            new_wt_tl.append(snap.model_copy(update={"triggered_by": new_trig}))
                            wt_tl_changed = True
                            continue
                    new_wt_tl.append(snap)
                if wt_tl_changed:
                    new_world_traits[wid] = wt.model_copy(update={"state_timeline": new_wt_tl})
                    wt_changed_any = True
                    changes.append(
                        f"Forwarded EVT renames/drops on world_trait '{wid}' state_timeline."
                    )
                else:
                    new_world_traits[wid] = wt
            patched_world_traits = new_world_traits if wt_changed_any else ws.world_traits
        else:
            patched_world_traits = ws.world_traits
    else:
        patched_world_traits = ws.world_traits

    # ----- Phase E: proposition / concern / belief-prop ops -----
    # Index existing propositions by id; apply add_propositions then
    # update_proposition_snapshots and commit_proposition_truth.
    prop_index: Dict[str, Proposition] = {
        p.proposition_id: p for p in ws.propositions
    }
    for pid, prop in patch.add_propositions.items():
        if pid in prop_index:
            changes.append(
                f"add_propositions: proposition '{pid}' already exists \u2014 skipped."
            )
            continue
        prop_index[pid] = prop
        changes.append(f"Added proposition '{pid}'.")

    # Forward EVT renames/drops onto incoming snapshot triggered_by
    # ids so the patch contract stays consistent with the rest of the
    # apply path.
    def _normalise_trigger(trig: Optional[str]) -> Optional[str]:
        if not trig:
            return trig
        new_trig = _r(trig)
        if new_trig in drop_evts:
            return None
        return new_trig

    for pid, snaps in patch.update_proposition_snapshots.items():
        prop = prop_index.get(pid)
        if prop is None:
            changes.append(
                f"update_proposition_snapshots: unknown PROP_ id '{pid}' \u2014 skipped."
            )
            continue
        normalised: List[PropositionSnapshot] = []
        for snap in snaps:
            new_trig = _normalise_trigger(snap.triggered_by)
            if new_trig != snap.triggered_by:
                normalised.append(snap.model_copy(update={"triggered_by": new_trig}))
            else:
                normalised.append(snap)
        merged = list(prop.state_timeline) + normalised
        merged.sort(key=lambda s: (s.fabula_time, s.triggered_by or ""))
        prop_index[pid] = prop.model_copy(update={"state_timeline": merged})
        changes.append(
            f"Appended {len(normalised)} proposition snapshot(s) to '{pid}'."
        )

    for pid, commits in patch.commit_proposition_truth.items():
        prop = prop_index.get(pid)
        if prop is None:
            changes.append(
                f"commit_proposition_truth: unknown PROP_ id '{pid}' \u2014 skipped."
            )
            continue
        new_truth = dict(prop.truth_at_fabula)
        for fab, val in commits.items():
            new_truth[int(fab)] = bool(val)
        prop_index[pid] = prop.model_copy(update={"truth_at_fabula": new_truth})
        changes.append(
            f"Committed truth for '{pid}' at fabula={sorted(int(f) for f in commits)}."
        )

    new_propositions = [prop_index[pid] for pid in sorted(prop_index)]

    # add_concerns + update_concern_snapshots + set_belief_proposition_ids
    # — fold onto entities. Build concern dedup keys per entity to make
    # the op idempotent.
    if (
        patch.add_concerns
        or patch.update_concern_snapshots
        or patch.set_belief_proposition_ids
    ):
        bp_index: Dict[Tuple[str, str], str] = {
            (a.entity_id, a.target_id): a.proposition_id
            for a in patch.set_belief_proposition_ids
        }
        rewritten: Dict[str, Entity] = {}
        for eid, ent in new_entities.items():
            ent_update: Dict[str, Any] = {}

            # Belief proposition_id backfill.
            ent_assignments = {
                tgt: prop for (e_id, tgt), prop in bp_index.items()
                if e_id == eid
            }
            if ent_assignments:
                new_beliefs: List[Belief] = []
                hit = 0
                for b in ent.beliefs:
                    if b.target_id in ent_assignments:
                        new_beliefs.append(
                            b.model_copy(update={
                                "proposition_id": ent_assignments[b.target_id],
                            })
                        )
                        hit += 1
                    else:
                        new_beliefs.append(b)
                if hit:
                    ent_update["beliefs"] = new_beliefs
                    changes.append(
                        f"Set proposition_id on {hit} belief(s) of entity '{eid}'."
                    )

            # add_concerns — append, dedup on (proposition_id, polarity).
            new_concerns_for_ent = patch.add_concerns.get(eid, [])
            updates_for_ent = patch.update_concern_snapshots.get(eid, {})

            current_concerns: List[Concern] = list(ent.concerns)
            concerns_dirty = False
            if new_concerns_for_ent:
                existing_keys = {
                    (c.proposition_id, c.polarity) for c in current_concerns
                }
                added = 0
                for c in new_concerns_for_ent:
                    if (c.proposition_id, c.polarity) in existing_keys:
                        continue
                    current_concerns.append(c)
                    existing_keys.add((c.proposition_id, c.polarity))
                    added += 1
                if added:
                    concerns_dirty = True
                    changes.append(
                        f"Added {added} concern(s) to entity '{eid}'."
                    )

            # update_concern_snapshots
            if updates_for_ent:
                concern_idx = {c.concern_id: i for i, c in enumerate(current_concerns)}
                touched = 0
                for ccn_id, snaps in updates_for_ent.items():
                    if ccn_id not in concern_idx:
                        changes.append(
                            f"update_concern_snapshots: unknown CCN_ id '{ccn_id}' "
                            f"on entity '{eid}' \u2014 skipped."
                        )
                        continue
                    idx = concern_idx[ccn_id]
                    concern = current_concerns[idx]
                    normalised: List[ConcernSnapshot] = []
                    for snap in snaps:
                        new_trig = _normalise_trigger(snap.triggered_by)
                        if new_trig != snap.triggered_by:
                            normalised.append(snap.model_copy(update={"triggered_by": new_trig}))
                        else:
                            normalised.append(snap)
                    merged = list(concern.state_timeline) + normalised
                    merged.sort(key=lambda s: (s.fabula_time, s.triggered_by or ""))
                    current_concerns[idx] = concern.model_copy(
                        update={"state_timeline": merged},
                    )
                    touched += len(normalised)
                if touched:
                    concerns_dirty = True
                    changes.append(
                        f"Appended {touched} concern snapshot(s) on entity '{eid}'."
                    )

            if concerns_dirty:
                ent_update["concerns"] = current_concerns

            rewritten[eid] = ent.model_copy(update=ent_update) if ent_update else ent
        new_entities = rewritten

    new_ws = WorldStateV1(
        locations=ws.locations,
        objects=ws.objects,
        entities=new_entities,
        events=new_events,
        world_traits=patched_world_traits,
        narrative_style=ws.narrative_style,
        causal_topology=new_causal,
        spatial_topology=new_spatial,
        channels=new_channels,
        social_topology=new_social,
        propositions=new_propositions,
        world_facts=list(ws.world_facts),
    )
    return new_ws, changes


def _build_correction_patch_agent(
    config: ExtractionConfig,
) -> Agent[None, WorldStatePatch]:
    """Construct the patch-based correction agent (preferred path).

    The agent receives the current WorldStateV1 + a list of programmatic
    errors and is asked to emit a *small* :class:`WorldStatePatch` that
    fixes only what the errors named. This is dramatically more robust
    than asking it to re-emit the entire WorldStateV1, which had the
    failure mode of silently dropping whole topology fields when the
    response token budget ran out.
    """
    return Agent(
        _resolve_model(config.model),
        output_type=NativeOutput(WorldStatePatch),
        system_prompt=_load_prompt("correction.md"),
        retries=config.output_retries,
    )


def _is_correction_regression(
    before: WorldStateV1, after: WorldStateV1,
) -> Optional[str]:
    """Return a human-readable reason if *after* has lost too much vs *before*.

    Used as a circuit-breaker on patch application: if the LLM somehow
    drops more than half of any topology or any entities, we reject the
    patch and keep the previous state. Returns ``None`` when the post-
    correction state is acceptable.
    """
    def _ratio(a: int, b: int) -> float:
        return (a / b) if b > 0 else 1.0

    if len(after.entities) < len(before.entities):
        return (
            f"entities shrank from {len(before.entities)} to "
            f"{len(after.entities)} (correction is not allowed to drop "
            f"entities)"
        )
    if _ratio(len(after.events), len(before.events)) < 0.8:
        return (
            f"events shrank from {len(before.events)} to "
            f"{len(after.events)} (>20% loss)"
        )
    if before.causal_topology and _ratio(len(after.causal_topology), len(before.causal_topology)) < 0.5:
        return (
            f"causal_topology shrank from {len(before.causal_topology)} "
            f"to {len(after.causal_topology)} (>50% loss)"
        )
    if before.social_topology and _ratio(len(after.social_topology), len(before.social_topology)) < 0.5:
        return (
            f"social_topology shrank from {len(before.social_topology)} "
            f"to {len(after.social_topology)} (>50% loss)"
        )
    if before.spatial_topology and _ratio(len(after.spatial_topology), len(before.spatial_topology)) < 0.5:
        return (
            f"spatial_topology shrank from {len(before.spatial_topology)} "
            f"to {len(after.spatial_topology)} (>50% loss)"
        )
    # Channels and world-traits are *narrative ontology* — a correction
    # patch dropping more than half of either is almost certainly a
    # destructive hallucination. Channel loss in particular silently
    # severs every belief / utterance provenance edge that pointed at
    # the dropped CHN_, which the existing belief-provenance warnings
    # only surface *after* corruption has been persisted.
    if before.channels and _ratio(len(after.channels), len(before.channels)) < 0.5:
        return (
            f"channels shrank from {len(before.channels)} to "
            f"{len(after.channels)} (>50% loss)"
        )
    if before.world_traits and _ratio(len(after.world_traits), len(before.world_traits)) < 0.5:
        return (
            f"world_traits shrank from {len(before.world_traits)} to "
            f"{len(after.world_traits)} (>50% loss)"
        )
    # Propositions / per-entity concerns are first-class affect-layer
    # ontology — the correction agent has no path to legitimately
    # remove >50% of either, so a regression there is almost certainly
    # the same destructive-rewrite failure mode this circuit-breaker
    # exists to catch.
    if before.propositions and _ratio(len(after.propositions), len(before.propositions)) < 0.5:
        return (
            f"propositions shrank from {len(before.propositions)} to "
            f"{len(after.propositions)} (>50% loss)"
        )
    before_concerns = sum(len(e.concerns) for e in before.entities.values())
    after_concerns = sum(len(e.concerns) for e in after.entities.values())
    if before_concerns and _ratio(after_concerns, before_concerns) < 0.5:
        return (
            f"total concerns shrank from {before_concerns} to "
            f"{after_concerns} (>50% loss)"
        )
    return None


_EVT_ID_RE = re.compile(r"\bEVT_[A-Za-z0-9_]+")
_ENT_ID_RE = re.compile(r"\bENT_[A-Za-z0-9_]+")
_CHN_ID_RE = re.compile(r"\bCHN_[A-Za-z0-9_]+")
_LOC_ID_RE = re.compile(r"\bLOC_[A-Za-z0-9_]+")
_OBJ_ID_RE = re.compile(r"\bOBJ_[A-Za-z0-9_]+")
_WORLD_ID_RE = re.compile(r"\bWORLD_[A-Za-z0-9_]+")


def _build_correction_subgraph(
    world_state: WorldStateV1,
    prog_errors: List["ValidationIssue"],
) -> Optional[str]:
    """Return a JSON subgraph that focuses on error-relevant nodes.

    Used by ``_run_correction_patch`` when the full WorldState exceeds
    the configured threshold and would otherwise crowd out the LLM's
    reasoning budget. The subgraph contains:

    * ontology header (locations, objects, entities, world_traits —
      keys + names + minimal metadata);
    * every event whose id appears in any error detail, plus one-hop
      causal neighbours;
    * the causal edges between any two seed/neighbour events;
    * every channel whose id appears in any error detail;
    * every spatial edge touching a seed location/entity;
    * every social edge touching a seed entity.

    Returns ``None`` only when NO ids of any kind could be extracted
    from the errors (the caller should then fall back to the full
    state, which may still be necessary for purely textual errors).
    """
    seed_evt: set[str] = set()
    seed_ent: set[str] = set()
    seed_chn: set[str] = set()
    seed_loc: set[str] = set()
    seed_obj: set[str] = set()
    seed_world: set[str] = set()
    for err in prog_errors:
        seed_evt.update(_EVT_ID_RE.findall(err.detail))
        seed_ent.update(_ENT_ID_RE.findall(err.detail))
        seed_chn.update(_CHN_ID_RE.findall(err.detail))
        seed_loc.update(_LOC_ID_RE.findall(err.detail))
        seed_obj.update(_OBJ_ID_RE.findall(err.detail))
        seed_world.update(_WORLD_ID_RE.findall(err.detail))

    if not (seed_evt or seed_ent or seed_chn or seed_loc or seed_obj or seed_world):
        return None

    valid_event_ids = {e.id for e in world_state.events}
    seed_evt &= valid_event_ids

    # One-hop causal expansion from seed events
    expanded_evt = set(seed_evt)
    for ce in world_state.causal_topology:
        if ce.source_id in seed_evt and ce.target_id in valid_event_ids:
            expanded_evt.add(ce.target_id)
        if ce.target_id in seed_evt and ce.source_id in valid_event_ids:
            expanded_evt.add(ce.source_id)

    relevant_events = [e for e in world_state.events if e.id in expanded_evt]
    relevant_causal = [
        ce for ce in world_state.causal_topology
        if ce.source_id in expanded_evt and ce.target_id in expanded_evt
    ]

    # Channels referenced in errors
    relevant_channels = [
        ch.model_dump(mode="json")
        for ch in world_state.channels if ch.id in seed_chn
    ]

    # Spatial edges touching any seed location or entity
    relevant_spatial = [
        se.model_dump(mode="json") for se in world_state.spatial_topology
        if se.source_id in seed_loc or se.target_id in seed_loc
        or se.source_id in seed_ent or se.target_id in seed_ent
    ]

    # Social edges touching any seed entity
    relevant_social = [
        sr.model_dump(mode="json") for sr in world_state.social_topology
        if sr.source_id in seed_ent or sr.target_id in seed_ent
    ]

    # Ontology header — names only, no nested timelines / beliefs.
    ontology = {
        "locations": {lid: loc.name for lid, loc in world_state.locations.items()},
        "objects": {oid: obj.name for oid, obj in world_state.objects.items()},
        "entities": {
            eid: {"name": ent.name, "location_id": ent.location_id, "status": ent.status}
            for eid, ent in world_state.entities.items()
        },
        "world_traits": {
            wid: wt.name for wid, wt in world_state.world_traits.items()
        },
    }

    import json as _json
    payload = {
        "_subgraph_note": (
            f"Error-relevant subgraph: {len(relevant_events)}/"
            f"{len(world_state.events)} events, "
            f"{len(relevant_channels)}/{len(world_state.channels)} channels, "
            f"{len(relevant_spatial)}/{len(world_state.spatial_topology)} spatial edges, "
            f"{len(relevant_social)}/{len(world_state.social_topology)} social edges shown. "
            f"Patch ids must target the full WorldState."
        ),
        "ontology_header": ontology,
        "events": [e.model_dump(mode="json") for e in relevant_events],
        "causal_topology": [ce.model_dump(mode="json") for ce in relevant_causal],
        "channels": relevant_channels,
        "spatial_topology": relevant_spatial,
        "social_topology": relevant_social,
    }
    return _json.dumps(payload, indent=2)


def _format_correction_history(
    history: Optional[List[Dict[str, Any]]],
) -> str:
    """Render previous correction-loop iterations into a prompt block.

    Returns ``""`` when there's no prior history (first iteration). Each
    history entry surfaces what the LLM tried last time and what the
    deterministic side did with it (applied / regression / empty /
    failed) so the agent can change strategy instead of re-emitting the
    same patch and oscillating against the regression guard.
    """
    if not history:
        return ""
    lines: List[str] = [
        "PRIOR CORRECTION ATTEMPTS (do NOT repeat strategies that "
        "previously failed or were rejected):"
    ]
    for entry in history:
        it = entry.get("iteration", "?")
        status = entry.get("status", "unknown")
        if entry.get("changes"):
            change_preview = "; ".join(entry["changes"][:5])
            if len(entry["changes"]) > 5:
                change_preview += f"; …+{len(entry['changes']) - 5} more"
            lines.append(
                f"  - iter {it} [{status}] applied: {change_preview}"
            )
        if entry.get("errors"):
            err_preview = "; ".join(entry["errors"][:5])
            if len(entry["errors"]) > 5:
                err_preview += f"; …+{len(entry['errors']) - 5} more"
            lines.append(
                f"    errors at start of iter {it}: {err_preview}"
            )
        notes = entry.get("note")
        if notes:
            lines.append(f"    note: {notes}")
    return "\n".join(lines) + "\n\n"


def _run_correction_patch(
    world_state: WorldStateV1,
    prog_errors: List["ValidationIssue"],
    config: ExtractionConfig,
    log_prefix: str,
    history: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[WorldStateV1, List[str], str]:
    """Run one correction-agent iteration and apply the resulting patch.

    Parameters
    ----------
    history : list of dicts, optional
        Compact log of previous correction iterations on this same
        ``world_state``. Each entry has keys ``iteration`` (1-based),
        ``status`` (one of the return statuses below), and either
        ``errors`` (list of ``"[category] detail"`` strings) or
        ``changes`` (list of repair lines). Surfaced verbatim in the
        correction prompt so the LLM can see what was already tried
        and avoid re-emitting patches that previously oscillated, were
        rejected by the regression guard, or returned empty.

    Returns ``(new_world_state, repairs_applied, status)`` where status
    is one of:

    * ``"applied"`` — patch applied successfully (``repairs_applied``
      is non-empty).
    * ``"empty_patch"`` — the agent intentionally returned an empty
      patch (it could not find a safe fix); the orchestrator should
      stop retrying because re-asking will produce the same result.
    * ``"regression"`` — patch was rejected by the regression guard;
      orchestrator may continue but the same errors will likely
      recur, so the oscillation guard should kick in.
    * ``"agent_failed"`` — transient LLM error; orchestrator may
      retry within the remaining budget.
    * ``"apply_failed"`` — patch could not be applied to the state;
      orchestrator may retry.

    On any non-``"applied"`` status the original *world_state* is
    returned unchanged.
    """
    try:
        agent = _build_correction_patch_agent(config)
    except Exception:
        logger.exception("%s Failed to build correction patch agent.", log_prefix)
        return world_state, [], "agent_failed"

    ws_json = world_state.model_dump_json(indent=2)
    error_summary = "\n".join(
        f"  [{e.category}] {e.detail}" for e in prog_errors
    )

    # Item #15: when the serialized state is too large to fit comfortably
    # in a single LLM context window, send only the error-relevant
    # subgraph (events mentioned in the error details + their immediate
    # causal neighbours + ontology header) instead of the full state.
    # The patch contract still applies to the full state on the way out.
    SUBGRAPH_THRESHOLD = config.correction_subgraph_threshold_chars
    state_payload = ws_json
    payload_note = ""
    if len(ws_json) > SUBGRAPH_THRESHOLD:
        subgraph_json = _build_correction_subgraph(world_state, prog_errors)
        if subgraph_json is not None and len(subgraph_json) < len(ws_json):
            state_payload = subgraph_json
            payload_note = (
                "\n\nNOTE: The full WorldState is too large to fit in a "
                "single prompt. Only an error-relevant SUBGRAPH is shown "
                "below (events referenced by the errors + their immediate "
                "causal neighbours + the ontology header). Your patch "
                "MUST still target ids that exist in the full WorldState; "
                "do NOT add edges that depend on context you cannot see "
                "here.\n"
            )
            logger.info(
                "%s Using subgraph payload (%d \u2192 %d chars) for correction.",
                log_prefix, len(ws_json), len(subgraph_json),
            )

    correction_msg = (
        f"The following {len(prog_errors)} programmatic error(s) were found "
        f"in the WorldStateV1 below. Emit a WorldStatePatch that fixes ONLY "
        f"these errors. Do not re-emit the entire world state. Do not drop "
        f"unrelated edges, events, or entities. If you cannot determine a "
        f"safe fix, leave the patch empty and explain why in `notes`.\n\n"
        f"{_format_correction_history(history)}"
        f"ERRORS:\n{error_summary}{payload_note}\n\n"
        f"WORLD STATE:\n{state_payload}"
    )

    try:
        result = agent.run_sync(correction_msg, **_user_kwargs())
    except Exception:
        logger.exception("%s Correction agent FAILED — keeping previous state.", log_prefix)
        return world_state, [], "agent_failed"

    patch: WorldStatePatch = result.output
    if (
        not patch.event_renames
        and not patch.channel_renames
        and not patch.drop_event_ids
        and not patch.update_event_fields
        and not patch.update_entity_location
        and not patch.add_state_timeline_entries
        and not patch.drop_causal_edges
        and not patch.add_causal_edges
        and not patch.drop_social_edges
        and not patch.add_social_edges
        and not patch.drop_spatial_edges
        and not patch.add_spatial_edges
        and not patch.drop_channel_ids
        and not patch.add_channels
    ):
        logger.info(
            "%s Correction agent returned an empty patch (notes: %r).",
            log_prefix, patch.notes,
        )
        return world_state, [], "empty_patch"

    try:
        new_ws, changes = _apply_world_state_patch(world_state, patch)
    except Exception:
        logger.exception(
            "%s Failed to apply correction patch — keeping previous state.",
            log_prefix,
        )
        return world_state, [], "apply_failed"

    regression = _is_correction_regression(world_state, new_ws)
    if regression:
        logger.warning(
            "%s Rejected correction patch: %s. Keeping previous state. "
            "Patch notes: %r",
            log_prefix, regression, patch.notes,
        )
        return world_state, [], "regression"

    logger.info(
        "%s Applied correction patch: %d change(s). Notes: %r",
        log_prefix, len(changes), patch.notes,
    )
    return new_ws, changes, "applied"


# =====================================================================
# Step 5 — Post-Assembly World Trait Timeline Extraction
# =====================================================================


class _WorldTraitTimelineDeps(BaseModel):
    """Dependencies for Step 5 — world trait timeline extraction."""
    model_config = {"protected_namespaces": ()}
    world_traits: Dict[str, GlobalTrait]
    events: List[EventNode]


def _build_world_trait_timeline_agent(
    config: ExtractionConfig,
) -> Agent[_WorldTraitTimelineDeps, WorldTraitTimelineExtraction]:
    """Construct the Step 5 World Trait Timeline agent."""
    agent: Agent[_WorldTraitTimelineDeps, WorldTraitTimelineExtraction] = Agent(
        _resolve_model(config.model),
        deps_type=_WorldTraitTimelineDeps,
        output_type=NativeOutput(WorldTraitTimelineExtraction),
        system_prompt=_load_prompt("world_trait_timeline.md"),
        retries=config.output_retries,
    )

    @agent.system_prompt
    def inject_world_traits_and_events(ctx: RunContext[_WorldTraitTimelineDeps]) -> str:
        # Format world traits
        trait_lines: List[str] = []
        for wid, wt in ctx.deps.world_traits.items():
            trait_lines.append(
                f"  {wid} ({wt.name}): category={wt.category}, "
                f"magnitude={wt.magnitude.value:.2f}, inertia={wt.magnitude.inertia:.2f}\n"
                f"    Description: {wt.description}"
            )
        traits_block = "\n".join(trait_lines) if trait_lines else "(No world traits.)"

        # Format event timeline (compact)
        event_lines: List[str] = []
        for evt in ctx.deps.events:
            actors = ", ".join(evt.actor_ids) if evt.actor_ids else "none"
            event_lines.append(
                f"  {evt.id} (fabula={evt.fabula_time}, type={evt.event_type}, "
                f"actors=[{actors}]): {evt.description}"
            )
        events_block = "\n".join(event_lines) if event_lines else "(No events.)"

        return (
            "=== WORLD TRAITS (from Step 1d) ===\n"
            f"WORLD TRAIT IDs: {sorted(ctx.deps.world_traits.keys())}\n\n"
            f"{traits_block}\n\n"
            "=== COMPLETE EVENT TIMELINE (from Steps 2-4) ===\n"
            f"{events_block}\n\n"
            "Identify inflection points where the above world traits change "
            "due to specific events. Only include traits that actually change."
        )

    @agent.output_validator
    def validate_world_trait_timeline(
        ctx: RunContext[_WorldTraitTimelineDeps],
        result: WorldTraitTimelineExtraction,
    ) -> WorldTraitTimelineExtraction:
        """Validate that all referenced IDs are valid."""
        valid_world_ids = set(ctx.deps.world_traits.keys())
        valid_event_ids = {e.id for e in ctx.deps.events}
        event_fabula_map = {e.id: e.fabula_time for e in ctx.deps.events}
        bad: List[str] = []

        for wid, snapshots in result.timelines.items():
            if wid not in valid_world_ids:
                bad.append(f"World trait ID '{wid}' is not in the register.")
                continue
            for snap in snapshots:
                if snap.triggered_by and snap.triggered_by not in valid_event_ids:
                    bad.append(
                        f"WorldTraitSnapshot for '{wid}' references "
                        f"unknown event '{snap.triggered_by}'."
                    )
                if snap.triggered_by and snap.triggered_by in event_fabula_map:
                    expected_ft = event_fabula_map[snap.triggered_by]
                    if snap.fabula_time != expected_ft:
                        bad.append(
                            f"WorldTraitSnapshot for '{wid}' has fabula_time={snap.fabula_time} "
                            f"but triggered_by event '{snap.triggered_by}' has "
                            f"fabula_time={expected_ft}. They must match."
                        )

        if bad:
            raise ModelRetry(
                "The following issues were found in the world trait timeline. "
                "Fix them:\n" + "\n".join(bad)
            )
        return result

    return agent


def extract_world_trait_timelines(
    ws: WorldStateV1,
    config: ExtractionConfig | None = None,
) -> WorldStateV1:
    """Step 5: Post-assembly world trait timeline extraction.

    Given a fully assembled ``WorldStateV1`` with all events resolved,
    runs a single focused LLM call to identify inflection points where
    world traits changed due to specific events.

    Returns the world state with ``state_timeline`` populated on each
    ``GlobalTrait`` that experienced changes.
    """
    config = config or ExtractionConfig()

    if not ws.world_traits:
        logger.info("[Step 5] No world traits — skipping timeline extraction.")
        return ws

    agent = _build_world_trait_timeline_agent(config)
    deps = _WorldTraitTimelineDeps(
        world_traits=ws.world_traits,
        events=ws.events,
    )

    logger.info(
        "[Step 5] Extracting world trait timelines (%d traits, %d events) …",
        len(ws.world_traits), len(ws.events),
    )

    try:
        result = agent.run_sync(
            f"Analyze the following {len(ws.world_traits)} world trait(s) against "
            f"{len(ws.events)} events and identify any inflection points.",
            deps=deps,
            **_user_kwargs(),
        )
        extraction = result.output
        log_agent_output(logger, "WorldTraitTimeline", extraction)
    except Exception:
        logger.exception("[Step 5] World trait timeline extraction FAILED — skipping.")
        return ws

    # Apply timelines to the world state
    updated_traits: Dict[str, GlobalTrait] = {}
    changes_applied = 0
    for wid, wt in ws.world_traits.items():
        if wid in extraction.timelines and extraction.timelines[wid]:
            sorted_timeline = sorted(extraction.timelines[wid], key=_snapshot_sort_key)
            updated_traits[wid] = wt.model_copy(update={"state_timeline": sorted_timeline})
            changes_applied += 1
            logger.info(
                "[Step 5] %s: %d inflection point(s) identified.",
                wid, len(sorted_timeline),
            )
        else:
            updated_traits[wid] = wt

    ws = ws.model_copy(update={"world_traits": updated_traits})
    logger.info(
        "[Step 5] World trait timeline extraction complete — %d/%d traits changed.",
        changes_applied, len(ws.world_traits),
    )
    return ws


async def extract_world_trait_timelines_async(
    ws: WorldStateV1,
    config: ExtractionConfig | None = None,
) -> WorldStateV1:
    """Async variant of :func:`extract_world_trait_timelines`."""
    config = config or ExtractionConfig()

    if not ws.world_traits:
        logger.info("[Step 5·Async] No world traits — skipping timeline extraction.")
        return ws

    agent = _build_world_trait_timeline_agent(config)
    deps = _WorldTraitTimelineDeps(
        world_traits=ws.world_traits,
        events=ws.events,
    )

    logger.info(
        "[Step 5·Async] Extracting world trait timelines (%d traits, %d events) …",
        len(ws.world_traits), len(ws.events),
    )

    try:
        result = await agent.run(
            f"Analyze the following {len(ws.world_traits)} world trait(s) against "
            f"{len(ws.events)} events and identify any inflection points.",
            deps=deps,
            **_user_kwargs(),
        )
        extraction = result.output
    except Exception:
        logger.exception("[Step 5·Async] World trait timeline extraction FAILED — skipping.")
        return ws

    updated_traits: Dict[str, GlobalTrait] = {}
    changes_applied = 0
    for wid, wt in ws.world_traits.items():
        if wid in extraction.timelines and extraction.timelines[wid]:
            sorted_timeline = sorted(extraction.timelines[wid], key=_snapshot_sort_key)
            updated_traits[wid] = wt.model_copy(update={"state_timeline": sorted_timeline})
            changes_applied += 1
            logger.info(
                "[Step 5·Async] %s: %d inflection point(s) identified.",
                wid, len(sorted_timeline),
            )
        else:
            updated_traits[wid] = wt

    ws = ws.model_copy(update={"world_traits": updated_traits})
    logger.info(
        "[Step 5·Async] World trait timeline extraction complete — %d/%d traits changed.",
        changes_applied, len(ws.world_traits),
    )
    return ws


# =====================================================================
# Step 5c — Entity Concern Extraction (Affect Unification, Step 2)
# =====================================================================


class ConcernRegister(BaseModel):
    """Per-entity Step-5c output — the entity's standing concerns."""
    model_config = {"protected_namespaces": ()}
    concerns: List[Concern] = Field(
        default_factory=list,
        description="3–7 standing fears / desires for this entity. May be empty.",
    )


class _ConcernExtractionDeps(BaseModel):
    """Per-entity dependencies for the Step-5c concern agent."""
    model_config = {"protected_namespaces": ()}
    entity: Entity
    propositions: List[Proposition]
    involving_events: List[EventNode]


def _build_concern_extraction_agent(
    config: ExtractionConfig,
) -> Agent[_ConcernExtractionDeps, ConcernRegister]:
    """Construct the Step 5c entity-concern agent."""
    agent: Agent[_ConcernExtractionDeps, ConcernRegister] = Agent(
        _resolve_model(config.model),
        deps_type=_ConcernExtractionDeps,
        output_type=NativeOutput(ConcernRegister),
        system_prompt=_load_prompt("concern_extraction.md"),
        retries=config.output_retries,
    )

    @agent.system_prompt
    def inject_entity_and_propositions(
        ctx: RunContext[_ConcernExtractionDeps],
    ) -> str:
        ent = ctx.deps.entity
        # Compact entity card.
        trait_lines = [
            f"  {k}: value={v.value:.2f}, inertia={v.inertia:.2f}"
            for k, v in ent.traits.items()
        ]
        traits_block = "\n".join(trait_lines) if trait_lines else "  (none)"
        belief_lines = [
            f"  re {b.target_id}: '{b.perceived_state}' (conf={b.confidence:.2f})"
            for b in ent.beliefs
        ]
        beliefs_block = "\n".join(belief_lines) if belief_lines else "  (none)"

        # Proposition register (compact, capped to keep prompt small).
        prop_cap = 120
        props = ctx.deps.propositions[:prop_cap]
        prop_lines = [
            f"  {p.proposition_id} (kind={p.kind}, stakes={p.stakes:.2f}): "
            f"{p.description}"
            for p in props
        ]
        prop_overflow = (
            f"  …(+{len(ctx.deps.propositions) - prop_cap} more propositions truncated)"
            if len(ctx.deps.propositions) > prop_cap else ""
        )
        prop_block = (
            "\n".join(prop_lines) + (("\n" + prop_overflow) if prop_overflow else "")
        )

        # Events involving this entity.
        evt_cap = 60
        evts = ctx.deps.involving_events[:evt_cap]
        evt_lines = [
            f"  {e.id} (fabula={e.fabula_time}, type={e.event_type}): "
            f"{e.description}"
            for e in evts
        ]
        evt_overflow = (
            f"  …(+{len(ctx.deps.involving_events) - evt_cap} more events truncated)"
            if len(ctx.deps.involving_events) > evt_cap else ""
        )
        evt_block = (
            "\n".join(evt_lines) + (("\n" + evt_overflow) if evt_overflow else "")
        )

        return (
            f"=== ENTITY: {ent.id} ({ent.name}) ===\n"
            f"status={ent.status}; location_id={ent.location_id}\n"
            f"traits:\n{traits_block}\n"
            f"beliefs:\n{beliefs_block}\n\n"
            f"=== EVENTS INVOLVING THIS ENTITY ({len(ctx.deps.involving_events)}) ===\n"
            f"{evt_block}\n\n"
            f"=== PROPOSITION REGISTER ({len(ctx.deps.propositions)}) ===\n"
            f"VALID PROPOSITION IDs: {sorted(p.proposition_id for p in ctx.deps.propositions)[:40]}…\n"
            f"{prop_block}\n\n"
            "Identify this entity's 3–7 standing concerns. Each concern's "
            "``proposition_id`` MUST appear in the register above."
        )

    @agent.output_validator
    def validate_concerns(
        ctx: RunContext[_ConcernExtractionDeps],
        result: ConcernRegister,
    ) -> ConcernRegister:
        valid_prop_ids = {p.proposition_id for p in ctx.deps.propositions}
        seen_pairs: set[tuple[str, str]] = set()
        seen_ccn_ids: set[str] = set()
        bad: List[str] = []
        for c in result.concerns:
            if c.proposition_id not in valid_prop_ids:
                bad.append(
                    f"Concern '{c.concern_id}' references unknown "
                    f"proposition_id '{c.proposition_id}'."
                )
            pair = (c.proposition_id, c.polarity)
            if pair in seen_pairs:
                bad.append(
                    f"Duplicate concern over ({c.proposition_id}, {c.polarity}) "
                    f"— collapse into one."
                )
            seen_pairs.add(pair)
            if c.concern_id in seen_ccn_ids:
                bad.append(f"Duplicate concern_id '{c.concern_id}'.")
            seen_ccn_ids.add(c.concern_id)
        # Validate counter_concern_ids only references siblings in same response.
        for c in result.concerns:
            for ccid in c.counter_concern_ids:
                if ccid not in seen_ccn_ids:
                    bad.append(
                        f"Concern '{c.concern_id}' has counter_concern_ids "
                        f"entry '{ccid}' not present in this response."
                    )
        if bad:
            raise ModelRetry(
                "The following issues were found in the concern register. "
                "Fix them:\n" + "\n".join(bad)
            )
        return result

    return agent


def _entities_eligible_for_concerns(
    ws: WorldStateV1, min_appearances: int,
) -> List[Entity]:
    """Return entities that appear in at least ``min_appearances`` events."""
    counts: Dict[str, int] = {}
    for evt in ws.events:
        for eid in set(evt.actor_ids) | set(evt.target_ids):
            counts[eid] = counts.get(eid, 0) + 1
    return [
        ent for eid, ent in ws.entities.items()
        if counts.get(eid, 0) >= min_appearances
    ]


def _events_involving(ws: WorldStateV1, entity_id: str) -> List[EventNode]:
    return [
        e for e in ws.events
        if entity_id in set(e.actor_ids) or entity_id in set(e.target_ids)
    ]


def _ensure_propositions_for_concerns(ws: WorldStateV1) -> None:
    """Synthesise propositions if absent — concerns reference them by id."""
    if ws.propositions:
        return
    # Lazy import to avoid a circular import at module load time.
    from shadow_loom.affect_unification import synthesise_propositions
    synthesise_propositions(ws)


def extract_entity_concerns(
    ws: WorldStateV1,
    config: ExtractionConfig | None = None,
) -> WorldStateV1:
    """Step 5c: Per-entity standing concern (fear/desire) extraction.

    For every entity that participates in at least
    ``config.concern_min_event_appearances`` events, runs one focused
    LLM call against the synthesised ``Proposition`` register and
    populates ``Entity.concerns``. Idempotent — entities with an
    already-populated ``concerns`` list are skipped.
    """
    config = config or ExtractionConfig()
    if not config.enable_concern_extraction:
        logger.info("[Step 5c] Concern extraction disabled — skipping.")
        return ws
    if not ws.entities:
        return ws

    _ensure_propositions_for_concerns(ws)
    eligible = _entities_eligible_for_concerns(
        ws, config.concern_min_event_appearances,
    )
    eligible = [e for e in eligible if not e.concerns]
    if not eligible:
        logger.info("[Step 5c] No eligible entities for concern extraction.")
        return ws

    agent = _build_concern_extraction_agent(config)
    logger.info(
        "[Step 5c] Extracting concerns for %d entit%s …",
        len(eligible), "y" if len(eligible) == 1 else "ies",
    )

    updated_entities: Dict[str, Entity] = dict(ws.entities)
    n_concerns_total = 0
    for ent in eligible:
        deps = _ConcernExtractionDeps(
            entity=ent,
            propositions=ws.propositions,
            involving_events=_events_involving(ws, ent.id),
        )
        try:
            result = agent.run_sync(
                f"Extract standing concerns for entity {ent.id} ({ent.name}).",
                deps=deps,
                **_user_kwargs(),
            )
            concerns = list(result.output.concerns)
            log_agent_output(logger, f"Concerns·{ent.id}", result.output)
        except Exception:
            logger.exception(
                "[Step 5c] Concern extraction FAILED for %s — leaving empty.",
                ent.id,
            )
            continue
        if concerns:
            updated_entities[ent.id] = ent.model_copy(update={"concerns": concerns})
            n_concerns_total += len(concerns)
            logger.info("[Step 5c] %s: %d concern(s).", ent.id, len(concerns))

    ws = ws.model_copy(update={"entities": updated_entities})
    logger.info(
        "[Step 5c] Concern extraction complete — %d concerns across %d entit%s.",
        n_concerns_total, len(eligible), "y" if len(eligible) == 1 else "ies",
    )
    return ws


async def extract_entity_concerns_async(
    ws: WorldStateV1,
    config: ExtractionConfig | None = None,
) -> WorldStateV1:
    """Async variant of :func:`extract_entity_concerns`.

    Concerns for distinct entities are independent so the LLM calls
    fan out concurrently (bounded by ``config.max_concurrent_chunks``
    when present, else unbounded — the per-entity prompts are small).
    """
    config = config or ExtractionConfig()
    if not config.enable_concern_extraction:
        logger.info("[Step 5c·Async] Concern extraction disabled — skipping.")
        return ws
    if not ws.entities:
        return ws

    _ensure_propositions_for_concerns(ws)
    eligible = _entities_eligible_for_concerns(
        ws, config.concern_min_event_appearances,
    )
    pre_covered = sum(1 for e in eligible if e.concerns)
    eligible = [e for e in eligible if not e.concerns]
    if pre_covered:
        logger.info(
            "[Step 5c\u00b7Async] Phase D gating: %d eligible entit%s already "
            "covered by Phase A/C catalogue seeds \u2014 LLM call skipped for them.",
            pre_covered, "y" if pre_covered == 1 else "ies",
        )
    if not eligible:
        logger.info("[Step 5c\u00b7Async] No eligible entities for concern extraction.")
        return ws

    agent = _build_concern_extraction_agent(config)
    logger.info(
        "[Step 5c·Async] Extracting concerns for %d entit%s …",
        len(eligible), "y" if len(eligible) == 1 else "ies",
    )

    sem_size = getattr(config, "max_concurrent_chunks", 0) or len(eligible)
    sem = asyncio.Semaphore(max(1, sem_size))

    async def _one(ent: Entity) -> tuple[str, List[Concern]]:
        deps = _ConcernExtractionDeps(
            entity=ent,
            propositions=ws.propositions,
            involving_events=_events_involving(ws, ent.id),
        )
        async with sem:
            try:
                result = await agent.run(
                    f"Extract standing concerns for entity {ent.id} ({ent.name}).",
                    deps=deps,
                    **_user_kwargs(),
                )
                return ent.id, list(result.output.concerns)
            except Exception:
                logger.exception(
                    "[Step 5c·Async] Concern extraction FAILED for %s — "
                    "leaving empty.", ent.id,
                )
                return ent.id, []

    results = await asyncio.gather(*(_one(e) for e in eligible))

    updated_entities: Dict[str, Entity] = dict(ws.entities)
    n_concerns_total = 0
    for eid, concerns in results:
        if concerns:
            updated_entities[eid] = updated_entities[eid].model_copy(
                update={"concerns": concerns},
            )
            n_concerns_total += len(concerns)
            logger.info("[Step 5c·Async] %s: %d concern(s).", eid, len(concerns))

    ws = ws.model_copy(update={"entities": updated_entities})
    logger.info(
        "[Step 5c·Async] Concern extraction complete — %d concerns across %d entit%s.",
        n_concerns_total, len(eligible), "y" if len(eligible) == 1 else "ies",
    )
    return ws


# =====================================================================
# Step 5d — Belief Proposition Clustering (Affect Unification, Step 3 LLM half)
# =====================================================================


class BeliefPropositionCluster(BaseModel):
    """One cluster of equivalent ``perceived_state`` strings → one Proposition."""
    model_config = {"protected_namespaces": ()}
    proposition: Proposition
    perceived_states: List[str] = Field(
        description=(
            "Verbatim ``Belief.perceived_state`` strings from the input that "
            "all map to ``proposition``. Every input string must appear in "
            "exactly one cluster."
        ),
    )


class BeliefClusterRegister(BaseModel):
    """Per-target Step-5d output — clusters of equivalent beliefs."""
    model_config = {"protected_namespaces": ()}
    clusters: List[BeliefPropositionCluster] = Field(default_factory=list)


class _BeliefClusteringDeps(BaseModel):
    """Per-target dependencies for the Step-5d clustering agent."""
    model_config = {"protected_namespaces": ()}
    target_id: str
    target_name: str
    target_kind: Literal["entity", "object", "location", "world_trait"]
    # Tuples of (entity_id, perceived_state, confidence) for context.
    held_by: List[tuple[str, str, float]] = Field(default_factory=list)
    # Distinct perceived_state strings to cluster.
    distinct_perceived_states: List[str] = Field(default_factory=list)


def _build_belief_clustering_agent(
    config: ExtractionConfig,
) -> Agent[_BeliefClusteringDeps, BeliefClusterRegister]:
    """Construct the Step 5d belief-proposition-clustering agent."""
    agent: Agent[_BeliefClusteringDeps, BeliefClusterRegister] = Agent(
        _resolve_model(config.model),
        deps_type=_BeliefClusteringDeps,
        output_type=NativeOutput(BeliefClusterRegister),
        system_prompt=_load_prompt("belief_proposition_clustering.md"),
        retries=config.output_retries,
    )

    @agent.system_prompt
    def inject_target_and_beliefs(
        ctx: RunContext[_BeliefClusteringDeps],
    ) -> str:
        held_lines = [
            f"  {eid} (conf={conf:.2f}): {state!r}"
            for eid, state, conf in ctx.deps.held_by
        ]
        distinct_lines = [f"  {i+1}. {s!r}" for i, s in enumerate(ctx.deps.distinct_perceived_states)]
        return (
            f"=== TARGET ===\n"
            f"id={ctx.deps.target_id}; name={ctx.deps.target_name}; "
            f"kind={ctx.deps.target_kind}\n\n"
            f"=== ALL BELIEFS HELD ABOUT THIS TARGET ({len(ctx.deps.held_by)}) ===\n"
            + ("\n".join(held_lines) if held_lines else "  (none)")
            + "\n\n"
            f"=== DISTINCT PERCEIVED_STATE STRINGS TO CLUSTER ({len(ctx.deps.distinct_perceived_states)}) ===\n"
            + ("\n".join(distinct_lines) if distinct_lines else "  (none)")
            + "\n\nCluster the distinct strings into propositions. Every "
            "supplied string MUST appear in exactly one cluster's "
            "``perceived_states``. Negation pairs are TWO propositions, "
            "not one."
        )

    @agent.output_validator
    def validate_clusters(
        ctx: RunContext[_BeliefClusteringDeps],
        result: BeliefClusterRegister,
    ) -> BeliefClusterRegister:
        bad: List[str] = []
        # 1. Every input string must appear exactly once across clusters.
        seen_strings: Dict[str, int] = {}
        for cl in result.clusters:
            for s in cl.perceived_states:
                seen_strings[s] = seen_strings.get(s, 0) + 1
        for required in ctx.deps.distinct_perceived_states:
            if required not in seen_strings:
                bad.append(f"perceived_state {required!r} is missing from all clusters.")
            elif seen_strings.get(required, 0) > 1:
                bad.append(
                    f"perceived_state {required!r} appears in "
                    f"{seen_strings[required]} clusters; must be exactly one."
                )
        # 2. Unique proposition_ids inside this response.
        seen_ids: set[str] = set()
        for cl in result.clusters:
            pid = cl.proposition.proposition_id
            if pid in seen_ids:
                bad.append(f"Duplicate proposition_id {pid!r} in this response.")
            seen_ids.add(pid)
            if not pid.startswith("PROP_"):
                bad.append(f"proposition_id {pid!r} must start with 'PROP_'.")
        # 3. The target_id should appear in each proposition's referent_ids.
        for cl in result.clusters:
            if ctx.deps.target_id not in cl.proposition.referent_ids:
                bad.append(
                    f"Proposition {cl.proposition.proposition_id!r} omits "
                    f"target {ctx.deps.target_id!r} from referent_ids."
                )
        if bad:
            raise ModelRetry(
                "The following issues were found in the belief-cluster register. "
                "Fix them:\n" + "\n".join(bad)
            )
        return result

    return agent


def _gather_unbound_beliefs(
    ws: WorldStateV1,
) -> Dict[str, List[tuple[str, str, float]]]:
    """Group unbound character beliefs by ``target_id``.

    Returns ``{target_id: [(entity_id, perceived_state, confidence), ...]}``
    for every belief without a ``proposition_id`` whose target is NOT an
    event (event-target beliefs are handled by the deterministic
    ``backfill_character_belief_propositions`` pass).
    """
    event_ids = {e.id for e in ws.events}
    out: Dict[str, List[tuple[str, str, float]]] = {}
    for eid, ent in ws.entities.items():
        # Audience entity is synthesised after this pass — skip if seen.
        if eid == "ENT_AUDIENCE":
            continue
        # Walk both initial beliefs and timeline beliefs_added.
        for b in ent.beliefs:
            if b.proposition_id or b.target_id in event_ids:
                continue
            out.setdefault(b.target_id, []).append(
                (eid, b.perceived_state, float(b.confidence)),
            )
        for snap in ent.state_timeline:
            for b in snap.beliefs_added:
                if b.proposition_id or b.target_id in event_ids:
                    continue
                out.setdefault(b.target_id, []).append(
                    (eid, b.perceived_state, float(b.confidence)),
                )
    return out


def _classify_target(ws: WorldStateV1, target_id: str) -> Literal[
    "entity", "object", "location", "world_trait",
]:
    if target_id in ws.entities:
        return "entity"
    if target_id in ws.objects:
        return "object"
    if target_id in ws.locations:
        return "location"
    return "world_trait"


def _target_name(ws: WorldStateV1, target_id: str) -> str:
    for store in (ws.entities, ws.objects, ws.locations, ws.world_traits):
        if target_id in store:
            obj = store[target_id]
            return getattr(obj, "name", target_id) or target_id
    return target_id


def _apply_belief_clusters(
    ws: WorldStateV1,
    target_id: str,
    clusters: List[BeliefPropositionCluster],
) -> int:
    """Assign ``proposition_id`` to every belief about ``target_id`` whose
    ``perceived_state`` matches a cluster member. Returns the number of
    beliefs updated.
    """
    # Build perceived_state → proposition_id index for this target.
    state_to_prop: Dict[str, str] = {}
    for cl in clusters:
        for s in cl.perceived_states:
            state_to_prop[s] = cl.proposition.proposition_id

    n = 0
    for ent in ws.entities.values():
        if ent.id == "ENT_AUDIENCE":
            continue
        for b in ent.beliefs:
            if b.proposition_id or b.target_id != target_id:
                continue
            pid = state_to_prop.get(b.perceived_state)
            if pid:
                b.proposition_id = pid
                n += 1
        for snap in ent.state_timeline:
            for b in snap.beliefs_added:
                if b.proposition_id or b.target_id != target_id:
                    continue
                pid = state_to_prop.get(b.perceived_state)
                if pid:
                    b.proposition_id = pid
                    n += 1
    return n


def cluster_belief_propositions(
    ws: WorldStateV1,
    config: ExtractionConfig | None = None,
) -> WorldStateV1:
    """Step 5d: LLM-driven proposition clustering for non-event beliefs.

    For every (non-event) belief target with at least
    ``config.belief_cluster_min_beliefs`` beliefs across entities,
    runs one focused LLM call to cluster equivalent ``perceived_state``
    strings into shared ``Proposition`` records. Newly emitted
    propositions are appended to ``ws.propositions`` and matched
    beliefs receive a ``proposition_id``.

    Idempotent — beliefs with an existing ``proposition_id`` are
    skipped, and previously clustered targets contribute zero
    distinct strings on a second pass.
    """
    config = config or ExtractionConfig()
    if not getattr(config, "enable_belief_clustering", True):
        logger.info("[Step 5d] Belief clustering disabled — skipping.")
        return ws

    # Always run the cheap deterministic event-target backfill first,
    # then the deterministic single-referent-match backfill. Both are
    # cheap, idempotent, and shrink the work the LLM clustering pass
    # has to do.
    from shadow_loom.affect_unification import (
        backfill_belief_propositions_by_referent,
        backfill_character_belief_propositions,
        synthesise_propositions,
    )
    if not ws.propositions:
        synthesise_propositions(ws)
    n_event_bound = backfill_character_belief_propositions(ws)
    if n_event_bound:
        logger.info(
            "[Step 5d] Deterministic event-target backfill bound %d beliefs.",
            n_event_bound,
        )
    n_referent_bound = backfill_belief_propositions_by_referent(ws)
    if n_referent_bound:
        logger.info(
            "[Step 5d] Deterministic single-referent backfill bound %d beliefs.",
            n_referent_bound,
        )

    grouped = _gather_unbound_beliefs(ws)
    min_beliefs = getattr(config, "belief_cluster_min_beliefs", 2)
    eligible_targets = [t for t, lst in grouped.items() if len(lst) >= min_beliefs]
    if not eligible_targets:
        logger.info("[Step 5d] No eligible targets for belief clustering.")
        return ws

    agent = _build_belief_clustering_agent(config)
    new_props: List[Proposition] = []
    n_total = 0
    for tid in eligible_targets:
        held = grouped[tid]
        distinct = sorted({state for _, state, _ in held})
        deps = _BeliefClusteringDeps(
            target_id=tid,
            target_name=_target_name(ws, tid),
            target_kind=_classify_target(ws, tid),
            held_by=held,
            distinct_perceived_states=distinct,
        )
        try:
            result = agent.run_sync(
                f"Cluster {len(distinct)} distinct perceived_state strings "
                f"about target {tid}.",
                deps=deps,
                **_user_kwargs(),
            )
            log_agent_output(logger, f"BeliefClusters·{tid}", result.output)
        except Exception:
            logger.exception(
                "[Step 5d] Belief clustering FAILED for %s — leaving unbound.",
                tid,
            )
            continue
        clusters = list(result.output.clusters)
        n_bound = _apply_belief_clusters(ws, tid, clusters)
        if n_bound:
            new_props.extend(cl.proposition for cl in clusters)
            n_total += n_bound
            logger.info(
                "[Step 5d] %s: %d clusters, %d beliefs bound.",
                tid, len(clusters), n_bound,
            )

    if new_props:
        # De-dupe by proposition_id against what's already on the world.
        existing = {p.proposition_id for p in ws.propositions}
        ws.propositions = list(ws.propositions) + [
            p for p in new_props if p.proposition_id not in existing
        ]

    logger.info(
        "[Step 5d] Belief clustering complete — %d beliefs bound to %d new propositions.",
        n_total, len(new_props),
    )
    return ws


async def cluster_belief_propositions_async(
    ws: WorldStateV1,
    config: ExtractionConfig | None = None,
) -> WorldStateV1:
    """Async variant of :func:`cluster_belief_propositions`."""
    config = config or ExtractionConfig()
    if not getattr(config, "enable_belief_clustering", True):
        logger.info("[Step 5d·Async] Belief clustering disabled — skipping.")
        return ws

    from shadow_loom.affect_unification import (
        backfill_belief_propositions_by_referent,
        backfill_character_belief_propositions,
        synthesise_propositions,
    )
    if not ws.propositions:
        synthesise_propositions(ws)
    n_event_bound = backfill_character_belief_propositions(ws)
    if n_event_bound:
        logger.info(
            "[Step 5d·Async] Deterministic event-target backfill bound %d beliefs.",
            n_event_bound,
        )
    n_referent_bound = backfill_belief_propositions_by_referent(ws)
    if n_referent_bound:
        logger.info(
            "[Step 5d·Async] Deterministic single-referent backfill bound %d beliefs.",
            n_referent_bound,
        )

    grouped = _gather_unbound_beliefs(ws)
    min_beliefs = getattr(config, "belief_cluster_min_beliefs", 2)
    eligible_targets = [t for t, lst in grouped.items() if len(lst) >= min_beliefs]
    if not eligible_targets:
        logger.info("[Step 5d·Async] No eligible targets for belief clustering.")
        return ws

    agent = _build_belief_clustering_agent(config)
    sem_size = getattr(config, "max_concurrent_chunks", 0) or len(eligible_targets)
    sem = asyncio.Semaphore(max(1, sem_size))

    async def _one(tid: str) -> tuple[str, List[BeliefPropositionCluster]]:
        held = grouped[tid]
        distinct = sorted({state for _, state, _ in held})
        deps = _BeliefClusteringDeps(
            target_id=tid,
            target_name=_target_name(ws, tid),
            target_kind=_classify_target(ws, tid),
            held_by=held,
            distinct_perceived_states=distinct,
        )
        async with sem:
            try:
                result = await agent.run(
                    f"Cluster {len(distinct)} distinct perceived_state "
                    f"strings about target {tid}.",
                    deps=deps,
                    **_user_kwargs(),
                )
                return tid, list(result.output.clusters)
            except Exception:
                logger.exception(
                    "[Step 5d·Async] Belief clustering FAILED for %s — "
                    "leaving unbound.", tid,
                )
                return tid, []

    results = await asyncio.gather(*(_one(t) for t in eligible_targets))

    new_props: List[Proposition] = []
    n_total = 0
    for tid, clusters in results:
        if not clusters:
            continue
        n_bound = _apply_belief_clusters(ws, tid, clusters)
        if n_bound:
            new_props.extend(cl.proposition for cl in clusters)
            n_total += n_bound
            logger.info(
                "[Step 5d·Async] %s: %d clusters, %d beliefs bound.",
                tid, len(clusters), n_bound,
            )

    if new_props:
        existing = {p.proposition_id for p in ws.propositions}
        ws.propositions = list(ws.propositions) + [
            p for p in new_props if p.proposition_id not in existing
        ]

    logger.info(
        "[Step 5d·Async] Belief clustering complete — %d beliefs bound to %d new propositions.",
        n_total, len(new_props),
    )
    return ws


# ----------------------------------------------------------------------
# Step 5e — synthesise ENT_AUDIENCE (Affect Unification, Step 4)
# ----------------------------------------------------------------------
def _maybe_synthesise_audience_entity(
    ws: WorldStateV1,
    config: ExtractionConfig | None = None,
) -> WorldStateV1:
    """Synthesise the reserved ``ENT_AUDIENCE`` entity, if enabled.

    Deterministic, no LLM cost. Walks the syuzhet stream and emits
    graded-confidence ``Belief`` snapshots so the unified affect scorers
    can read audience belief uniformly with character belief. See
    ``shadow_loom.affect_unification.synthesise_audience_entity`` for
    the substrate logic and ``/memories/repo/affect-unification-plan.md``
    Step 4 for context.
    """
    config = config or ExtractionConfig()
    if not getattr(config, "enable_audience_synthesis", True):
        logger.info("[Step 5e] Audience synthesis disabled — skipping.")
        return ws

    from shadow_loom.affect_unification import (
        AUDIENCE_ID,
        synthesise_audience_entity,
        synthesise_propositions,
    )

    if not ws.propositions:
        synthesise_propositions(ws)

    synthesise_audience_entity(ws)

    audience = ws.entities.get(AUDIENCE_ID)
    n_snapshots = len(audience.state_timeline) if audience is not None else 0
    n_beliefs = (
        sum(len(s.beliefs_added) for s in audience.state_timeline)
        if audience is not None else 0
    )
    logger.info(
        "[Step 5e] Audience synthesis complete — %s populated with %d "
        "state-timeline snapshots (%d belief updates).",
        AUDIENCE_ID, n_snapshots, n_beliefs,
    )
    return ws


def _build_compact_validation_view(ws: WorldStateV1) -> str:
    """Return a topology-preserving compact JSON view for LLM validation.

    Keeps every node/edge id, name, type and reference-bearing field so
    the auditor can still spot cross-element contradictions, but strips
    the deep nested arrays (entity beliefs / state_timelines / world
    trait timelines) that dominate serialised size on long manuscripts.
    Used by ``validate_world_state`` when the full ``model_dump_json``
    exceeds the LLM context budget.
    """
    import json as _json

    payload = {
        "_compact_note": (
            "Compact projection: nested entity beliefs / state_timelines "
            "and world-trait state_timelines are omitted. Counts are "
            "given so the auditor can still flag missing-data anomalies."
        ),
        "locations": {
            lid: {"name": loc.name} for lid, loc in ws.locations.items()
        },
        "objects": {
            oid: {"name": obj.name, "owner_id": obj.owner_id}
            for oid, obj in ws.objects.items()
        },
        "entities": {
            eid: {
                "name": ent.name,
                "status": ent.status,
                "location_id": ent.location_id,
                "n_beliefs": len(ent.beliefs),
                "n_state_timeline": len(ent.state_timeline),
                "trait_keys": sorted(ent.traits.keys()),
                # Belief summaries: keep the (target_id, perceived_state,
                # confidence) triple so the LLM auditor can still spot
                # internal contradictions like "two confident beliefs
                # about the same target with opposite perceived_state".
                # Provenance + inertia + evidence_strength are dropped.
                "beliefs_summary": [
                    {
                        "target_id": b.target_id,
                        "perceived_state": b.perceived_state,
                        "confidence": round(float(getattr(b, "confidence", 1.0) or 1.0), 2),
                    }
                    for b in ent.beliefs
                ],
                # State-timeline summary: just the per-tick triggered_by
                # + status / location transitions and the *count* of new
                # beliefs / trait deltas. Lets the auditor catch missing
                # status transitions ("alive entity referenced after
                # EVT_X_KILLS_Y") and movement / location inconsistencies.
                "timeline_summary": [
                    {
                        "fabula_time": s.fabula_time,
                        "triggered_by": s.triggered_by,
                        "status": s.status,
                        "location_id": s.location_id,
                        "n_traits_changed": len(s.traits or {}),
                        "n_beliefs_added": len(s.beliefs_added or []),
                        "n_beliefs_invalidated": len(s.beliefs_invalidated or []),
                    }
                    for s in ent.state_timeline
                ],
            }
            for eid, ent in ws.entities.items()
        },
        "world_traits": {
            wid: {
                "name": wt.name,
                "n_state_timeline": len(wt.state_timeline),
                "timeline_summary": [
                    {
                        "fabula_time": s.fabula_time,
                        "triggered_by": s.triggered_by,
                    }
                    for s in wt.state_timeline
                ],
            }
            for wid, wt in ws.world_traits.items()
        },
        "events": [e.model_dump(mode="json") for e in ws.events],
        "causal_topology": [ce.model_dump(mode="json") for ce in ws.causal_topology],
        "spatial_topology": [se.model_dump(mode="json") for se in ws.spatial_topology],
        "channels": {
            cid: ch.model_dump(mode="json") for cid, ch in ws.channels.items()
        },
        "social_topology": [
            edge.model_dump(mode="json") for edge in ws.social_topology
        ],
    }
    return _json.dumps(payload, indent=2)


def validate_world_state(
    ws: WorldStateV1,
    config: ExtractionConfig | None = None,
) -> ValidationReport:
    """
    Step 3: Run programmatic and LLM-based validation on the
    assembled world state.
    """
    config = config or ExtractionConfig()

    # Phase A: fast programmatic checks
    prog_issues = _programmatic_validation(ws)
    prog_errors = [i for i in prog_issues if i.severity == "error"]
    logger.info(
        "[Step 3·Programmatic] %d issues (%d errors, %d warnings).",
        len(prog_issues), len(prog_errors), len(prog_issues) - len(prog_errors),
    )

    # Phase B: LLM audit for semantic contradictions
    agent = _build_validation_agent(config)
    ws_json = ws.model_dump_json(indent=2)
    # When the full state is too large, send a *compact* projection that
    # preserves topology (events + edges + ontology header) but drops the
    # verbose nested state_timeline and beliefs payloads, which dominate
    # serialized size and rarely host the kind of cross-element semantic
    # contradictions the LLM auditor catches. Prior behaviour silently
    # truncated the JSON tail \u2014 invisible to the auditor and skewed
    # corrections toward front-loaded sections (audit item #9).
    max_chars = config.validation_payload_max_chars
    if len(ws_json) > max_chars:
        compact_payload = _build_compact_validation_view(ws)
        if len(compact_payload) < len(ws_json):
            logger.info(
                "[Step 3\u00b7LLM] WorldState too large (%d chars); using "
                "compact projection (%d chars) for LLM audit.",
                len(ws_json), len(compact_payload),
            )
            ws_json = compact_payload
        else:
            ws_json = ws_json[:max_chars] + "\n... [TRUNCATED]"

    logger.info("[Step 3·LLM] Running validation agent (%d chars) …", len(ws_json))

    # Summarise programmatic findings so the LLM doesn't duplicate them
    if prog_issues:
        prog_summary = "\n".join(
            f"  [{i.severity}/{i.category}] {i.detail}" for i in prog_issues
        )
        preamble = (
            "The following issues were ALREADY found by programmatic validation. "
            "Do NOT re-report them:\n" + prog_summary + "\n\n"
        )
    else:
        preamble = "Programmatic validation found 0 issues.\n\n"

    try:
        result = agent.run_sync(
            preamble + f"Validate the following WorldStateV1 JSON:\n\n{ws_json}",
            **_user_kwargs(),
        )
        llm_report = result.output
        llm_issues = llm_report.issues
        llm_suggestions = llm_report.suggestions
    except Exception as e:  # noqa: BLE001 — never crash import on validator failure
        logger.warning(
            "[Step 3·LLM] Validation agent failed (%s: %s) — "
            "falling back to programmatic-only validation.",
            type(e).__name__, e,
        )
        llm_issues = []
        llm_suggestions = []

    # Merge programmatic + LLM issues
    all_issues = prog_issues + llm_issues
    has_errors = any(i.severity == "error" for i in all_issues)

    merged = ValidationReport(
        is_valid=not has_errors,
        issues=all_issues,
        suggestions=llm_suggestions,
    )
    logger.info(
        "[Step 3] Validation complete — is_valid=%s, %d total issues.",
        merged.is_valid, len(merged.issues),
    )
    return merged


# =====================================================================
# Top-Level Orchestrator
# =====================================================================

async def run_extraction_async(
    text: str,
    config: ExtractionConfig | None = None,
    *,
    user_id: Optional[int] = None,
    project_id: Optional[int] = None,
    version_id: Optional[int] = None,
) -> Tuple[WorldStateV1, ValidationReport]:
    """Run the full 3-step extraction pipeline (async).
    
    Parameters
    ----------
    text : str
        Full narrative prose text.
    config : ExtractionConfig or None
        Pipeline configuration. Uses defaults if None.
    user_id : int, optional
        User ID for cost tracking and audit logging.
    project_id : int, optional
        Project ID for cost tracking context.
    version_id : int, optional
        Version ID for cost tracking context.
        
    Returns
    -------
    (WorldStateV1, ValidationReport)
        The assembled world state and its validation report.
    """

    config = config or ExtractionConfig()
    logger.info("[Pipeline·Async] Starting extraction with model=%s, strategy=%s", config.model, config.chunk_strategy)

    # Prepare user context for cost tracking
    user_context = {
        'user_id': user_id,
        'project_id': project_id,
        'version_id': version_id,
    }
    # ContextVar is async-task local under ``asyncio``; the ``with``
    # block sets it for the duration of this orchestrator and resets
    # on exit (including exceptions). Note: ``asyncio.to_thread`` calls below
    # automatically copy the current Context (via
    # ``contextvars.copy_context``), so the worker thread sees the same
    # ``_user_context_var`` as the orchestrator task.
    with _user_context_scope(user_context):
        # Audit fix #3 (final pass): try high-level pipeline checkpoint
        # before kicking off Step 1. If both register + topologies are
        # cached for this exact (text, config) pair we skip straight to
        # assembly, which is the dominant cost saving on a re-run.
        pipeline_ckpt = config.pipeline_checkpoint_dir
        fingerprint = _extraction_fingerprint(config)

        # Step 1: Global Ontology (parallel 1b + 1c)
        # Step 1: Extract ontology
        register = _load_register_checkpoint(pipeline_ckpt, text, fingerprint)
        if register is None:
            register = await extract_ontology_async(text, config, user_context)
            _save_register_checkpoint(
                pipeline_ckpt, text, fingerprint, register,
            )
        else:
            logger.info(
                "[Pipeline\u00b7Async] Step 1 skipped \u2014 register "
                "loaded from pipeline checkpoint.",
            )

        # Step 2: Chunk Topology (parallel chunks)
        ckpt_loaded = _load_topologies_checkpoint(
            pipeline_ckpt, text, fingerprint,
        )
        catalogue: Optional[PropositionCatalogue] = None
        if ckpt_loaded is None:
            topologies = None
        else:
            topologies, catalogue = ckpt_loaded
        if topologies is None:
            chunks = chunk_text(
                text,
                strategy=config.chunk_strategy,
                min_chunk_chars=config.min_chunk_chars,
                max_chunk_chars=config.max_chunk_chars,
            )
            logger.info("[Pipeline\u00b7Async] Text split into %d chunks.", len(chunks))
            # Phase A3: optional Proposition Catalogue between
            # ontology and per-chunk extraction. The result threads
            # PROP_ ids into every chunk's deps so beliefs / utterances
            # / outcome events carry canonical ``proposition_id`` /
            # ``asserts_proposition_id`` / ``denies_proposition_id`` /
            # ``resolves_proposition_ids`` at extraction time. Failure
            # degrades to an empty catalogue (legacy behaviour).
            if config.enable_proposition_catalogue:
                catalogue = await extract_proposition_catalogue_async(
                    text, register, config,
                )
            topologies = await extract_topology_async(
                chunks, register, config, catalogue=catalogue,
            )
            _save_topologies_checkpoint(
                pipeline_ckpt, text, fingerprint, topologies,
                catalogue=catalogue,
            )
        else:
            logger.info(
                "[Pipeline\u00b7Async] Step 2 skipped \u2014 %d chunk "
                "topologies loaded from pipeline checkpoint.",
                len(topologies),
            )

        # Step 3: Assembly + Normalize + Auto-Repair + Validation (same as sync)
        world_state = assemble_world_state(register, topologies)
        world_state = _normalize_fabula_times(world_state, config.fabula_time_spacing)
        world_state, repairs = _auto_repair(world_state)
        if repairs:
            logger.info("[Pipeline·Async] Auto-repaired %d issues before validation.", len(repairs))

        # Phase C: fold the Phase A3 catalogue + per-chunk Phase B4
        # affect outputs onto the assembled world (propositions /
        # truth_at_fabula / Concern records / Proposition+Concern
        # state_timelines). Runs after _auto_repair so any event id
        # rewrites the repair pass made are already reflected on the
        # ``triggered_by`` fields the reconciler reads. No-op when
        # catalogue is None and every topology has empty affect lists.
        try:
            world_state = reconcile_affect(world_state, catalogue, topologies)
        except Exception:
            logger.exception(
                "[Pipeline·Async] Phase C affect reconciliation FAILED — "
                "leaving propositions / concerns at their pre-reconciler state.",
            )

        # Step 5: Post-assembly world trait timeline extraction
        world_state = await extract_world_trait_timelines_async(world_state, config)

        # Step 5c: Per-entity standing concern extraction (Affect Unification, Step 2).
        try:
            world_state = await extract_entity_concerns_async(world_state, config)
        except Exception:
            logger.exception(
                "[Pipeline·Async] Step 5c concern extraction failed — leaving "
                "entity.concerns empty.",
            )

        # Step 5d: LLM proposition clustering for non-event beliefs
        # (Affect Unification, Step 3 LLM half).
        try:
            world_state = await cluster_belief_propositions_async(world_state, config)
        except Exception:
            logger.exception(
                "[Pipeline·Async] Step 5d belief clustering failed — leaving "
                "non-event beliefs without proposition_id.",
            )

        # Step 5e: synthesise ENT_AUDIENCE (Affect Unification, Step 4).
        # Deterministic, no LLM cost — walks the syuzhet stream and
        # emits graded-confidence Belief snapshots so the unified affect
        # scorers can read audience belief uniformly with characters.
        try:
            world_state = _maybe_synthesise_audience_entity(world_state, config)
        except Exception:
            logger.exception(
                "[Pipeline·Async] Step 5e audience synthesis failed — the "
                "unified scorers will synthesise lazily on first call.",
            )

        # Validation calls a sync LLM agent internally; offload it to a
        # worker thread so we don't block the event loop while it runs
        # (audit item #7 — same rationale for ``_run_correction_patch``
        # below).
        report = await asyncio.to_thread(validate_world_state, world_state, config)

        # Snapshot pre-correction event identity so we can detect rename /
        # drop / time-shift and re-run trait timelines once afterwards.
        pre_correction_event_signature = tuple(
            (e.id, e.fabula_time) for e in world_state.events
        )

        # --- Correction retry loop ---
        # Track the signature of the outstanding errors across iterations
        # so we can detect oscillation (the patch keeps fixing X and
        # breaking Y, then fixing Y and breaking X). Without this guard
        # the loop runs to ``max_correction_retries`` even when no
        # progress is being made, burning LLM tokens for no benefit.
        seen_error_signatures: set[Tuple[Tuple[str, str], ...]] = set()
        # Compact iteration log fed back into each correction prompt so
        # the LLM can see what was already tried (and rejected) and
        # avoid re-emitting the same patch shape.
        correction_history: List[Dict[str, Any]] = []
        for retry_num in range(config.max_correction_retries):
            prog_errors = [i for i in report.issues if i.severity == "error"]
            if not prog_errors:
                break

            error_signature = tuple(sorted(
                (i.category, i.detail) for i in prog_errors
            ))
            if error_signature in seen_error_signatures:
                logger.warning(
                    "[Pipeline·Async·Correction] Same error set recurred "
                    "after a correction iteration (oscillation detected) "
                    "— breaking out of the retry loop with %d error(s) "
                    "remaining.",
                    len(prog_errors),
                )
                break
            seen_error_signatures.add(error_signature)

            log_prefix = f"[Pipeline·Async·Correction {retry_num + 1}/{config.max_correction_retries}]"
            logger.info(
                "%s %d errors remain — running patch-based correction agent.",
                log_prefix, len(prog_errors),
            )

            iter_errors_compact = [
                f"[{i.category}] {i.detail}" for i in prog_errors
            ]

            # ``_run_correction_patch`` calls ``agent.run_sync`` internally;
            # wrap it in ``to_thread`` so concurrent extraction tasks under
            # the same event loop are not stalled by the LLM round-trip.
            new_world_state, change_log, patch_status = await asyncio.to_thread(
                _run_correction_patch,
                world_state, prog_errors, config, log_prefix,
                correction_history,
            )
            correction_history.append({
                "iteration": retry_num + 1,
                "status": patch_status,
                "errors": iter_errors_compact,
                "changes": list(change_log),
            })
            if patch_status == "empty_patch":
                # Agent intentionally signalled "no safe fix available" —
                # re-asking will return the same answer, so stop early
                # and let the remaining errors surface on the report.
                logger.info(
                    "%s Correction agent returned an intentional empty "
                    "patch; ending correction loop with %d error(s) "
                    "remaining.",
                    log_prefix, len(prog_errors),
                )
                break
            if not change_log:
                # Transient failure (agent_failed / apply_failed /
                # regression). Burn one retry slot but keep going so
                # the oscillation guard above can decide when to stop.
                continue
            world_state = new_world_state
            repairs.extend(change_log)

            world_state = _normalize_fabula_times(world_state, config.fabula_time_spacing)
            world_state, new_repairs = _auto_repair(world_state)
            if new_repairs:
                repairs.extend(new_repairs)
            report = await asyncio.to_thread(validate_world_state, world_state, config)

        post_correction_event_signature = tuple(
            (e.id, e.fabula_time) for e in world_state.events
        )
        if (
            post_correction_event_signature != pre_correction_event_signature
            and world_state.world_traits
        ):
            logger.info(
                "[Pipeline·Async] Re-running world-trait timeline extraction "
                "after correction touched events.",
            )
            try:
                world_state = await extract_world_trait_timelines_async(world_state, config)
                world_state, post_repairs = _auto_repair(world_state)
                if post_repairs:
                    repairs.extend(post_repairs)
                report = await asyncio.to_thread(validate_world_state, world_state, config)
            except Exception:
                logger.exception(
                    "[Pipeline·Async] Post-correction timeline re-extraction failed "
                    "— keeping pre-correction timelines.",
                )

        # Step 5c: re-run concern extraction if events changed during correction.
        if pre_correction_event_signature != tuple(
            (e.id, e.fabula_time) for e in world_state.events
        ):
            try:
                # Reset concerns so re-extraction sees the corrected event set.
                refreshed = {
                    eid: e.model_copy(update={"concerns": []})
                    for eid, e in world_state.entities.items()
                }
                world_state = world_state.model_copy(update={"entities": refreshed})
                world_state = await extract_entity_concerns_async(world_state, config)
            except Exception:
                logger.exception(
                    "[Pipeline·Async] Post-correction concern re-extraction failed "
                    "— leaving concerns empty.",
                )

            # Re-synthesise ENT_AUDIENCE so its belief snapshots reflect the
            # post-correction event set. Drop the prior audience entity
            # first so the rebuild starts from a clean slate.
            try:
                if "ENT_AUDIENCE" in world_state.entities:
                    new_entities = {
                        eid: e for eid, e in world_state.entities.items()
                        if eid != "ENT_AUDIENCE"
                    }
                    world_state = world_state.model_copy(update={"entities": new_entities})
                world_state = _maybe_synthesise_audience_entity(world_state, config)
            except Exception:
                logger.exception(
                    "[Pipeline·Async] Post-correction audience re-synthesis failed.",
                )

        # ------------------------------------------------------------------
        # Step 3d — optional, segregated external research (post-assembly)
        # ------------------------------------------------------------------
        try:
            world_state = await _run_research_step_async(world_state, config)
        except Exception:
            logger.exception("[Pipeline·Research·Async] unexpected failure — continuing without research.")

        try:
            world_state.narrative_style = infer_narrative_style(text)
            logger.info(
                "[Pipeline·Async] Narrative style inferred: format=%s, target=%d–%d words, density=%s.",
                world_state.narrative_style.format,
                world_state.narrative_style.target_word_min,
                world_state.narrative_style.target_word_max,
                world_state.narrative_style.prose_density,
            )
        except Exception:
            logger.exception("[Pipeline·Async] Narrative-style inference failed — continuing without it.")

        # Final-pass validation snapshot. Research + narrative-style do
        # not normally mutate topology, but a stale ``report`` would
        # silently misrepresent the returned WorldState if anything in
        # those steps did edit it. Cheap programmatic re-validation
        # ensures ``report`` always describes what we actually return.
        try:
            final_report = await asyncio.to_thread(
                validate_world_state, world_state, config,
            )
            # Preserve the accumulated repairs log across the refresh.
            final_report.repairs = list(repairs)
            report = final_report
        except Exception:
            logger.exception(
                "[Pipeline·Async] Final-pass validation failed — "
                "returning the pre-research report.",
            )
            report.repairs = list(repairs)

        return world_state, report



def run_extraction(
    text: str,
    config: ExtractionConfig | None = None,
    *,
    user_id: Optional[int] = None,
    project_id: Optional[int] = None,
    version_id: Optional[int] = None,
) -> Tuple[WorldStateV1, ValidationReport]:
    """Synchronous wrapper around :func:`run_extraction_async` (audit fix #1).

    The pipeline is async-native; this wrapper exists so callers in
    sync contexts (CLI scripts, notebooks, blocking integration tests)
    don't have to manage their own event loop. Raises ``RuntimeError``
    if invoked from inside an already-running event loop \u2014 in that
    case use :func:`run_extraction_async` directly.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop_running = False
    else:
        loop_running = True
    if loop_running:
        raise RuntimeError(
            "run_extraction() called from inside a running event loop. "
            "Await run_extraction_async(...) instead."
        )
    return asyncio.run(run_extraction_async(
        text, config,
        user_id=user_id, project_id=project_id, version_id=version_id,
    ))
