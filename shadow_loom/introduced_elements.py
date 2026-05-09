"""Per-cycle birth declarations for new top-level world elements.

Lives in its own module so both :mod:`shadow_loom.query_models` (for the
user-facing ``query.introduce`` payload) and :mod:`shadow_loom.generation`
(for the renderer's ``GeneratedScene.introduced_elements`` field) can
import these types without forming an import cycle.

Two channels carry introductions through the cycle:

1. **User-side** — the user pre-declares new elements on the query
   itself via ``query.introduce``. The pipeline pre-spawns these into
   the sandbox before physics so do-surgeries can target them safely.
2. **Renderer-side** — the renderer declares any *additional* new
   elements it had to invent in ``GeneratedScene.introduced_elements``.
   These flow through re-extraction as authoritative spawns.

Both channels are typed identically (the same :class:`IntroducedElements`
container) so the auditor and merge surfaces handle them uniformly.
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class _IntroducedSpecBase(BaseModel):
    """Common fields every IntroducedXxxSpec carries."""
    id: str = Field(
        description=(
            "Stable identifier following the project's prefix convention "
            "(``ENT_*`` for entities, ``LOC_*`` for locations, ``OBJ_*`` "
            "for objects, ``WT_*`` for world traits, ``PROP_*`` for "
            "propositions, ``CCN_*`` for concerns). MUST NOT collide with "
            "any id already in ``WorldStateV1`` \u2014 the merge will reject "
            "duplicates."
        ),
    )
    name: str = Field(
        description="Human-readable display name used in the prose.",
    )
    justification: str = Field(
        description=(
            "One sentence explaining why this element had to be invented "
            "rather than drawn from existing world state. Surfaces in the "
            "audit log; the auditor uses it to reject gratuitous "
            "invention (e.g. inventing a generic onlooker when an "
            "existing entity could have served)."
        ),
    )


class IntroducedEntitySpec(_IntroducedSpecBase):
    """Renderer- or user-declared new ``Entity``."""
    role: Optional[str] = Field(
        default=None,
        description=(
            "Short role description (e.g. ``\"messenger\"``, ``\"witness\"``, "
            "``\"henchman\"``). Used by the ingestion agents to seed the "
            "Entity's downstream traits."
        ),
    )
    located_in: Optional[str] = Field(
        default=None,
        description=(
            "Location id where this entity first appears. May reference "
            "an existing ``LOC_*`` id OR an id declared in this same "
            "``IntroducedElements`` payload."
        ),
    )
    initial_traits: Dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Optional seed trait values keyed by trait name; the "
            "ingestion layer will materialise these via the standard "
            "``EntityStateSnapshot`` path."
        ),
    )


class IntroducedLocationSpec(_IntroducedSpecBase):
    """Renderer- or user-declared new ``Location``."""
    description: Optional[str] = Field(default=None)
    parent_location: Optional[str] = Field(
        default=None,
        description="Containing location id, if hierarchical.",
    )


class IntroducedObjectSpec(_IntroducedSpecBase):
    """Renderer- or user-declared new ``NarrativeObject``."""
    description: Optional[str] = Field(default=None)
    located_in: Optional[str] = Field(
        default=None,
        description="Location id (or holder ``ENT_*`` id) where this object resides.",
    )


class IntroducedWorldTraitSpec(_IntroducedSpecBase):
    """Renderer- or user-declared new ``GlobalTrait``."""
    value: Optional[float] = Field(
        default=None,
        description="Initial scalar value. Free-form qualitative world traits omit this.",
    )
    description: Optional[str] = Field(default=None)


class IntroducedPropositionSpec(_IntroducedSpecBase):
    """Renderer- or user-declared new ``Proposition``.

    ``name`` carries the proposition's natural-language content; the
    superclass field name is kept for schema uniformity even though
    propositions are not "named" in the everyday sense.
    """
    kind: Optional[str] = Field(
        default=None,
        description=(
            "One of the project's proposition kinds (``event_outcome``, "
            "``identity``, ``relation``, ``trait``, ``epistemic``). The "
            "ingestion layer falls back to a heuristic when omitted."
        ),
    )
    truth_value: Optional[bool] = Field(
        default=None,
        description=(
            "If known at introduction time, the proposition's truth value "
            "in the *current* branch. Leave ``None`` for open questions "
            "(mystery / suspense / unresolved)."
        ),
    )


class IntroducedConcernSpec(_IntroducedSpecBase):
    """Renderer- or user-declared new ``Concern``.

    A concern is held by a specific entity *about* a specific
    proposition; both ids must already exist OR be declared in this
    same payload.
    """
    holder_entity_id: str = Field(
        description="Entity id that holds the concern.",
    )
    proposition_id: str = Field(
        description="Proposition id that the concern is about.",
    )
    polarity: Literal["positive", "negative"] = Field(
        description=(
            "``positive`` if the holder wants the proposition to be true, "
            "``negative`` if they want it false."
        ),
    )
    salience: float = Field(
        default=0.5,
        description="Concern weight in [0, 1].",
    )


class IntroducedElements(BaseModel):
    """Per-cycle birth list for new top-level world elements.

    Empty by default. The auditor checks every proper noun in ``prose``
    against ``WorldStateV1`` \u222a these lists; anything unresolved fires
    an ``undeclared_element`` violation. The merge treats these lists
    as authoritative spawns when re-ingesting the prose into the next
    ``WorldStateV1`` revision.

    User-pre-declared introductions (via ``query.introduce``) are
    pre-spawned into the sandbox before physics runs, so do-surgeries
    can target them. Renderer-declared introductions (via
    ``GeneratedScene.introduced_elements``) flow through re-extraction
    as authoritative spawns.
    """
    entities: List[IntroducedEntitySpec] = Field(default_factory=list)
    locations: List[IntroducedLocationSpec] = Field(default_factory=list)
    objects: List[IntroducedObjectSpec] = Field(default_factory=list)
    world_traits: List[IntroducedWorldTraitSpec] = Field(default_factory=list)
    propositions: List[IntroducedPropositionSpec] = Field(default_factory=list)
    concerns: List[IntroducedConcernSpec] = Field(default_factory=list)

    def is_empty(self) -> bool:
        return not (
            self.entities or self.locations or self.objects
            or self.world_traits or self.propositions or self.concerns
        )

    def declared_ids(self) -> set[str]:
        """All ids declared across every kind \u2014 useful for resolving
        cross-references (e.g. an entity's ``located_in`` pointing at a
        co-declared location) and for the auditor's name-resolution
        pass."""
        out: set[str] = set()
        for spec_list in (
            self.entities, self.locations, self.objects,
            self.world_traits, self.propositions, self.concerns,
        ):
            out.update(s.id for s in spec_list)
        return out

    def declared_names(self) -> set[str]:
        """Display names across every named kind. Propositions are
        excluded (their ``name`` is the content sentence, not a noun
        the renderer would use as a referent)."""
        out: set[str] = set()
        for spec_list in (
            self.entities, self.locations,
            self.objects, self.world_traits,
        ):
            out.update(s.name for s in spec_list)
        return out

    def merged_with(
        self, other: Optional["IntroducedElements"],
    ) -> "IntroducedElements":
        """Return a new :class:`IntroducedElements` containing the union
        of ``self`` and ``other``, deduplicated by id (``self`` wins on
        collisions). Used by the pipeline to fold the user's
        ``query.introduce`` channel together with the renderer's
        ``GeneratedScene.introduced_elements`` channel before the
        merge step so both surfaces of the cycle see one authoritative
        birth list."""
        if other is None or other.is_empty():
            return self
        seen_ids = self.declared_ids()

        def _extend(target: list, incoming: list) -> list:
            return target + [s for s in incoming if s.id not in seen_ids]

        return IntroducedElements(
            entities=_extend(self.entities, other.entities),
            locations=_extend(self.locations, other.locations),
            objects=_extend(self.objects, other.objects),
            world_traits=_extend(self.world_traits, other.world_traits),
            propositions=_extend(self.propositions, other.propositions),
            concerns=_extend(self.concerns, other.concerns),
        )
