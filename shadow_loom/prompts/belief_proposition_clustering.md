# Belief Proposition Clustering (Post-Assembly Pass)

You are an analytical narrative reasoner. You have been given **all character beliefs about a single target** in a story — that target is either an entity (a character or group), an object, a location, or a world-trait, NOT an event. Each belief is a string ``perceived_state`` plus the entity holding it.

Your task: cluster equivalent ``perceived_state`` strings into a small set of shared **propositions** so the affect-unification layer can compute audience-vs-character belief divergence on the *same* proposition (dramatic irony, surprise).

## What Counts as One Proposition

Two beliefs collapse into the same proposition when they assert the same fact about the target — even if phrased differently. Negations are NOT the same proposition: ``"Heathcliff is faithful"`` and ``"Heathcliff is unfaithful"`` are two propositions over the same domain (one is the *negation* of the other; the engine handles them as separate propositions because confidence operates per-proposition).

- Same: ``"Cup is safe"`` / ``"the goblet is fine"`` / ``"nothing is wrong with the cup"`` → ONE proposition (perhaps "Cup is unpoisoned").
- Same: ``"Macbeth will be king"`` / ``"Macbeth shall reign"`` → ONE proposition.
- Different: ``"Macbeth will be king"`` vs ``"Macbeth is currently king"`` (future vs present) → TWO propositions.
- Different: ``"Cordelia loves Lear"`` vs ``"Cordelia does NOT love Lear"`` → TWO propositions (negation pair).

## Output Format

Return a ``BeliefClusterRegister`` whose ``clusters`` field is a list of ``BeliefPropositionCluster`` records. Each cluster contains:

- **proposition** — a complete ``Proposition`` record. Set:
  - ``proposition_id`` — fresh ``PROP_*`` id (e.g. ``PROP_CORDELIA_LOVES_LEAR``). Must be unique across this response.
  - ``kind`` — one of ``"trait_holds"`` (a quality/state of the target), ``"relation_holds"`` (a relationship between target and another entity), ``"identity_is"`` (who/what the target IS), or ``"event_occurs"`` only if the belief is about a hypothetical future event. Do NOT use ``"outcome"`` here — that kind is reserved for choice-events.
  - ``referent_ids`` — list including the target_id and any other entity ids mentioned in the proposition.
  - ``description`` — short canonical statement of the proposition (e.g. "Cordelia loves Lear").
  - ``audience_default_prior`` — the audience's default confidence in this proposition before any narrative evidence. Use 0.5 unless the proposition is genuinely common-knowledge / common-trope.
  - ``stakes`` — how much resolution of this proposition matters narratively (0.0–1.0).
  - ``truth_at_fabula`` — leave empty (``{}``); ground-truth commitments are tracked separately.
- **perceived_states** — the exact list of ``perceived_state`` strings (verbatim, case-preserving) that this proposition covers, drawn from the supplied input list.

## Guidelines

1. Output **at most as many clusters as there are distinct viewpoints** on the target. A target about whom every character agrees has one cluster; a contested target has 2-4.
2. Do **not** drop input strings. Every supplied ``perceived_state`` must appear in exactly one cluster's ``perceived_states`` list. Fall-through propositions are fine — if a belief truly doesn't fit anywhere else, give it its own cluster.
3. **Negation pairs** must be two clusters, not one. The downstream KL math uses `p` and `1 - p` separately; collapsing negations destroys it.
4. Prefer concise canonical descriptions ("Heathcliff is cruel") over long paraphrases. The proposition_id should mirror the description ("PROP_HEATHCLIFF_CRUEL").
5. If a target has only one supplied belief, emit one cluster containing that belief.
6. Do not invent propositions not grounded in the supplied strings.
