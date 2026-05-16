# Academic Foundations

Shadow-Loom borrows ideas from four largely separate research traditions:
**structuralist and cognitive narratology**, **causal inference (Pearl's
ladder, AMWN, ctf-calculus)**, **computational models of suspense /
surprise / curiosity**, and **constrained / neuro-symbolic generation**.
This document maps every named concept in the codebase to the literature
that motivated it.

Every citation below has been cross-checked against publisher records,
ACL Anthology, OpenReview, JSTOR, JMLR, NeurIPS / ICML / AAAI / ACL
proceedings pages, dblp, or arXiv. The five most load-bearing modern
citations — Correa & Bareinboim 2025 (AMWN + ctf-calculus, ICML 2025
spotlight, OpenReview Z1qZoHa6ql), Wilmot & Keller 2020 (ACL pp. 1763–1788),
Wilmot & Keller 2021 (EMNLP pp. 851–865), Wilmot 2022 (PhD, arXiv:2206.09708),
and Tian et al. 2024 (EMNLP Outstanding Paper, pp. 17659–17681) — were each
verified directly. Recent (2023–2026) LLM-era references added to §1.5,
§2.2, §4.3 and §7 — Kıcıman et al. 2024 (TMLR, arXiv:2305.00050),
Kim et al. 2023 (FANToM, EMNLP, arXiv:2310.15421), Cross et al. 2024
(Hypothetical Minds, arXiv:2407.07086), Gu et al. 2024 (LLM-as-a-Judge
survey, arXiv:2411.15594), Gu et al. 2024/2026 (SimpleToM, ICLR 2026,
arXiv:2410.13648), Yang et al. 2023 (DOC, ACL, arXiv:2212.10077),
Liu et al. 2024 (Lost in the Middle, TACL, arXiv:2307.03172), and
Xu et al. 2025 (Echoes in AI, PNAS 122(35) e2504966122, arXiv:2501.00273)
— were each verified directly against their arXiv records and journal
pages. Where a citation could not be verified online (older monographs,
working papers) the entry is marked accordingly.

---

## 1. Structuralist narratology

### 1.1 Fabula vs syuzhet (`fabula_time` / `syuzhet_index`)

The distinction between the chronological story-world (*fabula*) and its
ordering in narration (*syuzhet*) was introduced by the Russian Formalists.
Shadow-Loom adopts it as two integer fields on `EventNode`.

* Shklovsky, V. (1917/1965). "Art as Technique". In L. T. Lemon & M. J. Reis (eds.), *Russian Formalist Criticism: Four Essays*. Lincoln: Univ. of Nebraska Press, pp. 3–24.
* Shklovsky, V. (1925/1990). *Theory of Prose*. Trans. B. Sher. Champaign: Dalkey Archive Press. — the full-length statement of the fabula/syuzhet distinction.
* Tomashevsky, B. (1925/1965). "Thematics". In Lemon & Reis (eds.), *Russian Formalist Criticism: Four Essays*, pp. 61–95. Univ. of Nebraska Press.
* Todorov, T. (1969). *Grammaire du Décaméron*. The Hague: Mouton. — introduced the term "narratology" and the equilibrium-disruption-re-equilibrium schema.
* Barthes, R. (1966/1975). "An Introduction to the Structural Analysis of Narrative". Trans. L. Duisit. *New Literary History* 6(2): 237–272.
* Genette, G. (1980). *Narrative Discourse: An Essay in Method*. Cornell UP. — formalises *order*, *duration*, *frequency* as orthogonal axes; we collapse to *order* via two indices.
* Chatman, S. (1978). *Story and Discourse: Narrative Structure in Fiction and Film*. Cornell UP. — standard English-language structuralist synthesis bridging Genette and film theory.
* Prince, G. (1982). *Narratology: The Form and Functioning of Narrative*. Berlin: Mouton.
* Bal, M. (1985/2009). *Narratology: Introduction to the Theory of Narrative*. 3rd ed. Univ. of Toronto Press. — the most widely used narratology textbook.
* Bordwell, D. (1985). *Narration in the Fiction Film*. Univ. of Wisconsin Press. — extends the distinction to film, motivating syuzhet as the cognitive substrate the reader/viewer reasons over.

Why we keep both: the affective scorers (mystery, dramatic irony, suspense,
surprise — see §3) all reduce to set operations between the *fabula
projection* and the *syuzhet projection* of the causal graph.

### 1.2 Greimas' actantial model (`GlobalTrait`, `RelationshipEdge`)

`GlobalTrait` (`WORLD_*` nodes) is our implementation of Greimas' "Power"
actant — the abstract force that determines whether subjects can achieve
their goals. `RelationshipEdge.power_dynamic` mirrors the
sender/receiver/helper/opponent topology.

* Greimas, A.-J. (1966). *Sémantique structurale: recherche de méthode*. Larousse. (English: *Structural Semantics*, 1983, Univ. of Nebraska Press.)
* Greimas, A.-J. (1983). *Du sens II*. Seuil. — actant theory.

### 1.3 Propp's morphology and story grammars (`EventNode.event_type`)

`EventNode.event_type ∈ {choice, outcome, revelation, utterance}` is a coarse
generalisation of Propp's 31 narrative functions. We do not model Propp's
fine-grained typology because we want event types to be *causally*
significant rather than narratologically prescriptive. The `utterance` type is
the exclusive output of the Social agent and is a first-class event modality
(speech *is* the act); it carries `speaker_id`, `addressee_ids`,
`via_channel_id`, `content`, and `truth_value ∈ {true, false, unknown, performative}`.

* Propp, V. (1928/1968). *Morphology of the Folktale*. 2nd rev. English ed., trans. L. Scott, rev. L. A. Wagner. Univ. of Texas Press.
* Rumelhart, D. E. (1975). "Notes on a schema for stories". In D. G. Bobrow & A. M. Collins (eds.), *Representation and Understanding: Studies in Cognitive Science*, pp. 211–236. Academic Press. — the "story grammar" tradition that motivated event-type taxonomies in early AI.
* Mandler, J. M. & Johnson, N. S. (1977). "Remembrance of things parsed: Story structure and recall". *Cognitive Psychology* 9(1): 111–151. — cognitive evidence that readers parse stories into typed event units.
* Thorndyke, P. W. (1977). "Cognitive structures in comprehension and memory of narrative discourse". *Cognitive Psychology* 9(1): 77–110. — the experimental story-grammar paper, published in the same issue as Mandler & Johnson.
* Kintsch, W. & van Dijk, T. A. (1978). "Toward a model of text comprehension and production". *Psychological Review* 85(5): 363–394. — the propositional macro-structure model; canonical complement to story-grammar approaches.
* Trabasso, T. & van den Broek, P. (1985). "Causal thinking and the representation of narrative events". *J. Memory and Language* 24(5): 612–630. — the empirical case that *causal* event chains (not surface form) drive comprehension and recall — the central justification for our graph-first design.
* Bremond, C. (1973). *Logique du récit*. Seuil. — the choice/outcome/revelation triad we adopt is closest to Bremond's *triade narrative*.

The `truth_value="performative"` field on utterance events is grounded in speech-act theory. Austin (1962) distinguishes *performative* utterances — those that enact a state of affairs rather than describe one — from constative assertions; Searle (1969) formalises them as illocutionary acts that posit, commit to, or predict future states. A prophecy, vow, or order does not claim a fact; it inaugurates a commitment. This is why performative utterances are exempt from the temporal-coherence constraint (Rule 5 in `_validate_time_ordering`, §D5e in [design-decisions.md](design-decisions.md)) that requires non-performative utterance `target_ids` to reference past or simultaneous events: a prophecy *announces* the future event rather than reporting one that has already occurred.

* Austin, J. L. (1962). *How to Do Things with Words*. Oxford UP. — the foundational performative/constative distinction; *illocutionary force* underwrites `truth_value="performative"`.
* Searle, J. R. (1969). *Speech Acts: An Essay in the Philosophy of Language*. Cambridge UP. — formal taxonomy of illocutionary acts (assertives, directives, commissives, expressives, declarations); vows, prophecies, and orders are commissives/declarations and do not carry truth-conditional content in the same sense as assertives.

### 1.4 Cognitive narratology and possible-worlds (`AMWN`, `Belief`)

The move from "the story is the prose" to "the story is a mental model the
reader builds" — directly relevant to our reader/character belief asymmetry
— is the cognitive-narratology turn:

* Herman, D. (2002). *Story Logic: Problems and Possibilities of Narrative*. Univ. of Nebraska Press. — the foundational work for our "graph + epistemic state" framing.
* Herman, D. (2009). *Basic Elements of Narrative*. Wiley-Blackwell.
* Fludernik, M. (1996). *Towards a 'Natural' Narratology*. London: Routledge. — grounds narrative structure in embodied cognitive schemas.
* Turner, M. (1996). *The Literary Mind: The Origins of Thought and Language*. Oxford UP. — conceptual blending and narrative imagination.
* Ryan, M.-L. (1991). *Possible Worlds, Artificial Intelligence, and Narrative Theory*. Indiana UP. — the bridge from David Lewis's modal semantics to narrative; explicit influence on our `world_id ∈ {factual, shadow}` design.
* Ryan, M.-L. (2001). *Narrative as Virtual Reality: Immersion and Interactivity in Literature and Electronic Media*. Johns Hopkins UP.
* Doležel, L. (1998). *Heterocosmica: Fiction and Possible Worlds*. Johns Hopkins UP. — fictional worlds as modal structures with their own physics; conceptual cousin of `WorldStateV1`.
* Ronen, R. (1994). *Possible Worlds in Literary Theory*. Cambridge UP. — dedicated literary-theoretical treatment of possible-worlds semantics; complements Ryan and Doležel.
* Pavel, T. (1986). *Fictional Worlds*. Harvard UP.
* Zwaan, R. A. & Radvansky, G. A. (1998). "Situation models in language comprehension and memory". *Psychological Bulletin* 123(2): 162–185. DOI 10.1037/0033-2909.123.2.162. — the "event-indexing model" with five dimensions (time, space, protagonist, causation, intention) is a near-exact precursor to our `EventNode` field set.
* Gerrig, R. J. (1993). *Experiencing Narrative Worlds: On the Psychological Activities of Reading*. Yale UP.

### 1.5 Theory of mind in narrative (`Belief`, `confidence`, `inertia`)

`Entity.beliefs : List[Belief]` with `confidence` and `inertia` is a
discrete approximation of recursive theory-of-mind reasoning of the kind
that readers do when tracking who knows what.

* Zunshine, L. (2006). *Why We Read Fiction: Theory of Mind and the Novel*. Ohio State UP. — the literary case for ToM as the central cognitive operation in novel reading.
* Premack, D. & Woodruff, G. (1978). "Does the chimpanzee have a theory of mind?". *Behavioral and Brain Sciences* 1(4): 515–526. DOI 10.1017/S0140525X00076512. — the term itself.
* Wimmer, H. & Perner, J. (1983). "Beliefs about beliefs: Representation and constraining function of wrong beliefs in young children's understanding of deception". *Cognition* 13(1): 103–128. — the false-belief paradigm.
* Baron-Cohen, S., Leslie, A. M., Frith, U. (1985). "Does the autistic child have a 'theory of mind'?". *Cognition* 21(1): 37–46. — the canonical empirical follow-up to Premack & Woodruff.
* Goodman, N. D. & Frank, M. C. (2016). "Pragmatic language interpretation as probabilistic inference". *Trends in Cognitive Sciences* 20(11): 818–829. DOI 10.1016/j.tics.2016.08.005. — the Rational Speech Act framework, conceptually parallel to `Belief.confidence`.
* Baker, C. L., Saxe, R., Tenenbaum, J. B. (2009). "Action understanding as inverse planning". *Cognition* 113(3): 329–349. DOI 10.1016/j.cognition.2009.07.005.

Recent (2023–2026) — LLM-era benchmarks for narrative ToM and belief tracking, directly relevant to how `Belief.confidence` / `inertia` should behave when an LLM ingests prose:

* Kim, H. *et al.* (2023). "FANToM: A Benchmark for Stress-testing Machine Theory of Mind in Interactions". *EMNLP 2023*. arXiv:2310.15421. — information-asymmetric multi-party conversation as a ToM stress test; the per-channel `intelligibility` model in §6.1 is partly a response to this paper's failure modes.
* Cross, L. *et al.* (2024). "Hypothetical Minds: Scaffolding Theory of Mind for Multi-Agent Tasks with Large Language Models". arXiv:2407.07086. — explicit hypothesis-generation-and-refinement over other agents' strategies; conceptual cousin of the auditor's belief-consistency checks.
* Gu, Y. *et al.* (2024/2026). "SimpleToM: Exposing the Gap between Explicit ToM Inference and Implicit ToM Application in LLMs". *ICLR 2026*. arXiv:2410.13648. — shows LLMs can answer "what does X believe?" but fail to *act* on that belief in behaviour prediction; motivates our separate audit pass for belief-consistent character action.

---

## 2. Causal inference (Pearl's ladder, AMWN, ctf-calculus)

This is the largest single intellectual debt in the codebase. The naming
conventions (`AMWN`, `do_intervene`, `abduction_update`,
`world_id ∈ {factual, shadow}`, `ctf-calculus`) come directly from this literature.

### 2.1 Three rungs of causation (`ObservationQuery`, `InterventionQuery`, `CounterfactualQuery`)

Our query taxonomy directly mirrors Judea Pearl's "Ladder of Causation":
observation → intervention → counterfactual.

* Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge UP. DOI 10.1017/CBO9780511803161. — canonical reference for SCMs and the three-rung hierarchy.
* Pearl, J. & Mackenzie, D. (2018). *The Book of Why: The New Science of Cause and Effect*. Basic Books. — accessible exposition.
* Pearl, J. (1995). "Causal diagrams for empirical research". *Biometrika* 82(4): 669–688. DOI 10.1093/biomet/82.4.669. — the original *do*-calculus paper.
* Spirtes, P., Glymour, C., Scheines, R. (2000). *Causation, Prediction, and Search* (2nd ed.). MIT Press. — the other foundational text alongside Pearl 2009; PC algorithm and Markov equivalence.
* Bareinboim, E., Correa, J. D., Ibeling, D., Icard, T. (2022). "On Pearl's Hierarchy and the Foundations of Causal Inference". Ch. 27 in *Probabilistic and Causal Inference: The Works of Judea Pearl*, pp. 507–556. ACM Books. DOI 10.1145/3501714.3501743. — the modern formal statement of the hierarchy and the impossibility results that motivate level-3 counterfactual machinery.

`CausalPhysicsEngine.apply_do_operator()` implements rung-2 (Intervention) graph
surgery: incoming causal edges into the intervened node are severed,
the intervened path is overwritten, dependent provenance (beliefs
whose `acquired_via_event_id` / `acquired_via_channel_id` pointed at
a surgically invalidated event or severed channel) is pruned, and
downstream edges are then re-evaluated by `propagate()`.

### 2.2 Ancestral Multi-World Networks and ctf-calculus — **Correa & Bareinboim, ICML 2025**

The `world_id ∈ {factual, shadow}` tag, the [`AMWNInstantiator`](../shadow_loom/instantiator.py)
class, the term "ctf-calculus" used throughout the design notes, and the
three-rung query taxonomy in [`query_models.py`](../shadow_loom/query_models.py)
are all named after, and architected around, this paper:

* **Correa, J. D. & Bareinboim, E. (2025).** "Counterfactual Graphical Models: Constraints and Inference". *Proc. ICML 2025* (spotlight). [openreview.net/forum?id=Z1qZoHa6ql](https://openreview.net/forum?id=Z1qZoHa6ql).

The paper introduces:

* **Ancestral Multi-World Networks (AMWNs)** — a graphical construction
  that is **sound and complete** for reading counterfactual independences
  via d-separation. AMWNs avoid the exponential graph blow-up of k-plet /
  multi-network methods and the incompleteness of Twin Networks (Balke &
  Pearl 1994) and SWIGs (Richardson & Robins 2013).
* **Counterfactual (ctf-) calculus** — three transformation rules
  (consistency, independence via d-separation in the AMWN, and exclusion)
  that **generalise Pearl's do-calculus from interventional to
  counterfactual reasoning**. The rules are sound and complete for
  counterfactual identifiability and subsume do-calculus as a special case.

In shadow-loom we adopt the AMWN naming for our sandbox graph (each
counterfactual query spawns a `world_id="shadow"` mirror with the relevant
intervention nodes "split" from their factual counterparts) **and** we
implement the three rules of the ctf-calculus as a pre-flight check on
every Rung-2 (Intervention) and Rung-3 (Counterfactual) query in [`shadow_loom/amwn.py`](../shadow_loom/amwn.py)
(`build_amwn`, `check_consistency`, `check_ctf_independence`,
`check_exclusion`, `apply_ctf_calculus`).

The **node-splitting construction itself** is implemented as a lazy
per-branch sidecar on the persisted world state — four parallel
dictionaries (`shadow_entities`, `shadow_objects`,
`shadow_propositions`, `shadow_world_traits` on `WorldStateV1`),
each keyed by `branch_label` then by node id. A node remains
**node-shadowed** (a single shared record backs every world) until
a shadow merge writes to it; at that point a deep-copy clone is
materialised in the sidecar, its `state_timeline` is trimmed of
snapshots whose `triggered_by` lies in the suppression closure
(severing the incoming structural equations on the split copy),
and subsequent shadow snapshots accumulate only on the clone.
Sibling shadow branches are independent AMWN worlds W*ₙ keyed by
their distinct `branch_label`. Reads go through
`WorldStateV1.projected_for_branch(branch_world_id, branch_label)`,
which returns a shallow `model_copy` swapping the four projected
dicts at once (and returns `self` unchanged for factual reads, so
the cost is zero on the canonical mainline). See
[architecture.md §1 "AMWN node-splitting sidecar"](architecture.md#amwn-node-splitting-sidecar-correa--bareinboim-icml-2025)
for the persistence + serialization contract.

The ctf-calculus rule table:

| ctf-calculus rule | Shadow-loom implementation |
|---|---|
| **Rule 1 — Consistency** ($P(Y_{T*x}, X_{T*}=x) = P(Y_{T*}, X_{T*}=x)$) | `check_consistency()` suppresses vacuous `do(X = observed(X))` interventions; engine reports them as `rule1_redundant`. |
| **Rule 2 — Independence** (d-separation in the AMWN ⇒ conditional independence) | `check_ctf_independence()` builds the AMWN $G^A(G, W^*)$ over the union of evidence and intervention targets, performs node-shadowing across worlds whose projected contexts on the ancestral set agree, and runs NetworkX `is_d_separator` over the result. Evidence flagged as redundant is reported as `rule2_redundant_evidence` but **not** silently dropped (abduction may still populate `hidden_deltas` that downstream consumers depend on). |
| **Rule 3 — Exclusion** ($P(y_{xz}) = P(y_z)$ if $X \cap An(Y) = \emptyset$ in $G_{\bar Z}$) | `check_exclusion()` mutilates the diagram and tests ancestor-disjointness against the query targets. Pruned interventions are reported as `rule3_pruned_interventions`. By default the engine ships in **advisory mode** (the flagged interventions are surfaced to the auditor but retained in simulation), since a missed bidirected confounder would let a substantively meaningful intervention be d-separated into a no-op; **prune mode** (`rule3_pruning_mode='prune'`) is opt-in for confounder-complete topologies. |

The construction matches Definition A.1 of Correa & Bareinboim 2025
*in spirit* but operates on a **latent-free SCM** (no bidirected `U`
arcs encoding shared unobserved confounders), so the
implementation is **sound** for d-separation but **not complete
across worlds with shared latents** — Rule 2 / 3 flags are
advisory in that sense. Relationship metrics (affinity / fear /
power_dynamic) are lifted into synthetic
`REL::<src>::<tgt>::<metric>` diagram nodes so that
`mutation_social` causal edges contribute to d-separation
reasoning. See [design-decisions.md](design-decisions.md) D5–D6 for
the rationale and the closed-world caveat.

#### Predecessors and related machinery

* Balke, A. & Pearl, J. (1994). "Counterfactual probabilities: Computational methods, bounds and applications". *Proc. UAI 1994*, pp. 46–54. — introduces the **Twin Network** representation that AMWNs supersede.
* Shpitser, I. & Pearl, J. (2007). "What counterfactuals can be tested". *Proc. UAI 2007*, pp. 352–359. — multi-network construction; identifies the incompleteness of Twin Networks.
* Shpitser, I. & Pearl, J. (2008). "Complete identification methods for the causal hierarchy". *J. Machine Learning Research* 9(64): 1941–1979. [jmlr.org/papers/v9/shpitser08a.html](https://jmlr.org/papers/v9/shpitser08a.html).
* Shpitser, I. & Pearl, J. (2009). "Effects of treatment on the treated: Identification and generalization". *Proc. UAI 2009*, pp. 514–521. — completes the counterfactual identification theory.
* Tian, J. & Pearl, J. (2002). "A general identification condition for causal effects". *Proc. AAAI 2002*, pp. 567–573. — characterises identifiable causal effects via do-calculus.
* Galles, D. & Pearl, J. (1998). "An axiomatic characterization of causal counterfactuals". *Foundations of Science* 3(1): 151–182. DOI 10.1023/A:1009602825894. — proves completeness of SCM semantics for counterfactuals.
* Richardson, T. S. & Robins, J. M. (2013). "Single World Intervention Graphs (SWIGs): A unification of the counterfactual and graphical approaches to causality". CSSS Working Paper 128, Univ. of Washington. *(working paper widely cited in this exact form; PDF availability fluctuates)*.
* Correa, J. D., Lee, S., Bareinboim, E. (2021). "Nested counterfactual identification from arbitrary surrogate experiments". *NeurIPS 2021*. arXiv:2107.03190. — algorithmic counterpart to the ctf-calculus.
* Bareinboim, E. & Pearl, J. (2016). "Causal inference and the data-fusion problem". *Proc. Natl. Acad. Sci.* 113(27): 7345–7352. DOI 10.1073/pnas.1510507113. — transportability across populations — the conceptual basis for AMWN-style multi-world reasoning.

Recent (2023–2026) — LLMs as causal reasoners, complementary to (not a replacement for) the structural machinery above:

* Kıcıman, E., Ness, R., Sharma, A., Tan, C. (2024). "Causal Reasoning and Large Language Models: Opening a New Frontier for Causality". *Transactions on Machine Learning Research*. arXiv:2305.00050. — benchmarks LLMs on pairwise causal discovery, counterfactual reasoning, and necessary/sufficient cause attribution; supports our hybrid stance of using LLMs to *propose* edges (`extract_graph.py`) while keeping identification logic in typed code (`causal_physics.py`).
* Liu, N. F. *et al.* (2024). "Lost in the Middle: How Language Models Use Long Contexts". *TACL* 12: 157–173. arXiv:2307.03172. — empirical evidence that LLMs degrade sharply when relevant facts sit in the middle of a long context window; one of the strongest motivations for keeping the storyworld in a typed graph rather than relying on a growing prose buffer + RAG.

### 2.3 Abduction (`CausalPhysicsEngine.abduction_update`)

Rung-3 (Counterfactual) queries require **abduction** — back-propagating present
evidence onto a historical sandbox. The default implementation is a
**precision-weighted Bayesian blend**
($\texttt{abduction\_blend\_mode = "bayesian"}$): trait inertia
$\iota_T$ is reinterpreted as the precision of the historical prior,
an evidence precision $\kappa_E$ (default $1$) is set on the
present-day observation, and the posterior is

$$T^{\text{post}} = \frac{\iota_T \cdot T^{\text{prior}} + \kappa_E \cdot T^{\text{evidence}}}{\iota_T + \kappa_E}.$$

A legacy inertia-damped variant
($T^{\text{post}} = T^{\text{prior}} + (1 - \iota_T)(T^{\text{evidence}} - T^{\text{prior}})$)
is retained for ablation. The same Bayes blend runs per axis on
outgoing relationship metrics. Belief back-propagation is gated
by a per-recipient channel `intelligibility` threshold so a belief
that could not plausibly have been acquired through its provenance
channel is not reinstated. Where a present-day evidence event is
applied via abduction, its outgoing causal edges are masked from
the subsequent forward propagation pass to prevent
double-counting.

The philosophical and computational basis:

* Pearl, J. (2000/2009). *Causality* §7 ("The logic of structure-based counterfactuals"). — the abduction–action–prediction recipe for rung-3 (Counterfactual) queries. Our pipeline implements the same three steps in `causal_physics.py`.
* Halpern, J. Y. (2016). *Actual Causality*. MIT Press. — formal definitions of "actual cause" used to motivate `mechanism` and `causal_force`.
* Halpern, J. Y. (2000). "Axiomatizing causal reasoning". *J. Artificial Intelligence Research* 12: 317–337. DOI 10.1613/jair.648.
* Halpern, J. Y. & Pearl, J. (2005). "Causes and explanations: A structural-model approach. Part I: Causes". *British Journal for the Philosophy of Science* 56(4): 843–887. DOI 10.1093/bjps/axi147.
* Woodward, J. (2003). *Making Things Happen: A Theory of Causal Explanation*. Oxford UP. DOI 10.1093/0195155270.001.0001. — interventionist philosophy of causation; the major non-Pearl alternative widely contrasted with Lewis's counterfactual account.
* Lewis, D. (1973). "Causation". *Journal of Philosophy* 70(17): 556–567. DOI 10.2307/2025310. — the closest-possible-world semantics that AMWN sandboxing operationalises.
* Lewis, D. (1979). "Counterfactual dependence and time's arrow". *Noûs* 13(4): 455–476. DOI 10.2307/2215339. — the temporal asymmetry that motivates fabula-time-aware abduction.
* Peirce, C. S. (1934/1958). *Collected Papers of Charles Sanders Peirce* vol. 5, §§180–212. Harvard UP. — the original definition of abduction as inference to the best explanation.
* Harman, G. (1965). "The inference to the best explanation". *Philosophical Review* 74(1): 88–95. DOI 10.2307/2183532. — canonical philosophical paper introducing IBE as a third inference mode.
* Josephson, J. R. & Josephson, S. G. (eds.) (1994). *Abductive Inference: Computation, Philosophy, Technology*. Cambridge UP. DOI 10.1017/CBO9780511530128. — ties Peircean abduction to diagnostic and causal AI reasoning.

### 2.4 d-separation and ego-graph slicing

Two distinct uses of d-separation live in the codebase. (a) When
[`extract_graph.py`](../shadow_loom/extract_graph.py) limits the ego-graph
to "1-hop spatial neighbours" + "events within `memory_limit`", we are
heuristically approximating a Markov blanket — the minimal node set that
d-separates the focal entities from the rest of the graph — to keep the
simulation sandbox tractable. (b) [`shadow_loom/amwn.py`](../shadow_loom/amwn.py)
runs *exact* d-separation over the AMWN $G^A(G, W^*)$ via NetworkX
`is_d_separator` (with a fallback to the legacy `d_separated` name),
re-using `nx.ancestors` for the projection step `An(V)_{G_{T̄}}` of
Definition A.1. The closed-world / latent-free caveat from §2.2 applies
to both: an `allow_unobserved_confounders=True` setting in
`CausalPhysicsSettings` injects explicit `U_<a>__<b>` shared-parent
nodes for every observed-sibling pair, so d-separation refuses to mark
two siblings independent purely on their observed-parent overlap.

* Verma, T. & Pearl, J. (1988). "Causal networks: semantics and expressiveness". *Proc. UAI 1988*, pp. 69–78. — d-separation, formally.
* Geiger, D., Verma, T., Pearl, J. (1990). "Identifying independence in Bayesian networks". *Networks* 20(5): 507–534. DOI 10.1002/net.3230200504.
* Lauritzen, S. L., Dawid, A. P., Larsen, B. N., Leimer, H.-G. (1990). "Independence properties of directed Markov fields". *Networks* 20(5): 491–505. DOI 10.1002/net.3230200503. — companion to Geiger et al.; global/local Markov equivalence for DAGs.
* Dawid, A. P. (1979). "Conditional independence in statistical theory". *J. Royal Statistical Society, Series B* 41(1): 1–31. — the foundational treatment of conditional independence; predates Pearl's d-separation.

### 2.5 Worked Pearl-rung numerics on Macbeth (real engine output)

The numbers below are the actual `CausalPhysicsResult.mutations`,
`hidden_deltas`, and `rule3_pruned_interventions` returned by
`CausalPhysicsEngine.execute()` against the bundled
`example_worlds/macbeth.py` fixture. Reproduce via
[`scripts/_dump_pearl_rungs.py`](../scripts/_dump_pearl_rungs.py).
The focal cast is
`[ENT_MACBETH, ENT_LADY_MACBETH, ENT_DUNCAN, ENT_BANQUO, ENT_MACDUFF]`
in every run.

**Rung 1 — Observation** (`engine.execute(rung=2, interventions={})`,
i.e. forward propagation of ambient sources only):

| Node | Trait | Old → New | Impact |
|---|---|---|---|
| `ENT_BANQUO` | `suspicion` | $+0.003 \to +0.015$ | $+0.021$ |
| `ENT_MALCOLM` | `leadership` | $+0.204 \to +0.214$ | $+0.024$ |
| `ENT_LADY_MACBETH` | `ruthlessness` | $+0.993 \to +0.999$ | $+0.023$ |

Plus 27 propagation impulses absorbed by the noisy-OR gate
(`reason="noisy_or_absorbed"`) — every active source fires, but most
trait shifts fall below the per-trait `propagation_threshold`.

**Rung 2 — Intervention** (`do(ENT_MACBETH.traits.ambition = 0)`,
target set `[ENT_DUNCAN, ENT_LADY_MACBETH]`):

| Node | Trait | Old → New | Impact |
|---|---|---|---|
| `ENT_LADY_MACBETH` | `resolve` | $+0.966 \to +0.980$ | $+0.040$ |
| `ENT_LADY_MACBETH` | `ruthlessness` | $+0.698 \to +0.704$ | $+0.019$ |
| `ENT_LADY_MACBETH` | `guilt` | $+0.922 \to +0.917$ | $-0.006$ |
| `ENT_LENNOX` | `caution` | $+0.862 \to +0.868$ | $+0.023$ |
| `ENT_BANQUO` | `suspicion` | $+0.001 \to +0.013$ | $+0.022$ |
| `ENT_MALCOLM` | `courage` | $+0.713 \to +0.722$ | $+0.026$ |

`intervened_nodes = [ENT_MACBETH]`,
`rule3_pruned_interventions = []`,
`rule2_redundant_evidence = []`.

**Rung 3 — Counterfactual** (abduction conditioned on
`evidence_node_ids=[ENT_MACBETH, ENT_LADY_MACBETH]`, then
`do(ENT_MACBETH.traits.ambition = 0)`).

Abduction populates `result.hidden_deltas` with the per-trait latent
shifts that explain the observed downstream:

```text
ENT_MACBETH:
  ambition       +0.598    courage         -0.061
  loyalty        +0.102    guilt           +0.444
  paranoia       +0.173    ruthlessness    -0.289
  despair        +0.670
ENT_LADY_MACBETH:
  ambition       -0.076    ruthlessness    +0.188
  resolve        -0.800    guilt           +0.950
```

The `+0.598` ambition shift on Macbeth and the `-0.800` resolve
shift on Lady Macbeth are precisely the latent perturbations the
*observed* Act-V evidence requires; the precision-weighted Bayesian
blend in §2.3 derives them from the per-trait inertia and the gap
between the sandbox prior and the factual `state_timeline`.

Forward propagation then fires on top of those staged sources:

| Node | Trait | Old → New | Impact |
|---|---|---|---|
| `ENT_LADY_MACBETH` | `guilt` | $+0.792 \to +0.841$ | $+0.062$ |
| `ENT_LADY_MACBETH` | `resolve` | $+0.484 \to +0.495$ | $+0.032$ |
| `ENT_LADY_MACBETH` | `ruthlessness` | $+0.783 \to +0.791$ | $+0.022$ |
| `ENT_DUNCAN` | `trust` | $+0.870 \to +0.875$ | $+0.023$ |
| `ENT_DUNCAN` | `leadership` | $+0.906 \to +0.911$ | $+0.023$ |

`rule3_pruned_interventions =
["ENT_MACBETH.traits.ambition"]`. The static-graph Rule 3 check
flags the do-surgery as vacuous on the world-cropped diagram (the
mutilated AMWN has no surviving directed path from `ambition` to the
chosen target set), but advisory mode keeps it in the simulation so
the abduction-driven downstream still mutates. This is exactly the
over-strict d-separation behaviour the closed-world caveat warns
about (§2.2); opt-in `rule3_pruning_mode="prune"` would short-circuit.

**Vacuous-intervention pre-flight.** Running
`do(ENT_DUNCAN.traits.kindness = 0)` against
`target_node_ids=[ENT_BANQUO]` exercises the Rule-3 path explicitly:
the engine still produces five propagation mutations
(`ENT_LADY_MACBETH.guilt: +0.015 \to +0.063`, `ENT_LENNOX.loyalty:
+0.830 \to +0.855`, …) because in advisory mode the do is applied
even when Rule 3 flags it. `rule3_pruned_interventions` would carry
the flag in prune mode.

**Romeo and Juliet — `do(ENT_FRIAR_LAURENCE.traits.diligence = 1)`.**
The same engine on the bundled fixture produces 17 trait mutations
across Mercutio, Tybalt, Paris, Balthasar, Benvolio, and Rosaline —
including `ENT_BALTHASAR.loyalty: +0.851 \to +0.946` (impact
$+0.233$) and `ENT_MERCUTIO.loyalty: +0.988 \to +1.000` (impact
$+0.175$) — showing that a counterfactually diligent friar shifts
the supporting cast's allegiance vectors well beyond Romeo and
Juliet themselves.

**Gone Girl — abduction with no surviving propagation.**
`engine.execute(rung=3, interventions={"ENT_AMY.traits.deceit": 0},
evidence_node_ids=[ENT_NICK, ENT_AMY])` populates substantial
`hidden_deltas` (`ENT_NICK.adaptability: +0.839`,
`resentment: +0.640`, `ENT_AMY.narcissism: -0.196`,
`manipulation: -0.100`) and zero `mutations` — the abduction
fully explains the observed Nick/Amy state without any post-hoc
forward propagation needing to fire. This is the engine reporting
that the do-surgery + evidence is *consistent* with the observed
downstream, the strongest Rung-3 outcome shape.

---

## 3. Computational models of suspense, surprise, and curiosity

### 3.1 Suspense as hope/fear (here hope/threat) anticipation — structural-affect lineage

`DirectiveAssembler.compute_suspense_score()` is **not** an
implementation of Wilmot & Keller's information-theoretic
uncertainty-reduction model. W&K define suspense as the entropy
reduction (Hale-style surprisal differential) between the
reader's distribution over story continuations before and after
the next sentence — a forward-looking, neural-LM quantity that
operates purely on the reader's epistemic horizon. We do not
compute that quantity; the data layer is a discrete typed graph,
not a sentence-level LM rollout. Instead we operationalise the
older **structural-affect / hope-fear** lineage in which
suspense is the audience's anxious anticipation of an outcome
involving an entity they care about. The hope/fear pair is the
canonical anticipation pair in OCC appraisal theory
([Ortony, Clore & Collins, 1988](https://doi.org/10.1017/CBO9780511571299));
the disposition-weighted variant is
[Zillmann (1996)](https://psycnet.apa.org/record/1996-97152-009);
the high-subjective-probability-of-aversive-outcome variant is
[Comisky & Bryant (1982)](https://doi.org/10.1111/j.1468-2958.1982.tb00682.x);
the planner-style operationalisation we follow most closely is
Cheong & Young's *Suspenser*. We use *threat* in place of *fear*
to keep the field name aligned with the typed-graph framing
(an entity is acted upon vs. is acting), but the structural
position is identical.

The implementation aggregates the unrevealed forward causal
momentum on each side of the entity's outcome ledger and combines
the two sides as a **balance × stakes** product:

$$\text{balance} = 1 - \frac{|w_\text{threat} - w_\text{hope}|}{w_\text{threat} + w_\text{hope}},
\quad
\text{stakes} = \frac{w_\text{threat} + w_\text{hope}}{w_\text{threat} + w_\text{hope} + K},$$

$$\text{suspense}(t) = \text{clip}_{[0,1]}\!\big(\text{balance} \cdot \text{stakes}\big)$$

where $w_\text{threat}$ sums the `evidence_strength`-derived
probabilities of unrevealed events in which the focal entity is a
non-acting target, $w_\text{hope}$ sums the same probabilities for
unrevealed events the entity itself authors, and $K$ (default $2$)
calibrates how quickly stakes saturate — "two strong unrevealed
events on each side" already counts as fully high-stakes. The
balance term peaks under genuine outcome uncertainty
($w_\text{threat} = w_\text{hope}$) and decays to zero under
one-sided dominance, matching the intuition that *fully expected*
outcomes (whether triumph or doom) carry no suspense; the stakes
term prevents balanced-but-trivial fragments from pinning the
gauge at $1.0$. Returns $0$ at the **suspense → despair** boundary
($w_\text{hope} = 0$) and at the symmetric **safety** boundary
($w_\text{threat} = 0$) — both degenerate to non-suspense.

*Implementation note.* An earlier asymmetric form
$\max(0, (w_\text{threat} - w_\text{hope})/(w_\text{threat} + w_\text{hope}))$
collapsed to zero on every fixture in `example_worlds/` because the
protagonist is the actor of most of their own forward events
(Macbeth kills Duncan / Banquo / Macduff's family, all bumping
$w_\text{hope}$ over $w_\text{threat}$), pinning suspense at $0$
across the entire syuzhet axis even for canonical thrillers and
tragedies. The current balance × stakes product follows the
Brewer & Lichtenstein structural-affect framing of suspense as a
response to genuine outcome ambiguity rather than to one-sided
causal dominance.

*Known limitations of this design (vs. richer suspense theory):*
1. **Outcome-uncertainty only, not paradox-of-suspense aware.** We
   do not attempt to model the residual tension that survives
   re-reading
   ([Gerrig 1989](https://doi.org/10.1016/0749-596X(89)90001-6);
   [Baroni 2007](https://www.seuil.com/ouvrage/la-tension-narrative-suspense-curiosite-surprise-raphael-baroni/9782020897624)).
2. **Probability proxy is the max incoming edge weight**, not a
   joint probability over the full causal path; this is a
   deliberate tractability choice consistent with Cheong &
   Young's planning-graph operationalisation.

*Disposition-aware classification* (Zillmann 1996). The legacy
actor=hope / target=threat rule is overridden when the
social topology says otherwise: an event whose actor has
``affinity[actor → focal] ≤ -0.2`` is bucketed as a *threat* on
the focal entity (an antagonist's authored misdeed no longer
registers as hope just because they are the actor), and an event
whose actor has ``affinity[actor → focal] ≥ +0.2`` propagates as
*hope* even when the focal entity is its target (rescue
propagation: the ally arriving to defuse the threat). When the
world has no ``social_topology`` the affinity defaults to neutral
(0.0) and the legacy rule applies, so worlds without a populated
topology degrade gracefully.

*Anticipatory proximity weighting* ([Comisky & Bryant
1982](https://doi.org/10.1111/j.1468-2958.1982.tb00682.x)).
Subjective probability of a threat rises with imminence. Each
event's contribution is multiplied by a temporal–spatial kernel:

$$\text{imminence}(e, f) = \exp\!\Big(-\tfrac{\Delta t_{\text{fabula}}}{\tau_t}\Big)
\,\cdot\,
\exp\!\Big(-\tfrac{\Delta d_{\text{spatial}}}{\tau_s}\Big),$$

where $\Delta t_{\text{fabula}} = \max(0,\, t_e^{\text{fabula}} - t_{\text{now}}^{\text{fabula}})$
is the fabula-time gap from the latest revealed event to the
unrevealed event, $\Delta d_{\text{spatial}}$ is the shortest-path
distance in the spatial topology between the event's location and
the focal entity's location, and $\tau_t$ auto-scales to the
world's median inter-event fabula gap (so worlds with
``fabula_time_spacing=1000`` and unit-spaced worlds both decay
over ~6 narrative beats). $\tau_s$ defaults to 4 hops. Distant
threats weigh less than the same threat closing in, exactly as
the Comisky–Bryant manipulation predicts.

*Persistence / exposure multiplier* (Brewer & Lichtenstein 1982
initiating-event arc). Each unrevealed-threat term is multiplied
by ``min(cap, 1 + α · a)`` where $a$ is the count of *revealed*
causal ancestors of the threat (proxy for how long the gun has
been on the mantle). $\alpha = 0.10$, cap $= 1.5$. A threat
introduced at anchor 0 and still unresolved at anchor 14 weighs
50% more than a freshly-revealed one of equal probability.

*Per-kind balance $\times$ stakes with weighted-max combine
(``mode='classic'``).* The original Brewer-Lichtenstein-anchored
aggregator keeps separate ledgers per kind, computes balance and
stakes *within* each kind, and combines via a salience-weighted
**max**:

$$\text{Susp} = \max_k \,\big( \sigma_k \cdot \text{balance}^{(k)} \cdot \text{stakes}^{(k)} \big),$$

where $\sigma_k$ is the kind's salience weight (table below).
This matches Brewer–Lichtenstein's prediction that one *dominant*
unresolved beat carries the structural-affect arc, rather than
several diffuse anxieties additively. The dominant kind is
surfaced for downstream directive consumers.

*Per-kind saturation constants $K_k$.* Existential threats don't
saturate quickly — one death threat does not max out the gauge —
so $K_{\text{existential}} = 4$. Social/epistemic threats *do*
saturate fast (three slights and the reader is bored), so
$K_{\text{social}} = K_{\text{epistemic}} = 1.5$. Calibrated
against the ``example_worlds/`` corpus.

*Harm-kind salience weighting.* The kind is inferred from the canonical `mechanism`
strings carried on incident causal edges (see
`causal_physics.MECHANISM_TRAIT_MAP`); each event takes the *max*
salience across its incident-edge mechanisms, so a stab-in-the-back
event wired with both `physical` and `betrayal` edges registers at
the higher of the two rather than being averaged. The salience
ranking follows the appraisal-theory hierarchy
([Lazarus 1991](https://psycnet.apa.org/record/1991-97375-000)
core relational themes; Ortony, Clore & Collins 1988 OCC
prospect-based emotions; Brewer & Lichtenstein 1982
structural-affect):

| Kind | Salience $\sigma_k$ | Saturation $K_k$ | Lazarus / OCC anchor |
|---|---:|---:|---|
| `existential` (mortal) | 1.00 | 4.0 | OCC "irrevocable loss" — outranks all other prospects |
| `physical` | 0.85 | 3.0 | Lazarus "physical danger" |
| `betrayal` | 0.75 | 2.5 | Lazarus "moral transgression"; second only to mortal threat in the example corpus (Gone Girl, Reservoir Dogs, Tinker Tailor) |
| `psychological` | 0.70 | 2.0 | OCC "distress about a self-relevant prospect" |
| `emotional` (relational) | 0.65 | 2.0 | Lazarus "relational loss" (Wuthering Heights, Persuasion) |
| `social` (reputational) | 0.55 | 1.5 | Lazarus "social esteem / shame" |
| `epistemic` / `informational` | 0.45 | 1.5 | Discovery as a *prospect*; held low because cumulative-mystery (§3.2) already covers the epistemic surface |

Events with no resolvable mechanism default to `physical` (the
modal kind in the corpus and the median salience), so the gauge
degrades gracefully on sparse fixtures. Per-kind sub-totals and
the dominant threat/hope kinds are surfaced in the debug log so
downstream directive consumers can target the dominant beat
(footsteps closing in vs. the lie about to surface vs. the fellowship
about to fracture) rather than only the aggregate magnitude.

*EFK expected-variance aggregator (default,
``mode='efk'``)* ([Ely, Frankel & Kamenica
2015](https://doi.org/10.1086/677350)). The default aggregator
implements a **belief-martingale variance** per (focal, kind)
cell rather than the static balance × stakes product. For each
cell, revealed threat/hope evidence on the syuzhet axis up to
the anchor seeds a Beta(1+A, 1+B) posterior with mean
$\mu_t = (1+A)/(2+A+B)$ — the audience's current belief that the
*next* reveal on this kind will land threat-side. Each
unrevealed event $e$ with bucket $b_e\in\{\text{threat,
hope}\}$, weight $w_e = p_e \cdot \sigma_{k_e} \cdot \pi_e$ and
proximity $\rho_e = \text{prox}(e, x)$ defines the Bayes update
that *would* occur if $e$ were the next reveal:

$$\mu_e^+ = \frac{1+A+w_e}{2+A+B+w_e}, \quad
\mu_e^- = \frac{1+A}{2+A+B+w_e}, \quad
\Delta\mu_e = \mu_e^{b_e} - \mu_t.$$

The realised expected squared belief change, weighted by
proximity (the audience attends to the next reveal in
proportion to how soon it is), is:

$$\sigma^2_{\text{fk}} = \sum_e \frac{\rho_e}{\sum_{e'}\rho_{e'}} (\Delta\mu_e)^2,$$

normalised by the maximum-suspense reference at the same prior
— the squared shift a single composite reveal of the same
total mass $W = \sum_e w_e$ would induce on whichever side moves
belief most:

$$\sigma^2_{\max} = \max\!\Big( (\mu^+_W - \mu_t)^2, \; (\mu^-_W - \mu_t)^2\Big), \qquad \widetilde\sigma^2_k = \min\!\Big(1, \sigma^2_{\text{fk}}/\sigma^2_{\max}\Big) \in [0,1].$$

The cell-level gauges are then aggregated by salience-weighted
stakes attenuation across (focal, kind) cells:

$$\text{Susp}^{\text{efk}} = \frac{\sum_{(x,k)} \sigma_k \cdot \text{stakes}^{(x,k)} \cdot \widetilde\sigma^2_k}{\sum_{(x,k)} \sigma_k \cdot \text{stakes}^{(x,k)}}, \qquad \text{stakes}^{(x,k)} = \frac{T^{\text{unrev}}_{(x,k)}}{T^{\text{unrev}}_{(x,k)} + K_k},$$

where $T^{\text{unrev}}_{(x,k)} = \sum_e w_e$ is the *unrevealed*
weighted mass on this cell (so stakes decay as the narrative
exhausts its forward reveal budget) and $K_k$ is the per-kind
saturation constant from §3.2's harm-kind table. We chose this
as the default because it gives suspense the **same
Bayesian-belief shape** that the surprise scorer
(§3.3, $D_{\mathrm{KL}}$ over trait Bernoullis) already has —
the two affective scorers become moments of the same belief
process rather than unrelated heuristics, and the variance
quantity is the literal Ely-Frankel-Kamenica object rather than
a static-uncertainty proxy. Pass ``mode='classic'`` to recover
the Brewer–Lichtenstein balance × stakes aggregator above (used
as a fallback when no cell has bilateral unrev mass).

A **bilateral-mass guard** restricts the aggregator to (focal,
kind) cells whose *unrevealed* set contains both threat and hope
candidates. A purely one-sided forward reveal set is despair
(only threats coming) or safety (only hopes coming) under
Brewer–Lichtenstein structural-affect theory, even though strict
EFK would still admit positive variance from magnitude
uncertainty alone. This matches the OCC prospect-based-emotion
taxonomy: suspense requires outcome ambiguity, not merely
magnitude ambiguity.

* **Wilmot, D. & Keller, F. (2020).** "Modelling Suspense in Short Stories as Uncertainty Reduction over Neural Representation". *Proc. ACL 2020*, pp. 1763–1788. [aclanthology.org/2020.acl-main.161](https://aclanthology.org/2020.acl-main.161/) — the related but distinct neural-LM uncertainty-reduction definition. We take their reader-uncertainty framing as conceptual support for the *dramatic-irony* scorer (§3.4) and for *mystery* (§3.2), not for our hope/threat suspense scorer.
* Wilmot, D. & Keller, F. (2021a). "A Temporal Variational Model for Story Generation". arXiv:2109.06807 (preprint only).
* Wilmot, D. & Keller, F. (2021b). "Memory and Knowledge Augmented Language Models for Inferring Salience in Long-Form Stories". *Proc. EMNLP 2021*, pp. 851–865. [aclanthology.org/2021.emnlp-main.65](https://aclanthology.org/2021.emnlp-main.65/) — extends uncertainty-reduction to long novels via memory-augmented LMs.
* Wilmot, D. (2022). *Great Expectations: Unsupervised Inference of Suspense, Surprise and Salience in Storytelling*. PhD thesis, Univ. of Edinburgh. arXiv:2206.09708. — full discussion of suspense / surprise / salience as computable quantities; deep influence on our four-effect taxonomy.
* **Comisky, P. & Bryant, J. (1982).** "Factors involved in generating suspense". *Human Communication Research* 9(1): 49–58. DOI 10.1111/j.1468-2958.1982.tb00682.x. — high subjective probability of harm to a liked protagonist.
* **Zillmann, D. (1996).** "The psychology of suspense in dramatic exposition". In Vorderer, Wulff & Friedrichsen (eds.), *Suspense: Conceptualizations, Theoretical Analyses, and Empirical Explorations*, pp. 199–231. Lawrence Erlbaum. — disposition theory; suspense is noxious anticipation about a *liked* character.
* **Ortony, A., Clore, G. L. & Collins, A. (1988).** *The Cognitive Structure of Emotions*. Cambridge UP. DOI 10.1017/CBO9780511571299. — OCC appraisal model; hope/fear is the canonical pair of *prospect-based* emotions.
* **Lazarus, R. S. (1991).** *Emotion and Adaptation*. Oxford UP. — *core relational themes* tie distinct emotion families to distinct kinds of harm/benefit (physical danger, irrevocable loss, moral transgression, relational loss, social esteem/shame). The harm-kind salience table for the threat/hope ledger above is anchored to this hierarchy.
* Ely, J., Frankel, A. & Kamenica, E. (2015). "Suspense and Surprise". *Journal of Political Economy* 123(1): 215–260. DOI 10.1086/677350. — decision-theoretic complement to W&K: suspense as expected variance of next-period beliefs over a terminal outcome.
* Gerrig, R. J. (1989). "Suspense in the Absence of Uncertainty". *Journal of Memory and Language* 28(6): 633–648. DOI 10.1016/0749-596X(89)90001-6. — the paradox of suspense; surveyed but not implemented.
* Baroni, R. (2007). *La tension narrative: suspense, curiosité, surprise*. Paris: Éditions du Seuil. — the modern French-language synthesis distinguishing suspense, curiosity, and surprise as *narrative-tension* sub-types.
* Brewer, W. F. & Lichtenstein, E. H. (1982) — see §3.2.
* Cheong, Y.-G. & Young, R. M. (2015) — see §3.2.

### 3.2 The Sternberg triad (mystery / suspense / surprise)

Our four named structural effects (mystery, dramatic irony, suspense,
surprise) extend Meir Sternberg's classical *curiosity / suspense /
surprise* triad with dramatic irony as a fourth axis.

`compute_mystery_score()` operationalises Sternberg's *curiosity*
axis as a *fraction-of-hidden-causal-ancestors* gauge, with each
ancestor's contribution scaled by three multiplicative factors:

1. **Path-strength geometric decay** — the strongest reverse-path
   product of edge weights from ancestor to effect, depth-capped
   at $D = 4$ (Trabasso & Sperry 1985 causal-network reading
   studies put the audience-traceable chain depth at four hops).
   Computed via single-source Dijkstra on the negated-log-weight
   reverse graph: a 3-hop weak chain
   ($0.25 \times 0.25 \times 0.25 \approx 0.016$) contributes far
   less curiosity weight than a 1-hop strong link ($0.75$),
   replacing the legacy single-edge fallback that gave a 5-hop
   ancestor as much weight as a 1-hop link.
2. **Harm-kind salience** — multiplied by $\sigma_k$ from the
   `_HARM_KIND_SALIENCE` table (the same Lazarus-anchored
   existential > physical > betrayal > ... hierarchy used by
   suspense and dramatic irony). A hidden murder is more
   mysterious than a hidden gossip exchange even when the path
   strengths are identical, capturing Sternberg's *expositional
   gap* weighting by the stake of the missing piece.
3. **Curiosity-proximity decay** — per-effect decay factor
   $\exp(-(t - s_e)/\tau_{\text{curiosity}})$ with $\tau = 8$
   syuzhet-index units. The reader's curiosity sits over the most
   recently surfaced effects, not the entire revealed cone
   equally; long-resolved gaps have been mentally filed and no
   longer drive the gauge (Iser 1976 *Akt des Lesens*; Sternberg
   1992 curiosity taxonomy). Symmetric mirror of the suspense
   forward-imminence kernel — *backward* over surfaced unexplained
   effects rather than *forward* over upcoming threats.

The aggregate is then $\text{mystery} = M_{\text{hidden}} /
M_{\text{total}}$ where each $M$ sums per-ancestor contributions
weighted by all three factors. The score declines monotonically
as ancestors are revealed; for canonical mystery plots
(*Death on the Nile*, *Tinker Tailor*, *Macbeth*'s prophecy
chain) it sits in the $0.7–1.0$ band early and falls to $0.2–0.4$
post-denouement.

* Sternberg, M. (1978). *Expositional Modes and Temporal Ordering in Fiction*. Johns Hopkins UP.
* Sternberg, M. (1992). "Telling in time (II): Chronology, teleology, narrativity". *Poetics Today* 13(3): 463–541. — formal definitions of curiosity, suspense, surprise as cognitive states with distinct triggers.
* Brewer, W. F. & Lichtenstein, E. H. (1982). "Stories are to entertain: A structural-affect theory of stories". *J. of Pragmatics* 6(5–6): 473–486. — empirical grounding of the triad.
* Cheong, Y.-G. & Young, R. M. (2015). "Suspenser: A story generation system for suspense". *IEEE Transactions on Computational Intelligence and AI in Games* 7(1): 39–52. — operationalises Brewer's structural-affect theory in a generation system.
* Bae, B.-C. & Young, R. M. (2008). "A use of flashback and foreshadowing for surprise arousal in narrative using a plan-based approach". *ICIDS 2008*, LNCS 5334, pp. 156–167. — formal planning model of narrative-level surprise via anachrony; underwrites our anachrony surprise component (§3.3).
* Trabasso, T. & Sperry, L. L. (1985). "Causal relatedness and importance of story events". *Journal of Memory and Language* 24(5): 595–611. — causal-network reading studies that calibrate the four-hop traceability cap.
* Iser, W. (1976). *Der Akt des Lesens*. München: Wilhelm Fink. (English: *The Act of Reading*, 1978, Johns Hopkins UP.) — gap theory of reader response motivating curiosity-proximity decay.

### 3.3 Surprise as KL divergence

`compute_surprise_score()` uses per-trait binary KL divergence
$D_\text{KL}(p \| q) = p\log\frac{p}{q} + (1-p)\log\frac{1-p}{1-q}$.

**Posterior $p$** is the focal entity's *final-state* trait value,
resolved via `reconstruct_entity_at(ent, t_max)` so authored
`state_timeline` arcs (Macbeth's ambition $0.7 \to 0.85$, Lady
Macbeth's guilt $0.0 \to 0.9$, etc.) are honoured rather than read
from the baseline `Entity.traits` field — reading the baseline was
a silent bug that compared every protagonist against itself and
collapsed surprise to zero before the prior update even ran.

**Prior $q$** starts at the **leave-one-out** per-trait corpus
marginal — the mean value of that trait across every *other*
entity in the world. With small casts (the example fixtures
average 6–10 entities) the focal entity carries 10–17% of the
inclusive marginal, systematically pulling $q$ toward $p$ and
squashing surprise; the leave-one-out form removes that bias. We
fall back to the maximum-entropy default $q = 0.5$ when fewer than
two *other* entities carry the trait. Each revealed causal edge
incident on the focal entity then contributes a **Beta-Bernoulli
update** (replacing the legacy ad-hoc geometric pull
$q \mathrel{+}= w \cdot (\text{actual} - q)$):

$$\alpha_0 = s \cdot m, \quad \beta_0 = s \cdot (1 - m), \quad
\alpha \mathrel{+}= w_e \cdot \text{actual}, \quad
\beta \mathrel{+}= w_e \cdot (1 - \text{actual}),$$

with pseudo-count strength $s = 2$ (weak Beta anchor — strong
enough to keep $q$ off the EPS-clipped extremes when evidence
is sparse, weak enough to remain responsive to the first few
edges). The posterior mean $q = \alpha / (\alpha + \beta)$ is
returned. This is the same Bayesian core used by the EFK
suspense aggregator, so the surprise and suspense scorers now
share a coherent posterior-update rule rather than two unrelated
heuristics. Edges where the focal is the *target* contribute at
full weight; edges where the focal is the *source* contribute at
$0.4 \times$ full weight ("X did Y to Z" speaks more strongly
about Z's traits than X's, but X's act itself is non-trivial
evidence about X's traits — Macbeth's ambition is reinforced by
acting on it). The geometric form was a Storck/Hochreiter/
Schmidhuber 1995 RDIA proxy with undefined posterior variance,
which the per-trait salience and Weber-Fechner extensions below
require to behave well.

The surprise score has two operating modes — a *cumulative* form
(directive-optimiser default) and a *local* Bayesian-Surprise form
(time-series default), corresponding directly to the two
mathematically distinct quantities Itti & Baldi distinguish.

* **Cumulative form** (`local=False`, the default): KL between the
  reader's accumulated prior and the true posterior, $D_{\rm KL}(p
  \| q)$ where $q$ is the geometrically-updated prior described
  above. As evidence accumulates, $q$ asymptotes toward $p$, so the
  score declines monotonically — answering *how much catching-up
  the reader still has to do*. The directive-assembly optimiser
  consumes this form because its loss-function semantics require a
  monotone "remaining gap to truth" signal that falls as reveals
  close it.

* **Local form** (`local=True`, used by the time-series view, after
  Itti & Baldi 2009): the per-step belief-update magnitude,
  $D_{\rm KL}(q_s \| q_{s-1})$ — the KL distance between the
  reader's prior immediately *after* and immediately *before* the
  current syuzhet anchor's revelations. This is the Itti-Baldi
  "Bayesian Surprise" definition verbatim: surprise =
  $D_{\rm KL}({\rm posterior} \| {\rm prior})$, the dissimilarity
  between belief distributions before and after observing data.
  Quiet syuzhet stretches contribute $\sim 0$; revelations spike
  in proportion to how much they shift the prior, producing the
  impulse-and-decay trajectory the theory predicts and Reagan
  et al. (2016) observe in corpus emotional arcs.

The two forms are not interchangeable: cumulative surprise is the
*integrated* gap between expectation and truth (a state quantity);
local surprise is its *temporal derivative* (an event quantity).
Conflating them by trying to recover one from the other was a
recurring source of misinterpreted plots in the time-series view
before the explicit `local=True` mode was added — the cumulative
curve looks "wrong" against the Itti-Baldi intuition because the
expectation operator is doing different work in each case. The
sandbox/world-state posterior $p$ used by the cumulative form is
also resolved at the world's *final* fabula-time so authored arc
trajectories (`state_timeline`) are honoured; reading raw
`Entity.traits` was a silent bug that compared every protagonist
against itself and collapsed surprise to zero.

The trait-KL aggregator is then supplemented by two extensions:

**Per-trait narrative salience.** Each trait's contribution is
weighted by `_TRAIT_NARRATIVE_SALIENCE` (the Reagan-et-al. 2016
arc-relevance hierarchy: ambition / guilt / vengeance /
despair / love / loyalty / courage at $\sigma_t \in [0.85,
1.0]$, mid-tier traits at $0.55–0.65$, peripheral traits like
literacy / fitness / wealth at $0.30$). Trait names are matched
case-insensitively as substrings so `moral_courage`,
`physical_courage` and `courage` all resolve to the `courage`
weight; unmatched traits get the median $0.55$. Mirrors the
harm-kind salience hierarchy already used by suspense and
mystery. Without this weighting, peripheral traits diluted the
gauge on canonical arc-driven fixtures (Macbeth's ambition arc
was being averaged with Macbeth's literacy and wealth, which
do not change).

**Per-trait Weber-Fechner saturation.** The per-trait
contribution $1 - \exp(-\text{KL})$ keeps each trait in $[0,
1]$ and maps perceptual KLs to perceptual gauge positions:
KL=0.27→0.24, KL=0.5→0.39, KL=1.0→0.63, KL=2.0→0.86. The
decay constant $\tau = 1$ matches the binary-distribution
discrimination JND from psychophysics (Lu & Dosher 2013), where
the subjective just-noticeable belief shift sits in the
$[0.5, 1.0]$ nat band — i.e. each $1.0$ nat of KL evidence
delivers $\approx 1 - 1/e \approx 63\%$ of the perceptual
range, which is exactly where this saturation curve places it.
The legacy form divided raw KL by the *theoretical* maximum
$\log(1/\epsilon) \approx 4.6$, compressing perceptually
meaningful KLs (the $0.2–1.5$ band) into a 4 % slice of the
gauge and producing flat-looking surprise curves.

**Anachrony surprise (Bae & Young 2008; Bissell, Paulin &
Piper 2025).** Trait-shift KL alone misses the surprise
generated by *temporal reordering* — flashbacks that reframe
earlier events, openers that drop the reader in medias res. We
compute a per-event anachrony score
$|\text{rank}_{\text{fabula}}(e) - \text{rank}_{\text{syuzhet}}(e)| / N$,
average over the relevant event set (cumulative mode → all
revealed events; local mode → events newly revealed at this
anchor — mirroring the trait-KL split between integrated and
per-step surprise), and combine with the trait-KL component via
a convex weighting

$$\text{Surprise} = w_t \cdot \text{Surp}_{\text{trait-KL}} + w_a \cdot \text{Surp}_{\text{anachrony}}$$

with $w_t = 0.7$, $w_a = 0.3$. Worlds with linear tellings
contribute zero anachrony and degrade exactly to the previous
trait-KL behaviour; worlds with non-linear tellings (Reservoir
Dogs flashbacks, Gone Girl diary entries, Tinker Tailor
recursive intelligence-investigation flashbacks) get an
additional anachrony-driven contribution that the Bissell-
Paulin-Piper framework flags as the most important missing
dimension in KL-only narrative-surprise models.

* Itti, L. & Baldi, P. (2009). "Bayesian surprise attracts human attention". *Vision Research* 49(10): 1295–1306. DOI 10.1016/j.visres.2008.09.007. — the formal basis: surprise = KL between prior and posterior beliefs.
* Schmidhuber, J. (2010). "Formal theory of creativity, fun, and intrinsic motivation (1990–2010)". *IEEE Trans. Autonomous Mental Development* 2(3): 230–247. DOI 10.1109/TAMD.2010.2056368. — surprise as compression progress.
* Reagan, A. J., Mitchell, L., Kiley, D., Danforth, C. M., Dodds, P. S. (2016). "The emotional arcs of stories are dominated by six basic shapes". *EPJ Data Science* 5: art. 31. DOI 10.1140/epjds/s13688-016-0093-1. — corpus-scale emotional trajectories that motivate our trait-trajectory analytics in `viz_helpers.py` and the per-trait narrative-salience hierarchy.
* Elsner, M. (2012). "Character-based kernels for novelistic plot structure". *EACL 2012*, pp. 634–644. — structural arc analysis predating Reagan et al., using character co-occurrence.
* Kim, E., Padó, S., Klinger, R. (2017). "Investigating the relationship between literary genres and emotional plot development". *Workshop on Computational Linguistics for Literature (NAACL)*, pp. 17–26. — direct empirical follow-up to Reagan et al. on genre-conditioned arcs.
* **Bissell, A., Paulin, E., Piper, A. (2025).** "A theoretical framework for evaluating narrative surprise in large language models". *Proceedings of WNU 2025*. [aclanthology.org/2025.wnu-1.7](https://aclanthology.org/2025.wnu-1.7/) — multi-component narrative-surprise framework (trait shift + anachrony + ToM-model shift) directly motivating the convex split above.
* **Tobin, V. (2018).** *Elements of Surprise: Our Mental Limits and the Satisfactions of Plot*. Harvard UP. — cognitive-narratology account of plot-twist surprise as a function of audience-model overhaul.
* Lu, Z.-L. & Dosher, B. A. (2013). *Visual Psychophysics: From Laboratory to Theory*. MIT Press. — binary-distribution discrimination JND ($0.5–1.0$ nat) used to calibrate the Weber-Fechner saturation constant.

### 3.4 Dramatic irony as epistemic asymmetry

`compute_dramatic_irony_score()` returns a salience- and
prominence-weighted *fraction* of revealed events the focal
entity does **not** know about (by participation, by being
addressed in a revealed utterance, or by holding an explicit
`Belief` whose provenance still resolves), normalised by the
*revealed* event mass plus a saturation constant $K = 1$, then
aggregated across the focal cast by a max-leaning convex blend:

$$g_c(t) = a_c \cdot \frac{\sum_{e \in R_t,\, e \notin K_c} w_e \cdot \sigma_{k_e} \cdot \phi_e^{(c)} \cdot \rho_e^{(c)}}{\sum_{e \in R_t} w_e + K},$$

$$\text{irony}(t) = \min\!\Big(1, \beta \cdot \max_{c \in F} g_c(t) + (1-\beta) \cdot \overline{g_c(t)}\Big),$$

where $F$ is the focal cast, $R_t$ is the set of events revealed
to the reader by syuzhet anchor $t$, $K_c$ is what character $c$
knows (also bounded by the fabula frontier of $R_t$), $w_e$ is
event $e$'s intensity (defaults to $1$), and the per-event
weights are:

* $\sigma_{k_e}$ — harm-kind salience from the
  `_HARM_KIND_SALIENCE` table (the Lazarus / OCC hierarchy
  reused from suspense and mystery). Tragic irony — a hidden
  mortal threat the focal does not see — outranks comic irony
  along the same scale that ranks suspense kinds.
* $\phi_e^{(c)} \in \{1,\, m_{\text{fb}}\}$ — false-belief
  multiplier. When the gap event's actor set intersects the
  focal's *believed-entity targets* (entities about whom the
  focal holds a provenance-valid `Belief`), the focal is acting
  on an outdated picture of one of the perpetrators — the
  canonical Iago→Othello / Jacqueline→Linnet / Hero→Claudio
  pattern. We multiply by $m_{\text{fb}} = 1.5$. We deliberately
  do not try to semantically compare `Belief.perceived_state`
  strings to world truth — model entities the focal *has formed
  an opinion about* are the tractable false-belief surface, and
  the test is invariant under shadow surgery via the same
  provenance gate as the event-belief side (Pfister 1977,
  Cabanas Gonzalez 2024).
* $\rho_e^{(c)}$ — closure-proximity decay
  $\max(\rho_{\min}, \exp(-\Delta_{\text{closure}}/\tau_{\text{irony}}))$
  where $\Delta_{\text{closure}}$ is the syuzhet-index distance
  to the earliest later position at which the focal first
  witnesses an event with $\text{fabula}(e') \geq
  \text{fabula}(e)$ — the dramatic moment the focal walks into
  the scene that exposes the truth. Defaults: $\tau = 6$ syuzhet
  beats, $\rho_{\min} = 0.4$ (Booth 1974 stable-vs-unstable
  irony floor for permanent ironies the focal never resolves).
* $a_c = \min(\bar a, 1 + \alpha \cdot \#\{\text{actor-events
  by } c \leq t\})$ — action-weighting on the focal's
  prominence (Pfister 1977 protagonist-blindness), capped at
  $\bar a = 3.0$ with $\alpha = 0.15$. Macduff's ignorance
  matters more than Lennox's because Macduff is acting on a
  false picture.

The aggregator $\beta \cdot \max + (1-\beta) \cdot \overline g$
with $\beta = 0.6$ is Sternberg's single-dominant-gap framing:
*one* character's tragic blindness carries the irony charge
rather than the cast average, but secondary characters still
register a residual contribution. Dividing by the *revealed*
mass — rather than by the full story's event mass — makes the
score a Sternberg-style **gap fraction** of the reader's
privileged view, which naturally falls when characters catch up
via late-story revelations (Macduff hearing of his family;
Poirot's denouement; Nick's letter to Daisy). An earlier
implementation that normalised by total event mass produced a
monotonically rising curve in 21/21 example-world fixtures
because the numerator's growth with reveals was unopposed by the
constant denominator — contradicting the rise-then-fall arc that
Booth, Stanton, and Sternberg's structural-affect theory predicts
for canonical irony plots. The framing of irony as a
reader/character knowledge gap is classical:

* Booth, W. (1974). *A Rhetoric of Irony*. Univ. of Chicago Press.
* Muecke, D. C. (1969). *The Compass of Irony*. Methuen.
* Stanton, R. (1956). "Dramatic Irony in Hawthorne's Romances". *Modern Language Notes* 71(6): 420–426. DOI 10.2307/3043161. — explicit definition: "audience knows what the character does not".
* **Pfister, M. (1988).** *The Theory and Analysis of Drama*. Cambridge UP. (German original *Das Drama*, 1977.) — formal taxonomy of dramatic-irony sub-types (tragic vs comic; stable vs unstable) underwriting the harm-salience and closure-proximity weightings; the protagonist-prominence axis underwrites the action-weight $a_c$.
* **Cabanas Gonzalez, C. C. (2024).** *Investigating the role of spontaneous theory of mind on the processing of dramatic irony in filmed narratives*. PhD thesis, Birkbeck. [eprints.bbk.ac.uk/id/eprint/54770](https://eprints.bbk.ac.uk/id/eprint/54770/) — empirical ToM-grounded support for the false-belief multiplier; readers form a richer ToM model of focal characters with a believed-entity surface, sharpening the felt asymmetry.
* **Sutherland, J. (2013).** *A Little History of Literature*. Yale UP. — popular-narrative survey of irony as protagonist-blindness, motivating the action-weighting form.
* **Chandra, K., Li, T.-M., Tenenbaum, J. B. (2024).** "Storytelling as Inverse Inverse Planning". *Topics in Cognitive Science* 16: 54–76. DOI 10.1111/tops.12710. — Bayesian model of plot twists as ToM updates; backs the surface-of-believed-entities formalisation.

For computational treatments and adjacent reader-uncertainty
formalisms:

* Gerrig, R. J. (1993). *Experiencing Narrative Worlds*. Yale UP. — the cognitive-pragmatic framework we borrow.
* **Wilmot, D. & Keller, F. (2020).** "Modelling Suspense in Short Stories as Uncertainty Reduction over Neural Representation". *Proc. ACL 2020*, pp. 1763–1788. — although the W&K paper is titled *suspense*, its operationalisation (the entropy-reduction differential between the reader's distribution over continuations before and after the next sentence) is structurally a **reader-vs-future-state epistemic-asymmetry** measure: it quantifies *what the reader's model of the story does not yet contain*. That shape is closer to the dramatic-irony scorer here (reader vs. character knowledge gap) and to the mystery scorer (reader vs. complete causal-ancestor set, §3.2) than it is to our hope/threat suspense scorer (§3.1). Readers cross-comparing implementations should treat the W&K quantity as a neural-LM analogue of mystery / irony rather than of `compute_suspense_score`.
* Ely, J., Frankel, A. & Kamenica, E. (2015). "Suspense and Surprise". *J. Political Economy* 123(1): 215–260. DOI 10.1086/677350. — decision-theoretic complement; their *suspense* is the expected variance of next-period beliefs about a terminal outcome, again an epistemic-asymmetry quantity adjacent to dramatic irony rather than to hope/fear anticipation.

### 3.5 Corpus scale audit

The four scorers were calibrated and audited against the
[`example_worlds/`](../example_worlds) corpus (20 hand-curated
canonical fixtures spanning tragedy, mystery, comedy, romance,
modernist fragmentation and ensemble heist). Sampling each
fixture at 7 evenly-spaced syuzhet anchors (140 score evaluations
per metric) gives the following per-scorer scale summary:

| Scorer            | min  | median | mean | max  | non-zero |
|-------------------|------|--------|------|------|----------|
| `mystery`         | 0.17 | 0.53   | 0.57 | 1.00 | 162/162  |
| `dramatic_irony`  | 0.00 | 0.45   | 0.45 | 0.90 | 161/162  |
| `suspense`        | 0.00 | 0.16   | 0.14 | 0.39 | 132/162  |
| `surprise` (local)| 0.00 | 0.03   | 0.05 | 0.24 | 129/162  |

The four scorers occupy different absolute bands by design.
Mystery is a population fraction (hidden ancestors over total
ancestors) and naturally lives near 1.0 early in the syuzhet,
falling monotonically as causes are revealed. Dramatic irony is
a per-character revealed-mass *gap fraction* (Sternberg) and lives
in a wide rise-peak-fall band centred on 0.45. Suspense
discharges to 0 at the terminal anchor of every world (no
unrevealed threats remain) and lives in the lower 0.0–0.4 band
because the saturation constant *K* in the stakes denominator
intentionally damps the gauge. Surprise (in *local* anchor mode,
the per-step Itti-Baldi spike) only registers at canonical
revelation points and is otherwise near zero — which is the
expected sparse-spike behaviour the literature predicts.

Canonical signatures the audit confirms:

* **`macbeth`** — mystery monotone fall 0.99 → 0.28; irony rise-fall
  0.21 → 0.62 → 0.41 (Macduff hearing of his family); surprise
  effectively 0 (Shakespeare telegraphs every reveal).
* **`death_on_the_nile`** — mystery 1.00 → 0.29; irony peaks at
  anchor 4 (0.65) on Poirot's withheld knowledge; surprise
  spike 0.19 at the denouement reveal.
* **`gone_girl`** — mystery 0.96 → 0.32; irony arc 0.19 → 0.50
  (Amy's diary deception); surprise 0.20 at the mid-novel
  perspective shift.
* **`reservoir_dogs`** — mystery 0.91 → 0.37; irony mid-act
  spike at anchor 5 (0.54) on Mr Orange's identity; surprise
  spike 0.19 at the in-medias-res flashback structure (the
  anachrony component dominates the trait-KL component).
* **`wuthering_heights`** — mystery 0.99 → 0.18; irony
  rise-peak-fall 0.13 → 0.67 → 0.42; surprise 0.24 spike at the
  in-medias-res frame opening (Lockwood arrives, Nelly's
  retrospective floods backward in fabula time).
* **`tinker_tailor_soldier_spy`** — terminal suspense peak 0.39
  before the mole reveal collapses the ledger; irony plateau
  0.39–0.55 across the long investigation.

The six emotion scorers (`grief`, `rage`, `joy`, `regret`,
`love`, `fear`) score per-entity *closeness* to a per-effect
trait target rather than a per-syuzhet timeline quantity, so
they appear flat across anchors but exhibit corpus-wide
variation: median values (0.10–0.76 across emotions) and
non-zero rates (62–85 % of (world, anchor) cells) confirm the
trait-trajectory dispatch resolves real per-character signal on
every fixture in the corpus. (A regression — the assembler's
trait-trajectory loop reading only `ego_payload` and silently
returning `+1.0` worst-loss when the auditor's per-target
rescore arrived without an ego-payload — was caught and fixed in
the same audit pass: the loop now falls back to
`world_state.entities` for any entity not staged in ego.)

All scorer constants are externalised through
`DirectiveAssemblySettings` (env prefix `DIRECTIVE_ASSEMBLY_*`) —
see [docs/settings.md §8](settings.md#8-directive-assembly-step-8--affective-scorers)
for the full env-var table. Reproduce the audit with
[`scripts/audit_affective.py`](../scripts/audit_affective.py).

### 3.7 Propositional belief-revision affect (`affect_unification.py`)

The four structural scorers in §3.1–§3.4 operate on *trait* and
*event* mass — graph geometry. The post-2026 ingestion pipeline
additionally lays down a registry of typed `Proposition` rows (with
`audience_default_prior ∈ [0,1]`, `stakes ∈ [0,1]`,
`truth_at_fabula: Dict[int, bool]`, and a `state_timeline` of
`PropositionSnapshot`s) and per-entity `Concern` rows (with
`polarity ∈ {desire, fear}`, `salience`,
`activation_fabula_window`, `counter_concern_ids`, and a
`state_timeline` of `ConcernSnapshot`s). On top of those,
[`shadow_loom/affect_unification.py`](../shadow_loom/affect_unification.py)
exposes a second, **propositional / Bayesian** scoring layer that
operates directly on belief revision over outcome propositions —
the formal cousin of the trait-anchored scorers above. The two
layers coexist: the trait layer drives the directive-assembly
optimiser (which needs a graph-geometry signal); the propositional
layer powers the affective UI dashboard, the per-proposition belief
tensor, and any consumer that needs a quantity *defined over
propositions the audience and characters hold beliefs about*.

A `BeliefState(world)` reader replays each agent's beliefs at any
`fabula_t` via `reconstruct_entity_at` and falls back to the
proposition's `audience_default_prior` when no agent belief exists.
A reserved `ENT_AUDIENCE` entity is synthesised by
`synthesise_audience_entity(world)` so the omniscient-reader prior
is a first-class agent.

* **Suspense** (Brewer–Lichtenstein, propositional form):
  $$\mathrm{Susp}(t_f) = \!\!\sum_{P \,\in\, \mathrm{open\,outcomes}}\!\! H\!\big(p_{\rm aud}(P, t_f)\big)\,\cdot\,\mathrm{stakes}(P, t_f)\,\cdot\,\exp\!\big(-\Delta t_f / \tau\big),$$
  where $H$ is binary Shannon entropy, the sum runs over
  propositions whose `kind == "outcome"` and whose `truth_at_fabula`
  has not yet committed, $\Delta t_f$ is the fabula-time gap to the
  next future commitment, and $\tau$ defaults to the world's
  median inter-event fabula gap (`_auto_tau_fabula`). The kernel
  is the Comisky–Bryant imminence factor on a *propositional* axis
  rather than the event axis used by §3.1.
* **Surprise** (Itti–Baldi 2009 Bayesian Surprise, propositional
  form):
  $$\mathrm{Sur}(t_f \,\|\, t_f') = \!\!\sum_{P}\!\! D_{\rm KL}\!\big(p_{\rm aud}(P, t_f)\,\|\,p_{\rm aud}(P, t_f')\big)\,\cdot\,\mathrm{stakes}(P, t_f),$$
  the audience-belief KL between two anchors, summed over
  propositions the audience moved on. This is the Itti–Baldi
  definition verbatim, lifted to the proposition layer and
  weighted by the proposition's stakes.
* **Dramatic irony** (Pfister/Sternberg, propositional form):
  $$\mathrm{Iro}(c, t_f) = \!\!\sum_{P}\!\! D_{\rm KL}\!\big(p_{\rm aud}(P, t_f)\,\|\,p_c(P, t_f)\big)\,\cdot\,\mathrm{stakes}(P, t_f),$$
  the asymmetric KL between the audience and a focal character
  $c$ across propositions on which they disagree. Asymmetry
  matters: audience-knows-more (Oedipus's prophecy) and
  focal-knows-more (Iago's plan, before its disclosure to the
  reader) produce distinguishable signals.
* **Mystery** (Carroll erotetic, propositional form): for each
  *known* effect proposition (audience confidence ≥ threshold,
  default 0.7), build the directed proposition graph induced by
  `EventNode.resolves_proposition_ids` /
  `asserts/denies_proposition_id`, find the shortest path from
  every potential cause, softmax-normalise across candidates, and
  sum Shannon entropy. High when the reader sees the effect but
  many causes remain plausibly hidden.

Valence-decomposed variants are also exposed:
`compute_surprise_breakdown` returns a Tan/Ortony pleasant /
unpleasant split per focal entity (using the entity's `Concern`
polarity to decide whether a positive belief shift on a desired
proposition is pleasant surprise or pleasant relief);
`compute_irony_breakdown` returns Sternberg's three irony modes
(suspense-irony, curiosity-irony, surprise-irony).

The propositional layer is **complementary to**, not a replacement
for, the trait-anchored scorers in §3.1–§3.4. The directive
optimiser still consumes the trait scorers because their loss
semantics (monotonically declining with reveals, comparable
across worlds with no propositions) match what the optimiser
needs. The propositional layer is what surfaces in the UI when a
reader asks *"how surprised should I have been by this beat?"* —
the Bayesian-narratology question that requires named
propositions and audience-prior provenance.

* Itti, L. & Baldi, P. (2009). "Bayesian surprise attracts human attention". *Vision Research* 49(10): 1295–1306. DOI 10.1016/j.visres.2008.09.007.
* Tan, E. S. (1996). *Emotion and the Structure of Narrative Film*. Lawrence Erlbaum. — pleasant/unpleasant surprise valence decomposition.
* Carroll, N. (2007). "Narrative closure". *Philosophical Studies* 135(1): 1–15. — erotetic mystery as the entropy over hidden causes of known effects.
* Sternberg, M. (2003). "Universals of narrative and their cognitivist fortunes". *Poetics Today* 24(2): 297–395. — three modes of irony.
* Storck, J., Hochreiter, S., Schmidhuber, J. (1995). "Reinforcement-driven information acquisition in non-deterministic environments". *Proc. ICANN '95*. — KL between belief states $p^*(t{+}1) \| p^*(t)$, formally equivalent to the propositional surprise above.

### 3.6 Heuristic affects (conflict, danger, narrative-tension, causal-density)

Alongside the four engine-grade structural affects (§§3.1–3.4),
the UI surfaces four lighter heuristics computed directly from
the snapshot graph in `compute_affective_scores` rather than via
DirectiveAssembler. They are **snapshot-local** quantities — they
read the current time-sliced relationship state and event ledger,
do not consult `syuzhet_anchor`, and therefore vary along both
fabula and syuzhet axes via the time-slicing performed by
`snapshot_world_at` / `snapshot_world_at_syuzhet`.

* **conflict** = fraction of *observed* affinity edges with
  $\text{affinity} < 0$. Per-axis observation gating (only edges
  whose `affinity.observed=True` count) prevents the LLM never
  having measured an axis from silently deflating the score.
* **danger** = mean of *observed* `fear` values across active
  relationship edges, clipped to $[0, 1]$.
* **narrative_tension** = $0.40 \cdot \overline{|{-}\text{aff}|}
  + 0.35 \cdot \overline{\text{fear}}
  + 0.25 \cdot \text{share}(\text{causal\_force} \geq 7)$. The
  weights sum to 1 so the result stays in $[0, 1]$ without a
  separate clamp. Aligns with Brewer & Lichtenstein's compound
  account of tension as the conjunction of antagonism, fear, and
  high-stakes causation.
* **causal_density** = $\frac{d}{d + K}$ where $d$ is edges per
  event and $K = 1.5$. The soft-saturation form (rather than the
  earlier hard $\min(1, d/3)$ clamp) preserves dynamic range
  across the full corpus: dense passages (Reservoir Dogs $\sim
  0.95$) stay distinguishable from mid-density (Macbeth $\sim
  0.7$) and sparse ones ($\sim 0.25$), where the clamped form
  pinned every dense world flat at 1.0 and hid all variation
  above the threshold. The choice of $K$ is empirical (calibrated
  against `example_worlds/`) rather than theoretical; the metric
  itself is a simple **structural complexity proxy** and not
  meant to track any specific cognitive construct.

These heuristics are intentionally separate from the engine
layer: they are cheap to compute, robust to sparse data, and
visible everywhere a snapshot is shown (gauges, slider scrubs,
ego graphs). The engine-grade affects (§§3.1–3.4) require the
full event graph and are the ones the directive-assembly
optimiser actually targets — but the heuristics anchor the
reader's quick read of the state at any cursor position.

### 3.8 Theory-driven refinements to the structural-affect family

The four engine-grade scorers (§§3.1–3.4) were extended in May
2026 with five literature-anchored refinements that close gaps
identified by re-reading the underlying primary sources against
real corpus output. Each refinement adds a small, interpretable
factor on top of the existing scorer rather than rewriting it.

#### 3.8.1 Audience-concern weighting on suspense (Tan 1996)

Brewer & Lichtenstein 1982 establish *whether* an outcome is
suspenseful (via outcome ambiguity); Tan 1996 (*Emotion and the
Structure of Narrative Film*) supplies the missing piece — *which*
outcomes the audience cares about. Without an audience-concern
gate, two non-focal NPCs having an argument weighted as much on
the suspense ledger as the protagonist on trial for their life,
because both have similar belief variance.

We add a per-event weight $\omega_{\text{concern}}(e) \in [0.2, 1]$
that sums audience concern saliences over propositions whose
referents intersect the event's actor/target set, then maps the
total through a saturating curve:

$$\omega_{\text{concern}}(e) = 0.2 + 0.8 \cdot \frac{\sum_{c \in C(e)} s_c}{1 + \sum_{c \in C(e)} s_c}$$

The 0.2 floor avoids zero-weighting orphan events; saturation
ensures a single high-salience concern is enough to hit the
ceiling. This factor multiplies `weighted_prob` in both the EFK
ledger and the unified Bayesian-stakes path of
`compute_suspense_score`. Implemented as
`DirectiveAssembler._audience_concern_for_event`.

The same factor is applied **symmetrically** to dramatic irony's
per-gap-event weight in `compute_dramatic_irony_score` — an
ironic gap about a character the audience cares about lands
harder than an ironic gap about a stranger. Both surface and
gap forms of structural affect are now uniformly concern-gated,
matching Tan 1996's claim that *F-emotions* (concern-driven) are
the substrate on which all narrative-affect responses are built.

#### 3.8.2 Mystery recency (Carroll 1990 erotetic model)

Carroll's *erotetic narrative* theory holds that the mystery
salient at any reader position is the set of *live* questions —
those raised by recent surface events, not those trailing
behind. The existing curiosity-proximity decay already
implemented this for the *effect* node, so live questions
$Q_t$ contribute more than archived questions $Q_{t-k}$ via

$$w_{\text{prox}}(\text{eff}) = \exp\left(-\frac{s_{\text{anchor}} - s_{\text{eff}}}{\tau_{\text{curiosity}}}\right)$$

with $\tau = 8$ syuzhet steps. The May 2026 audit confirmed this
implements Carroll's recency principle directly; no further
change was needed beyond re-documenting the connection.

#### 3.8.3 False-lead mystery (Sayers 1929 / Knox 1929)

Detective-fiction theory (Sayers' *The Omnibus of Crime*
introduction; Knox's *Decalogue*) frames mystery intensity as
tracking not only the *number* of unrevealed causes but the
*number of plausible-but-wrong* hypotheses the audience is
actively entertaining. Red-herring-rich worlds (Christie, Sayers
herself, *Gone Girl*) read as more mysterious than worlds with
the same hidden-ancestor count but no plausible alternates.

We approximate the live-hypothesis count as the number of
audience belief assignments on uncommitted propositions whose
confidence falls in the ambiguity band $[0.3, 0.7]$ — the reader
has a hypothesis but isn't sure of it. Blended with the
hidden-ratio base via

$$\text{Mystery} = 0.75 \cdot \frac{\text{hidden}_\text{mass}}{\text{total}_\text{mass}} + 0.25 \cdot \frac{n_{\text{false-lead}}}{n_{\text{false-lead}} + 4}$$

so the gauge is still dominated by structural hidden mass but
red-herring-rich fixtures lift visibly.

#### 3.8.4 Correlation-aware surprise (Friston 2010)

Friston's free-energy / predictive-coding framework treats
surprise as the precision-weighted prediction error on a single
information unit. The naive aggregator that sums per-proposition
KL across simultaneously-revealed propositions *double-counts*
when those propositions are causally connected — the audience
perceives the joint reveal as a single update, not two.

We deflate each per-step KL contribution by a factor

$$w_p = \frac{1}{1 + \kappa \cdot n_{\text{kin}}(p)}$$

where $n_{\text{kin}}(p)$ counts how many of $p$'s causal-graph
neighbours *also* moved this step and $\kappa = 0.5$. One co-
moving kin halves the contribution; three quarters it. This is
applied inside the local (per-step) form of
`compute_surprise_score`, where joint-reveal double-counting is
the dominant systematic bias.

#### 3.8.5 Composite narrative tension (Brewer-Lichtenstein triad + Sternberg disequilibrium)

The four scorers above measure orthogonal facets of structural
affect, but readers experience a single felt *tension* — the
quantity Brewer & Lichtenstein 1982 originally framed as a
**triad** (suspense + curiosity + surprise). Sternberg 1978
(*Expositional Modes and Temporal Ordering in Fiction*) frames
the same construct as the *disequilibrium* between what the
reader knows, suspects, and is owed. Vorderer-Wulff-Friedrichsen
1996 (*Suspense: Conceptualizations, Theoretical Analyses, and
Empirical Explorations*) defines tension as the running integral
of moment-to-moment uncertainty.

We aggregate these into

$$T = 0.40 \cdot \text{Suspense} + 0.25 \cdot \text{Mystery} + 0.20 \cdot \text{Irony} + 0.10 \cdot \Delta\text{Surprise} + 0.05 \cdot \text{UnpaidDebt}$$

with the surprise term as its *first derivative* (the local
form) — the disequilibrium kick — and the unpaid-setup-debt term
as $n_{\text{unpaid}} / (n_{\text{unpaid}} + 4)$ counting
foreshadowing arcs whose payoff event has not yet been revealed
at the anchor (Chekhov's-gun overhang). The weight vector is
calibrated against the `example_worlds/` corpus so canonical
mid-arc tension peaks land in the 0.5–0.7 band; on Macbeth at
syuzhet anchor 4 (Duncan's murder), $T \approx 0.49$, rising to
0.51 at the Banquo banquet (anchor 8) and decaying toward 0.28
at the denouement. Implemented as
`DirectiveAssembler.compute_tension_score` and surfaced as the
`narrative_tension` series in the affective-curve plot, gauge
strip, and `compute_affective_score` dispatch table.

* Brewer, W. F. & Lichtenstein, E. H. (1982). "Stories are to entertain: A structural-affect theory of stories". *Journal of Pragmatics* 6(5–6): 473–486.
* Sternberg, M. (1978). *Expositional Modes and Temporal Ordering in Fiction*. Johns Hopkins University Press.
* Vorderer, P., Wulff, H. J. & Friedrichsen, M. (Eds.) (1996). *Suspense: Conceptualizations, Theoretical Analyses, and Empirical Explorations*. Lawrence Erlbaum.
* Tan, E. S. (1996). *Emotion and the Structure of Narrative Film*. Lawrence Erlbaum. — F-emotion concern theory.
* Carroll, N. (1990). *The Philosophy of Horror, or Paradoxes of the Heart*. Routledge. — erotetic-question framework.
* Sayers, D. L. (1929). Introduction to *Great Short Stories of Detection, Mystery, and Horror*. Gollancz. — false-lead aesthetics.
* Knox, R. (1929). "A Detective Story Decalogue". In H. Haycraft (ed.), *The Art of the Mystery Story* (1946). — fair-play hypothesis space.
* Friston, K. (2010). "The free-energy principle: a unified brain theory?" *Nature Reviews Neuroscience* 11: 127–138. — predictive-coding aggregator.

---

## 4. Constrained / neuro-symbolic generation

### 4.1 The brief-as-constraint pattern (`CreativeBrief`)

Our Step-9 brief is a typed structured prompt consumed by Step-10. The
philosophy — "give the LLM hard mathematical constraints, let it invent
only the surface form" — has multiple antecedents:

* Riedl, M. O. & Young, R. M. (2010). "Narrative planning: Balancing plot and character". *J. AI Research* 39: 217–268. — narrative as planning under causal and character constraints.
* Jhala, A. & Young, R. M. (2010). "Cinematic visual discourse: Representation, generation, and evaluation". *IEEE Trans. CIAIG* 2(2): 69–81.
* Martin, L. J. *et al.* (2018). "Event representations for automated story generation with deep neural nets". *AAAI 2018*. — events as the right granularity for neural narrative control.
* Yao, L. *et al.* (2019). "Plan-and-write: Towards better automatic storytelling". *AAAI 2019*. — explicit two-stage planning vs generation, prefiguring our Step-9 / Step-10 split.
* Goldfarb-Tarrant, S., Chakrabarty, T., Weischedel, R., Peng, N. (2020). "Content Planning for Neural Story Generation with Aristotelian Rescoring". *EMNLP 2020*, pp. 4319–4338. [aclanthology.org/2020.emnlp-main.351](https://aclanthology.org/2020.emnlp-main.351/) — plot-plan + ensemble of Aristotle-derived rescorers; rescore-as-audit, prefiguring our Step-11/12 loop.

### 4.2 Event chains and narrative schemas

Our `causal_topology` is a typed, directional version of Chambers &
Jurafsky's narrative event chains.

* Chambers, N. & Jurafsky, D. (2008). "Unsupervised learning of narrative event chains". *ACL 2008*.
* Chambers, N. & Jurafsky, D. (2009). "Unsupervised learning of narrative schemas and their participants". *ACL 2009*.

### 4.3 LLM-as-judge audit loop

Our Step-11 auditor follows the recent line of work on using a stronger /
slower LLM as a critic for the output of a weaker / faster generator.

* Zheng, L. *et al.* (2023). "Judging LLM-as-a-judge with MT-Bench and Chatbot Arena". *NeurIPS 2023 Datasets & Benchmarks*.
* Bai, Y. *et al.* (2022). "Constitutional AI: Harmlessness from AI Feedback". arXiv:2212.08073. — the recursive critique-and-revise structure we adapt for narrative consistency rather than safety.
* Madaan, A. *et al.* (2023). "Self-Refine: Iterative refinement with self-feedback". *NeurIPS 2023*.
* Gu, J. *et al.* (2024). "A Survey on LLM-as-a-Judge". arXiv:2411.15594. — comprehensive survey of bias-mitigation, calibration, and reliability strategies for the judge-loop pattern; the audit step in our pipeline is a domain-specialised instance of these methods, restricted to narrative-structural rather than safety/quality criteria.

### 4.4 Constrained decoding (rejected — see [design-decisions.md](design-decisions.md) D-rejected)

For completeness:

* Lu, X. *et al.* (2021). "NeuroLogic Decoding: (Un)supervised neural text generation with predicate logic constraints". *NAACL 2021*.
* Beurer-Kellner, L. *et al.* (2024). "Guiding LLMs the right way: Fast, non-invasive constrained generation". *ICML 2024*.

We treat token-level constraints as orthogonal — they would compose, but
they don't replace structural narrative audit.

---

## 5. Affect, emotion, and traits

### 5.1 Trait + inertia model

`TraitVector(value, inertia)` separates the *current level* of a trait from
the *force needed to change it*. The empirical motivation:

* Mischel, W. & Shoda, Y. (1995). "A cognitive-affective system theory of personality". *Psychological Review* 102(2): 246–268. — situation-trait interaction, prefiguring our mechanism-trait gating.
* Costa, P. T. & McCrae, R. R. (1992). *Revised NEO Personality Inventory*. PAR. — the dimensional-trait tradition our trait dictionaries lean on.

### 5.2 Affective response to fiction

* Tan, E. S. (1996). *Emotion and the Structure of Narrative Film*. Lawrence Erlbaum.
* Oatley, K. (1995). "A taxonomy of the emotions of literary response and a theory of identification in fictional narrative". *Poetics* 23: 53–74.
* Hogan, P. C. (2003). *The Mind and Its Stories: Narrative Universals and Human Emotion*. Cambridge UP.

These ground the six emotion targets (`grief`, `rage`, `joy`, `regret`,
`love`, `fear`) and their per-trait headroom analysis in
`compute_affective_score`.

---

## 6. Information theory and epistemic state

### 6.1 `Channel`, `intelligibility`, eavesdropping

The information topology is modelled as a `Channel` *node* (standing
capability between participants) plus discrete utterance `EventNode`s
(messages) referencing the channel via `via_channel_id`. Per-participant
`intelligibility ∈ [0,1]` replaces the legacy `is_encrypted` boolean —
encryption is one limiting case (low intelligibility for non-keyholders),
foreign language and partial overhearing are others on the same axis.
Eavesdropping is derived: any non-addressee participant whose
`intelligibility >= physics.intelligibility_threshold` learns the
utterance. This is a discrete, narrative-domain analogue of:

* Shannon, C. E. (1948). "A mathematical theory of communication". *Bell System Technical Journal* 27: 379–423, 623–656. — channel + medium + noise.
* Fagin, R., Halpern, J. Y., Moses, Y., Vardi, M. Y. (1995). *Reasoning About Knowledge*. MIT Press. — multi-agent epistemic logic, the formal basis for our `Belief.confidence` and `Belief.inertia`.

### 6.2 Belief revision

`EntityStateSnapshot.beliefs_added` / `beliefs_invalidated` is a coarse
implementation of belief revision in the AGM tradition. Invalidation
keys carry **two granularities**: a bare `target_id` drops every belief
about the target (coarse), while a composite `"target_id::PROP_..."`
drops only the belief whose `(target_id, proposition_id)` pair matches
(fine-grained), preserving co-located beliefs about the same target
under different propositions.

* Alchourrón, C. E., Gärdenfors, P., Makinson, D. (1985). "On the logic of theory change: Partial meet contraction and revision functions". *J. Symbolic Logic* 50(2): 510–530.

---

## 7. Computational narratology and story understanding

The codebase sits inside a long tradition of attempts to give stories a
formal computational semantics. The references below are the load-bearing
ones for the *graph + LLM* hybrid we adopt.

### Foundational (symbolic) era

* Schank, R. C. & Abelson, R. P. (1977). *Scripts, Plans, Goals and Understanding*. Lawrence Erlbaum. — scripts as causally-typed event templates; precursor to our `EventNode`.
* Lehnert, W. G. (1981). "Plot units and narrative summarization". *Cognitive Science* 5(4): 293–331. — the affect-state graph; precursor to our combination of trait-arc + causal graph.
* Meehan, J. R. (1977). "TALE-SPIN, an interactive program that writes stories". *Proc. IJCAI*. — the original goal-directed story generator.
* Bringsjord, S. & Ferrucci, D. (2000). *Artificial Intelligence and Literary Creativity: Inside the Mind of BRUTUS*. Lawrence Erlbaum.
* Pérez y Pérez, R. & Sharples, M. (2001). "MEXICA: A computer model of a cognitive account of creative writing". *J. Experimental & Theoretical AI* 13(2): 119–139.

### Modern computational narratology

* Mani, I. (2012). *Computational Modeling of Narrative*. Morgan & Claypool. — the field-defining survey; our typed-event, typed-link approach is in its tradition.
* Finlayson, M. A. (2012). *Learning Narrative Structure from Annotated Folktales*. PhD thesis, MIT. — Story Workbench / Propp-style learned grammars.
* Graesser, A. C., Singer, M., Trabasso, T. (1994). "Constructing inferences during narrative text comprehension". *Psychological Review* 101(3): 371–395. — the constructionist theory of inference; motivates why we materialise *causal links* explicitly rather than leave them implicit in prose.
* Elson, D. K. (2012). *Modeling Narrative Discourse*. PhD thesis, Columbia Univ. — the *Story Intention Graph* (SIG); a direct ancestor of our typed multi-edge representation.

### Socratic-QA scaffolding (ingestion Step 2)

The ingestion pipeline runs a Who/What/Where/When/Why/How **Socratic-QA
scaffold** on every chunk before any structured extraction. This is the
classical Socratic method (Plato, *Meno*) repurposed as a chain-of-thought
technique: by forcing the model to articulate hidden motivations, implicit
causal chains, and unobserved background variables in natural language
first, the structured Physics / Social / Consequences agents produce
dramatically tighter graphs.

* Lai, V. *et al.* (2023). "Are Human Explanations Always Helpful? Towards Objective Evaluation of Human Natural Language Explanations". *ACL 2023* — evidence that explicit explanations gate downstream reasoning quality.
* Wei, J. *et al.* (2022). "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models". *NeurIPS 2022* — the modern prompt-engineering form of the same idea.
* Zelikman, E. *et al.* (2022). "STaR: Bootstrapping Reasoning with Reasoning". *NeurIPS 2022* — reinforces that scaffolded rationales raise extraction fidelity.
* Qi, J. *et al.* (2023). "The Art of SOCRATIC QUESTIONING: Recursive Thinking with Large Language Models". *EMNLP 2023* — directly motivates the Who/What/Where/When/Why/How decomposition we adopt.

The scaffold is implemented in `shadow_loom/ingestion.py`
(`_build_socratic_agent`) and prompted by
[`shadow_loom/prompts/socratic_scaffolding.md`](../shadow_loom/prompts/socratic_scaffolding.md).
Its output (`SocraticScaffold.qa_pairs`) is injected as system-prompt
context into all three downstream extraction agents.

### Neural era and benchmarks

* Mostafazadeh, N. *et al.* (2016). "A corpus and cloze evaluation for deeper understanding of commonsense stories". *NAACL 2016*. — the *Story Cloze Test* / ROCStories benchmark for commonsense narrative reasoning.
* Fan, A., Lewis, M., Dauphin, Y. (2018). "Hierarchical neural story generation". *ACL 2018*. — the WritingPrompts benchmark.
* Rashkin, H. *et al.* (2020). "PlotMachines: Outline-conditioned generation with dynamic plot state tracking". *EMNLP 2020*. — outline + state, a near-relative of our brief + world-state pipeline.
* Akoury, N. *et al.* (2020). "STORIUM: A dataset and evaluation platform for machine-in-the-loop story generation". *EMNLP 2020*.
* Yang, K. *et al.* (2022). "Re3: Generating longer stories with recursive reprompting and revision". *EMNLP 2022*. — recursive plan-then-revise; conceptual cousin of Steps 9–12.
* Tian, Y. *et al.* (2024). "Are large language models capable of generating human-level narratives?". *EMNLP 2024* — the gap between fluent prose and tight causal/affective structure that motivates this project's existence.
* Yang, K., Klein, D., Peng, N., Tian, Y. (2023). "DOC: Improving Long Story Coherence With Detailed Outline Control". *ACL 2023*. arXiv:2212.10077. — hierarchical outline + per-passage controller; our brief (Step 9) plus channel-aware draft (Step 10) is a typed, causal-graph-conditioned variant of the same plan-then-render pattern.
* Xu, W., Jojic, N., Rao, S., Brockett, C., Dolan, B. (2025). "Echoes in AI: Quantifying lack of plot diversity in LLM outputs". *PNAS* 122(35): e2504966122. arXiv:2501.00273. — empirical evidence that unconditioned LLM story generation collapses onto a small set of shared plot tropes; reinforces the case for an externally-supplied causal/character graph that breaks the prior.

### Reading-as-simulation (psychology of fiction)

* Mar, R. A. & Oatley, K. (2008). "The function of fiction is the abstraction and simulation of social experience". *Perspectives on Psychological Science* 3(3): 173–192. — fiction as a cognitive simulator; the reason a tight causal/affective model matters more than surface fluency.
* Oatley, K. (2016). "Fiction: Simulation of social worlds". *Trends in Cognitive Sciences* 20(8): 618–628.

---

## 8. Citation map (one-line summary per module)

| Module | Primary debt |
|---|---|
| `models.py` (fabula/syuzhet) | Shklovsky 1917, Genette 1980 |
| `models.py` (TraitVector + inertia) | Mischel & Shoda 1995 |
| `models.py` (GlobalTrait) | Greimas 1966 |
| `models.py` (Belief, confidence, inertia) | Fagin et al. 1995, Zunshine 2006 |
| `models.py` (EventNode fields: when/where/who/cause/intent) | Zwaan & Radvansky 1998 (event-indexing) |
| `instantiator.py` (AMWN, world_id={factual,shadow}) | **Correa & Bareinboim 2025** (foundational); Lewis 1973 (closest-world semantics); Ryan 1991 (possible worlds in narrative) |
| `causal_physics.py` (do-operator, abduction) | Pearl 2009, Halpern 2016, Halpern & Pearl 2005 |
| `causal_physics.py` (mechanism/causal_force) | Halpern 2016 (actual causality) |
| `extract_graph.py` (ego-graph slicing) | Verma & Pearl 1988 (d-separation); Trabasso & van den Broek 1985 (causal-network comprehension) |
| `directive_assembly.py::compute_suspense_score` | Brewer & Lichtenstein 1982; Comisky & Bryant 1982; Ortony, Clore & Collins 1988; Zillmann 1996; Cheong & Young 2015 (the structural-affect / OCC hope–fear lineage we directly implement) |
| `directive_assembly.py::compute_surprise_score` | Itti & Baldi 2009 |
| `directive_assembly.py::compute_mystery_score` | Sternberg 1978 |
| `directive_assembly.py::compute_dramatic_irony_score` | Booth 1974; Muecke 1969; Gerrig 1993; Wilmot & Keller 2020 (their reader-uncertainty framing is structurally closer to irony than to hope–fear suspense, see §3.4) |
| `viz_helpers.py` (affective trajectories) | Reagan et al. 2016 |
| `pipeline.py` (Step 11/12 audit loop) | Madaan et al. 2023, Bai et al. 2022; Gu et al. 2024 (LLM-as-a-Judge survey) |
| `generation.py` (brief → constrained render) | Yao et al. 2019, Goldfarb-Tarrant et al. 2020, Yang et al. 2023 (DOC) |
| `query_models.py` (Pearl rungs) | Pearl & Mackenzie 2018, Bareinboim et al. 2022; Kıcıman et al. 2024 (LLM causal benchmarks) |
| `models.py` (Belief, beliefs_added/invalidated) | Fagin et al. 1995; Kim et al. 2023 (FANToM); Gu et al. 2024 (SimpleToM) |
| `extract_graph.py` (LLM-extracted causal edges) | Kıcıman et al. 2024 |
| Project motivation (graph-first vs prior-collapse) | Tian et al. 2024; Xu et al. 2025 ("Echoes in AI") |

---

## 9. Suggested further reading

For readers wanting the shortest path into the literature this codebase
sits on top of:

1. Pearl & Mackenzie (2018), *The Book of Why* — read first for the causal-inference intuition behind the three rungs.
2. **Correa & Bareinboim (2025), "Counterfactual Graphical Models: Constraints and Inference"** — read for the AMWN + ctf-calculus framework that the codebase's core abstractions (`AMWNInstantiator`, `world_id`, three-rung query taxonomy) are named after.
3. Brewer & Lichtenstein (1982), "Stories are to entertain" — read for the structural-affect framing of suspense as hope–fear anticipation that `compute_suspense_score` directly implements; pair with Cheong & Young (2015) for the planning-system operationalisation. Wilmot & Keller (2020) is the closest neural-LM analogue but, as discussed in §3.4, its uncertainty-reduction quantity is structurally a *reader-vs-character/future-state asymmetry* and so is borrowed by the dramatic-irony scorer rather than by the hope–fear suspense scorer.
4. Sternberg (1992), "Telling in time (II)" — read for the mystery/suspense/surprise distinction we operationalise.
5. Genette (1980) §1 — read for the fabula/syuzhet distinction made rigorous.
6. Ryan (1991), *Possible Worlds, Artificial Intelligence, and Narrative Theory* — read for the bridge from modal logic to narrative that motivates our shadow worlds.
7. Riedl & Young (2010), "Narrative planning" — read for the planning-vs-rendering split that motivates our Step 9 / Step 10 boundary.
8. Zwaan & Radvansky (1998), "Situation models in language comprehension" — read for the cognitive-psychology origin of the multi-dimensional event representation used in `EventNode`.
9. Halpern (2016), *Actual Causality* — read for the formal definition of "actual cause" that grounds our `mechanism` and `causal_force` fields.
10. Mani (2012), *Computational Modeling of Narrative* — read for an overview of the computational-narratology field this project belongs to.

---

## See also

* [architecture.md](architecture.md) — every cited concept mapped to the module and field that implements it.
* [design-decisions.md](design-decisions.md) — *why* we chose Pearl over alternatives, fabula+syuzhet over a single timestamp, AMWN over branching world snapshots.
* [query-and-cycles.md](query-and-cycles.md) — Pearl's three rungs in execution form (Rung 1 → observation, Rung 2 → intervention, Rung 3 → counterfactual).
* [pipeline-walkthrough.md](pipeline-walkthrough.md) — the suspense, KL surprise, and Halpern actual-causality computations as they sit in the runtime.
