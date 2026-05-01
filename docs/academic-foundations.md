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
verified directly. Where a citation could not be verified online (older
monographs, working papers) the entry is marked accordingly.

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

`EventNode.event_type ∈ {choice, outcome, revelation}` is a coarse
generalisation of Propp's 31 narrative functions. We do not model Propp's
fine-grained typology because we want event types to be *causally*
significant rather than narratologically prescriptive.

* Propp, V. (1928/1968). *Morphology of the Folktale*. 2nd rev. English ed., trans. L. Scott, rev. L. A. Wagner. Univ. of Texas Press.
* Rumelhart, D. E. (1975). "Notes on a schema for stories". In D. G. Bobrow & A. M. Collins (eds.), *Representation and Understanding: Studies in Cognitive Science*, pp. 211–236. Academic Press. — the "story grammar" tradition that motivated event-type taxonomies in early AI.
* Mandler, J. M. & Johnson, N. S. (1977). "Remembrance of things parsed: Story structure and recall". *Cognitive Psychology* 9(1): 111–151. — cognitive evidence that readers parse stories into typed event units.
* Thorndyke, P. W. (1977). "Cognitive structures in comprehension and memory of narrative discourse". *Cognitive Psychology* 9(1): 77–110. — the experimental story-grammar paper, published in the same issue as Mandler & Johnson.
* Kintsch, W. & van Dijk, T. A. (1978). "Toward a model of text comprehension and production". *Psychological Review* 85(5): 363–394. — the propositional macro-structure model; canonical complement to story-grammar approaches.
* Trabasso, T. & van den Broek, P. (1985). "Causal thinking and the representation of narrative events". *J. Memory and Language* 24(5): 612–630. — the empirical case that *causal* event chains (not surface form) drive comprehension and recall — the central justification for our graph-first design.
* Bremond, C. (1973). *Logique du récit*. Seuil. — the choice/outcome/revelation triad we adopt is closest to Bremond's *triade narrative*.

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

---

## 2. Causal inference (Pearl's ladder, AMWN, ctf-calculus)

This is the largest single intellectual debt in the codebase. The naming
conventions (`AMWN`, `do_intervene`, `abduction_update`, `world_id∈{factual,
shadow}`, `ctf-calculus`) come directly from this literature.

### 2.1 Three rungs of causation (`ObservationQuery`, `InterventionQuery`, `CounterfactualQuery`)

Our query taxonomy directly mirrors Judea Pearl's "Ladder of Causation":
observation → intervention → counterfactual.

* Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge UP. DOI 10.1017/CBO9780511803161. — canonical reference for SCMs and the three-rung hierarchy.
* Pearl, J. & Mackenzie, D. (2018). *The Book of Why: The New Science of Cause and Effect*. Basic Books. — accessible exposition.
* Pearl, J. (1995). "Causal diagrams for empirical research". *Biometrika* 82(4): 669–688. DOI 10.1093/biomet/82.4.669. — the original *do*-calculus paper.
* Spirtes, P., Glymour, C., Scheines, R. (2000). *Causation, Prediction, and Search* (2nd ed.). MIT Press. — the other foundational text alongside Pearl 2009; PC algorithm and Markov equivalence.
* Bareinboim, E., Correa, J. D., Ibeling, D., Icard, T. (2022). "On Pearl's Hierarchy and the Foundations of Causal Inference". Ch. 27 in *Probabilistic and Causal Inference: The Works of Judea Pearl*, pp. 507–556. ACM Books. DOI 10.1145/3501714.3501743. — the modern formal statement of the hierarchy and the impossibility results that motivate level-3 counterfactual machinery.

`CausalPhysicsEngine.do_intervene()` implements rung-2 graph surgery:
incoming causal edges into the intervened node are severed, downstream
edges re-evaluated.

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
intervention nodes "split" from their factual counterparts) and use the
three-rule structure of ctf-calculus as the conceptual model for our query
pipeline:

| ctf-calculus rule | Shadow-loom analogue |
|---|---|
| **Consistency** (relate observed and intervened values) | `_intervene_state` keeps factual values where surgery doesn't apply. |
| **Independence** (d-separation in the AMWN ⇒ conditional independence) | Ego-graph slicing in `extract_graph.py` (§2.5 below) is a heuristic Markov-blanket cut. |
| **Exclusion** (eliminate interventions that don't reach a target) | `MECHANISM_TRAIT_MAP` gates which mechanisms can affect which trait families; affordance checks gate cross-location influence. |

*We do not (yet) implement the algorithm of Correa & Bareinboim 2025
literally* — our graph is heavily typed for narrative use rather than
rigorous SCM identification — but the mental model and naming are theirs.
See [design-decisions.md](design-decisions.md) D5–D6 for how we adapt it to
a narrative setting.

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

### 2.3 Abduction (`CausalPhysicsEngine.abduction_update`)

Rung-3 counterfactuals require **abduction** — back-propagating present
evidence onto a historical sandbox. Our implementation blends entity
traits 50 % toward observed factual values, weighted by `evidence_strength`,
and gates by mechanism.

The philosophical and computational basis:

* Pearl, J. (2000/2009). *Causality* §7 ("The logic of structure-based counterfactuals"). — the abduction–action–prediction recipe for rung-3 queries. Our pipeline implements the same three steps in `causal_physics.py`.
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

When [`extract_graph.py`](../shadow_loom/extract_graph.py) limits the
ego-graph to "1-hop spatial neighbours" + "events within `memory_limit`", we
are heuristically approximating a Markov blanket: the minimal node set that
d-separates the focal entities from the rest of the graph. With the AMWN of
Correa & Bareinboim 2025, this becomes a *counterfactual* Markov-blanket
argument — we are reading counterfactual conditional independences off the
local subgraph rather than the full causal diagram.

* Verma, T. & Pearl, J. (1988). "Causal networks: semantics and expressiveness". *Proc. UAI 1988*, pp. 69–78. — d-separation, formally.
* Geiger, D., Verma, T., Pearl, J. (1990). "Identifying independence in Bayesian networks". *Networks* 20(5): 507–534. DOI 10.1002/net.3230200504.
* Lauritzen, S. L., Dawid, A. P., Larsen, B. N., Leimer, H.-G. (1990). "Independence properties of directed Markov fields". *Networks* 20(5): 491–505. DOI 10.1002/net.3230200503. — companion to Geiger et al.; global/local Markov equivalence for DAGs.
* Dawid, A. P. (1979). "Conditional independence in statistical theory". *J. Royal Statistical Society, Series B* 41(1): 1–31. — the foundational treatment of conditional independence; predates Pearl's d-separation.

---

## 3. Computational models of suspense, surprise, and curiosity

### 3.1 Suspense as uncertainty reduction — **Wilmot & Keller, ACL 2020**

`DirectiveAssembler.compute_suspense_score()` is directly inspired by
Wilmot & Keller's framing of suspense as a *forward-looking* uncertainty
measure rather than a *backward-looking* surprise measure. Our
implementation is:

$$\text{suspense}(t) = P(\text{threat}) - P(\text{hope})$$

where threat / hope are derived from unrevealed forward causal paths whose
target / source is the focal entity, and `evidence_strength` is the
probability proxy. We return 0 when hope is extinguished — the
**suspense → despair** boundary that Wilmot's neural model also exhibits in
its annotated short-story corpus.

* **Wilmot, D. & Keller, F. (2020).** "Modelling Suspense in Short Stories as Uncertainty Reduction over Neural Representation". *Proc. ACL 2020*, pp. 1763–1788. [aclanthology.org/2020.acl-main.161](https://aclanthology.org/2020.acl-main.161/) — the central reference.
* Wilmot, D. & Keller, F. (2021a). "A Temporal Variational Model for Story Generation". arXiv:2109.06807 (preprint only).
* Wilmot, D. & Keller, F. (2021b). "Memory and Knowledge Augmented Language Models for Inferring Salience in Long-Form Stories". *Proc. EMNLP 2021*, pp. 851–865. [aclanthology.org/2021.emnlp-main.65](https://aclanthology.org/2021.emnlp-main.65/) — extends uncertainty-reduction to long novels via memory-augmented LMs.
* Wilmot, D. (2022). *Great Expectations: Unsupervised Inference of Suspense, Surprise and Salience in Storytelling*. PhD thesis, Univ. of Edinburgh. arXiv:2206.09708. — full discussion of suspense / surprise / salience as computable quantities; deep influence on our four-effect taxonomy.

### 3.2 The Sternberg triad (mystery / suspense / surprise)

Our four named structural effects (mystery, dramatic irony, suspense,
surprise) extend Meir Sternberg's classical *curiosity / suspense /
surprise* triad with dramatic irony as a fourth axis.

* Sternberg, M. (1978). *Expositional Modes and Temporal Ordering in Fiction*. Johns Hopkins UP.
* Sternberg, M. (1992). "Telling in time (II): Chronology, teleology, narrativity". *Poetics Today* 13(3): 463–541. — formal definitions of curiosity, suspense, surprise as cognitive states with distinct triggers.
* Brewer, W. F. & Lichtenstein, E. H. (1982). "Stories are to entertain: A structural-affect theory of stories". *J. of Pragmatics* 6(5–6): 473–486. — empirical grounding of the triad.
* Cheong, Y.-G. & Young, R. M. (2015). "Suspenser: A story generation system for suspense". *IEEE Transactions on Computational Intelligence and AI in Games* 7(1): 39–52. — operationalises Brewer's structural-affect theory in a generation system.
* Bae, B.-C. & Young, R. M. (2008). "A use of flashback and foreshadowing for surprise arousal in narrative using a plan-based approach". *ICIDS 2008*, LNCS 5334, pp. 156–167. — formal planning model of narrative-level surprise via anachrony.

### 3.3 Surprise as KL divergence

`compute_surprise_score()` uses per-trait binary KL divergence
$D_\text{KL}(p \| q) = p\log\frac{p}{q} + (1-p)\log\frac{1-p}{1-q}$ with a
maximum-entropy prior $q = 0.5$ updated toward truth by revealed causal
edges.

* Itti, L. & Baldi, P. (2009). "Bayesian surprise attracts human attention". *Vision Research* 49(10): 1295–1306. DOI 10.1016/j.visres.2008.09.007. — the formal basis: surprise = KL between prior and posterior beliefs.
* Schmidhuber, J. (2010). "Formal theory of creativity, fun, and intrinsic motivation (1990–2010)". *IEEE Trans. Autonomous Mental Development* 2(3): 230–247. DOI 10.1109/TAMD.2010.2056368. — surprise as compression progress.
* Reagan, A. J., Mitchell, L., Kiley, D., Danforth, C. M., Dodds, P. S. (2016). "The emotional arcs of stories are dominated by six basic shapes". *EPJ Data Science* 5: art. 31. DOI 10.1140/epjds/s13688-016-0093-1. — corpus-scale emotional trajectories that motivate our trait-trajectory analytics in `viz_helpers.py`.
* Elsner, M. (2012). "Character-based kernels for novelistic plot structure". *EACL 2012*, pp. 634–644. — structural arc analysis predating Reagan et al., using character co-occurrence.
* Kim, E., Padó, S., Klinger, R. (2017). "Investigating the relationship between literary genres and emotional plot development". *Workshop on Computational Linguistics for Literature (NAACL)*, pp. 17–26. — direct empirical follow-up to Reagan et al. on genre-conditioned arcs.

### 3.4 Dramatic irony as epistemic asymmetry

`compute_dramatic_irony_score()` walks revealed causal edges where the
source event is **not** in the focal entity's belief set at
`temporal_anchor`. The framing of irony as a reader/character knowledge gap
is classical:

* Booth, W. (1974). *A Rhetoric of Irony*. Univ. of Chicago Press.
* Muecke, D. C. (1969). *The Compass of Irony*. Methuen.

For computational treatments:

* Gerrig, R. J. (1993). *Experiencing Narrative Worlds*. Yale UP. — the cognitive-pragmatic framework we borrow.

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
`intelligibility \u2208 [0,1]` replaces the legacy `is_encrypted` boolean —
encryption is one limiting case (low intelligibility for non-keyholders),
foreign language and partial overhearing are others on the same axis.
Eavesdropping is derived: any non-addressee participant whose
`intelligibility >= physics.intelligibility_threshold` learns the
utterance. This is a discrete, narrative-domain analogue of:

* Shannon, C. E. (1948). "A mathematical theory of communication". *Bell System Technical Journal* 27: 379–423, 623–656. — channel + medium + noise.
* Fagin, R., Halpern, J. Y., Moses, Y., Vardi, M. Y. (1995). *Reasoning About Knowledge*. MIT Press. — multi-agent epistemic logic, the formal basis for our `Belief.confidence` and `Belief.inertia`.

### 6.2 Belief revision

`EntityStateSnapshot.beliefs_added` / `beliefs_invalidated` is a coarse
implementation of belief revision in the AGM tradition.

* Alchourrón, C. E., Gärdenfors, P., Makinson, D. (1985). "On the logic of theory change: Partial meet contraction and revision functions". *J. Symbolic Logic* 50(2): 510–530.

---

## 6.5 Computational narratology and story understanding

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

### Reading-as-simulation (psychology of fiction)

* Mar, R. A. & Oatley, K. (2008). "The function of fiction is the abstraction and simulation of social experience". *Perspectives on Psychological Science* 3(3): 173–192. — fiction as a cognitive simulator; the reason a tight causal/affective model matters more than surface fluency.
* Oatley, K. (2016). "Fiction: Simulation of social worlds". *Trends in Cognitive Sciences* 20(8): 618–628.

---

## 7. Citation map (one-line summary per module)

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
| `directive_assembly.py::compute_suspense_score` | **Wilmot & Keller 2020** |
| `directive_assembly.py::compute_surprise_score` | Itti & Baldi 2009 |
| `directive_assembly.py::compute_mystery_score` | Sternberg 1978 |
| `directive_assembly.py::compute_dramatic_irony_score` | Gerrig 1993, Booth 1974 |
| `viz_helpers.py` (affective trajectories) | Reagan et al. 2016 |
| `pipeline.py` (Step 11/12 audit loop) | Madaan et al. 2023, Bai et al. 2022 |
| `generation.py` (brief → constrained render) | Yao et al. 2019, Goldfarb-Tarrant et al. 2020 |
| `query_models.py` (Pearl rungs) | Pearl & Mackenzie 2018, Bareinboim et al. 2022 |

---

## 8. Suggested further reading

For readers wanting the shortest path into the literature this codebase
sits on top of:

1. Pearl & Mackenzie (2018), *The Book of Why* — read first for the causal-inference intuition behind the three rungs.
2. **Correa & Bareinboim (2025), "Counterfactual Graphical Models: Constraints and Inference"** — read for the AMWN + ctf-calculus framework that the codebase's core abstractions (`AMWNInstantiator`, `world_id`, three-rung query taxonomy) are named after.
3. Wilmot & Keller (2020) ACL paper — read the introduction and §3 for the suspense framing we directly inherit in `compute_suspense_score`.
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
