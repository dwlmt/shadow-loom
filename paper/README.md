<!--
SPDX-FileCopyrightText: 2026 David Rae Wilmot
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Shadow-Loom — overview paper

This directory contains a short (4-page main body + appendix) project
overview paper in ACL style.

## Files

- [shadow_loom.tex](shadow_loom.tex) — main LaTeX source.
- [references.bib](references.bib) — BibTeX in ACL `acl_natbib` style.

## Building

The paper uses the official ACL style files. Download them from
<https://github.com/acl-org/acl-style-files> and place `acl.sty` and
`acl_natbib.bst` next to `shadow_loom.tex` (or install them onto your
TeX path), then:

```bash
pdflatex shadow_loom
bibtex   shadow_loom
pdflatex shadow_loom
pdflatex shadow_loom
```

## Scope

Per the brief, the paper:

- Focuses on the main features (typed world graph, causal physics,
  narrative physics, constrained renderer + auditor).
- Frames the work as **experimental** — combining graphical causal
  reasoning with a physics-like graph model of narrative — rather
  than as a benchmarked NLP system.
- Argues relevance to future NLP / reasoning work and to
  computational social science.
- Keeps the main body short (around 5 pages).
- Defers all definitions and equations (schema, ingestion, AMWN
  sandbox, causal physics, narrative physics, generation, audit) to
  Appendix A.
- Provides an extended end-to-end narrative walkthrough of the whole
  pipeline on the *Macbeth* fixture in Appendix B
  (`app:walkthrough`), every intermediate object shown.
- Provides per-stage worked examples drawn from the bundled plot
  fixtures in Appendix C (`app:examples`).
- Provides a dedicated appendix on the authoring user interface ---
  every tab, the version sidebar, the cross-tab event bus, and
  example sessions on real fixtures --- in Appendix D (`app:ui`).

## Cross-references to the Markdown docs

The paper appendices are intentionally aligned with the Markdown
documentation in `../docs/`:

| Paper appendix | Companion markdown doc |
|---|---|
| App. A (`app:defs`) | [`docs/architecture.md`](../docs/architecture.md), [`docs/academic-foundations.md`](../docs/academic-foundations.md) |
| App. B (`app:walkthrough`) | [`docs/pipeline-walkthrough.md`](../docs/pipeline-walkthrough.md), [`docs/pipeline-by-example.md`](../docs/pipeline-by-example.md) |
| App. C (`app:examples`) | [`docs/model-examples.md`](../docs/model-examples.md), [`docs/use-cases.md`](../docs/use-cases.md) |
| App. D (`app:ui`) | [`docs/ui-guide.md`](../docs/ui-guide.md) |
