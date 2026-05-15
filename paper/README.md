<!--
SPDX-FileCopyrightText: 2026 David Rae Wilmot
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Shadow-Loom — overview paper

This directory contains a short main-body project overview paper plus
extensive appendices, in ACL style.

## Files

- [shadow_loom.tex](shadow_loom.tex) — main LaTeX source.
- [references.bib](references.bib) — BibTeX in ACL `acl_natbib` style.
- [acl.sty](acl.sty), [acl_natbib.bst](acl_natbib.bst) — bundled
  copies of the official ACL style files
  (<https://github.com/acl-org/acl-style-files>).

## Building locally

The paper uses the bundled ACL style files; no extra installation is
required as long as the `paper/` directory is the current working
directory:

```bash
cd paper
pdflatex shadow_loom
bibtex   shadow_loom
pdflatex shadow_loom
pdflatex shadow_loom
```

## Building an arXiv submission

A reproducible source tarball is built by:

```bash
make arxiv          # produces paper/shadow_loom_arxiv.tar.gz
```

The resulting tarball contains `shadow_loom.tex`, `references.bib`,
`acl.sty`, `acl_natbib.bst`, and the pre-built `shadow_loom.bbl`
(arXiv does not always re-run BibTeX). Logs, intermediate `.aux` files,
and the locally compiled PDF are excluded.

## Scope

Per the brief, the paper:

- Focuses on the main features (typed world graph, causal physics,
  narrative physics, constrained renderer + auditor).
- Frames the work as **experimental** — combining graphical causal
  reasoning with a physics-like graph model of narrative — rather
  than as a benchmarked NLP system.
- Argues relevance to future NLP / reasoning work and to
  computational social science.
- Keeps the main body short (around five pages).
- Defers all definitions and equations (schema, ingestion, AMWN
  sandbox, causal physics, narrative physics, generation, audit) to
  Appendix A.
- Provides an extended end-to-end narrative walkthrough of the whole
  pipeline on the *Macbeth* fixture in Appendix B
  (`app:walkthrough`), with every intermediate object shown.
- Provides per-stage worked examples drawn from the bundled plot
  fixtures in Appendix C (`app:examples`).
- Provides a dedicated appendix on the authoring user interface ---
  every tab, the version sidebar, the chat / command bar, the
  Answer panel, the cross-tab event bus, and example sessions on
  real fixtures --- in Appendix D (`app:ui`).

## Cross-references to the Markdown docs

The paper appendices are intentionally aligned with the Markdown
documentation in `../docs/`:

| Paper appendix | Companion markdown doc |
|---|---|
| App. A (`app:defs`) | [`docs/architecture.md`](../docs/architecture.md), [`docs/academic-foundations.md`](../docs/academic-foundations.md) |
| App. B (`app:walkthrough`) | [`docs/pipeline-walkthrough.md`](../docs/pipeline-walkthrough.md), [`docs/pipeline-by-example.md`](../docs/pipeline-by-example.md) |
| App. C (`app:examples`) | [`docs/model-examples.md`](../docs/model-examples.md), [`docs/use-cases.md`](../docs/use-cases.md) |
| App. D (`app:ui`) | [`docs/ui-guide.md`](../docs/ui-guide.md) |
