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
- Keeps the main body to 4 pages.
- Defers all definitions and equations (schema, ingestion, AMWN
  sandbox, causal physics, narrative physics, generation, audit) to
  the appendix.
