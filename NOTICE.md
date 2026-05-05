# Shadow Loom — Third-party notices and attributions

<!--
SPDX-FileCopyrightText: 2026 David Rae Wilmot
SPDX-License-Identifier: AGPL-3.0-or-later
-->

This NOTICE accompanies the AGPLv3-licensed distribution of Shadow
Loom and supplements (but does not modify) [`LICENSE`](LICENSE) and
[`COMMERCIAL-LICENSE.md`](COMMERCIAL-LICENSE.md).

The Shadow Loom **code** (the Python packages under `shadow_loom/`,
`shadow_loom_mcp/`, and `shadow_loom_ui/`, plus everything in
`scripts/`, `tests/`, and the build / CI tooling) is © 2026
David Rae Wilmot and contributors, licensed under
AGPL-3.0-or-later. The structural scaffolding of the example
fixtures (the schema, edges, traits, and pipeline-orchestration
code) is likewise covered by AGPLv3.

The sections below identify material in this repository whose
**underlying creative content** is owned by third parties. Shadow
Loom claims no copyright in those underlying works. Their inclusion
here is for non-commercial academic research, study, criticism and
review under fair-use / fair-dealing doctrines (US 17 U.S.C. § 107;
UK CDPA 1988 §§ 29–30; equivalent provisions in other
jurisdictions).

If you are a rights-holder and believe any item below exceeds
fair-use / fair-dealing, please email `david.wilmot@gmail.com`
with the subject `Shadow Loom — takedown request` and the affected
material will be removed or amended promptly.

---

## 1. Plot summaries and example worlds

The files under [`example_worlds/`](example_worlds/) and
[`sample_plots/`](sample_plots/) contain plot scaffolds, character
graphs, and short summaries of pre-existing literary, dramatic, and
cinematic works. These are research fixtures used to validate the
extraction, propagation, and counterfactual-reasoning pipeline
against well-understood narrative ground truth.

The structural representation (entities, events, edges, trait
vectors, world-trait fields) is original code authored for Shadow
Loom and licensed under AGPLv3. The **underlying narrative**
remains the property of the respective rights-holders.

### 1.1 Public-domain works (no third-party rights claimed)

The following are out of copyright in most jurisdictions and are
included without restriction beyond the project's AGPLv3 licence
on the surrounding code:

| File | Underlying work | Author | Year |
|---|---|---|---|
| `frankenstein.py` / `frankenstein.txt` | *Frankenstein; or, The Modern Prometheus* | Mary Shelley | 1818 |
| `wuthering_heights.py` / `wuthering_heights.txt` | *Wuthering Heights* | Emily Brontë | 1847 |
| `persuasion.py` / `persuasion.txt` | *Persuasion* | Jane Austen | 1817 |
| `great_expectations.py` / `great_expectations.txt` | *Great Expectations* | Charles Dickens | 1861 |
| `romeo_and_juliet.py` / `romeo_and_juliet.txt` | *Romeo and Juliet* | William Shakespeare | c. 1597 |
| `macbeth.py` / `macbeth.txt` | *Macbeth* | William Shakespeare | c. 1606 |
| `the_great_gatsby.py` / `great_gatsby.txt` | *The Great Gatsby* | F. Scott Fitzgerald | 1925 (US public domain since 2021) |
| `death_on_the_nile.py` / `death_on_the_nile.txt` | *Death on the Nile* | Agatha Christie | 1937 (PD in life+70 jurisdictions from 2047; included as an academic abstract only) |

### 1.2 Works still under copyright in major jurisdictions

The structural fixtures below summarise works that **remain in
copyright** in the US, UK, EU, and most other Berne signatories.
The summaries are short, transformative abstractions used solely to
exercise the pipeline; no protected expression (dialogue, prose,
shot sequences) is reproduced. Rights in the underlying works
belong to the listed rights-holders or their successors.

| File | Underlying work | Rights-holder (best information) |
|---|---|---|
| `the_lion_the_witch_and_the_wardrobe.py` | *The Lion, the Witch and the Wardrobe* (1950), C. S. Lewis | The C. S. Lewis Company Ltd / HarperCollins |
| `nineteen_eighty_four.py` | *Nineteen Eighty-Four* (1949), George Orwell | The Estate of the late Sonia Brownell Orwell / Penguin Random House (PD in some jurisdictions from 2021; in copyright in the US until 2044) |
| `dads_army.py` | *Dad's Army* (1968–1977), Jimmy Perry & David Croft | The Estates of Jimmy Perry and David Croft / BBC |
| `tinker_tailor_soldier_spy.py` | *Tinker Tailor Soldier Spy* (1974), John le Carré | The Estate of David Cornwell / Penguin Random House |
| `apocalypse_now.py` | *Apocalypse Now* (1979), Francis Ford Coppola & John Milius | American Zoetrope / Lionsgate (loosely adapted from Joseph Conrad's *Heart of Darkness*, 1899, which is itself in the public domain) |
| `once_upon_a_time_in_the_west.py` | *Once Upon a Time in the West* (1968), Sergio Leone | Paramount Pictures / Rafran Cinematografica |
| `brief_encounter.py` | *Brief Encounter* (1945), David Lean (after Noël Coward, *Still Life*, 1936) | ITV Studios / Cineguild (PD status varies) |
| `a_fish_called_wanda.py` | *A Fish Called Wanda* (1988), Charles Crichton & John Cleese | MGM |
| `reservoir_dogs.py` | *Reservoir Dogs* (1992), Quentin Tarantino | Live Entertainment / Lionsgate |
| `the_devil_wears_prada.py` | *The Devil Wears Prada* (novel 2003, film 2006) | Lauren Weisberger / Doubleday; 20th Century Studios (film) |
| `gone_girl.py` | *Gone Girl* (2012), Gillian Flynn | Crown Publishing / 20th Century Studios |
| `a_court_of_thorn_and_roses.py` | *A Court of Thorns and Roses* (2015), Sarah J. Maas | Bloomsbury Publishing |

If you are running Shadow Loom in a context where summarising
in-copyright works is not covered by your jurisdiction's
fair-use / fair-dealing exceptions (for example, certain
commercial-redistribution scenarios), you should remove the files
in § 1.2 from your distribution and rely on § 1.1 plus your own
public-domain or licensed inputs.

---

## 2. Software dependencies

Runtime and development dependencies are declared in
[`pyproject.toml`](pyproject.toml) and pinned in [`uv.lock`](uv.lock).
Each dependency is distributed under its own licence; consult the
individual project for terms. Notable copyleft / weak-copyleft
dependencies are flagged in [`docs/architecture.md`](docs/architecture.md)
where they affect the project boundary.

The list of license identifiers appearing across the source tree is
maintained in [`REUSE.toml`](REUSE.toml); run `reuse lint` to verify
SPDX coverage.

---

## 3. Trademarks

"Shadow Loom" and the Shadow Loom marks are trademarks of David Rae
Wilmot. See [`TRADEMARK.md`](TRADEMARK.md) for the trademark
policy. Names of third-party works listed in § 1 are trademarks
and/or registered marks of their respective rights-holders and are
used here only to identify the source material of each fixture for
academic-research purposes.

---

## 4. Outputs

Outputs of running Shadow Loom on user-supplied inputs (graphs,
summaries, narrative drafts, audit reports) are not encumbered by
this NOTICE or by AGPLv3. You own them, subject to (a) any rights
held by upstream LLM providers (OpenAI, Anthropic, OpenRouter,
Together, etc. — see each provider's terms), (b) any rights held in
your input materials by their authors, and (c) the underlying-work
rights described in § 1 if you generate output derived from those
fixtures.

See [`COMMERCIAL-LICENSE.md` § 7](COMMERCIAL-LICENSE.md#7-scope-of-the-licensed-work-derivative-works--integrations)
for the maintainer's interpretive position on outputs.
