# SPDX-License-Identifier: AGPL-3.0-or-later
"""Generate the paper's empirical figures from live engine output.

Writes PDF figures into paper/figures/ for inclusion in shadow_loom.tex:

  * affective_arcs.pdf  — mystery / dramatic-irony / suspense / surprise
    trajectories across syuzhet for four bundled fixtures, showing the
    rise-peak-fall structure the affective calculus predicts.
  * macbeth_traits.pdf  — ENT_MACBETH trait state (guilt / ambition /
    paranoia) reconstructed over fabula time, illustrating Bayesian
    noisy-OR propagation and inertia-gated state change.

All numbers are real engine output; nothing is hand-authored.
Run: python scripts/_make_paper_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from example_worlds import (  # noqa: E402
    macbeth, death_on_the_nile, romeo_and_juliet, reservoir_dogs,
)
from shadow_loom.causal_physics import reconstruct_entity_at  # noqa: E402
from shadow_loom_ui.viz_helpers import affective_timeseries_syuzhet  # noqa: E402

OUT = REPO / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Colour-blind-safe palette (Wong 2011).
C = {
    "mystery": "#0072B2",
    "dramatic_irony": "#D55E00",
    "suspense": "#009E73",
    "surprise": "#CC79A7",
    "guilt": "#D55E00",
    "ambition": "#0072B2",
    "paranoia": "#009E73",
}


def _focal(ws, n=6):
    counts: dict[str, int] = {}
    for ev in ws.events:
        for a in ev.actor_ids or []:
            counts[a] = counts.get(a, 0) + 1
        for t in ev.target_ids or []:
            counts[t] = counts.get(t, 0) + 1
    return [eid for eid, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:n]]


def affective_arcs():
    fixtures = [
        (macbeth.world_state, "Macbeth"),
        (death_on_the_nile.world_state, "Death on the Nile"),
        (romeo_and_juliet.world_state, "Romeo and Juliet"),
        (reservoir_dogs.world_state, "Reservoir Dogs"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6), sharey=True)
    for ax, (ws, name) in zip(axes.flat, fixtures):
        times, series = affective_timeseries_syuzhet(
            ws, samples=11, entity_ids=_focal(ws)
        )
        # normalise syuzhet anchors to [0, 1] story progress
        lo, hi = times[0], times[-1]
        x = [(t - lo) / (hi - lo) if hi > lo else 0.0 for t in times]
        for key, style in [
            ("mystery", "-"),
            ("dramatic_irony", "-"),
            ("suspense", "-"),
        ]:
            y = series.get(key, [])
            if y:
                ax.plot(x, y, style, color=C[key], lw=1.6,
                        label=key.replace("_", " "))
        sur = series.get("surprise", [])
        if sur:
            ax.plot(x, sur, ":", color=C["surprise"], lw=1.2,
                    marker="o", ms=2.5, label="surprise")
        ax.set_title(name, fontsize=9)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(0, 1)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, lw=0.4)
    for ax in axes[-1]:
        ax.set_xlabel("syuzhet progress", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("score", fontsize=8)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = OUT / "affective_arcs.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def macbeth_traits():
    m = macbeth.world_state.entities["ENT_MACBETH"]
    fts = [0, 2000, 5000, 6000, 10000, 11000, 13000, 17000, 19000]

    def val(ft, name):
        tr = reconstruct_entity_at(m, ft)["traits"].get(name)
        return tr["value"] if tr else None

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    xs = [ft / 1000 for ft in fts]
    for name in ["guilt", "ambition", "paranoia"]:
        ys = [val(ft, name) for ft in fts]
        ax.step(xs, ys, where="post", color=C[name], lw=1.8, marker="o",
                ms=3, label=name)
    events = {6: "Duncan\nmurder", 11: "Banquo\nghost", 13: "2nd\nprophecy"}
    for xt, lbl in events.items():
        ax.axvline(xt, color="#999999", ls="--", lw=0.7)
        ax.text(xt, 1.02, lbl, fontsize=6, ha="center", va="bottom",
                color="#555555")
    ax.set_xlabel("fabula time (×1000)", fontsize=9)
    ax.set_ylabel("trait value", fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25, lw=0.4)
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    fig.tight_layout()
    path = OUT / "macbeth_traits.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    affective_arcs()
    macbeth_traits()
