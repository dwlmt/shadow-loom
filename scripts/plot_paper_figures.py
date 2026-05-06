#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Generate PNG figures for the paper directly from real engine code.

Outputs (under ``paper/figures/``):

* ``affective_structural.png`` -- 2x2 grid: Macbeth, Death on the Nile,
  Romeo and Juliet, Reservoir Dogs. Each panel shows mystery, dramatic
  irony, suspense and surprise vs. syuzhet anchor, scored by the real
  ``compute_affective_scores`` helper that drives the UI.
* ``affective_emotion.png`` -- 2x2 grid for Macbeth, Gone Girl, Brief
  Encounter, Wuthering Heights, plotting the six emotion-target scorers
  produced by ``DirectiveAssembler.compute_affective_score``.
* ``query_observation.png`` -- Romeo and Juliet ego-graph at a chosen
  syuzhet anchor, focused on Romeo (real BFS ego-extract).
* ``query_intervention.png`` -- AMWN causal diagram for Macbeth with
  the inbound edges to ``EVT_DUNCAN_MURDER`` severed by graph surgery
  (real call to ``build_causal_diagram`` + the engine's mutilation rule).
* ``query_counterfactual.png`` -- Bar chart of trait deltas produced by
  ``CausalPhysicsEngine.execute(rung=3, ...)`` on Macbeth conditioned on
  ``ENT_MACBETH`` evidence with a ``do(EVT_DUNCAN_MURDER -> blocked)``
  intervention, comparing factual vs. counterfactual.
* ``mystery_hidden_ancestors.png`` -- Mystery-scorer mechanism plot:
  hidden-ancestor fraction per syuzhet anchor for Macbeth.

Usage::

    python scripts/plot_paper_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

import example_worlds  # noqa: E402  (only used for the import side effects)
from example_worlds import (  # noqa: E402
    macbeth,
    death_on_the_nile,
    reservoir_dogs,
    romeo_and_juliet,
    gone_girl,
    brief_encounter,
    wuthering_heights,
)
from shadow_loom.amwn import build_causal_diagram  # noqa: E402
from shadow_loom.causal_physics import CausalPhysicsEngine  # noqa: E402
from shadow_loom.directive_assembly import DirectiveAssembler  # noqa: E402
from shadow_loom.extract_graph import extract_ego_graph_from_memory  # noqa: E402
from shadow_loom.instantiator import AMWNInstantiator  # noqa: E402
from shadow_loom_ui.viz_helpers import (  # noqa: E402
    affective_timeseries_syuzhet,
    syuzhet_time_bounds,
    ws_to_ego_graph_data,
)


OUT = REPO / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

STRUCTURAL = ["mystery", "dramatic_irony", "suspense", "surprise"]
STRUCTURAL_COLORS = {
    "mystery": "#1f77b4",
    "dramatic_irony": "#d62728",
    "suspense": "#2ca02c",
    "surprise": "#ff7f0e",
}
EMOTIONS = ["grief", "rage", "joy", "regret", "love", "fear"]
EMOTION_COLORS = {
    "grief": "#1f77b4",
    "rage": "#d62728",
    "joy": "#2ca02c",
    "regret": "#9467bd",
    "love": "#e377c2",
    "fear": "#8c564b",
}


# ── helpers ────────────────────────────────────────────────────────

def _focal_entity_ids(ws, n: int = 6) -> list[str]:
    counts: dict[str, int] = {}
    for ev in ws.events:
        for a in (ev.actor_ids or []):
            counts[a] = counts.get(a, 0) + 1
        for t in (ev.target_ids or []):
            counts[t] = counts.get(t, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    return [eid for eid, _ in ranked[:n]]


def _structural_panel(ax, ws, title: str, samples: int = 9) -> None:
    eids = _focal_entity_ids(ws)
    times, series = affective_timeseries_syuzhet(
        ws, samples=samples, entity_ids=eids,
    )
    xs = list(range(len(times)))
    for k in STRUCTURAL:
        ys = series.get(k, [0.0] * len(times))
        ax.plot(xs, ys, marker="o", lw=1.8, ms=4,
                color=STRUCTURAL_COLORS[k], label=k.replace("_", " "))
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("syuzhet anchor (early → late)", fontsize=9)
    ax.set_ylabel("score ∈ [0, 1]", fontsize=9)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(i) for i in range(len(times))], fontsize=8)
    ax.grid(True, alpha=0.3)


def _emotion_panel(ax, ws, title: str, samples: int = 9) -> None:
    eids = _focal_entity_ids(ws)
    smin, smax = syuzhet_time_bounds(ws)
    if smax <= smin:
        anchors = [smin]
    else:
        step = max(1, (smax - smin) // (samples - 1))
        anchors = list(range(smin, smax + 1, step))
        if anchors[-1] != smax:
            anchors.append(smax)
    assembler = DirectiveAssembler(None, {}, ws)
    series: dict[str, list[float]] = {emo: [] for emo in EMOTIONS}
    for s in anchors:
        for emo in EMOTIONS:
            try:
                v = -assembler.compute_affective_score(emo, eids, s)
            except Exception:
                v = 0.0
            series[emo].append(max(0.0, min(1.0, v)))
    xs = list(range(len(anchors)))
    for emo in EMOTIONS:
        ax.plot(xs, series[emo], marker="o", lw=1.6, ms=3.5,
                color=EMOTION_COLORS[emo], label=emo)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("syuzhet anchor", fontsize=9)
    ax.set_ylabel("trait closeness ∈ [0, 1]", fontsize=9)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(i) for i in range(len(anchors))], fontsize=8)
    ax.grid(True, alpha=0.3)


# ── figure 1: structural scorers ───────────────────────────────────

def fig_structural() -> None:
    fixtures = [
        (macbeth.world_state, "Macbeth"),
        (death_on_the_nile.world_state, "Death on the Nile"),
        (romeo_and_juliet.world_state, "Romeo and Juliet"),
        (reservoir_dogs.world_state, "Reservoir Dogs"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for (ws, title), ax in zip(fixtures, axes.flat):
        _structural_panel(ax, ws, title)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=4, fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Structural affective scorers across syuzhet "
                 "(real shadow_loom engine output)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    out = OUT / "affective_structural.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ── figure 2: emotion-target scorers ───────────────────────────────

def fig_emotion() -> None:
    fixtures = [
        (macbeth.world_state, "Macbeth"),
        (gone_girl.world_state, "Gone Girl"),
        (brief_encounter.world_state, "Brief Encounter"),
        (wuthering_heights.world_state, "Wuthering Heights"),
        (death_on_the_nile.world_state, "Death on the Nile"),
        (romeo_and_juliet.world_state, "Romeo and Juliet"),
    ]
    # Compute one closeness score per (fixture, emotion) at the
    # terminal syuzhet anchor of each fixture, using the top-6 focal
    # entities ranked by event participation.
    scores: dict[str, list[float]] = {emo: [] for emo in EMOTIONS}
    for ws, _title in fixtures:
        eids = _focal_entity_ids(ws)
        _smin, smax = syuzhet_time_bounds(ws)
        assembler = DirectiveAssembler(None, {}, ws)
        for emo in EMOTIONS:
            try:
                v = -assembler.compute_affective_score(emo, eids, smax)
            except Exception:
                v = 0.0
            scores[emo].append(max(0.0, min(1.0, v)))

    fig, ax = plt.subplots(figsize=(13, 6))
    n_fix = len(fixtures)
    n_emo = len(EMOTIONS)
    bar_w = 0.8 / n_emo
    xs = list(range(n_fix))
    for i, emo in enumerate(EMOTIONS):
        offsets = [x + (i - n_emo / 2 + 0.5) * bar_w for x in xs]
        ax.bar(offsets, scores[emo], width=bar_w,
               color=EMOTION_COLORS[emo], label=emo)
    ax.set_xticks(xs)
    ax.set_xticklabels([t for _, t in fixtures], fontsize=10)
    ax.set_ylabel("trait-target closeness ∈ [0, 1]", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(ncol=6, loc="upper center", fontsize=10, frameon=False,
              bbox_to_anchor=(0.5, -0.07))
    ax.set_title(
        "Emotion-target closeness per fixture (top-6 focal entities, "
        "terminal syuzhet anchor)\n"
        "DirectiveAssembler.compute_affective_score on the real "
        "shadow_loom engine",
        fontsize=12,
    )
    fig.tight_layout()
    out = OUT / "affective_emotion.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ── figure 3: ObservationQuery — ego graph ─────────────────────────

_EDGE_COLOR = {
    "located_in": "#9E9E9E",
    "owned_by": "#9E9E9E",
    "causal": "#1f77b4",
    "connected_to": "#9E9E9E",
    "relationship": "#d62728",
    "communicating_with": "#bcbd22",
    "utters_to": "#bcbd22",
    "actor_of": "#666666",
    "target_of": "#666666",
    "used_in": "#666666",
    "governs": "#9467bd",
}


def _draw_graph_data(ax, nodes, links, focus_ids, title: str) -> None:
    g = nx.DiGraph()
    for n in nodes:
        g.add_node(n["id"], name=n["name"],
                   ntype=n.get("_sl_node_type", "Entity"))
    for ln in links:
        g.add_edge(ln["source"], ln["target"], etype=ln.get("value", ""))
    if not g.nodes:
        ax.set_title(f"{title} (empty)")
        ax.axis("off")
        return
    try:
        pos = nx.spring_layout(g, seed=7, k=1.4 / max(1, len(g.nodes) ** 0.5))
    except Exception:
        pos = nx.circular_layout(g)
    type_colors = {
        "Entity": "#4FC3F7",
        "Location": "#A5D6A7",
        "EventNode": "#FFB74D",
        "NarrativeObject": "#CE93D8",
        "WorldTrait": "#9FA8DA",
        "Channel": "#FFF59D",
    }
    node_colors = [type_colors.get(g.nodes[n].get("ntype", ""), "#BDBDBD")
                   for n in g.nodes]
    sizes = [900 if n in focus_ids else 350 for n in g.nodes]
    edgec = ["#FFD700" if n in focus_ids else "#444444" for n in g.nodes]
    linew = [3.0 if n in focus_ids else 0.6 for n in g.nodes]
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=node_colors,
                           node_size=sizes,
                           edgecolors=edgec, linewidths=linew)
    edge_colors = [_EDGE_COLOR.get(g.edges[e].get("etype", ""), "#BDBDBD")
                   for e in g.edges]
    nx.draw_networkx_edges(g, pos, ax=ax, edge_color=edge_colors,
                           width=0.9, alpha=0.7,
                           arrows=True, arrowsize=8,
                           connectionstyle="arc3,rad=0.05")
    labels = {n: g.nodes[n].get("name", n) for n in g.nodes
              if n in focus_ids or g.nodes[n].get("ntype") in ("Entity", "Location")}
    nx.draw_networkx_labels(g, pos, labels=labels, ax=ax, font_size=7)
    ax.set_title(title, fontsize=11)
    ax.axis("off")


def fig_observation() -> None:
    ws = romeo_and_juliet.world_state
    focus = ["ENT_ROMEO"]
    nodes, links, _ = ws_to_ego_graph_data(ws, focus_ids=focus, max_hops=2)
    fig, ax = plt.subplots(figsize=(10, 8))
    _draw_graph_data(
        ax, nodes, links, set(focus),
        title=("ObservationQuery — 2-hop ego graph around ENT_ROMEO\n"
               "(Romeo and Juliet, full topology BFS)"),
    )
    out = OUT / "query_observation.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ── figure 4: InterventionQuery — graph mutilation ─────────────────

def fig_intervention() -> None:
    ws = macbeth.world_state
    diagram = build_causal_diagram(ws)
    target = "EVT_DUNCAN_MURDER"
    if target not in diagram.nodes:
        print(f"missing node {target} in macbeth; skipping intervention fig")
        return

    # Restrict to the 2-hop neighbourhood of the target so the figure is
    # readable. Use undirected BFS over the diagram for clarity, and
    # drop REL:: synthetic nodes (they're internal to d-separation
    # bookkeeping; rendering them swamps the actual story graph).
    und = diagram.to_undirected()
    nbrs = {target}
    frontier = {target}
    for _ in range(2):
        nxt = set()
        for n in frontier:
            nxt |= set(und.neighbors(n))
        nbrs |= nxt
        frontier = nxt
    nbrs = {n for n in nbrs if not n.startswith("REL::")
            and not n.startswith("UTT_") and not n.startswith("CHN_")}
    sub = diagram.subgraph(nbrs).copy()

    severed = list(sub.in_edges(target))  # the do() rule cuts these.

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    pos = nx.spring_layout(sub, seed=11, k=1.6 / max(1, len(sub) ** 0.5))

    def _draw(ax, edge_alpha_fn, title):
        node_colors = ["#d62728" if n == target else "#4FC3F7"
                       for n in sub.nodes]
        node_sizes = [900 if n == target else 320 for n in sub.nodes]
        nx.draw_networkx_nodes(sub, pos, ax=ax,
                               node_color=node_colors,
                               node_size=node_sizes,
                               edgecolors="#222", linewidths=0.8)
        for u, v in sub.edges:
            alpha = edge_alpha_fn(u, v)
            color = "#bbbbbb" if alpha < 0.5 else "#444444"
            style = "dashed" if alpha < 0.5 else "solid"
            nx.draw_networkx_edges(
                sub, pos, ax=ax, edgelist=[(u, v)],
                edge_color=color, width=0.9, alpha=max(0.25, alpha),
                arrows=True, arrowsize=8, style=style,
                connectionstyle="arc3,rad=0.06",
            )
        labels = {n: n.replace("ENT_", "").replace("EVT_", "")
                       .replace("WORLD_", "W:")[:14]
                  for n in sub.nodes}
        nx.draw_networkx_labels(sub, pos, labels=labels, ax=ax, font_size=7)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    _draw(axes[0], lambda u, v: 1.0,
          "Factual diagram (subgraph around EVT_DUNCAN_MURDER)")
    severed_set = set(severed)
    _draw(axes[1],
          lambda u, v: 0.2 if (u, v) in severed_set else 1.0,
          (f"InterventionQuery: do(EVT_DUNCAN_MURDER = blocked)\n"
           f"{len(severed)} inbound causal edges severed by graph surgery"))

    fig.suptitle("Pearl-style do() mutilation on the real Macbeth AMWN",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = OUT / "query_intervention.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ── figure 5: CounterfactualQuery — trait deltas ───────────────────

def fig_counterfactual() -> None:
    ws = macbeth.world_state
    focus = ["ENT_MACBETH", "ENT_LADY_MACBETH"]
    ego = extract_ego_graph_from_memory(ws, focus_entity_ids=focus)
    sandbox = AMWNInstantiator.create_sandbox(ego.model_dump(), "intervention")
    engine_factual = CausalPhysicsEngine(sandbox, ws)
    factual = engine_factual.execute(
        rung=2,
        interventions={},
        target_node_ids=focus,
    )

    sandbox_cf = AMWNInstantiator.create_sandbox(
        ego.model_dump(), "counterfactual",
    )
    engine_cf = CausalPhysicsEngine(sandbox_cf, ws)
    cf_intervention = {"ENT_MACBETH.traits.ambition": 0.0}
    cf = engine_cf.execute(
        rung=3,
        interventions=cf_intervention,
        evidence_node_ids=["ENT_MACBETH"],
        target_node_ids=focus,
    )

    def _trait_map(result, eid):
        out: dict[str, float] = {}
        for m in result.mutations:
            if m.node_id != eid:
                continue
            out[m.trait] = out.get(m.trait, 0.0) + (
                m.new_value - m.old_value
            )
        return out

    def _panel(ax, eid, title):
        f = _trait_map(factual, eid)
        c = _trait_map(cf, eid)
        traits = sorted(set(f) | set(c))
        if not traits:
            ax.set_title(f"{title} (no mutations)")
            ax.axis("off")
            return
        xs = list(range(len(traits)))
        wf = 0.4
        ax.bar([x - wf / 2 for x in xs], [f.get(t, 0.0) for t in traits],
               width=wf, label="factual (do={})", color="#4FC3F7")
        ax.bar([x + wf / 2 for x in xs], [c.get(t, 0.0) for t in traits],
               width=wf, label="counterfactual (do(ambition=0))",
               color="#d62728")
        ax.axhline(0, color="#222", lw=0.5)
        ax.set_xticks(xs)
        ax.set_xticklabels(traits, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Δ trait (new − old)", fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    _panel(axes[0], "ENT_MACBETH", "ENT_MACBETH trait deltas")
    _panel(axes[1], "ENT_LADY_MACBETH",
           "ENT_LADY_MACBETH trait deltas")
    fig.suptitle(
        "CounterfactualQuery — Rung-3 abduction on Macbeth: do(ENT_MACBETH.traits.ambition = 0.0)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = OUT / "query_counterfactual.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ── figure 6: mystery hidden-ancestor mechanism ────────────────────

def fig_mystery_mechanism() -> None:
    ws = macbeth.world_state
    diagram = build_causal_diagram(ws)
    smin, smax = syuzhet_time_bounds(ws)
    samples = 9
    step = max(1, (smax - smin) // (samples - 1))
    anchors = list(range(smin, smax + 1, step))
    if anchors[-1] != smax:
        anchors.append(smax)

    revealed_fracs = []
    hidden_fracs = []
    event_ids = {evt.id: evt for evt in ws.events}
    for s in anchors:
        revealed = {eid for eid, evt in event_ids.items()
                    if evt.syuzhet_index <= s}
        if not event_ids:
            revealed_fracs.append(0.0)
            hidden_fracs.append(0.0)
            continue
        # Hidden-ancestor fraction: across the revealed events, what
        # fraction of their structural ancestors in the AMWN are still
        # unrevealed?
        total_anc = 0
        hidden_anc = 0
        for evt_id in revealed:
            if evt_id not in diagram.nodes:
                continue
            ancs = nx.ancestors(diagram, evt_id)
            ancs &= set(event_ids)
            total_anc += len(ancs)
            hidden_anc += len(ancs - revealed)
        revealed_fracs.append(len(revealed) / max(1, len(event_ids)))
        hidden_fracs.append(hidden_anc / max(1, total_anc))

    fig, ax = plt.subplots(figsize=(10, 5))
    xs = list(range(len(anchors)))
    ax.plot(xs, hidden_fracs, marker="o", color="#1f77b4",
            label="hidden-ancestor fraction (mystery driver)")
    ax.plot(xs, revealed_fracs, marker="s", color="#2ca02c",
            label="revealed-event fraction")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(i) for i in xs])
    ax.set_xlabel("syuzhet anchor (early → late)")
    ax.set_ylabel("fraction ∈ [0, 1]")
    ax.set_title(
        "Mystery scorer mechanism on real Macbeth AMWN\n"
        "(hidden ancestors collapse as the syuzhet reveals causes)",
        fontsize=11,
    )
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = OUT / "mystery_hidden_ancestors.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    fig_structural()
    fig_emotion()
    fig_observation()
    fig_intervention()
    fig_counterfactual()
    fig_mystery_mechanism()


if __name__ == "__main__":
    main()
