# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Social tab — characters' inner & inter-personal layer.

A unified surface for the four entangled audience-side data classes:

  * **Beliefs**          — what a character thinks is true.
  * **Concerns**         — what they desire / fear about a proposition.
  * **Propositions**     — the storyworld facts they hold beliefs about.
  * **Relationships**    — affinity / fear / power_dynamic between dyads.

Plus per-character trait trajectories so the inner state of every
character is visible alongside the social/epistemic structure that
state moves through.

All views are sliced by the global fabula-time cursor (shared with the
World, Causality, and Affective tabs via :class:`StateEvent`).

View modes
----------
- **Network** — combined entity / proposition / concern / relationship
  graph (force or circular layout). Belief and concern edges colour by
  confidence and salience respectively.
- **Beliefs** — per-character belief cards (scrollable claims with
  conviction bands, provenance, time-since-acquired).
- **Concerns** — per-character concern cards (desire / fear,
  salience-weighted, activation-window-aware).
- **Propositions** — per-proposition cards (truth-at-cursor, stakes,
  audience prior, # believers / # concerns referencing).
- **Relationships** — per-dyad cards (affinity / fear / power) and the
  full entity×entity heatmap for the chosen metric.
- **Trait Trajectories** — every character's traits over fabula time
  (how their inner state evolves through the story).
- **Tables** — propositions and concerns as simple data tables.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from nicegui import ui

from shadow_loom_ui.state import AppState, StateEvent
from shadow_loom_ui.viz import (
    render_channels_overview,
    render_concern_salience_heatmap,
    render_entity_belief_chart,
    render_entity_concern_card,
    render_entity_concerns_grid,
    render_entity_trait_trajectory,
    render_epistemic_grid,
    render_knowledge_asymmetry_heatmap,
    render_proposition_state_card,
    render_proposition_stake_timeline,
    render_proposition_truth_sparkline,
    render_propositions_grid,
    render_relationship_heatmap,
    render_relationship_heatmap_timeline,
    render_relationship_quadrant,
    render_relationship_state_grid,
    render_social_layer_graph,
    render_social_layer_legend,
    render_trait_trajectories_grid,
    with_expand,
)
from shadow_loom_ui.viz_helpers import (
    _set_slider_bounds,
    fabula_time_bounds,
    list_believers,
    list_concern_holders,
    snapshot_world_at,
    ws_to_belief_rows,
    ws_to_channel_rows,
    ws_to_concern_rows,
    ws_to_proposition_rows,
    ws_to_social_rows,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


_VIEW_MODES = {
    "network": "Network",
    "ego": "Ego Graph",
    "intermental": "Intermental",
    "asymmetry": "Knowledge Asymmetry",
    "beliefs": "Beliefs",
    "concerns": "Concerns",
    "propositions": "Propositions",
    "channels": "Channels",
    "relationships": "Relationships",
    "trajectories": "Trait Trajectories",
}


def build_social_tab(state: AppState) -> None:
    """Build the Social tab layout."""

    with ui.column().classes("w-full h-full"):
        # ── Toolbar ───────────────────────────────────────────────
        with ui.row().classes("w-full items-center q-pa-sm gap-2"):
            view_mode = ui.toggle(
                _VIEW_MODES,
                value="network",
            ).props("dense no-caps")

            from shadow_loom_ui.components.help_popover import help_popover
            help_popover(
                title="Social — characters, beliefs, concerns, propositions",
                body_md=(
                    "Everything *about characters and the storyworld"
                    " facts they care about*, time-sliced by the"
                    " fabula cursor.\n\n"
                    "### View modes\n"
                    "- **Network** — combined graph of characters,"
                    " propositions (diamonds, sized by stakes) and"
                    " their epistemic / motivational links."
                    " Edges: relationship affinity"
                    " (entity\u2194entity), beliefs (solid amber"
                    " arrow entity\u2192proposition, coloured by"
                    " confidence), and desires/fears (dashed"
                    " green/red arrow entity\u2192proposition,"
                    " sized by salience).\n"
                    "- **Ego Graph** — same network restricted to"
                    " one focus character and its N-hop social /"
                    " epistemic neighbourhood. Pick the character"
                    " from the combo and adjust the hop count.\n"
                    "- **Beliefs** — per-character belief cards"
                    " (scrollable claims with conviction bands and"
                    " provenance).\n"
                    "- **Concerns** — per-character concern cards"
                    " (desire/fear, salience-weighted,"
                    " activation-window aware).\n"
                    "- **Propositions** — per-proposition cards with"
                    " truth-at-cursor, stakes, audience prior, and"
                    " counts of believers + concerns referencing them.\n"
                    "- **Relationships** — per-dyad affinity / fear /"
                    " power cards plus the entity\u00d7entity heatmap"
                    " (static or animated through fabula time).\n"
                    "- **Trait Trajectories** — every character's"
                    " traits as a multi-line plot over fabula time so"
                    " you can see *how the inner state of each"
                    " character moves* alongside the social and"
                    " epistemic structure.\n\n"
                    "### Time cursor\n"
                    "Drag to slice every panel: relationship metrics"
                    " honour ``last_updated_fabula \u2264 t``, beliefs"
                    " honour ``established_at_fabula \u2264 t``,"
                    " concerns replay their snapshots and"
                    " activation windows, and proposition truth"
                    " values reflect ``truth_at_fabula`` commitments"
                    " at or before the cursor."
                ),
                tooltip="What is this tab?",
            )

            # Per-mode selectors.
            believer_select = ui.select(
                options=[], label="Believers",
                multiple=True,
            ).classes("w-64").props("use-chips clearable")
            believer_select.set_visibility(False)

            concern_holder_select = ui.select(
                options=[], label="Concern holders",
                multiple=True,
            ).classes("w-64").props("use-chips clearable")
            concern_holder_select.set_visibility(False)

            traj_select = ui.select(
                options=[], label="Characters",
                multiple=True,
            ).classes("w-72").props("use-chips clearable")
            traj_select.set_visibility(False)

            kind_select = ui.select(
                options={
                    "event_occurs": "event_occurs",
                    "trait_holds": "trait_holds",
                    "relation_holds": "relation_holds",
                    "identity_is": "identity_is",
                    "outcome": "outcome",
                },
                label="Proposition kinds",
                multiple=True,
            ).classes("w-72").props("use-chips clearable")
            kind_select.set_visibility(False)

            # Network-mode controls.
            net_layout = ui.toggle(
                {"force": "Force", "circular": "Circular"},
                value="force",
            ).props("dense no-caps").tooltip("Network layout")
            net_layout.set_visibility(False)
            net_show_rels = ui.checkbox(
                "Relationships", value=True,
            )
            net_show_rels.set_visibility(False)
            net_show_beliefs = ui.checkbox(
                "Beliefs", value=True,
            )
            net_show_beliefs.set_visibility(False)
            net_show_concerns = ui.checkbox(
                "Concerns", value=True,
            )
            net_show_concerns.set_visibility(False)
            net_show_props = ui.checkbox(
                "Propositions", value=True,
            )
            net_show_props.set_visibility(False)

            # Ego-graph controls.
            ego_select = ui.select(
                options={}, label="Ego character",
            ).classes("w-64").props("dense outlined clearable")
            ego_select.set_visibility(False)
            ego_hops = ui.number(
                label="Hops", value=1, min=1, max=4, step=1,
            ).classes("w-24").props("dense outlined")
            ego_hops.set_visibility(False)

            # POV / focalisation control (shared by Network mode).
            pov_select = ui.select(
                options={}, label="POV (focalise)",
            ).classes("w-56").props("dense outlined clearable")
            pov_select.set_visibility(False)

            # Intermental multi-select.
            intermental_select = ui.select(
                options={}, label="Intermental egos",
                multiple=True,
            ).classes("w-72").props("use-chips clearable")
            intermental_select.set_visibility(False)
            intermental_thresh = ui.number(
                label="Conf \u2265", value=0.4, min=0.0, max=1.0, step=0.1,
            ).classes("w-24").props("dense outlined")
            intermental_thresh.set_visibility(False)

            # Relationship-mode controls.
            rel_metric = ui.select(
                {
                    "affinity": "Affinity  (\u20131 hate \u2194 +1 love)",
                    "fear": "Fear  (0 calm \u2192 1 terrified)",
                    "power_dynamic": "Power dynamic  (\u20131 \u2194 +1)",
                },
                value="affinity",
                label="Heatmap metric",
            ).classes("w-64").props("dense outlined")
            rel_metric.set_visibility(False)
            rel_animate = ui.checkbox(
                "Animate over fabula time", value=False,
            )
            rel_animate.set_visibility(False)
            rel_quadrant = ui.checkbox(
                "Affinity\u00d7power scatter", value=False,
            )
            rel_quadrant.set_visibility(False)

            ui.button(
                "Refresh", icon="refresh",
                on_click=lambda: _refresh(),
            ).props("flat dense")

            def _on_mode_change():
                m = view_mode.value
                believer_select.set_visibility(m == "beliefs")
                concern_holder_select.set_visibility(m == "concerns")
                traj_select.set_visibility(m == "trajectories")
                kind_select.set_visibility(m == "propositions")
                net_layout.set_visibility(m in ("network", "ego"))
                net_show_rels.set_visibility(m in ("network", "ego"))
                net_show_beliefs.set_visibility(m in ("network", "ego"))
                net_show_concerns.set_visibility(m in ("network", "ego"))
                net_show_props.set_visibility(m in ("network", "ego"))
                ego_select.set_visibility(m == "ego")
                ego_hops.set_visibility(m == "ego")
                pov_select.set_visibility(m == "network")
                intermental_select.set_visibility(m == "intermental")
                intermental_thresh.set_visibility(m == "intermental")
                rel_metric.set_visibility(m == "relationships")
                rel_animate.set_visibility(m == "relationships")
                rel_quadrant.set_visibility(m == "relationships")
                _refresh()

            view_mode.on("update:model-value", _on_mode_change)
            for w in (
                net_layout, net_show_rels, net_show_beliefs,
                net_show_concerns, net_show_props,
                ego_select, ego_hops, pov_select,
                intermental_select, intermental_thresh,
                rel_metric, rel_animate, rel_quadrant,
                believer_select, concern_holder_select,
                traj_select, kind_select,
            ):
                w.on("update:model-value", lambda _e=None: _refresh())

        # ── Fabula timeline slider (shared cursor) ────────────────
        _slider_state = {
            "local_origin": False,
            "rendering": False,
            "pending": False,
            "event_index": [],
        }

        slider_row = ui.row().classes(
            "w-full items-center q-px-md q-pb-sm gap-3 "
            "bg-slate-50 border-b border-slate-200"
        )
        with slider_row:
            ui.icon("schedule", color="primary")
            time_axis_label = ui.label("Fabula time:").classes(
                "text-sm text-slate-600"
            )
            time_label = ui.label("live").classes(
                "text-sm font-mono text-slate-700 w-12"
            )
            time_slider = ui.slider(min=0, max=1, value=0, step=1).props(
                "color=primary label-always dense"
            ).classes("flex-grow")
            ui.button(
                "Live", icon="bolt",
                on_click=lambda: _set_live(),
            ).props("flat dense no-caps color=secondary")

        def _on_global_axis_change(**_kw):
            time_axis_label.text = (
                "Syuzhet idx:" if state.time_axis == "syuzhet"
                else "Fabula time:"
            )
            _refresh()
        state.on(StateEvent.TIME_AXIS_CHANGED, _on_global_axis_change)
        # Sync label on initial render too.
        time_axis_label.text = (
            "Syuzhet idx:" if state.time_axis == "syuzhet"
            else "Fabula time:"
        )

        def _set_live():
            _slider_state["local_origin"] = False
            state.set_active_cursor(None)

        def _on_slider_change():
            try:
                t = int(time_slider.value)
            except (TypeError, ValueError):
                return
            if state.active_cursor == t:
                return
            _slider_state["local_origin"] = True
            state.set_active_cursor(t)

        def _nearest_event_label(t: int) -> str:
            idx = _slider_state["event_index"]
            prefix = "s" if state.time_axis == "syuzhet" else "t"
            if not idx:
                return f"{prefix}={t}"
            best = min(
                idx,
                key=lambda et: (abs(et[0] - t), 0 if et[0] <= t else 1),
            )
            label = best[1]
            if len(label) > 80:
                label = label[:77] + "\u2026"
            return f"{prefix}={t} \u2014 {label}"

        def _push_label_value(text: str) -> None:
            safe = text.replace('"', "'").replace("\n", " ")
            time_slider.props(f'label-value="{safe}"')

        def _on_slider_input():
            try:
                t = int(time_slider.value)
            except (TypeError, ValueError):
                return
            _push_label_value(_nearest_event_label(t))
            prefix = "s" if state.time_axis == "syuzhet" else "t"
            label_text = f"{prefix}={t}"
            if time_label.text != label_text:
                time_label.text = label_text

        time_slider.on(
            "update:model-value", lambda _e=None: _on_slider_input()
        )
        time_slider.on("change", lambda: _on_slider_change())

        # ── Graph container ───────────────────────────────────────
        graph_container = ui.column().classes("w-full flex-grow p-4")

        # ── Raw data tables ───────────────────────────────────────
        with ui.expansion(
            "Raw Data (Propositions \u2022 Concerns)",
            icon="table_chart",
        ).classes(
            "w-full mx-4 mb-4 bg-white border border-slate-200 rounded-xl"
        ):
            _build_data_tables(state)

        def _on_graph_click(e):
            data = e.args if isinstance(e.args, dict) else {}
            node_data = data.get("data", {})
            node_id = node_data.get("id") or data.get("name")
            node_type = node_data.get("_sl_node_type")
            if node_id and node_type:
                state.select_node(node_id, node_type)

        # ── Slider widget sync ────────────────────────────────────
        def _sync_slider_widget(ws) -> None:
            from shadow_loom_ui.viz_helpers import (
                axis_bounds, event_axis_value,
            )
            axis = state.time_axis
            tmin, tmax = axis_bounds(ws, axis)
            if tmax <= tmin:
                slider_row.set_visibility(False)
                return
            slider_row.set_visibility(True)
            _set_slider_bounds(time_slider, tmin, tmax)
            event_index = []
            for evt in (getattr(ws, "events", None) or []):
                ft = event_axis_value(evt, axis)
                desc = (
                    getattr(evt, "description", None)
                    or getattr(evt, "content", None)
                    or getattr(evt, "id", None)
                    or ""
                ).strip()
                if not desc:
                    continue
                event_index.append((int(ft), desc))
            event_index.sort(key=lambda x: x[0])
            _slider_state["event_index"] = event_index
            cursor_value = state.active_cursor
            cur_prefix = "s" if state.time_axis == "syuzhet" else "t"
            if cursor_value is None:
                desired = tmax
                label_text = "live"
            else:
                desired = max(tmin, min(tmax, cursor_value))
                label_text = f"{cur_prefix}={desired}"
            if not _slider_state["local_origin"]:
                try:
                    cur = int(time_slider.value or 0)
                except (TypeError, ValueError):
                    cur = -1
                if cur != desired:
                    time_slider.value = desired
            if time_label.text != label_text:
                time_label.text = label_text
            _push_label_value(_nearest_event_label(int(desired)))
            _slider_state["local_origin"] = False

        # ── Refresh ───────────────────────────────────────────────
        def _refresh(**kw):
            if _slider_state["rendering"]:
                _slider_state["pending"] = True
                return
            _slider_state["rendering"] = True
            try:
                _do_refresh(**kw)
            finally:
                _slider_state["rendering"] = False
            if _slider_state["pending"]:
                _slider_state["pending"] = False
                ui.timer(0.01, lambda: _refresh(), once=True)

        def _do_refresh(**_kw):
            graph_container.clear()
            ws = state.world_state
            if ws is None:
                slider_row.set_visibility(False)
                with graph_container:
                    ui.label("No world model loaded.").classes(
                        "text-sm text-slate-400 italic q-pa-lg"
                    )
                return

            _sync_slider_widget(ws)
            from shadow_loom_ui.viz_helpers import axis_bounds
            _, tmax = axis_bounds(ws, state.time_axis)

            # The active cursor on the current axis (None => live / latest).
            fabula_t = state.active_cursor
            if fabula_t is None:
                fabula_t_eff = tmax if tmax > 0 else None
            else:
                fabula_t_eff = fabula_t

            # Reading-order (syuzhet) re-mapping: when the global
            # axis is "syuzhet", treat the cursor as a syuzhet index
            # and resolve it to the *latest* fabula_time among events
            # with ``syuzhet_index <= cursor``. This implements
            # Genette's order/anachrony filter \u2014 \"what does the
            # audience know by the time syuzhet=k is told?\"
            if state.time_axis == "syuzhet" and fabula_t_eff is not None:
                from shadow_loom_ui.viz_helpers import resolve_cursor
                fabula_t_eff = resolve_cursor(
                    ws, "syuzhet", int(fabula_t_eff)
                )

            # Snapshot the world for entity / belief / relationship
            # views (so beliefs reflect the cursor — same pattern as
            # the World tab).
            ws_snap = ws
            if fabula_t_eff is not None and tmax > 0:
                try:
                    ws_snap = snapshot_world_at(ws, fabula_t_eff)
                except Exception:
                    logger.exception(
                        "Snapshot failed; falling back to live"
                    )
                    ws_snap = ws

            # Refresh selectors against the live model (so options
            # don't shrink when scrubbing back through fabula time).
            believer_opts = {
                eid: name for (eid, name, _c) in list_believers(ws)
            }
            believer_select.options = believer_opts
            concern_opts = {
                eid: name for (eid, name, _c) in list_concern_holders(ws)
            }
            concern_holder_select.options = concern_opts
            entity_opts = {eid: ent.name for eid, ent in ws.entities.items()}
            traj_select.options = entity_opts
            ego_select.options = entity_opts
            pov_select.options = entity_opts
            intermental_select.options = entity_opts
            if ego_select.value not in entity_opts:
                # Default to first entity if nothing valid is selected.
                ego_select.value = (
                    next(iter(entity_opts), None)
                    if entity_opts else None
                )

            mode = view_mode.value
            with graph_container:
                try:
                    if mode == "network":
                        with ui.expansion(
                            "Combined social-layer network",
                            icon="hub",
                            value=True,
                        ).classes(
                            "w-full bg-white border border-slate-200 "
                            "rounded-xl mb-2"
                        ):
                            render_social_layer_legend()
                            with_expand(
                                lambda h, pid=pov_select.value: (
                                    render_social_layer_graph(
                                        ws_snap,
                                        on_click=_on_graph_click,
                                        height=h,
                                        layout=net_layout.value or "force",
                                        include_relationships=bool(
                                            net_show_rels.value
                                        ),
                                        include_beliefs=bool(
                                            net_show_beliefs.value
                                        ),
                                        include_concerns=bool(
                                            net_show_concerns.value
                                        ),
                                        include_propositions=bool(
                                            net_show_props.value
                                        ),
                                        fabula_t=fabula_t_eff,
                                        event_t=fabula_t_eff,
                                        pov_id=pid,
                                    )
                                ),
                                title=(
                                    "Characters \u2022 Propositions"
                                    " \u2022 Concerns \u2022 Beliefs"
                                    " \u2022 Relationships"
                                ),
                                height="540px",
                            )

                    elif mode == "ego":
                        ego_id = ego_select.value
                        try:
                            hops = max(1, int(ego_hops.value or 1))
                        except (TypeError, ValueError):
                            hops = 1
                        ego_name = (
                            ws_snap.entities[ego_id].name
                            if ego_id and ego_id in ws_snap.entities
                            else (ego_id or "—")
                        )
                        with ui.expansion(
                            f"Ego graph \u2014 {ego_name} (\u2264{hops} hop"
                            f"{'s' if hops != 1 else ''})",
                            icon="account_tree",
                            value=True,
                        ).classes(
                            "w-full bg-white border border-slate-200 "
                            "rounded-xl mb-2"
                        ):
                            render_social_layer_legend()
                            if not ego_id:
                                ui.label(
                                    "Choose a character above to focus the"
                                    " ego graph."
                                ).classes(
                                    "text-sm text-slate-500 italic q-pa-md"
                                )
                            else:
                                with_expand(
                                    lambda h, eid=ego_id, hp=hops: (
                                        render_social_layer_graph(
                                            ws_snap,
                                            on_click=_on_graph_click,
                                            height=h,
                                            layout=(
                                                net_layout.value or "force"
                                            ),
                                            include_relationships=bool(
                                                net_show_rels.value
                                            ),
                                            include_beliefs=bool(
                                                net_show_beliefs.value
                                            ),
                                            include_concerns=bool(
                                                net_show_concerns.value
                                            ),
                                            include_propositions=bool(
                                                net_show_props.value
                                            ),
                                            fabula_t=fabula_t_eff,
                                            event_t=fabula_t_eff,
                                            ego_id=eid,
                                            ego_max_hops=hp,
                                        )
                                    ),
                                    title=f"Ego graph \u2014 {ego_name}",
                                    height="540px",
                                )

                    elif mode == "intermental":
                        sel = intermental_select.value
                        sel_ids = (
                            list(sel) if isinstance(sel, list) and sel
                            else []
                        )
                        try:
                            thresh = float(intermental_thresh.value or 0.4)
                        except (TypeError, ValueError):
                            thresh = 0.4
                        with ui.expansion(
                            "Intermental network \u2014 shared & "
                            "divergent beliefs (Palmer)",
                            icon="diversity_3",
                            value=True,
                        ).classes(
                            "w-full bg-white border border-slate-200 "
                            "rounded-xl mb-2"
                        ):
                            render_social_layer_legend()
                            if len(sel_ids) < 2:
                                ui.label(
                                    "Pick \u2265 2 characters above to"
                                    " visualise shared / divergent"
                                    " beliefs."
                                ).classes(
                                    "text-sm text-slate-500 italic q-pa-md"
                                )
                            else:
                                with_expand(
                                    lambda h, ids=sel_ids, th=thresh: (
                                        render_social_layer_graph(
                                            ws_snap,
                                            on_click=_on_graph_click,
                                            height=h,
                                            layout=(
                                                net_layout.value or "force"
                                            ),
                                            fabula_t=fabula_t_eff,
                                            event_t=fabula_t_eff,
                                            intermental_ids=ids,
                                            intermental_threshold=th,
                                        )
                                    ),
                                    title="Intermental graph",
                                    height="540px",
                                )

                    elif mode == "asymmetry":
                        with ui.expansion(
                            "Knowledge asymmetry \u2014 dramatic-irony"
                            " lens (Sternberg)",
                            icon="visibility",
                            value=True,
                        ).classes(
                            "w-full bg-white border border-slate-200 "
                            "rounded-xl mb-2"
                        ):
                            with_expand(
                                lambda h: (
                                    render_knowledge_asymmetry_heatmap(
                                        ws_snap,
                                        fabula_t=fabula_t_eff,
                                        height=h,
                                    )
                                ),
                                title=(
                                    "Audience prior strip + character"
                                    " \u00d7 proposition asymmetry"
                                ),
                                height="540px",
                            )

                    elif mode == "beliefs":
                        sel = believer_select.value
                        sel_ids = list(sel) if isinstance(sel, list) and sel else None
                        render_epistemic_grid(
                            ws_snap, selected_ids=sel_ids,
                        )

                    elif mode == "concerns":
                        sel = concern_holder_select.value
                        sel_ids = list(sel) if isinstance(sel, list) and sel else None
                        # Salience heatmap above the per-character
                        # cards: lets the user spot high-salience
                        # concerns at a glance before drilling into
                        # individual horizon strips.
                        render_concern_salience_heatmap(
                            ws,
                            fabula_t=fabula_t_eff,
                            selected_ids=sel_ids,
                        )
                        ui.separator().classes("q-my-md")
                        render_entity_concerns_grid(
                            ws,  # use live ws so concerns aren't lost
                            selected_ids=sel_ids,
                            fabula_t=fabula_t_eff,
                        )

                    elif mode == "propositions":
                        sel = kind_select.value
                        kinds = list(sel) if isinstance(sel, list) and sel else None
                        render_propositions_grid(
                            ws,
                            fabula_t=fabula_t_eff,
                            selected_kinds=kinds,
                        )
                        # Stake / audience-prior / surprise-spike
                        # timeline for each surfaced proposition.
                        # Mirrors the grid filter so the user can
                        # scan the same set as a time-series.
                        props_for_timeline = list(ws.propositions or [])
                        if kinds:
                            wanted = set(kinds)
                            props_for_timeline = [
                                p for p in props_for_timeline if p.kind in wanted
                            ]
                        if props_for_timeline:
                            ui.separator().classes("q-my-md")
                            ui.label("Stake & truth-commit timeline").classes(
                                "text-sm font-semibold text-slate-700"
                            )
                            for p in props_for_timeline[:6]:
                                render_proposition_stake_timeline(
                                    ws, p.proposition_id,
                                )
                                # Compact binary truth sparkline
                                # right under each stake chart so
                                # the reveal pattern (one-shot vs
                                # oscillating) is visible at a glance.
                                with ui.row().classes(
                                    "w-full items-center gap-2"
                                ):
                                    ui.label("truth:").classes(
                                        "text-[10px] uppercase "
                                        "text-slate-500 tracking-wide"
                                        " w-12 shrink-0"
                                    )
                                    render_proposition_truth_sparkline(
                                        ws, p.proposition_id,
                                    )

                    elif mode == "channels":
                        # First-class Channel rendering: medium,
                        # directionality, per-participant
                        # intelligibility (decode probability), and
                        # lifespan with a strike-through when severed
                        # at or before the cursor. Channels were
                        # invisible in the UI prior to this panel.
                        render_channels_overview(
                            ws_snap, fabula_t=fabula_t_eff, state=state,
                        )

                    elif mode == "relationships":
                        chosen_metric = rel_metric.value or "affinity"
                        _metric_titles = {
                            "affinity": (
                                "Affinity heatmap  (\u20131 hate \u2194 +1 love)"
                            ),
                            "fear": (
                                "Fear heatmap  (0 calm \u2192 1 terrified)"
                            ),
                            "power_dynamic": (
                                "Power dynamic  (\u20131 \u2194 +1)"
                            ),
                        }
                        animate = bool(rel_animate.value)
                        heatmap_title = _metric_titles.get(
                            chosen_metric, "Relationship heatmap"
                        )
                        axis_used = state.time_axis or "fabula"
                        if animate:
                            heatmap_title = (
                                f"{heatmap_title} \u2014 over "
                                + ("syuzhet (reading order)"
                                   if axis_used == "syuzhet"
                                   else "fabula time")
                            )
                        with ui.expansion(
                            heatmap_title, icon="grid_on", value=True,
                        ).classes(
                            "w-full bg-white border border-slate-200 "
                            "rounded-xl mb-2"
                        ):
                            if animate:
                                with_expand(
                                    lambda h, m=chosen_metric, a=axis_used: (
                                        render_relationship_heatmap_timeline(
                                            ws, metric=m, height=h, axis=a,
                                        )
                                    ),
                                    title=heatmap_title, height="520px",
                                )
                            else:
                                with_expand(
                                    lambda h, m=chosen_metric: (
                                        render_relationship_heatmap(
                                            ws_snap, metric=m, height=h,
                                        )
                                    ),
                                    title=heatmap_title, height="420px",
                                )
                        with ui.expansion(
                            "Per-dyad snapshot cards "
                            "(affinity \u2022 fear \u2022 power)",
                            icon="favorite", value=True,
                        ).classes(
                            "w-full bg-white border border-slate-200 "
                            "rounded-xl mb-2"
                        ):
                            render_relationship_state_grid(ws_snap)
                        if bool(rel_quadrant.value):
                            with ui.expansion(
                                "Affinity \u00d7 power quadrant"
                                " (Greimas actantial space)",
                                icon="scatter_plot", value=True,
                            ).classes(
                                "w-full bg-white border border-slate-200 "
                                "rounded-xl mb-2"
                            ):
                                with_expand(
                                    lambda h: (
                                        render_relationship_quadrant(
                                            ws_snap,
                                            fabula_t=fabula_t_eff,
                                            height=h,
                                        )
                                    ),
                                    title=(
                                        "Per-dyad scatter; size = fear"
                                    ),
                                    height="480px",
                                )

                    elif mode == "trajectories":
                        sel = traj_select.value
                        sel_ids = list(sel) if isinstance(sel, list) and sel else None
                        render_trait_trajectories_grid(
                            ws, selected_ids=sel_ids,
                        )

                except Exception as e:
                    logger.exception("Social tab rendering failed")
                    ui.label(f"Render error: {e}").classes("text-negative")

        # Initial render + subscriptions.
        _refresh()

        _SOCIAL_PATH = "social"
        _social_dirty = {"on": False}

        def _social_gated(*args, **kw):
            if not state.is_path_visible(_SOCIAL_PATH):
                _social_dirty["on"] = True
                return
            _social_dirty["on"] = False
            _refresh()

        def _on_path(**kw):
            if _social_dirty["on"] and state.is_path_visible(_SOCIAL_PATH):
                _social_dirty["on"] = False
                _refresh()

        state.on(StateEvent.WORLD_STATE_CHANGED, _social_gated)
        state.on(StateEvent.ACTIVE_PATH_CHANGED, _on_path)
        state.on(StateEvent.FABULA_CURSOR_CHANGED, _social_gated)
        state.on(StateEvent.SYUZHET_CURSOR_CHANGED, _social_gated)
        state.on(StateEvent.TIME_AXIS_CHANGED, _social_gated)


# =====================================================================
# Raw data tables — Propositions and Concerns only.
# =====================================================================

def _build_data_tables(state: AppState) -> None:
    """Build proposition and concern data tables as sub-tabs.

    Per the data model these two layers don't fit cleanly in the World
    tab's typed-graph tables (which are entity / event / object /
    edges); they live here next to the network and card views that
    visualise them.
    """
    from shadow_loom_ui.components._subtab_help import subtab_help

    cursor = {"t": None}  # follows the global fabula cursor

    with ui.tabs().classes("w-full").props("dense") as data_tabs:
        ui.tab("propositions", label="Propositions", icon="forum")
        ui.tab("concerns", label="Concerns", icon="psychology")
        ui.tab("beliefs", label="Beliefs", icon="lightbulb")
        ui.tab("relationships", label="Relationships", icon="favorite")
        ui.tab("channels", label="Channels", icon="hearing")
        ui.tab("traits", label="Traits", icon="tune")

    with ui.row().classes(
        "w-full items-center px-3 py-1 gap-3 text-xs text-slate-500"
    ):
        ui.icon("schedule", size="sm")
        cursor_label = ui.label("rows reflect: live").classes("font-mono")
        ui.space()
        only_active = ui.switch("Concerns: only active@t", value=False).props(
            "dense"
        ).classes("text-xs")

    _table_props = "dense flat bordered"

    with ui.tab_panels(data_tabs, value="propositions").classes("w-full"):
        with ui.tab_panel("propositions"):
            subtab_help("social.propositions")
            prop_table = ui.table(
                columns=[
                    {"name": "id", "label": "ID", "field": "id", "sortable": True},
                    {"name": "kind", "label": "Kind", "field": "kind", "sortable": True},
                    {"name": "description", "label": "Description@t", "field": "description"},
                    {"name": "truth_at", "label": "Truth@t", "field": "truth_at", "sortable": True},
                    {"name": "stakes", "label": "Stakes@t", "field": "stakes", "sortable": True},
                    {"name": "audience_default_prior", "label": "Aud. prior@t", "field": "audience_default_prior", "sortable": True},
                    {"name": "concerns_referencing", "label": "#Concerns", "field": "concerns_referencing", "sortable": True},
                    {"name": "beliefs_referencing", "label": "#Beliefs", "field": "beliefs_referencing", "sortable": True},
                    {"name": "referent_ids", "label": "Referents", "field": "referent_ids"},
                    {"name": "world_id", "label": "Branch", "field": "world_id", "sortable": True},
                ],
                rows=[],
                pagination={"rowsPerPage": 15},
            ).props(_table_props).classes("w-full")

        with ui.tab_panel("concerns"):
            subtab_help("social.concerns")
            concern_table = ui.table(
                columns=[
                    {"name": "entity", "label": "Entity", "field": "entity", "sortable": True},
                    {"name": "concern_id", "label": "Concern", "field": "concern_id", "sortable": True},
                    {"name": "polarity", "label": "Polarity@t", "field": "polarity", "sortable": True},
                    {"name": "salience", "label": "Salience@t", "field": "salience", "sortable": True},
                    {"name": "kind", "label": "Kind@t", "field": "kind"},
                    {"name": "active", "label": "Active@t", "field": "active", "sortable": True},
                    {"name": "proposition_id", "label": "Proposition", "field": "proposition_id", "sortable": True},
                    {"name": "proposition_desc", "label": "About", "field": "proposition_desc"},
                    {"name": "world_id", "label": "Branch", "field": "world_id", "sortable": True},
                ],
                rows=[],
                pagination={"rowsPerPage": 20},
            ).props(_table_props).classes("w-full")

        with ui.tab_panel("beliefs"):
            subtab_help("social.beliefs")
            belief_table = ui.table(
                columns=[
                    {"name": "believer", "label": "Believer", "field": "believer", "sortable": True},
                    {"name": "target", "label": "Target", "field": "target", "sortable": True},
                    {"name": "target_kind", "label": "Kind", "field": "target_kind", "sortable": True},
                    {"name": "perceived_state", "label": "Perceived state", "field": "perceived_state"},
                    {"name": "confidence", "label": "Conf.", "field": "confidence", "sortable": True},
                    {"name": "inertia", "label": "Inertia", "field": "inertia", "sortable": True},
                    {"name": "established_at_fabula", "label": "Est. @t", "field": "established_at_fabula", "sortable": True},
                    {"name": "acquired_via_channel", "label": "Via channel", "field": "acquired_via_channel"},
                    {"name": "acquired_via_event_id", "label": "Via event", "field": "acquired_via_event_id"},
                    {"name": "proposition_id", "label": "Proposition", "field": "proposition_id"},
                ],
                rows=[],
                pagination={"rowsPerPage": 20},
            ).props(_table_props).classes("w-full")

        with ui.tab_panel("relationships"):
            subtab_help("social.relationships")
            rel_table = ui.table(
                columns=[
                    {"name": "source", "label": "Source", "field": "source", "sortable": True},
                    {"name": "target", "label": "Target", "field": "target", "sortable": True},
                    {"name": "affinity", "label": "Affinity", "field": "affinity", "sortable": True},
                    {"name": "fear", "label": "Fear", "field": "fear", "sortable": True},
                    {"name": "power", "label": "Power", "field": "power", "sortable": True},
                    {"name": "inertia", "label": "Inertia", "field": "inertia", "sortable": True},
                    {"name": "evidence", "label": "Evidence", "field": "evidence", "sortable": True},
                    {"name": "axes_observed", "label": "Axes", "field": "axes_observed", "sortable": True},
                    {"name": "last_updated_fabula", "label": "Updated @t", "field": "last_updated_fabula", "sortable": True},
                    {"name": "world_id", "label": "Branch", "field": "world_id", "sortable": True},
                ],
                rows=[],
                pagination={"rowsPerPage": 20},
            ).props(_table_props).classes("w-full")

        with ui.tab_panel("channels"):
            subtab_help("social.channels")
            channel_table = ui.table(
                columns=[
                    {"name": "id", "label": "ID", "field": "id", "sortable": True},
                    {"name": "name", "label": "Name", "field": "name", "sortable": True},
                    {"name": "medium", "label": "Medium", "field": "medium", "sortable": True},
                    {"name": "directionality", "label": "Direction", "field": "directionality", "sortable": True},
                    {"name": "participants", "label": "Participants", "field": "participants"},
                    {"name": "min_intelligibility", "label": "Min intel.", "field": "min_intelligibility", "sortable": True},
                    {"name": "established_at_fabula", "label": "Est. @t", "field": "established_at_fabula", "sortable": True},
                    {"name": "world_id", "label": "Branch", "field": "world_id", "sortable": True},
                ],
                rows=[],
                pagination={"rowsPerPage": 20},
            ).props(_table_props).classes("w-full")

        with ui.tab_panel("traits"):
            subtab_help("social.traits")
            trait_table = ui.table(
                columns=[
                    {"name": "entity", "label": "Entity", "field": "entity", "sortable": True},
                    {"name": "trait", "label": "Trait", "field": "trait", "sortable": True},
                    {"name": "value", "label": "Value@t", "field": "value", "sortable": True},
                    {"name": "observed", "label": "Observed", "field": "observed", "sortable": True},
                    {"name": "inertia", "label": "Inertia", "field": "inertia", "sortable": True},
                    {"name": "world_id", "label": "Branch", "field": "world_id", "sortable": True},
                ],
                rows=[],
                pagination={"rowsPerPage": 25},
            ).props(_table_props).classes("w-full")

    def _refresh_tables(**_kw):
        ws = state.world_state
        if ws is None:
            prop_table.rows = []
            concern_table.rows = []
            belief_table.rows = []
            rel_table.rows = []
            channel_table.rows = []
            trait_table.rows = []
            return
        # Cursor lives on the active axis; resolve to a fabula time for
        # the row queries (they snapshot in story-world time).
        from shadow_loom_ui.viz_helpers import resolve_cursor
        raw_cursor = state.active_cursor
        cursor["t"] = resolve_cursor(ws, state.time_axis, raw_cursor)
        if raw_cursor is None:
            t_disp = "live"
        else:
            prefix = "s" if state.time_axis == "syuzhet" else "t"
            t_disp = f"{prefix}={raw_cursor}"
        cursor_label.text = f"rows reflect: {t_disp}"
        prop_table.rows = ws_to_proposition_rows(ws, fabula_t=cursor["t"])
        concern_table.rows = ws_to_concern_rows(
            ws, fabula_t=cursor["t"],
            only_active=bool(only_active.value),
        )
        belief_table.rows = ws_to_belief_rows(ws, fabula_t=cursor["t"])
        # Snapshot the world at the cursor for all remaining tables so
        # relationship metrics, channels, and trait values reflect the
        # chosen fabula time rather than the final-frame values.
        snap_ws = ws
        if cursor["t"] is not None:
            try:
                snap_ws = snapshot_world_at(ws, cursor["t"])
            except Exception:
                snap_ws = ws
        # Relationship rows: snapshot_world_at time-slices social_topology
        # axes to last_updated_fabula <= t and replays mutation_social edges,
        # so passing snap_ws gives cursor-accurate per-axis metric values.
        rel_table.rows = ws_to_social_rows(snap_ws)
        # Channel rows: snapshot_world_at already filters out channels
        # established after t or terminated at/before t.
        channel_table.rows = ws_to_channel_rows(snap_ws)
        trait_rows: list[dict] = []
        for eid, ent in snap_ws.entities.items():
            for tname, tdata in (ent.traits or {}).items():
                trait_rows.append({
                    "entity": ent.name,
                    "trait": tname,
                    "value": (
                        round(float(tdata.value), 3)
                        if isinstance(tdata.value, (int, float))
                        else str(tdata.value)
                    ),
                    "observed": bool(getattr(tdata, "observed", True)),
                    "inertia": round(
                        float(getattr(tdata, "inertia", 0.0) or 0.0), 3,
                    ),
                    "world_id": getattr(ent, "world_id", "") or "",
                })
        trait_rows.sort(key=lambda r: (r["entity"], r["trait"]))
        trait_table.rows = trait_rows

    only_active.on("update:model-value", lambda _e=None: _refresh_tables())
    _refresh_tables()
    state.on(StateEvent.WORLD_STATE_CHANGED, _refresh_tables)
    state.on(StateEvent.FABULA_CURSOR_CHANGED, _refresh_tables)
    state.on(StateEvent.SYUZHET_CURSOR_CHANGED, _refresh_tables)
    state.on(StateEvent.TIME_AXIS_CHANGED, _refresh_tables)
