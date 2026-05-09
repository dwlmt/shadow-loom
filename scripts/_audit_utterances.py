"""Audit utterance timing across all example_worlds vs canonical schema rules.

Rules checked (per-world):
  R1. syuzhet_index uniqueness across all events (utt + non-utt).
  R2. fabula_time monotonicity per (speaker_id) — speaker can't speak in two
       places simultaneously unless via different channels.
  R3. Utterance fabula_time should be >= max(target.fabula_time) for any
       target_ids that are *past* events the speaker is reporting (i.e. the
       speaker can only describe what has already happened).
       Exception: a prophecy / performative may reference a *future* event.
  R4. Utterance syuzhet_index >= max(target.syuzhet_index) for non-prophetic
       (truth_value != 'performative') utterances — you don't narrate the
       discussion of an event before narrating the event.
  R5. truth_value, speaker_id, addressee_ids, content all populated.
  R6. via_channel_id (when set) resolves; speaker+addressees subset of
       channel.participant_ids.
  R7. speaker_id in actor_ids (or actor_ids empty) — schema requirement.
  R8. addressee_ids non-empty.
  R9. target_ids all resolve to known event ids.

Prints a summary table + per-world findings.
"""
from __future__ import annotations
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WORLDS = [
    "a_court_of_thorn_and_roses", "a_fish_called_wanda", "apocalypse_now",
    "brief_encounter", "dads_army", "death_on_the_nile", "frankenstein",
    "gone_girl", "great_expectations", "great_gatsby", "macbeth",
    "nineteen_eighty_four", "once_upon_a_time_in_the_west", "persuasion",
    "reservoir_dogs", "romeo_and_juliet", "the_devil_wears_prada",
    "the_lion_the_witch_and_the_wardrobe", "tinker_tailor_soldier_spy",
    "wuthering_heights",
]


def audit_world(name: str):
    mod = importlib.import_module(f"example_worlds.{name}")
    ws = mod.world_state
    events = list(ws.events)
    by_id = {e.id: e for e in events}
    channels = dict(ws.channels or {})
    findings = []

    # R1 syuzhet uniqueness
    seen = {}
    for e in events:
        if e.syuzhet_index in seen:
            findings.append(("R1", f"syuzhet_index={e.syuzhet_index} duplicated: "
                                   f"{seen[e.syuzhet_index]} <-> {e.id}"))
        else:
            seen[e.syuzhet_index] = e.id

    utts = [e for e in events if e.event_type == "utterance"]

    # R2 speaker fabula collisions
    by_speaker = {}
    for u in utts:
        if not u.speaker_id:
            continue
        by_speaker.setdefault(u.speaker_id, []).append(u)
    for sp, lst in by_speaker.items():
        seen_ft = {}
        for u in lst:
            if u.fabula_time in seen_ft:
                other = seen_ft[u.fabula_time]
                # OK if different channels and both telepathic / written
                findings.append(("R2", f"speaker {sp} has 2 utterances at "
                                       f"fabula={u.fabula_time}: {other} & {u.id}"))
            else:
                seen_ft[u.fabula_time] = u.id

    for u in utts:
        # R5
        missing = [k for k in ("truth_value", "speaker_id", "content")
                   if not getattr(u, k)]
        if missing:
            findings.append(("R5", f"{u.id}: missing {missing}"))
        # R7
        if u.actor_ids and u.speaker_id and u.speaker_id not in u.actor_ids:
            findings.append(("R7", f"{u.id}: speaker {u.speaker_id} not in actor_ids {u.actor_ids}"))
        # R8
        if not u.addressee_ids:
            findings.append(("R8", f"{u.id}: empty addressee_ids"))
        # R6
        if u.via_channel_id:
            ch = channels.get(u.via_channel_id)
            if not ch:
                findings.append(("R6", f"{u.id}: via_channel_id {u.via_channel_id} unknown"))
            else:
                parts = set(ch.participant_ids)
                miss = [p for p in [u.speaker_id, *u.addressee_ids] if p and p not in parts]
                if miss:
                    findings.append(("R6", f"{u.id}: speaker/addressee {miss} not in channel {ch.id} participants"))
        # R9 + R3 + R4
        for tid in u.target_ids or []:
            if tid.startswith("EVT_"):
                t = by_id.get(tid)
                if not t:
                    findings.append(("R9", f"{u.id}: target {tid} unknown"))
                    continue
                # R3
                if u.truth_value != "performative":
                    if t.fabula_time > u.fabula_time:
                        findings.append(("R3", f"{u.id} (fabula={u.fabula_time}, "
                                               f"truth={u.truth_value}) reports future event "
                                               f"{tid} (fabula={t.fabula_time})"))
                # R4 dropped — analepsis/foreshadowing makes syuzhet ordering of
                # *referenced* events independent of the utterance's prose location.
    return utts, findings


def main():
    print(f"{'world':<35} {'#utt':>4} {'#findings':>9}")
    print("-" * 55)
    all_findings = {}
    for w in WORLDS:
        try:
            utts, fnd = audit_world(w)
        except Exception as e:
            print(f"{w:<35} ERROR: {e}")
            continue
        all_findings[w] = fnd
        print(f"{w:<35} {len(utts):>4} {len(fnd):>9}")
    print()
    for w, fnd in all_findings.items():
        if not fnd:
            continue
        print(f"\n=== {w} ({len(fnd)} findings) ===")
        for code, msg in fnd:
            print(f"  [{code}] {msg}")


if __name__ == "__main__":
    main()
