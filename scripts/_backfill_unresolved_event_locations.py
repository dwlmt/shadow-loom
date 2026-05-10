# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""One-shot backfill of ``EventNode.at_location_id`` for events that the
primary-actor fallback cannot resolve (deaths, off-stage scenes, weather).

Mappings here were derived by reading each event's description in context
and choosing the most defensible LOC_ id from the world's location set.
A handful of events with no plausible canonical location in the fixture
(e.g. Isabella dying "in the south", off in another county) are
intentionally left unmapped.

Run once from the repo root::

    python scripts/_backfill_unresolved_event_locations.py [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import libcst as cst

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._backfill_event_locations import _EventNodeRewriter  # noqa: E402

WORLDS_DIR = ROOT / "example_worlds"

# Fixture → {EVT_id: LOC_id}. See module docstring for the rationale.
MAPPINGS: dict[str, dict[str, str]] = {
    "a_fish_called_wanda": {
        "EVT_MRS_COADY_DIES_HEART_ATTACK": "LOC_COADY_FLAT",
        "EVT_GEORGE_CONVICTED": "LOC_OLD_BAILEY",
        "EVT_OTTO_BLOWN_OFF_DURING_TAKEOFF": "LOC_AIRPORT_RUNWAY",
    },
    "apocalypse_now": {
        "EVT_MR_CLEAN_DEATH": "LOC_PBR_RIVER",
        "EVT_CHIEF_SPEARED": "LOC_PBR_RIVER",
        "EVT_UTT_DOSSIER_PROFILES_KURTZ": "LOC_PBR_RIVER",
    },
    "frankenstein": {
        "EVT_CAROLINE_DIES": "LOC_GENEVA",
        "EVT_JUSTINE_EXECUTED": "LOC_GENEVA",
        "EVT_ALPHONSE_DIES": "LOC_GENEVA",
        "EVT_UTT_VICTOR_NOTES_REVEAL_ORIGIN": "LOC_DELACEY_COTTAGE",
    },
    "gone_girl": {
        "EVT_MEDIA_TURNS": "LOC_MEDIA_CIRCUS",
        "EVT_AMY_ROBBED": "LOC_HIDEOUT_OZARKS",
    },
    "great_expectations": {
        "EVT_CONVICTS_RECAPTURED": "LOC_KENT_MARSHES",
        "EVT_MRS_JOE_DIES": "LOC_FORGE",
        "EVT_HAVISHAM_BURNS": "LOC_SATIS_HOUSE",
        "EVT_MAGWITCH_DIES": "LOC_NEWGATE",
        "EVT_DRUMMLE_DIES": "LOC_LONDON",
        "EVT_HAVISHAM_BEQUEST_TO_POCKETS": "LOC_JAGGERS_OFFICE",
    },
    "macbeth": {
        # Scone is canonical for crownings but not in the fixture's location
        # set; fall back to Forres (royal court) and Dunsinane (Macbeth's
        # final seat) which are the in-world ceremonial venues.
        "EVT_MACBETH_CROWNED": "LOC_FORRES_COURT",
        "EVT_BANQUO_GHOST": "LOC_FORRES_COURT",
        "EVT_MACDUFF_LEARNS_OF_MASSACRE": "LOC_ENGLAND",
        "EVT_LADY_MACBETH_SLEEPWALKING": "LOC_DUNSINANE_CASTLE",
        "EVT_LADY_MACBETH_DEATH": "LOC_DUNSINANE_CASTLE",
        "EVT_MALCOLM_CROWNED": "LOC_DUNSINANE_CASTLE",
    },
    "nineteen_eighty_four": {
        "EVT_PARSONS_DENOUNCED": "LOC_MINILOVE",
    },
    "reservoir_dogs": {
        "EVT_ORANGE_SHOT": "LOC_ORANGE_CAR",
        # Blue is killed off-screen in flight from the heist; the diamond
        # store is the closest in-world anchor.
        "EVT_BLUE_KILLED_OFFSCREEN": "LOC_DIAMOND_STORE",
    },
    "romeo_and_juliet": {
        "EVT_PLAGUE_QUARANTINE_FRIAR_JOHN": "LOC_VERONA_STREETS",
        "EVT_JULIET_AWAKES": "LOC_CAPULET_CRYPT",
    },
    "the_lion_the_witch_and_the_wardrobe": {
        "EVT_SPRING_THAW": "LOC_LANTERN_WASTE",
        "EVT_ASLAN_RESURRECTION": "LOC_STONE_TABLE",
        "EVT_CORONATION": "LOC_CAIR_PARAVEL",
    },
    "tinker_tailor_soldier_spy": {
        "EVT_PRIDEAUX_SHOT": "LOC_HUNGARY",
        "EVT_PRIDEAUX_CAPTURED": "LOC_HUNGARY",
        "EVT_PRIDEAUX_KILLED_FALSE": "LOC_THE_CIRCUS",
        "EVT_CONTROL_FORCED_RETIREMENT": "LOC_THE_CIRCUS",
        "EVT_ALLELINE_BECOMES_CHIEF": "LOC_THE_CIRCUS",
        "EVT_CONTROL_DIES": "LOC_ENGLAND_EXILE",
        "EVT_BORIS_MURDERED": "LOC_ISTANBUL",
        "EVT_IRINA_CAPTURED": "LOC_ISTANBUL",
        "EVT_SMILEY_RESTORED_CHIEF": "LOC_THE_CIRCUS",
    },
    "wuthering_heights": {
        "EVT_OLD_EARNSHAW_DIES": "LOC_WUTHERING_HEIGHTS",
        "EVT_FRANCES_DIES": "LOC_WUTHERING_HEIGHTS",
        "EVT_LINTON_PARENTS_DIE": "LOC_THRUSHCROSS_GRANGE",
        "EVT_CATHERINE_DIES": "LOC_THRUSHCROSS_GRANGE",
        "EVT_HINDLEY_DIES": "LOC_WUTHERING_HEIGHTS",
        # EVT_ISABELLA_DIES intentionally omitted — "in the south", no
        # in-world LOC fits.
        "EVT_EDGAR_DIES": "LOC_THRUSHCROSS_GRANGE",
        "EVT_LINTON_DIES": "LOC_WUTHERING_HEIGHTS",
        "EVT_HEATHCLIFF_DIES": "LOC_WUTHERING_HEIGHTS",
        "EVT_UTT_LOCKWOOD_READS_DIARY": "LOC_WUTHERING_HEIGHTS",
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    grand_applied = 0
    for world, mapping in MAPPINGS.items():
        path = WORLDS_DIR / f"{world}.py"
        src = path.read_text(encoding="utf-8")
        module = cst.parse_module(src)
        rewriter = _EventNodeRewriter(mapping)
        new_module = module.visit(rewriter)
        if not args.dry_run and rewriter.applied:
            path.write_text(new_module.code, encoding="utf-8")
        verb = "would patch" if args.dry_run else "patched"
        print(f"{world}: {verb} {len(rewriter.applied)}/{len(mapping)} events")
        missed = sorted(set(mapping) - rewriter.applied)
        for eid in missed:
            print(f"  ! not patched: {eid}")
        grand_applied += len(rewriter.applied)
    suffix = " (dry-run)" if args.dry_run else ""
    print(f"\nTotal: {grand_applied}{suffix}")


if __name__ == "__main__":
    main()
