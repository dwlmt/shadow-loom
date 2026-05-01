#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025-2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Run the ingestion pipeline on a sample plot and dump diagnostics."""

import json
import logging
import sys
import time
from pathlib import Path

# Setup logging to see pipeline progress
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)

from shadow_loom.ingestion import ExtractionConfig, run_extraction

def run_on_file(plot_path: str, output_dir: str = "pipeline_output"):
    """Run the pipeline on a single plot file and dump results."""
    text = Path(plot_path).read_text(encoding="utf-8")
    stem = Path(plot_path).stem
    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    config = ExtractionConfig(
        model="ollama:qwen3.6:27b",
        chunk_strategy="act_headings",
        fabula_time_spacing=100,  # match gold-standard fixtures
        output_retries=5,
        max_correction_retries=1,
    )

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Processing: {stem} ({len(text)} chars)", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    t0 = time.time()
    ws, report = run_extraction(text, config)
    elapsed = time.time() - t0

    # Dump world state JSON
    ws_path = out / f"{stem}_world_state.json"
    ws_path.write_text(ws.model_dump_json(indent=2), encoding="utf-8")

    # Dump validation report
    report_path = out / f"{stem}_validation.json"
    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    # Print summary
    n_locs = len(ws.locations)
    n_objs = len(ws.objects)
    n_ents = len(ws.entities)
    n_events = len(ws.events)
    n_causal = len(ws.causal_topology)
    n_spatial = len(ws.spatial_topology)
    n_channels = len(ws.channels)
    n_utterances = sum(1 for e in ws.events if e.event_type == "utterance")
    n_social = len(ws.social_topology)

    # Count causality types
    ct_counts = {}
    for e in ws.causal_topology:
        ct_counts[e.causality_type] = ct_counts.get(e.causality_type, 0) + 1

    # Count entities with state_timeline, beliefs, constants
    ents_with_timeline = sum(1 for e in ws.entities.values() if e.state_timeline)
    ents_with_beliefs = sum(1 for e in ws.entities.values() if e.beliefs)
    ents_with_constants = sum(1 for e in ws.entities.values() if e.constants)
    avg_traits = sum(len(e.traits) for e in ws.entities.values()) / max(n_ents, 1)

    # Validation stats
    errors = [i for i in report.issues if i.severity == "error"]
    warnings = [i for i in report.issues if i.severity == "warning"]

    summary = f"""
{'='*60}
PIPELINE RESULT: {stem}
{'='*60}
Elapsed: {elapsed:.1f}s

ONTOLOGY:
  Locations: {n_locs}
  Objects:   {n_objs}
  Entities:  {n_ents} (avg {avg_traits:.1f} traits)
    with state_timeline: {ents_with_timeline}/{n_ents}
    with beliefs:        {ents_with_beliefs}/{n_ents}
    with constants:      {ents_with_constants}/{n_ents}

EVENTS: {n_events}

TOPOLOGY:
  Causal edges:  {n_causal}
    {', '.join(f'{k}: {v}' for k, v in sorted(ct_counts.items()))}
  Spatial edges:  {n_spatial}
  Channels:       {n_channels}
  Utterances:     {n_utterances}
  Social edges:   {n_social}

VALIDATION: {'PASS' if report.is_valid else 'FAIL'}
  Errors:   {len(errors)}
  Warnings: {len(warnings)}
"""

    if errors:
        summary += "\nERRORS:\n"
        for e in errors:
            summary += f"  [{e.category}] {e.detail}\n"

    if warnings:
        summary += "\nWARNINGS:\n"
        for w in warnings:
            summary += f"  [{w.category}] {w.detail}\n"

    if report.suggestions:
        summary += "\nSUGGESTIONS:\n"
        for s in report.suggestions:
            summary += f"  - {s}\n"

    print(summary, file=sys.stderr)

    # Also dump summary to file
    (out / f"{stem}_summary.txt").write_text(summary, encoding="utf-8")

    return ws, report


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_pipeline.py <plot_file> [plot_file2 ...]", file=sys.stderr)
        sys.exit(1)

    for path in sys.argv[1:]:
        run_on_file(path)
