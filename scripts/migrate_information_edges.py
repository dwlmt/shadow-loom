"""One-shot migrator: rewrite legacy ``information_topology=[InformationEdge(...)]``
blocks in ``example_worlds/*.py`` to the new ``channels={CHN_X: Channel(...)}``
form, plus optionally an ``EVT_*`` utterance event for one-shot messages.

The rules:
  * If ``terminated_at_fabula == established_at_fabula`` and not None → the
    edge represents a single discrete message. We emit a Channel that
    spans only that instant AND an ``EventNode(event_type="utterance")``
    with ``via_channel_id`` set.
  * Otherwise (including ``terminated_at_fabula = None``) → standing
    capability; emit only a Channel.

The script is idempotent: it skips files that no longer reference
``InformationEdge`` in their imports.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "example_worlds"


def _slug_to_chn_id(source: str, targets: list[str], medium: str, idx: int) -> str:
    base = re.sub(r"[^A-Z0-9]+", "_", medium.upper()).strip("_") or "CHANNEL"
    return f"CHN_{base}_{idx:02d}"


def _slug_to_evt_id(source: str, targets: list[str], medium: str, fabula: int) -> str:
    base = re.sub(r"[^A-Z0-9]+", "_", medium.upper()).strip("_") or "UTTERANCE"
    src = source.replace("ENT_", "").replace("OBJ_", "")
    return f"EVT_UTT_{base}_{src}_{fabula}"


def _migrate_file(path: Path) -> bool:
    src = path.read_text()
    if "InformationEdge" not in src:
        return False

    tree = ast.parse(src)

    # Find the keyword `information_topology=[...]` inside the WorldStateV1 call.
    info_keyword = None
    ws_call = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "WorldStateV1"
        ):
            ws_call = node
            for kw in node.keywords:
                if kw.arg == "information_topology":
                    info_keyword = kw
                    break
            break

    if info_keyword is None or not isinstance(info_keyword.value, ast.List):
        return False

    channels: list[tuple[str, dict]] = []
    utterances: list[tuple[str, dict]] = []

    for idx, elt in enumerate(info_keyword.value.elts, start=1):
        if not (isinstance(elt, ast.Call) and isinstance(elt.func, ast.Name)
                and elt.func.id == "InformationEdge"):
            continue
        kw_map = {k.arg: k.value for k in elt.keywords}
        try:
            source_id = ast.literal_eval(kw_map["source_id"])
            target_ids = ast.literal_eval(kw_map["target_ids"])
            medium = ast.literal_eval(kw_map["medium"])
            established = ast.literal_eval(kw_map.get(
                "established_at_fabula", ast.Constant(0),
            ))
            terminated_node = kw_map.get("terminated_at_fabula")
            terminated = (
                ast.literal_eval(terminated_node)
                if terminated_node is not None else None
            )
            evidence = ast.literal_eval(kw_map.get(
                "evidence_strength", ast.Constant("moderate"),
            ))
            is_encrypted = ast.literal_eval(kw_map.get(
                "is_encrypted", ast.Constant(False),
            ))
            syuzhet = ast.literal_eval(kw_map.get(
                "discovered_at_syuzhet", ast.Constant(0),
            ))
        except Exception:
            continue

        participants = [source_id] + [t for t in target_ids if t != source_id]
        intelligibility = (
            {t: 0.2 for t in target_ids} if is_encrypted else {}
        )
        chn_id = _slug_to_chn_id(source_id, target_ids, medium, idx)
        ch_dict = {
            "id": chn_id,
            "name": medium.replace("_", " "),
            "medium": medium,
            "participant_ids": participants,
            "directionality": "broadcast" if len(target_ids) > 1 else "duplex",
            "intelligibility": intelligibility,
            "established_at_fabula": established,
            "terminated_at_fabula": terminated,
            "evidence_strength": evidence,
        }
        channels.append((chn_id, ch_dict))

        # If the edge collapses to a single instant, also emit an utterance event.
        if terminated is not None and terminated == established:
            evt_id = _slug_to_evt_id(source_id, target_ids, medium, established)
            utt = {
                "id": evt_id,
                "event_type": "utterance",
                "description": f"{source_id} → {', '.join(target_ids)} ({medium})",
                "speaker_id": source_id,
                "addressee_ids": list(target_ids),
                "actor_ids": [source_id],
                "target_ids": [],
                "via_channel_id": chn_id,
                "truth_value": "false" if "lie" in medium.lower()
                              or "false" in medium.lower()
                              or "fabricat" in medium.lower() else "true",
                "fabula_time": established,
                "syuzhet_index": syuzhet,
            }
            utterances.append((evt_id, utt))

    # Render the replacement code (keep simple, deterministic formatting).
    def _render_dict_lit(d: dict) -> str:
        # Lay out one key per line for readability.
        parts = []
        for k, v in d.items():
            parts.append(f"            {k!r}: {v!r}")
        return "{\n" + ",\n".join(parts) + ",\n        }"

    channels_text = "channels={\n"
    for cid, cd in channels:
        # Build Channel(...) call with kwargs for clarity.
        kwargs = ", ".join(f"{k}={v!r}" for k, v in cd.items())
        channels_text += f"        {cid!r}: Channel({kwargs}),\n"
    channels_text += "    }"

    # Splice into source. Use raw character ranges from AST.
    info_segment = ast.get_source_segment(src, info_keyword)
    if info_segment is None:
        return False

    new_src = src.replace(
        f"information_topology={info_segment.split('=', 1)[1]}",
        channels_text,
        1,
    )
    if new_src == src:
        # Fallback: replace the whole keyword segment.
        new_src = src.replace(info_segment, channels_text, 1)
    if new_src == src:
        return False

    # Update the import line: drop InformationEdge, add Channel.
    new_src = re.sub(
        r"\bInformationEdge\b(,\s*)?",
        "",
        new_src,
    )
    if "Channel" not in new_src.split("from shadow_loom.models import", 1)[1].split("\n", 1)[0]:
        new_src = re.sub(
            r"(from shadow_loom\.models import\s*\()",
            r"\1\n    Channel,",
            new_src,
            count=1,
        )

    # Append utterance events to the existing events=[...] list (best-effort).
    if utterances:
        utt_lines = []
        for eid, ud in utterances:
            kwargs = ", ".join(f"{k}={v!r}" for k, v in ud.items())
            utt_lines.append(f"        EventNode({kwargs}),")
        block = "\n".join(utt_lines)
        # Find the events=[...] block and inject before its closing bracket.
        m = re.search(r"events=\[", new_src)
        if m:
            depth = 0
            i = m.end()
            while i < len(new_src):
                c = new_src[i]
                if c == "[":
                    depth += 1
                elif c == "]":
                    if depth == 0:
                        new_src = new_src[:i] + "\n" + block + "\n    " + new_src[i:]
                        break
                    depth -= 1
                i += 1

    path.write_text(new_src)
    return True


def main() -> None:
    for py in sorted(ROOT.glob("*.py")):
        if py.name.startswith("__"):
            continue
        try:
            changed = _migrate_file(py)
        except Exception as exc:
            print(f"[FAIL] {py.name}: {exc}")
            continue
        print(f"[{'OK ' if changed else '--'}] {py.name}")


if __name__ == "__main__":
    main()
