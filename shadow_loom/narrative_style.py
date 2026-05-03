# SPDX-FileCopyrightText: 2025-2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Source-text style profiler.

A small, deterministic heuristic that looks at the *form* of the
ingested text and produces a :class:`NarrativeStyle` profile. The
profile is then carried through the pipeline so the directive
assembler, renderer, and auditor all condition on the same source
register — a plot-summary seed produces summary-length output, a
short-story seed produces a fully written scene, and so on.

This module is intentionally LLM-free: all signals come from token
counts, sentence lengths, dialogue density, and a small bag of
summary-diction markers ("Later,", "In September,", "Days later,"
…). LLM-based refinement of the profile is left to a future iteration
— the goal here is to give every render a usable target.
"""

from __future__ import annotations

import re
from typing import Tuple

from shadow_loom.models import NarrativeStyle


_SUMMARY_MARKERS = (
    r"\bLater\b", r"\bAfterwards?\b", r"\bSubsequently\b",
    r"\bDays later\b", r"\bWeeks later\b", r"\bMonths later\b",
    r"\bYears later\b", r"\bMeanwhile\b", r"\bEventually\b",
    r"\bIn (?:January|February|March|April|May|June|July|"
    r"August|September|October|November|December)\b",
    r"\bThe next (?:day|morning|evening|week|month|year)\b",
    r"\bOne (?:day|night|morning|evening)\b",
    r"\bUpon (?:hearing|learning|discovering|realising|realizing)\b",
)
_SUMMARY_RE = re.compile("|".join(_SUMMARY_MARKERS))

_SCREENPLAY_RE = re.compile(
    r"^(INT\.|EXT\.|FADE IN:|FADE OUT|CUT TO:)", re.MULTILINE,
)
_VERSE_RE = re.compile(r"\n[ \t]*\n")  # heuristic; refined below
_DIALOGUE_RE = re.compile(r"[\"\u201c\u201d\u2018\u2019].+?[\"\u201c\u201d\u2018\u2019]")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# --- Non-narrative / discursive form detectors ----------------------
# Each detector is a tuple of (compiled regex, "weight" hits required).
# We keep these intentionally cheap and explicit so the profiler stays
# LLM-free.

_NEWS_MARKERS = (
    r"\b(?:Reuters|AP|AFP|Bloomberg|BBC|CNN|Associated Press)\b",
    r"\bsaid (?:in a statement|on (?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday))\b",
    r"\baccording to (?:officials|sources|analysts|the (?:report|statement))\b",
    r"\bspokesperson\b", r"\bpress (?:conference|release|briefing)\b",
    r"\b(?:filed|reported|reporting) (?:from|in) [A-Z][a-z]+\b",
    # Datelines like "WASHINGTON —" or "LONDON, May 3 (Reuters)".
    r"^[A-Z][A-Z\s]{2,}[\u2014\-,]",
)
_NEWS_RE = re.compile("|".join(_NEWS_MARKERS), re.MULTILINE)

_HISTORY_MARKERS = (
    r"\bin (?:1[0-9]{3}|20[0-2][0-9])\b",        # year refs
    r"\bcentury\b", r"\bdynasty\b", r"\bempire\b",
    r"\bhistorian[s]?\b", r"\barchives?\b",
    r"\b(?:reign|rule|regime) of\b",
    r"\b(?:treaty|battle|siege|war) of\b",
    r"\bBC(?:E)?\b|\bAD\b|\bCE\b",
)
_HISTORY_RE = re.compile("|".join(_HISTORY_MARKERS))

_THOUGHT_EXPERIMENT_MARKERS = (
    r"\b(?:imagine|suppose|consider|posit|assume) (?:that|a|an|the|you)\b",
    r"\bthought experiment\b",
    r"\bif (?:we|you|one) (?:were|imagine|suppose|grant)\b",
    r"\bwhat (?:if|would happen)\b",
    r"\bgedanken\w*\b",
    # Classic philosophical setups.
    r"\b(?:trolley|brain in a vat|veil of ignorance|Chinese room|"
    r"twin earth|teletransport)\b",
)
_THOUGHT_EXPERIMENT_RE = re.compile(
    "|".join(_THOUGHT_EXPERIMENT_MARKERS), re.IGNORECASE,
)

_ESSAY_STRONG_MARKERS = (
    # Explicit thesis / first-person argumentative declarations. These
    # almost never appear in fiction plot summaries.
    r"\bI (?:argue|claim|contend|maintain|propose|will (?:argue|show))\b",
    r"\bin this (?:essay|paper|article|piece)\b",
    r"\b(?:my|the) thesis\b",
    r"\bthe (?:central |main )?argument (?:is|of this)\b",
    r"\b(?:in conclusion|to conclude|in summary)\b",
)
_ESSAY_STRONG_RE = re.compile("|".join(_ESSAY_STRONG_MARKERS), re.IGNORECASE)

_ESSAY_WEAK_MARKERS = (
    # Discursive signposting. These DO appear in plot summaries, so
    # they only count when at least one strong marker is also present.
    r"\b(?:firstly|secondly|moreover|furthermore|however|"
    r"nevertheless|therefore|thus|hence)\b",
    r"\b(?:on the one hand|on the other hand)\b",
)
_ESSAY_WEAK_RE = re.compile("|".join(_ESSAY_WEAK_MARKERS), re.IGNORECASE)

_CASE_STUDY_MARKERS = (
    r"\bcase study\b",
    r"\b(?:patient|client|subject|the company|the firm|the team)\b",
    r"\b(?:background|methodology|findings|outcomes?|"
    r"recommendations?|lessons learned)\b",
    r"\b(?:Q[1-4] \d{4}|fiscal year)\b",
)
_CASE_STUDY_RE = re.compile("|".join(_CASE_STUDY_MARKERS), re.IGNORECASE)

# Speaker-tag transcripts: "ALICE:" or "Interviewer:" at line starts.
_TRANSCRIPT_RE = re.compile(
    r"^[A-Z][A-Z a-z\.\-]{1,40}:\s", re.MULTILINE,
)


def _sentence_stats(text: str) -> Tuple[int, float]:
    """Return (sentence_count, mean_words_per_sentence)."""
    sentences = [s for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    if not sentences:
        return 0, 0.0
    word_counts = [len(s.split()) for s in sentences]
    return len(sentences), sum(word_counts) / len(sentences)


def _dialogue_fraction(text: str) -> float:
    """Crude fraction of characters inside quoted dialogue."""
    if not text:
        return 0.0
    quoted_chars = sum(len(m.group(0)) for m in _DIALOGUE_RE.finditer(text))
    return quoted_chars / max(len(text), 1)


def _looks_like_verse(text: str) -> bool:
    """Heuristic: many short lines and few trailing periods."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 8:
        return False
    short_lines = sum(1 for ln in lines if len(ln.split()) <= 10)
    period_lines = sum(1 for ln in lines if ln.rstrip().endswith((".", "!", "?")))
    return short_lines / len(lines) > 0.7 and period_lines / len(lines) < 0.4


def _exemplar(text: str, max_chars: int = 600) -> str:
    """Pull a representative snippet for tone-matching."""
    snippet = text.strip()
    if len(snippet) <= max_chars:
        return snippet
    # Prefer cutting on a sentence boundary.
    cut = snippet[:max_chars]
    last_period = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"))
    if last_period > max_chars // 2:
        return cut[: last_period + 1].strip()
    return cut.strip() + "…"


def infer_narrative_style(text: str) -> NarrativeStyle:
    """Produce a :class:`NarrativeStyle` profile for ``text``.

    The profile is a best-effort summary of the source register based
    on token/sentence/dialogue statistics. It is *not* authoritative —
    callers (or a future LLM step) may overwrite individual fields.
    """
    if not text or not text.strip():
        return NarrativeStyle()

    word_count = len(text.split())
    sent_count, mean_sent_words = _sentence_stats(text)
    dialogue_frac = _dialogue_fraction(text)
    summary_marker_hits = len(_SUMMARY_RE.findall(text))
    summary_density = summary_marker_hits / max(sent_count, 1)

    # Non-narrative / discursive scoring -------------------------------
    # Each detector contributes a normalised score; the highest non-zero
    # score above its threshold wins. We compute these up front so the
    # narrative branches only run when no non-narrative form fits.
    head = text[:8000]  # cheap window for marker scans
    news_hits = len(_NEWS_RE.findall(head))
    history_hits = len(_HISTORY_RE.findall(head))
    thought_hits = len(_THOUGHT_EXPERIMENT_RE.findall(head))
    essay_hits = len(_ESSAY_WEAK_RE.findall(head))
    essay_strong_hits = len(_ESSAY_STRONG_RE.findall(head))
    case_hits = len(_CASE_STUDY_RE.findall(head))
    transcript_lines = len(_TRANSCRIPT_RE.findall(head))

    fmt: str
    density: str
    target_min: int
    target_max: int
    voice_parts: list[str] = []

    # Non-narrative form detection -------------------------------------
    # Order matters: transcript and screenplay are structural and win
    # outright; the rest compete by hit-density.
    if transcript_lines >= 4 and dialogue_frac < 0.10:
        # Q&A / interview / dialectic transcript with explicit speaker
        # tags but no quoted dialogue prose.
        fmt = "transcript"
        density = "moderate"
        target_min = max(200, min(word_count, 400))
        target_max = max(target_min + 200, min(int(word_count * 1.2), 1500))
        voice_parts.append(
            "speaker-tagged transcript; alternating turns with explicit "
            "SPEAKER: prefixes; preserve register of each speaker"
        )
    elif _SCREENPLAY_RE.search(text):
        fmt = "screenplay"
        density = "moderate"
        target_min, target_max = 300, 1200
        voice_parts.append("screenplay format with sluglines and action lines")
    elif _looks_like_verse(text):
        fmt = "verse"
        density = "sparse"
        target_min, target_max = 80, 400
        voice_parts.append("verse with short, line-broken cadence")
    elif news_hits >= 2 and dialogue_frac < 0.15:
        # Current-affairs / journalism: datelines, attributions, and
        # named outlets. Rendered length tracks the source.
        fmt = "news_article"
        density = "moderate"
        target_min = max(200, min(word_count, 400))
        target_max = max(target_min + 200, min(int(word_count * 1.2), 1200))
        voice_parts.append(
            "third-person past-tense journalistic register; inverted-"
            "pyramid structure (lede first); attributed quotes and "
            "named sources; neutral diction"
        )
    elif (
        thought_hits >= 2
        and thought_hits >= essay_hits  # essay-style signposting wins ties
    ):
        # Philosophy / thought experiment: hypothetical framings,
        # second-person address, classic setups.
        fmt = "thought_experiment"
        density = "moderate"
        target_min = max(200, min(word_count, 400))
        target_max = max(target_min + 200, min(max(word_count, 600), 1500))
        voice_parts.append(
            "discursive philosophical register; hypothetical framing "
            "(\"Suppose…\", \"Imagine…\"); second-person or impersonal "
            "address; analytical rather than dramatic"
        )
    elif case_hits >= 3 and dialogue_frac < 0.10:
        fmt = "case_study"
        density = "moderate"
        target_min = max(300, min(word_count, 500))
        target_max = max(target_min + 300, min(int(word_count * 1.2), 1800))
        voice_parts.append(
            "third-person analytical case-study register; "
            "background/methodology/findings/recommendations structure; "
            "neutral diction with concrete particulars"
        )
    elif essay_hits >= 3 and essay_strong_hits >= 1 and dialogue_frac < 0.10:
        fmt = "essay"
        density = "moderate"
        target_min = max(300, min(word_count, 500))
        target_max = max(target_min + 300, min(int(word_count * 1.2), 2000))
        voice_parts.append(
            "first-person essayistic register; explicit thesis and "
            "argument structure; signposted reasoning (firstly/however/"
            "therefore); cited or paraphrased examples"
        )
    elif (
        history_hits >= 3
        and dialogue_frac < 0.10
    ):
        fmt = "historical_account"
        density = "moderate"
        target_min = max(200, min(word_count, 500))
        target_max = max(target_min + 200, min(int(max(word_count, 600) * 1.2), 2000))
        voice_parts.append(
            "third-person past-tense historiographic register; dated "
            "events with named actors and places; analytical framing "
            "rather than scene-by-scene dramatisation"
        )
    # Format detection (narrative fallback) ----------------------------
    elif word_count < 200:
        fmt = "outline"
        density = "sparse"
        target_min, target_max = 80, 300
        voice_parts.append("terse outline of beats")
    elif summary_density >= 0.20 and dialogue_frac < 0.05:
        # Wikipedia-style plot summary: many temporal markers,
        # essentially no dialogue.
        fmt = "plot_summary"
        density = "sparse"
        # Match the source order of magnitude (~1× source length per
        # render), capped to avoid runaway outputs.
        target_min = max(120, min(word_count, 250))
        target_max = max(target_min + 80, min(int(word_count * 1.2), 1000))
        voice_parts.append(
            "third-person past-tense plot summary; condensed beat-by-beat "
            "diction; no quoted dialogue; uses temporal connectors "
            "(\"Later,\", \"The next day,\")"
        )
    elif word_count < 1500 and dialogue_frac < 0.05:
        fmt = "synopsis"
        density = "sparse"
        target_min = max(150, word_count // 3)
        target_max = max(target_min + 100, min(word_count, 600))
        voice_parts.append("synoptic narration; no dialogue; condensed scene description")
    elif word_count < 4000:
        fmt = "scene" if dialogue_frac >= 0.05 else "short_story"
        density = "moderate"
        target_min, target_max = 600, 1800
        voice_parts.append(
            "scene-level prose with some dialogue and sensory detail"
            if dialogue_frac >= 0.05
            else "short-story prose, mostly narration"
        )
    else:
        fmt = "novel_excerpt"
        density = "rich"
        target_min, target_max = 800, 2500
        voice_parts.append(
            "novelistic prose with full sensory texture, interiority, "
            "and varied sentence rhythm"
        )

    # POV / tense hints (lightweight) ---------------------------------
    lower = text[:4000].lower()
    if re.search(r"\bi (?:was|am|had|saw|felt|thought)\b", lower):
        voice_parts.append("first-person POV")
    elif re.search(r"\byou (?:are|were|see|feel)\b", lower):
        voice_parts.append("second-person POV")
    else:
        voice_parts.append("third-person POV")

    if re.search(r"\b(?:was|were|had|said|went|came|saw|felt|thought)\b", lower):
        voice_parts.append("past tense")
    elif re.search(r"\b(?:is|are|sees|feels|walks|says)\b", lower):
        voice_parts.append("present tense")

    if mean_sent_words and mean_sent_words < 12:
        voice_parts.append("short, clipped sentences")
    elif mean_sent_words and mean_sent_words > 24:
        voice_parts.append("long, flowing sentences")

    voice = "; ".join(voice_parts)

    return NarrativeStyle(
        format=fmt,  # type: ignore[arg-type]
        target_word_min=int(target_min),
        target_word_max=int(target_max),
        prose_density=density,  # type: ignore[arg-type]
        voice=voice,
        style_exemplar=_exemplar(text),
        source_word_count=word_count,
    )


def format_narrative_style_block(style: NarrativeStyle, *, header: str = "STYLE FIDELITY (HARD)") -> str:
    """Render a NarrativeStyle as a prompt section.

    Used by both the rendering prompt (Step 10) and the audit prompt
    (Step 11) so the same fidelity contract drives generation and
    evaluation.
    """
    lines = [f"=== {header} ==="]
    lines.append(
        "The rendered prose MUST mirror the source text's *form* — "
        "matching its length envelope, prose density, and narrative "
        "voice. Do not inflate a plot summary into a short story, "
        "and do not condense a short story into a synopsis."
    )
    lines.append(f"  Source format: {style.format}")
    lines.append(
        f"  Target length: {style.target_word_min}\u2013{style.target_word_max} words"
    )
    lines.append(f"  Prose density: {style.prose_density}")
    if style.voice:
        lines.append(f"  Voice: {style.voice}")
    if style.source_word_count is not None:
        lines.append(f"  Source word count: {style.source_word_count}")
    if style.style_exemplar:
        lines.append("  Style exemplar (mirror cadence and diction; do NOT copy content):")
        for ln in style.style_exemplar.splitlines():
            lines.append(f"    | {ln}")
    return "\n".join(lines)
