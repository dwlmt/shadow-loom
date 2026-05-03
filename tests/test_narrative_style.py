# SPDX-FileCopyrightText: 2025-2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the narrative-style profiler."""

from __future__ import annotations

from shadow_loom.models import NarrativeStyle
from shadow_loom.narrative_style import (
    format_narrative_style_block,
    infer_narrative_style,
)


PLOT_SUMMARY = (
    "In spring 1922, Nick Carraway, a Yale alumnus and World War I "
    "veteran, journeys to New York to obtain employment as a bond "
    "salesman. He rents a bungalow in West Egg next to Jay Gatsby's "
    "estate. One evening, Nick dines with his cousin Daisy Buchanan "
    "and her husband Tom in East Egg. Jordan Baker confides that Tom "
    "keeps a mistress named Myrtle Wilson. Days later, Nick attends "
    "one of Gatsby's elaborate parties and meets the host. Later, "
    "Gatsby asks Nick to stage a reunion with Daisy. The next day, "
    "Gatsby and Daisy reunite and embark on an affair. In September, "
    "Tom discovers the affair and confronts Gatsby at the Plaza Hotel. "
    "Upon hearing of Gatsby's bootlegging fortune, Daisy hesitates, "
    "and Tom scornfully orders Gatsby to drive her home. The next "
    "morning, Daisy strikes Myrtle with Gatsby's car. Eventually, "
    "George Wilson is told the car belonged to Gatsby. The next "
    "evening, George shoots Gatsby in his swimming pool and then "
    "kills himself. Days later, Nick organises the funeral, which "
    "almost no one attends. Months later, Nick returns to the Midwest, "
    "disillusioned by the moral emptiness of the Eastern elite."
)

SHORT_STORY = (
    "Nick stepped onto the wooden dock as the green light shivered "
    "across the bay. \"You came,\" Gatsby said quietly. He stood a "
    "little apart, hands in his pockets, watching the water. The night "
    "air smelled of salt and cut grass and the faint, expensive smoke "
    "of someone else's cigar. Nick wanted to ask the obvious question "
    "and could not. Gatsby smiled, the way a man smiles when he has "
    "rehearsed the moment for years. \"Daisy is just across there,\" "
    "he said, lifting one finger toward the dark. The light blinked, "
    "and blinked, and the moment held them both for a long time before "
    "either of them dared to move."
) * 4  # ~600 words of dialogue-bearing scene prose


def test_infer_plot_summary():
    # Use a real sample plot whose form is unambiguously a Wikipedia-
    # style summary so the heuristic has enough material to work on.
    from pathlib import Path
    src = Path(__file__).resolve().parent.parent / "sample_plots" / "great_gatsby.txt"
    ns = infer_narrative_style(src.read_text())
    assert ns.format == "plot_summary"
    assert ns.prose_density == "sparse"
    # Length budget tracks the source order of magnitude.
    assert ns.target_word_min < ns.target_word_max
    assert ns.target_word_max <= 1000
    assert "third-person" in ns.voice
    assert "past tense" in ns.voice
    assert ns.style_exemplar
    assert ns.source_word_count is not None


def test_infer_short_story():
    ns = infer_narrative_style(SHORT_STORY)
    # Either short_story or scene depending on dialogue density.
    assert ns.format in {"short_story", "scene"}
    assert ns.prose_density == "moderate"
    assert ns.target_word_max >= 1000


def test_infer_outline_short_text():
    ns = infer_narrative_style("A man arrives. He leaves.")
    assert ns.format == "outline"
    assert ns.prose_density == "sparse"


def test_infer_empty_text():
    ns = infer_narrative_style("")
    assert isinstance(ns, NarrativeStyle)
    assert ns.format == "unknown"


def test_format_block_includes_targets_and_density():
    ns = infer_narrative_style(PLOT_SUMMARY)
    block = format_narrative_style_block(ns)
    assert "STYLE FIDELITY (HARD)" in block
    assert f"{ns.target_word_min}" in block
    assert f"{ns.target_word_max}" in block
    assert ns.prose_density in block
    assert "Style exemplar" in block


def test_word_range_validation():
    import pytest

    with pytest.raises(ValueError):
        NarrativeStyle(target_word_min=500, target_word_max=200)


def test_creative_brief_carries_style():
    """The CreativeBrief must accept and round-trip a NarrativeStyle."""
    from shadow_loom.directive_assembly import CreativeBrief

    ns = NarrativeStyle(
        format="plot_summary",
        target_word_min=200,
        target_word_max=600,
        prose_density="sparse",
        voice="third-person past tense",
    )
    brief = CreativeBrief(
        target_effect="observation",
        target_entities=["ENT_NICK"],
        narrative_style=ns,
    )
    assert brief.narrative_style is not None
    assert brief.narrative_style.format == "plot_summary"
    assert brief.narrative_style.target_word_max == 600


def test_assemble_rendering_prompt_includes_style_block():
    from shadow_loom.directive_assembly import CreativeBrief
    from shadow_loom.generation import assemble_rendering_prompt

    ns = NarrativeStyle(
        format="plot_summary",
        target_word_min=200,
        target_word_max=500,
        prose_density="sparse",
        voice="third-person past tense plot summary",
    )
    brief = CreativeBrief(
        target_effect="observation",
        target_entities=["ENT_X"],
        narrative_style=ns,
    )
    prompt = assemble_rendering_prompt(brief)
    assert "STYLE FIDELITY" in prompt
    assert "200" in prompt and "500" in prompt
    assert "sparse" in prompt


# ---------------------------------------------------------------------
# Non-narrative / discursive form detection
# ---------------------------------------------------------------------

NEWS_ARTICLE = (
    "WASHINGTON, May 3 (Reuters) — The Treasury Department on Monday "
    "announced new sanctions targeting a network of shell companies "
    "alleged to have funnelled cryptocurrency to a sanctioned regime, "
    "according to officials familiar with the matter. \n\n"
    "A spokesperson for the agency said in a statement that the "
    "designations would take immediate effect. Analysts at the "
    "Brookings Institution said the move was the broadest enforcement "
    "action of the year. \n\n"
    "The press conference, held shortly after markets closed, drew "
    "questions about the reach of the new authorities. According to "
    "sources, additional designations are expected within weeks."
)

HISTORICAL_ACCOUNT = (
    "In 1453, the fall of Constantinople brought an end to the "
    "Byzantine Empire and reshaped the political map of the eastern "
    "Mediterranean. Sultan Mehmed II, then twenty-one, had spent the "
    "previous winter assembling siege artillery on a scale not "
    "previously seen in the region. Historians have noted that the "
    "treaty of 1454, signed in the months that followed, formalised "
    "Ottoman control over much of the surrounding territory. The "
    "reign of Constantine XI, last of the Palaiologos dynasty, ended "
    "with his death during the final assault. The siege of the city "
    "was the culmination of a war of conquest begun by his father, "
    "Murad II. Subsequent rulers expanded the empire across three "
    "continents over the following century."
)

THOUGHT_EXPERIMENT = (
    "Imagine that you are standing beside a railway switch. A runaway "
    "trolley is hurtling down the main track toward five workers who "
    "cannot escape. Suppose that you can divert the trolley onto a "
    "side track, where it will strike and kill one worker instead. "
    "What if, instead of a switch, you were standing on a footbridge "
    "above the track, and the only way to stop the trolley were to "
    "push a heavy stranger onto the rails? Consider whether your "
    "moral intuition shifts between the two cases. The classic "
    "trolley problem is a thought experiment designed to probe "
    "exactly this asymmetry. If we were to grant that consequences "
    "alone determine permissibility, the two cases must be equivalent."
)

ESSAY = (
    "In this essay I argue that the modern attention economy has "
    "fundamentally altered the conditions under which democratic "
    "deliberation can take place. The thesis is straightforward: "
    "when the dominant medium rewards engagement over accuracy, "
    "the epistemic commons erodes. Firstly, consider the structural "
    "incentives at work. Secondly, examine the empirical record of "
    "the past decade. However, one might object that earlier media "
    "regimes faced analogous pressures. Nevertheless, the scale and "
    "personalisation of contemporary platforms mark a genuine "
    "discontinuity. Therefore, in conclusion, reform of the "
    "underlying incentive structure — not merely of content "
    "moderation — is the necessary response."
)

CASE_STUDY = (
    "Case study: Northwind Logistics. Background: the company, a "
    "mid-sized regional carrier, faced a 30% rise in fuel costs "
    "during fiscal year Q3 2024. Methodology: the team conducted "
    "structured interviews with twelve dispatchers and reviewed "
    "route-level telemetry across the firm's operations. Findings: "
    "the analysis identified three principal drivers of cost "
    "overrun. Outcomes: targeted routing changes reduced "
    "consumption by 11% within two quarters. Recommendations: the "
    "patient — a similarly structured regional carrier — should "
    "adopt the same instrumentation. Lessons learned were "
    "circulated to subject teams across the firm."
)

TRANSCRIPT = (
    "INTERVIEWER: Could you describe what happened on the morning of "
    "the launch?\n"
    "ALICE: I arrived at the control room just before six.\n"
    "INTERVIEWER: And the others?\n"
    "ALICE: Bob and Carol were already there. They had been running "
    "diagnostics overnight.\n"
    "BOB: That's correct. We had completed the third pass by five "
    "thirty.\n"
    "INTERVIEWER: When did you first notice the anomaly?\n"
    "CAROL: At six twelve. The telemetry feed dropped a frame.\n"
)


def test_infer_news_article():
    ns = infer_narrative_style(NEWS_ARTICLE)
    assert ns.format == "news_article"
    assert ns.prose_density == "moderate"
    assert "journalistic" in ns.voice


def test_infer_historical_account():
    ns = infer_narrative_style(HISTORICAL_ACCOUNT)
    assert ns.format == "historical_account"
    assert "historiographic" in ns.voice or "historiographic" in ns.voice.lower()


def test_infer_thought_experiment():
    ns = infer_narrative_style(THOUGHT_EXPERIMENT)
    assert ns.format == "thought_experiment"
    assert "philosophical" in ns.voice


def test_infer_essay():
    ns = infer_narrative_style(ESSAY)
    assert ns.format == "essay"
    assert "essayistic" in ns.voice or "thesis" in ns.voice


def test_infer_case_study():
    ns = infer_narrative_style(CASE_STUDY)
    assert ns.format == "case_study"


def test_infer_transcript():
    ns = infer_narrative_style(TRANSCRIPT)
    assert ns.format == "transcript"
    assert "transcript" in ns.voice or "speaker" in ns.voice.lower()


def test_non_narrative_does_not_misfire_on_fiction():
    """A pure short-story passage must not be classified as essay/news/etc."""
    ns = infer_narrative_style(SHORT_STORY)
    assert ns.format in {"short_story", "scene"}
