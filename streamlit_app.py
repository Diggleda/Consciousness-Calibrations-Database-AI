"""Streamlit UI for the CCD AI."""

from __future__ import annotations

from typing import Dict, List, Tuple

import streamlit as st

from ccd_ai import (
    MatchDict,
    average_calibration,
    calibration_range,
    database,
    format_entry,
    geometric_mean_entries,
    run_pipeline,
)


st.set_page_config(page_title="Truth is Free™ — a Content Mapping Tool™", layout="wide")
st.title("Truth is Free™")
st.markdown("### A Content Mapping Tool™")
st.markdown(
    """
    <style>
        body, .stApp, .main {
            background-color: #ffffff;
            color: #000000;
        }
        .stMarkdown, .stMarkdown p, label, .stCaption, .stExpander, .stExpander div, .stText {
            color: #000000 !important;
        }
        .stTextArea textarea, .stTextInput input, .stSelectbox div[data-baseweb="select"],
        .stMultiSelect div[data-baseweb="select"], .stNumberInput input {
            background-color: #ffffff;
            color: #000000;
            border: 1px solid #000000;
        }
        .stButton button {
            background-color: #ffffff;
            color: #000000;
            border: 1px solid #000000;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .stButton button:focus, .stButton button:hover {
            background-color: #f0f0f0;
            border-color: #000000;
        }
        div[data-testid="stHeader"] {
            background-color: #ffffff;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
st.caption(
    "Enter any statement, phrase, or concept to retrieve calibration matches from the CCD database. "
    "If no direct match exists, the app will automatically expand your query via GPT (OpenAI API "
    "and/or GPT‑2) and try secondary and tertiary keyword searches."
)


def render_stage(stage_name: str, matches: MatchDict, keywords: List[str]) -> None:
    with st.expander(stage_name, expanded=False):
        if keywords:
            st.write("**Keywords**: ", ", ".join(keywords))
        if matches:
            for entry in matches.values():
                st.write(f"- {format_entry(entry)}")
        else:
            st.write("_No matches._")


def render_suggestion_stage(stage_matches: MatchDict, suggestions: List[Tuple[str, str]]) -> None:
    with st.expander("Database suggestions", expanded=False):
        with st.expander("Matched entries from suggestions", expanded=False):
            if stage_matches:
                for entry in stage_matches.values():
                    st.write(f"- {format_entry(entry)}")
            else:
                st.write("_No matches were retrieved directly from the suggestions._")

        with st.expander("Suggestions & reasoning", expanded=False):
            if suggestions:
                seen: Set[str] = set()
                for suggestion in suggestions:
                    if isinstance(suggestion, (list, tuple)) and len(suggestion) >= 2:
                        name, reason = suggestion[0], suggestion[1]
                    else:
                        name, reason = str(suggestion), ""
                    normalized = name.lower()
                    if normalized in seen:
                        continue
                    seen.add(normalized)
                    st.write(f"- {name}: {reason or 'heuristic similarity'}")
            else:
                st.write("_No suggestions were provided._")


def render_results(
    matches: MatchDict,
    stages: List[Tuple[str, MatchDict]],
    secondary_keywords: List[str],
    tertiary_keywords: List[str],
    gpt_suggestions_with_reason: List[Tuple[str, str]],
) -> None:
    direct_stage = next(
        (stage_matches for stage_name, stage_matches in stages if stage_name == "Direct statement" and stage_matches),
        None,
    )
    if direct_stage:
        with st.expander("Direct statement matches", expanded=True):
            for entry in direct_stage.values():
                st.success(format_entry(entry))

    secondary_rendered = False
    tertiary_rendered = False
    for stage_name, stage_matches in stages:
        keywords: List[str] = []
        if stage_name == "Secondary keywords":
            keywords = secondary_keywords
            secondary_rendered = True
        elif stage_name == "Tertiary keywords":
            keywords = tertiary_keywords
            tertiary_rendered = True
        elif stage_name == "Database suggestions":
            render_suggestion_stage(stage_matches, gpt_suggestions_with_reason)
            continue
        render_stage(stage_name, stage_matches, keywords)

    if not secondary_rendered and secondary_keywords:
        render_stage("Secondary keywords", {}, secondary_keywords)
    if not tertiary_rendered and tertiary_keywords:
        render_stage("Tertiary keywords", {}, tertiary_keywords)
    if gpt_suggestions_with_reason and all(stage_name != "Database suggestions" for stage_name, _ in stages):
        render_suggestion_stage({}, gpt_suggestions_with_reason)

    avg = average_calibration(matches) or geometric_mean_entries(matches)
    cal_range = calibration_range(matches)
    if avg is None:
        st.info("No calibration guesstimate could be made with the current database. This could be wrong.")
    else:
        qualifier = ""
        if len(matches) < 3:
            qualifier = " _(only a few matches available)_"
        range_text = ""
        if cal_range:
            lo, hi = cal_range
            range_text = f" ({lo:.0f} - {hi:.0f})"
        st.success(
            f"Geometric-average calibration across {len(matches)} match(es): **{avg:.2f}**{range_text}{qualifier}. "
            "Remember the scale is logarithmic; geometric means are used. This could be wrong."
        )


with st.form("statement_form"):
    statement = st.text_area(
        "Statement to analyze",
        placeholder="e.g. Should I microwave my food?",
        height=120,
    ).strip()
    submitted = st.form_submit_button("Analyze")

if submitted:
    if not statement:
        st.error("Please provide a statement to analyze.")
    else:
        with st.spinner("Analyzing..."):
            matches, stages, secondary_keywords, tertiary_keywords, gpt_suggestions = run_pipeline(
                statement, database
            )
        render_results(
            matches,
            stages,
            secondary_keywords,
            tertiary_keywords,
            gpt_suggestions,
        )
else:
    st.info("Enter a statement above and click **Analyze** to get started.")
