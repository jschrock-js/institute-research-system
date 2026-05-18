from pathlib import Path
import streamlit as st

from main import run_research

# --------------------------------------------------
# Page Setup
# --------------------------------------------------
st.set_page_config(
    page_title="Institute Research System",
    layout="wide"
)

st.title("Institute of Evidence-based Policymaking Research System")
st.subheader("Multi-Agent Research Assistant")

# --------------------------------------------------
# Sidebar Inputs
# --------------------------------------------------
st.sidebar.header("Project Setup")

topic = st.sidebar.text_input(
    "Research Topic",
    value=""
)

main_question = st.sidebar.text_area(
    "Main Research Question",
    value="",
    height=200,
    placeholder="What does the best available evidence suggest regarding this issue?"
)

output_type = st.sidebar.selectbox(
    "Output Type",
    [
        "Research Coach Only",
        "Full Report",
        "Research Coach + Full Report",
        "Literature Review Only",
        "End-to-End Workflow",
    ]
)

research_depth = st.sidebar.selectbox(
    "Research Depth",
    [
        "Quick Scan",
        "Standard Review",
        "Deep Literature Review",
    ],
    index=1
)

include_data_sources = st.sidebar.checkbox(
    "Include Suggested Data Sources",
    value=True
)

include_systems_analysis = st.sidebar.checkbox(
    "Include Systems Analysis",
    value=True
)

research_coach_principles = st.sidebar.text_area(
    "Research Coach Principles",
    value="""Seek disconfirming evidence.
Focus on causal mechanisms.
Consider incentives and tradeoffs.
Identify second- and third-order effects.
Highlight key uncertainties.
Ask what evidence would most change the conclusion.""",
    height=180
)

project_instructions = st.sidebar.text_area(
    "Additional Project Instructions",
    value="""Use a neutral, evidence-based, clear, concise, and accessible style.
Distinguish exploratory insights, evidence-supported findings, and final conclusions.
Do not overstate conclusions beyond what the evidence supports.""",
    height=220
)

research_approach_instructions = st.sidebar.text_area(
    "Research Approach Instructions",
    value="",
    height=180,
    placeholder=(
        "Example: Prioritize peer-reviewed studies and government data. "
        "Use advocacy reports only when they contain unique data. "
        "Approach the issue primarily from an economist's perspective."
    )
)

methodology_question = st.sidebar.text_area(
    "Methodology Question (Optional)",
    value="",
    height=120,
    placeholder="Example: Does the system appropriately downgrade advocacy-based and speculative evidence?"
)

uploaded_files = st.sidebar.file_uploader(
    "Upload Supporting Documents",
    accept_multiple_files=True
)

# --------------------------------------------------
# Save Uploaded Files
# --------------------------------------------------
docs_dir = Path("inputs/docs")
docs_dir.mkdir(parents=True, exist_ok=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = docs_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

# --------------------------------------------------
# Run Analysis
# --------------------------------------------------
if st.sidebar.button("Run Analysis"):

    state = {
        "topic": topic,
        "main_question": main_question,
        "project_instructions": project_instructions,
        "research_approach_instructions": research_approach_instructions,
        "research_coach_principles": research_coach_principles,
        "methodology_question": methodology_question,
        "research_depth": research_depth,
        "include_data_sources": include_data_sources,
        "include_systems_analysis": include_systems_analysis,
        "report_type": output_type,
        "mode": "reasoning",
    }

    with st.spinner("Running analysis... This may take several minutes."):
        output = run_research(state)

    st.success("Analysis complete.")

    tabs = st.tabs([
        "Research Coach",
        "Methodology Notes",
        "Final Report",
        "Downloads",
    ])

    # Research Coach Tab
    with tabs[0]:
        st.markdown(
            output.get(
                "research_coach_notes",
                "No Research Coach Notes generated."
            )
        )

    # Methodology Notes Tab
    with tabs[1]:
        st.markdown(
            output.get(
                "methodology_notes",
                "No Methodology Notes generated."
            )
        )

    # Final Report Tab
    with tabs[2]:
        if output_type == "Research Coach Only":
            st.info("Research Coach Only selected.")
        else:
            st.markdown(
                output.get(
                    "synthesis",
                    "No final report generated."
                )
            )

    # Downloads Tab
    with tabs[3]:
        if output.get("saved_file"):
            st.write("Report:", output["saved_file"])

        if output.get("saved_excel_file"):
            st.write("Excel:", output["saved_excel_file"])

        if output.get("saved_evidence_csv"):
            st.write("CSV:", output["saved_evidence_csv"])