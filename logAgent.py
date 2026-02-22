from operator import add
from typing import List, Optional, Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st

# set your OpenAI API key
# # make a set env function
# import os, getpass

# def _set_env(var: str):
#     if not os.environ.get(var):
#         os.environ[var] = getpass.getpass(f"{var}: ")

# _set_env("LANGCHAIN_API_KEY")
# _set_env("OPENAI_API_KEY")
# _set_env("ANTHROPIC_API_KEY")

# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_PROJECT"] = "langchain-academy"


# The structure of the logs
class Log(TypedDict):
    id: str
    question: str
    docs: Optional[List]
    answer: str
    grade: Optional[int]
    grader: Optional[str]
    feedback: Optional[str]

# Failure Analysis Sub-graph
class FailureAnalysisState(TypedDict):
    cleaned_logs: List[Log]
    failures: List[Log]
    fa_summary: str
    processed_logs: List[str]

class FailureAnalysisOutputState(TypedDict):
    fa_summary: str
    processed_logs: List[str]
    
# Helper function to summarize failures or clean logs
def make_summaries(logs: List[Log]) -> str:
    """Summarize the failed logs using the OpenAI model."""
    logs_str = "\n".join([
        f"Log ID: {log['id']}\nQuestion: {log['question']}\nAnswer: {log['answer']}" 
        for log in logs
    ])
    prompt_template = ChatPromptTemplate([
        ("system", "You are a helpful assistant that summarizes failure logs in a concise manner."),
        ("user", "{logs_str}"),
    ])
    formatted_prompt = prompt_template.format_prompt(logs_str=logs_str)
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    response = model.invoke(formatted_prompt)
    return response.content

def make_reports(summary: str) -> str:
    """Generate a report based on the summary."""
    prompt_template = ChatPromptTemplate([
        ("system", "You are a helpful assistant that generates reports based on summaries. You also are an expert at providing relevant information to help the question and answer, like code snippets to solve the problem and help in any way possible with steps to solve the problem."),
        ("user", "{summary}"),
    ])
    formatted_prompt = prompt_template.format_prompt(summary=summary)
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    response = model.invoke(formatted_prompt)
    return response.content

def get_failures(state):
    """Get logs that contain a failure."""
    cleaned_logs = state["cleaned_logs"]
    failures = [log for log in cleaned_logs if "grade" in log]
    return {"failures": failures}

def generate_summary(state):
    """Generate summary of failures."""
    failures = state["failures"]
    fa_summary = make_summaries(failures)
    return {"fa_summary": fa_summary, "processed_logs": [f"failure-analysis-on-log-{failure['id']}" for failure in failures]}

fa_builder = StateGraph(input=FailureAnalysisState, output=FailureAnalysisOutputState)
fa_builder.add_node("get_failures", get_failures)
fa_builder.add_node("generate_summary", generate_summary)
fa_builder.add_edge(START, "get_failures")
fa_builder.add_edge("get_failures", "generate_summary")
fa_builder.add_edge("generate_summary", END)

# Summarization subgraph
class QuestionSummarizationState(TypedDict):
    cleaned_logs: List[Log]
    qs_summary: str
    report: str
    processed_logs: List[str]

class QuestionSummarizationOutputState(TypedDict):
    report: str
    processed_logs: List[str]

def generate_summary(state):
    cleaned_logs = state["cleaned_logs"]
    summary = make_summaries(cleaned_logs)
    return {"qs_summary": summary, "processed_logs": [f"summary-on-log-{log['id']}" for log in cleaned_logs]}

def send_to_slack(state):
    qs_summary = state["qs_summary"]
    report = make_reports(qs_summary)
    return {"report": report}

qs_builder = StateGraph(input=QuestionSummarizationState, output=QuestionSummarizationOutputState)
qs_builder.add_node("generate_summary", generate_summary)
qs_builder.add_node("send_to_slack", send_to_slack)
qs_builder.add_edge(START, "generate_summary")
qs_builder.add_edge("generate_summary", "send_to_slack")
qs_builder.add_edge("send_to_slack", END)

# Entry Graph
class EntryGraphState(TypedDict):
    raw_logs: List[Log]
    cleaned_logs: List[Log]
    fa_summary: str  # Generated in the FA sub-graph
    report: str      # Generated in the QS sub-graph
    processed_logs: Annotated[List[int], add]  # Generated in BOTH sub-graphs

def clean_logs(state):
    raw_logs = state["raw_logs"]
    cleaned_logs = raw_logs
    return {"cleaned_logs": cleaned_logs}

entry_builder = StateGraph(EntryGraphState)
entry_builder.add_node("clean_logs", clean_logs)
entry_builder.add_node("question_summarization", qs_builder.compile())
entry_builder.add_node("failure_analysis", fa_builder.compile())
entry_builder.add_edge(START, "clean_logs")
entry_builder.add_edge("clean_logs", "failure_analysis")
entry_builder.add_edge("clean_logs", "question_summarization")
entry_builder.add_edge("failure_analysis", END)
entry_builder.add_edge("question_summarization", END)

graph = entry_builder.compile()


# --- Streamlit App with Enhanced UI ---

# Add custom CSS and animations to enhance the UI
st.markdown(
"""
<style>
/* Animated title and description */
.big-title {
    font-size: 3em;
    font-weight: bold;
    text-align: center;
    color: #4CAF50;
    animation: fadeIn 2s;
    margin-bottom: 10px;
}
.description {
    font-size: 1.2em;
    text-align: center;
    color: #333;
    margin-bottom: 30px;
    animation: fadeIn 3s;
}

/* Card style for sections */
.card {
    background-color: #f9f9f9;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    margin: 20px 0;
    animation: slideIn 1s;
}

/* Keyframes for animations */
@keyframes fadeIn {
  from { opacity: 0; }
  to   { opacity: 1; }
}
@keyframes slideIn {
  from { transform: translateX(-100%); opacity: 0; }
  to { transform: translateX(0); opacity: 1; }
}
</style>
""", unsafe_allow_html=True)

# New, improved title and description with instructions
st.markdown('<div class="big-title">Intelligent Log Analyzer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="description">Welcome to the Intelligent Log Analyzer! This agent processes log entries, summarizes failures, and generates insightful reports. '
    'Simply add your log entries (with optional grade, grader, and feedback) via the sidebar, then click "Execute Agent" and watch the magic unfold.</div>',
    unsafe_allow_html=True
)

# Initialize session state variables if they do not exist
if "logs" not in st.session_state:
    st.session_state["logs"] = []  # Holds the list of raw log entries
if "results" not in st.session_state:
    st.session_state["results"] = None  # Holds the output from the agent

# Log Entry Input (Sidebar)
st.sidebar.header("Add a New Log Entry")
with st.sidebar.form(key="log_form"):
    log_id = st.text_input("Log ID")
    question = st.text_area("Question", height=100)
    answer = st.text_area("Answer", height=100)
    grade_input = st.text_input("Grade (Optional)", value="")
    grader = st.text_input("Grader (Optional)", value="")
    # Updated the height to 70 pixels to meet the minimum requirement
    feedback = st.text_area("Feedback (Optional)", value="", height=70)
    
    # Submit Button for the log entry
    submit_log = st.form_submit_button("Submit Log Entry")
    
    if submit_log:
        if log_id and question and answer:
            try:
                grade = int(grade_input) if grade_input.strip() != "" else None
            except ValueError:
                st.sidebar.error("Grade must be an integer.")
                grade = None
            new_log = {
                "id": log_id,
                "question": question,
                "docs": None,
                "answer": answer,
                "grade": grade,
                "grader": grader if grader.strip() != "" else None,
                "feedback": feedback if feedback.strip() != "" else None,
            }
            st.session_state.logs.append(new_log)
            st.sidebar.success("Log added!")
        else:
            st.sidebar.error("Please fill in the Log ID, Question, and Answer before submitting.")

# Display Current Logs
st.header("Current Log Entries")
if st.session_state.logs:
    st.write("Below are your added log entries:")
    st.json(st.session_state.logs)
else:
    st.info("No logs have been added yet.")

# Execute Agent
st.header("Run Agent")
if st.button("Execute Agent"):
    if not st.session_state.logs:
         st.warning("Please add at least one log entry before running the agent.")
    else:
         input_state = {"raw_logs": st.session_state.logs}
         try:
             result = graph.invoke(input_state)
             st.session_state.results = result
             st.success("Agent executed successfully!")
         except Exception as e:
             st.error(f"Agent execution failed: {e}")

# Display Agent Output in a streaming manner
if st.session_state.results:
    import time  # Imported here to keep changes minimal
    st.header("Agent Output")
    
    # Stream Failure Analysis Summary
    fa_text = st.session_state.results.get("fa_summary", "N/A")
    fa_placeholder = st.empty()
    fa_placeholder.subheader("Failure Analysis Summary")
    fa_text_placeholder = st.empty()
    displayed_text = ""
    for i in range(0, len(fa_text), 100):
         displayed_text += fa_text[i:i+100]
         fa_text_placeholder.write(displayed_text)
         time.sleep(0.2)
    
    # Stream Question Summarization Report
    qs_text = st.session_state.results.get("report", "N/A")
    qs_placeholder = st.empty()
    qs_placeholder.subheader("Question Summarization Report")
    qs_text_placeholder = st.empty()
    displayed_text = ""
    for i in range(0, len(qs_text), 100):
         displayed_text += qs_text[i:i+100]
         qs_text_placeholder.write(displayed_text)
         time.sleep(0.2)

    # Stream Processed Logs
    pl_text = str(st.session_state.results.get("processed_logs", "N/A"))
    pl_placeholder = st.empty()
    pl_placeholder.subheader("Processed Logs")
    pl_text_placeholder = st.empty()
    displayed_text = ""
    for i in range(0, len(pl_text), 100):
         displayed_text += pl_text[i:i+100]
         pl_text_placeholder.write(displayed_text)
         time.sleep(0.2)
    
    # Display Cleaned Logs all at once (JSON) with dropdown closed if available
    if "cleaned_logs" in st.session_state.results:
        st.subheader("Cleaned Logs")
        with st.expander("View Cleaned Logs", expanded=False):
            st.json(st.session_state.results.get("cleaned_logs"))
