__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import pandas as pd
import chromadb
import json
import os
from langgraph.graph import StateGraph
from ibm_watson_machine_learning.foundation_models import ModelInference
from fpdf import FPDF

##########################################################
#############    PUT API KEY HERE     ####################
##########################################################
wml_credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "apikey": "api_key"
}
model = ModelInference(credentials=wml_credentials, project_id="project_id",model_id="model_id")

# ChromaDB Setup
chroma_client = chromadb.PersistentClient(path="./chromadb_store")
collection = chroma_client.get_or_create_collection(name="dna_analysis")

def load_and_preprocess(file):
    """Load and preprocess the uploaded genomic data."""
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        df = pd.read_excel(file)
    elif file.name.endswith('.txt'):
        df = pd.read_csv(file, delimiter="\t")
    else:
        return None
    return df

def query_llm(category, data):
    """Query IBM Granite LLM with retrieved DNA data insights."""
    prompt = f"Analyze the following DNA data under the category {category}: {data}"
    response = model.generate_text(prompt)
    return response.text

# Define Graph for LangGraph
class DNAAnalysisState:
    def __init__(self, data, results=None):
        self.data = data
        self.results = results or {}

graph = StateGraph(DNAAnalysisState)

# Define Analysis Nodes
def analyze_genomic_disorders(state):
    insights = query_llm("Genomic Disorders", state.data)
    state.results["Genomic Disorders"] = insights
    return state

def analyze_physical_traits(state):
    insights = query_llm("Physical Characteristics", state.data)
    state.results["Physical Characteristics"] = insights
    return state

def analyze_mental_traits(state):
    insights = query_llm("Mental Characteristics", state.data)
    state.results["Mental Characteristics"] = insights
    return state

def analyze_personality(state):
    insights = query_llm("Personality", state.data)
    state.results["Personality"] = insights
    return state

def analyze_disease_risk(state):
    insights = query_llm("Future Disease Risks", state.data)
    state.results["Future Disease Risks"] = insights
    return state

def analyze_ancestry(state):
    insights = query_llm("Ancestry & Heritage", state.data)
    state.results["Ancestry & Heritage"] = insights
    return state

def analyze_forensic_insights(state):
    insights = query_llm("Forensic DNA Insights", state.data)
    state.results["Forensic DNA Insights"] = insights
    return state

def analyze_dna_matching(state, second_data):
    """Analyze relationship between two DNA datasets."""
    prompt = f"Compare the following two DNA datasets and determine the relationship: {state.data} and {second_data}"
    insights = model.generate_text(prompt).text
    state.results["DNA Matching"] = insights
    return state

# Add Nodes to Graph
graph.add_node("genomic_disorders", analyze_genomic_disorders)
graph.add_node("physical_traits", analyze_physical_traits)
graph.add_node("mental_traits", analyze_mental_traits)
graph.add_node("personality", analyze_personality)
graph.add_node("disease_risk", analyze_disease_risk)
graph.add_node("ancestry", analyze_ancestry)
graph.add_node("forensic_insights", analyze_forensic_insights)

graph.add_edge("genomic_disorders", "physical_traits")
graph.add_edge("physical_traits", "mental_traits")
graph.add_edge("mental_traits", "personality")
graph.add_edge("personality", "disease_risk")
graph.add_edge("disease_risk", "ancestry")
graph.add_edge("ancestry", "forensic_insights")

graph.set_entry_point("genomic_disorders")

# Streamlit UI
st.title("DNA Analysis Using AI")
uploaded_file = st.file_uploader("Upload your genomic data (CSV, XLSX, TXT)", type=["csv", "xlsx", "txt"])

if uploaded_file:
    df = load_and_preprocess(uploaded_file)
    if df is not None:
        st.dataframe(df.head())
        if st.button("Start Analysis"):
            state = DNAAnalysisState(df.to_json())
            result = graph.run(state)
            st.session_state["analysis_results"] = result.results
            st.success("Analysis completed!")
    else:
        st.error("Invalid file format.")

if "analysis_results" in st.session_state:
    results = st.session_state["analysis_results"]
    for category, insight in results.items():
        with st.expander(f"{category}"):
            st.write(insight)

    if st.button("Download Report as PDF"):
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, "DNA Analysis Report", ln=True, align="C")
        for category, insight in results.items():
            pdf.add_page()
            pdf.cell(200, 10, category, ln=True, align="C")
            pdf.multi_cell(0, 10, insight)
        pdf_path = "DNA_Analysis_Report.pdf"
        pdf.output(pdf_path)
        with open(pdf_path, "rb") as f:
            st.download_button("Download PDF", f, file_name=pdf_path, mime="application/pdf")

st.header("DNA Matching")
file1 = st.file_uploader("Upload First DNA Dataset", type=["csv", "xlsx", "txt"], key="file1")
file2 = st.file_uploader("Upload Second DNA Dataset", type=["csv", "xlsx", "txt"], key="file2")

if file1 and file2:
    df1 = load_and_preprocess(file1)
    df2 = load_and_preprocess(file2)
    if df1 is not None and df2 is not None:
        if st.button("Compare DNA"):
            state = DNAAnalysisState(df1.to_json())
            result = analyze_dna_matching(state, df2.to_json())
            st.session_state["dna_matching_result"] = result.results["DNA Matching"]
            st.success("DNA Matching completed!")

if "dna_matching_result" in st.session_state:
    st.subheader("DNA Matching Results")
    st.write(st.session_state["dna_matching_result"])