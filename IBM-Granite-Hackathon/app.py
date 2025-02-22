import streamlit as st
import pandas as pd
import chromadb
import json
import os
from langgraph.graph import StateGraph
from ibm_watson_machine_learning.foundation_models import ModelInference
from fpdf import FPDF
from ibm_watsonx_ai import Credentials

##########################################################
#############    PUT API KEY HERE     ####################
##########################################################

wml_credentials = {
    "apikey": "LJbZPxwGd6tVgUfD8A8RnEnz5Yh9ql6PQvilbuxNeGYr",
    "url": "https://eu-de.ml.cloud.ibm.com"
}

MODEL_ID = "ibm/granite-13b-chat-v2"
PROJECT_ID = "123e4567-e89b-12d3-a456-426614174000"  

model = ModelInference(
    model_id=MODEL_ID,
    credentials=wml_credentials,
    project_id=PROJECT_ID
)

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
    
    prompts = {
        "Genomic Disorders": f"""
        You are a genetic expert analyzing potential genomic disorders in a given DNA dataset.
        
        ### DNA Data:
        {data}
        
        ### Instructions:
        - Identify any gene mutations linked to known disorders.
        - Explain the significance of these mutations.
        - Highlight potential health risks associated with them.
        - Reference genetic studies where applicable.

        Provide results in an easy-to-understand format for non-experts.
        """,

        "Physical Characteristics": f"""
        You are a genetics researcher analyzing DNA data to determine physical traits.

        ### DNA Data:
        {data}

        ### Instructions:
        - Identify genetic markers for physical traits (e.g., eye color, hair type, height).
        - Explain how these markers influence phenotypic expression.
        - Compare the findings with known genomic studies.

        Provide an understandable summary.
        """,

        "Mental Characteristics": f"""
        Analyze the following DNA data for potential cognitive traits and mental characteristics.

        ### DNA Data:
        {data}

        ### Instructions:
        - Identify gene markers associated with intelligence, memory, and cognitive functions.
        - Explain potential genetic influences on mental abilities.
        - Highlight any relevant studies supporting these findings.

        Provide a scientifically sound but accessible response.
        """,

        "Personality": f"""
        Analyze DNA data to predict potential personality traits.

        ### DNA Data:
        {data}

        ### Instructions:
        - Identify genetic variations linked to temperament and personality.
        - Explain the impact of these genes on behavior.
        - Discuss the role of genetics vs. environment.

        Keep it evidence-based and objective.
        """,

        "Future Disease Risks": f"""
        You are analyzing DNA data for potential future disease risks.

        ### DNA Data:
        {data}

        ### Instructions:
        - Identify genetic predispositions to common diseases (e.g., diabetes, heart disease, cancer).
        - Explain how these genes increase or decrease risk.
        - Provide preventive measures based on genetic findings.

        Maintain clarity while ensuring scientific accuracy.
        """,

        "Ancestry & Heritage": f"""
        Analyze DNA data to determine ancestry and heritage.

        ### DNA Data:
        {data}

        ### Instructions:
        - Compare genetic markers with global population databases.
        - Identify potential ancestral origins.
        - Explain how specific gene sequences are inherited.

        Provide insights in a culturally sensitive manner.
        """,

        "Forensic DNA Insights": f"""
        You are conducting a forensic analysis of DNA data.

        ### DNA Data:
        {data}

        ### Instructions:
        - Identify any forensic markers that can assist in identification.
        - Explain potential familial connections or lineage.
        - Highlight forensic applications of the findings.

        Ensure a balanced and ethical analysis.
        """,

        "DNA Matching": f"""
        Compare two DNA datasets to determine genetic relationships.

        ### DNA Data 1:
        {data}

        ### Instructions:
        - Identify shared genetic markers.
        - Calculate the likelihood of biological relationships.
        - Explain whether the individuals could be related (parent-child, siblings, distant relatives).

        Use probability-based genetic analysis methods.
        """
    }

    prompt = prompts.get(category, f"Analyze the following DNA data under the category {category}: {data}")
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
    insights = query_llm("DNA Matching", f"{state.data} and {second_data}")
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

if "analysis_results" in st.session_state:
    results = st.session_state["analysis_results"]
    for category, insight in results.items():
        with st.expander(f"{category}"):
            st.write(insight)
