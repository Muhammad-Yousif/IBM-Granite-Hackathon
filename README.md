# RAG-Based DNA Analyzer - IBM Granite Hackathon

## Overview
This project is a **RAG-Based DNA Analyzer** built for the IBM Granite Hackathon. The goal is to create an agentic Retrieval-Augmented Generation (RAG) system that allows users to upload their genomic data and gain various insights based on AI-driven analysis.

## Features
- **Genomic Data Upload**: Users can upload their DNA data for analysis.
- **Genetic Insights**:
  - Ancestry and relationship mapping
  - Genetic traits and relatives
- **Health Insights**:
  - Predict heart disease, neuro disorders, and other conditions
- **Genetic Disorder Detection**
- **Physical Structure Analysis**
- **Behavioral & Personality Analytics**
- **DNA Identification & Matching**

## How It Works
The project employs an **Agentic RAG Architecture** powered by **LangChain** and **IBM Granite API**, where multiple specialized agents handle different DNA-related tasks.

- One agent specializes in DNA matching
- Another agent focuses on physical characteristics mapping
- Other agents analyze health risks, personality traits, and genetic relationships

## Tech Stack
- **Backend**: Python, LangChain, IBM Granite API
- **Frontend**: Streamlit or JavaScript (for deployment readiness)
- **Database**: ChromaDB (vector database)
- **Embeddings**: If provided by IBM Granite, they will be used; otherwise, compatible embeddings will be integrated.

## Installation
To set up the project, follow these steps:

### Prerequisites
- Python 3.8+
- IBM Granite API key
- Required dependencies (listed in `requirements.txt`)

### Steps
```sh
# Clone the repository
git clone https://github.com/your-repo/ibm-granite-dna-analyzer.git
cd ibm-granite-dna-analyzer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py  # If using Streamlit
```

## Usage
1. Upload your genomic data file (e.g., `.vcf`, `.txt`, or other formats).
2. Select the type of analysis you want to perform.
3. View insights and predictions generated by the system.

## Contributing
Feel free to submit pull requests and contribute to the project.

## License
MIT License

## Contact
For queries or support, contact [Muhammad Yousif] at [hellomyousif@gmail.com].

