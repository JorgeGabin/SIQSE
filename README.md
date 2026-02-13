[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg?style=for-the-badge)](https://www.python.org/downloads/)

<br />
<div align="center">
  <a href="https://github.com/JorgeGabin/SIQSE">
    <img src="assets/logo.png" alt="Logo" width="300">
  </a>

  <h3 align="center">SIQSE</h3>

  <p align="center">
    <strong>Beyond Top-k: Simulation-Based Interactive Evaluation for Query Suggestions</strong>
    <br />
    A Python package for simulating user interaction in query suggestion selection using Large Language Models.
    <br />
    <br />
    <br />
    <a href="https://github.com/JorgeGabin/SIQSE/issues">Report Bug</a>
    ·
    <a href="https://github.com/JorgeGabin/SIQSE/issues">Request Feature</a>
  </p>
</div>

---

## Overview

SIQSE simulates realistic user interaction for query suggestion selection in information retrieval systems. Instead of simply using top-e suggestions, it leverages Large Language Models (LLMs) to simulate how users would actually choose which suggestions to add to their queries, considering factors like:

- **User intent and information needs**
- **Document feedback from search results**
- **Interactive, iterative selection** (selecting suggestions one-by-one based on evolving results)
- **Realistic stopping behavior** (knowing when to stop selecting)

This enables more realistic evaluation of query suggestion systems by simulating the interactive nature of real user behavior.

---

## Features

- **User Simulation**: LLM-based simulation of realistic user suggestion selection behavior
- **Batch Mode**: Simulate selecting all suggestions at once (non-interactive)
- **Iterative Mode**: Simulate interactive selection where users see results and decide incrementally
- **Document Feedback**: Simulate users considering search results when choosing suggestions
- **Intent-Aware**: Incorporate user search intent to guide selection decisions
- **Automatic Stopping**: LLM decides when to stop selecting (simulating user satisfaction)
- **OpenAI-Compatible API**: Works with Ollama, vLLM, and other OpenAI-compatible endpoints
- **PyTerrier Integration**: Easy integration with PyTerrier for IR experiments

---

## Installation

```bash
pip install siqse
```

Or install from source:

```bash
cd SIQSE
pip install -e .
```

---

## Quick Start

### Basic Usage - Simulating User Selection

```python
from siqse import LLMSuggestionSelector

# Initialize the user simulator
simulator = LLMSuggestionSelector(
    model="llama3.1:70b",
    llm_api_url="http://localhost:11434",
    selection_mode="batch",
    use_feedback_docs=False
)

# Simulate which suggestions a user would select
query = "machine learning applications"
suggestions = ["neural networks", "deep learning", "AI ethics", "data mining"]

selected = simulator.select_suggestions(
    query=query,
    suggestions=suggestions,
    n=2  # Simulate selecting top 2
)

print(selected)  # e.g., ["neural networks", "deep learning"]
```

### Simulating Interactive Selection with Feedback

```python
def get_retrieval_docs(query: str, selected_keywords: list) -> list:
    """
    Retrieve documents to show to the simulated user.
    Returns list of dicts with keys: doc_id, score, rank, is_relevant, text
    """
    # Your retrieval logic here
    return [
        {
            "doc_id": "doc1",
            "score": 0.95,
            "rank": 1,
            "is_relevant": True,
            "text": "Document text here..."
        },
        # ... more documents
    ]

# Simulate iterative user behavior with feedback
simulator = LLMSuggestionSelector(
    model="llama3.1:70b",
    selection_mode="iterative",
    use_feedback_docs=True
)

selected = simulator.select_suggestions(
    query=query,
    suggestions=suggestions,
    user_intent="Find recent research papers on neural network architectures",
    n=3,
    get_docs=get_retrieval_docs
)
```

### Simulating Automatic Stopping (No Fixed k)

```python
# Simulate user deciding when to stop selecting
selected = simulator.select_suggestions(
    query=query,
    suggestions=suggestions,
    n=None  # No limit - simulated user decides when to stop
)
```

---

## Selection Modes

### Batch Mode (Non-Interactive Simulation)
Simulates a user selecting all suggestions at once without seeing intermediate results. Faster but less realistic for interactive search scenarios.

```python
simulator = LLMSuggestionSelector(selection_mode="batch")
```

### Iterative Mode (Interactive Simulation)
Simulates realistic interactive behavior where users select suggestions one-by-one, see updated results after each selection, and decide whether to continue. This better reflects actual user behavior in interactive search systems.

```python
simulator = LLMSuggestionSelector(selection_mode="iterative")
```

---

## PyTerrier Integration

SIQSE integrates seamlessly with PyTerrier for realistic query expansion experiments:

```python
import pyterrier as pt
from siqse import LLMSuggestionSelector

# Initialize
if not pt.started():
    pt.init()

index = pt.IndexFactory.of("/path/to/index")
simulator = LLMSuggestionSelector(
    model="llama3.1:8b",
    selection_mode="iterative",
    use_feedback_docs=True
)
bm25 = pt.BatchRetrieve(index, wmodel="BM25")

# Define expansion function that simulates user selection
def expand(df):
    def fb(q, s):
        # Show results to simulated user for decision-making
        expanded = q + " " + " ".join(s) if s else q
        return bm25.search(expanded).head(10).to_dict('records')
    
    df = df.copy()
    df["query"] = df.apply(
        lambda r: " ".join([r["query"]] + simulator.select_suggestions(
            query=r["query"],
            suggestions=r["suggestions"],
            user_intent=r.get("intent"),
            get_docs=fb
        )),
        axis=1
    )
    return df

# Build pipeline: simulate user selection then retrieve
pipeline = pt.apply.generic(expand) >> bm25
results = pipeline.transform(queries_df)
```

See `examples/pyterrier_integration.py` and `examples/pyterrier_minimal.py` for complete working examples.

---

## API Reference

### LLMSuggestionSelector

**Constructor Parameters:**
- `model` (str): LLM model name (default: "llama3.1:70b")
- `llm_api_url` (str): LLM API endpoint (default: "http://localhost:11434")
- `llm_api_key` (str): API key (default: "ollama")
- `temperature` (float): LLM temperature (default: 0.0)
- `selection_mode` (str): "batch" or "iterative" (default: "batch")
- `use_feedback_docs` (bool): Include document feedback (default: True)

**Methods:**

#### `select_suggestions(query, suggestions, user_intent=None, n=None, get_docs=None)`

Simulate user selection of query suggestions.

**Parameters:**
- `query` (str): User's search query
- `suggestions` (List[str]): Available query suggestions
- `user_intent` (str, optional): User's search intent/information need
- `n` (int, optional): Number of suggestions to select (None = simulated user decides when to stop)
- `get_docs` (Callable, optional): Function to retrieve documents for feedback (simulates showing results to user)

**Returns:**
- `List[str]`: Selected suggestions (simulating user choices)

---

## Utilities

```python
from siqse import (
    clean_query_for_pyterrier,
    format_document_for_display,
    count_relevant_documents
)

# Clean query for IR systems
clean_query = clean_query_for_pyterrier("Parkinson's disease")

# Format document for LLM prompt
doc = {
    "doc_id": "doc1",
    "score": 0.95,
    "rank": 1,
    "is_relevant": True,
    "text": "Long document text..."
}
formatted = format_document_for_display(doc, snippet_chars=300)

# Count relevant docs
docs = [...]  # list of doc dicts
num_relevant = count_relevant_documents(docs)
```

---

## Authors

**Jorge Gabín** (jorge.gabin@udc.es, [ORCID: 0000-0002-5494-0765](https://orcid.org/0000-0002-5494-0765))  
IRLab, CITIC, Universidade da Coruña, A Coruña, Spain

**Javier Parapar** (javier.parapar@udc.es, [ORCID: 0000-0002-5997-8252](https://orcid.org/0000-0002-5997-8252))  
IRLab, CITIC, Universidade da Coruña, A Coruña, Spain

**Xi Wang** (xi.wang@sheffield.ac.uk, [ORCID: 0000-0001-5936-9919](https://orcid.org/0000-0001-5936-9919))  
University of Sheffield, Sheffield, United Kingdom

---

## Citation

If you use this work, please cite:

```bibtex
@article{gabin2026beyond,
  title={Beyond Top-e: Simulation-Based Interactive Evaluation for Query Suggestions},
  author={Gabín, Jorge and Parapar, Javier and Wang, Xi},
  year={2026},
  note={Paper under review}
}
```

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## Contact

For questions and support, please open an issue on GitHub.

<!-- MARKDOWN LINKS & IMAGES -->
[contributors-shield]: https://img.shields.io/github/contributors/JorgeGabin/SIQSE.svg?style=for-the-badge
[contributors-url]: https://github.com/JorgeGabin/SIQSE/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/JorgeGabin/SIQSE.svg?style=for-the-badge
[forks-url]: https://github.com/JorgeGabin/SIQSE/network/members
[stars-shield]: https://img.shields.io/github/stars/JorgeGabin/SIQSE.svg?style=for-the-badge
[stars-url]: https://github.com/JorgeGabin/SIQSE/stargazers
[issues-shield]: https://img.shields.io/github/issues/JorgeGabin/SIQSE.svg?style=for-the-badge
[issues-url]: https://github.com/JorgeGabin/SIQSE/issues
[license-shield]: https://img.shields.io/github/license/JorgeGabin/SIQSE.svg?style=for-the-badge
[license-url]: https://github.com/JorgeGabin/SIQSE/blob/master/LICENSE
