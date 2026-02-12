"""
Minimal PyTerrier integration example matching the compact style.

This shows the exact pattern for integrating SIQSE into a PyTerrier pipeline.
"""

import pandas as pd
import pyterrier as pt
from siqse import LLMSuggestionSelector

# Initialize PyTerrier
if not pt.started():
    pt.init()

# Load index (replace with your index path)
index_path = "/path/to/your/index"
index = pt.IndexFactory.of(index_path)

# Initialize selector and retriever
selector = LLMSuggestionSelector(
    model="llama3.1:8b",
    llm_api_url="http://localhost:11434",
    selection_mode="iterative",
    use_feedback_docs=True,
)
bm25 = pt.BatchRetrieve(index, wmodel="BM25")


# Define expansion function
def expand(df):
    """Expand queries using SIQSE with document feedback."""

    # Feedback function that retrieves docs for selection
    def fb(q, s):
        expanded = q + " " + " ".join(s) if s else q
        return bm25.search(expanded).head(10).to_dict("records")

    # Expand each query
    df = df.copy()
    df["query"] = df.apply(
        lambda r: " ".join(
            [r["query"]]
            + selector.select_suggestions(
                query=r["query"],
                suggestions=r["suggestions"],
                user_intent=r.get("intent"),
                n=None,  # Auto selection
                get_docs=fb,
            )
        ),
        axis=1,
    )
    return df


# Example usage
if __name__ == "__main__":
    # Prepare queries with suggestions
    queries_df = pd.DataFrame(
        [
            {
                "qid": "1",
                "query": "climate change",
                "suggestions": ["global warming", "emissions", "temperature rise"],
                "intent": "Environmental impacts of climate change",
            }
        ]
    )

    # Build pipeline: expand queries then retrieve
    pipeline = pt.apply.generic(expand) >> bm25

    # Run retrieval
    results = pipeline.transform(queries_df)

    print("Expanded query:", results["query"].iloc[0])
    print("\nTop results:")
    print(results[["qid", "docno", "score", "rank"]].head())

    # Run retrieval without expansion for comparison
    print("\nRetrieval without expansion:")
    no_expansion_results = bm25.transform(queries_df[["qid", "query"]])
    print(no_expansion_results[["qid", "docno", "score", "rank"]].head())
