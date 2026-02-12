"""
Example of integrating SIQSE with PyTerrier for query expansion.

This example demonstrates:
1. Loading a PyTerrier index
2. Using SIQSE with PyTerrier's BatchRetrieve for document feedback
3. Applying query expansion in a retrieval pipeline
4. Comparing baseline vs. expanded queries
"""

import pandas as pd
import pyterrier as pt
from siqse import LLMSuggestionSelector


def main():
    # Initialize PyTerrier
    if not pt.started():
        pt.init()

    # Load your PyTerrier index
    # Replace with your actual index path
    index_path = "/path/to/your/index"
    index = pt.IndexFactory.of(index_path)

    # Initialize BM25 retriever
    bm25 = pt.BatchRetrieve(index, wmodel="BM25")

    # Initialize SIQSE selector
    selector = LLMSuggestionSelector(
        model="llama3.1:8b",
        llm_api_url="http://localhost:11434",
        selection_mode="iterative",
        use_feedback_docs=True,
        temperature=0.0,
    )

    # Define query expansion function
    def expand_queries(df):
        """
        Expand queries using SIQSE selector with document feedback.

        Args:
            df: DataFrame with columns: qid, query, suggestions, (optional) intent

        Returns:
            DataFrame with expanded queries
        """

        def get_feedback_docs(query, selected_keywords):
            """Retrieve documents for feedback using BM25."""
            # Build expanded query
            expanded_query = query
            if selected_keywords:
                expanded_query = query + " " + " ".join(selected_keywords)

            # Retrieve top documents
            results = bm25.search(expanded_query).head(10)

            # Convert to format expected by SIQSE
            docs = []
            for idx, row in results.iterrows():
                docs.append(
                    {
                        "doc_id": row.get("docno", ""),
                        "score": row.get("score", 0.0),
                        "rank": row.get("rank", idx + 1),
                        "is_relevant": False,  # Set to True if you have qrels
                        "text": row.get("text", ""),  # Include document text if available
                    }
                )

            return docs

        # Apply selection to each query
        def expand_single_query(row):
            selected = selector.select_suggestions(
                query=row["query"],
                suggestions=row["suggestions"],
                user_intent=row.get("intent"),
                n=None,  # Let LLM decide how many
                get_docs=get_feedback_docs,
            )

            # Combine original query with selected suggestions
            if selected:
                expanded = row["query"] + " " + " ".join(selected)
            else:
                expanded = row["query"]

            return expanded

        # Create a copy to avoid modifying original
        df = df.copy()
        df["query"] = df.apply(expand_single_query, axis=1)

        return df

    # Example queries DataFrame
    # In practice, this would come from your topic file
    queries_df = pd.DataFrame(
        [
            {
                "qid": "1",
                "query": "climate change impacts",
                "suggestions": [
                    "global warming",
                    "greenhouse gases",
                    "carbon emissions",
                    "sea level rise",
                ],
                "intent": "Find information about environmental effects of climate change",
            },
            {
                "qid": "2",
                "query": "machine learning applications",
                "suggestions": [
                    "neural networks",
                    "deep learning",
                    "artificial intelligence",
                    "computer vision",
                ],
                "intent": "Discover practical uses of machine learning",
            },
            {
                "qid": "3",
                "query": "renewable energy sources",
                "suggestions": ["solar power", "wind energy", "hydroelectric", "geothermal"],
                "intent": None,  # No intent provided
            },
        ]
    )

    print("=" * 70)
    print("PyTerrier + SIQSE Query Expansion Example")
    print("=" * 70)
    print()

    # Show original queries
    print("Original Queries:")
    for _, row in queries_df.iterrows():
        print(f"  {row['qid']}: {row['query']}")
    print()

    # Create retrieval pipeline with query expansion
    print("Expanding queries with SIQSE...")
    print()

    expansion_pipeline = pt.apply.generic(expand_queries) >> bm25

    # Run retrieval
    results = expansion_pipeline.transform(queries_df)

    print("=" * 70)
    print("Retrieval Results")
    print("=" * 70)
    print()

    # Show top results per query
    for qid in queries_df["qid"].unique():
        query_results = results[results["qid"] == qid].head(5)
        original_query = queries_df[queries_df["qid"] == qid]["query"].iloc[0]
        expanded_query = (
            query_results["query"].iloc[0] if len(query_results) > 0 else original_query
        )

        print(f"Query {qid}:")
        print(f"  Original: {original_query}")
        print(f"  Expanded: {expanded_query}")
        print(f"  Top documents:")

        for idx, row in query_results.iterrows():
            print(
                f"    {row.get('rank', 0)}. {row.get('docno', 'N/A')} (score: {row.get('score', 0):.4f})"
            )
        print()

    print("=" * 70)
    print("Comparison: Baseline vs. Expanded")
    print("=" * 70)
    print()

    # Run baseline (no expansion) for comparison
    baseline_results = bm25.transform(queries_df)

    # Compare MAP if you have qrels
    # If you have qrels, you can evaluate:
    # from pyterrier.measures import MAP, nDCG
    # qrels = pd.read_csv("path/to/qrels.txt", sep="\s+", names=["qid", "iter", "docno", "label"])
    # baseline_map = pt.Utils.evaluate(baseline_results, qrels, metrics=[MAP])
    # expanded_map = pt.Utils.evaluate(results, qrels, metrics=[MAP])
    # print(f"Baseline MAP: {baseline_map}")
    # print(f"Expanded MAP: {expanded_map}")

    print("✓ Pipeline execution complete!")


if __name__ == "__main__":
    main()
