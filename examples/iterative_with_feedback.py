"""
Example of using SIQSE with iterative selection and document feedback.

This example demonstrates:
1. Iterative selection mode
2. Integration with a mock retrieval system
3. Document feedback to guide selection
"""

from siqse import LLMSuggestionSelector
from typing import List


def mock_retrieval_system(query: str, keywords: List[str]) -> List[dict]:
    """
    Mock retrieval function that simulates document retrieval.

    In a real system, this would:
    1. Combine query with selected keywords
    2. Retrieve documents from an index
    3. Return ranked results with relevance judgments

    Args:
        query: The base query
        keywords: Currently selected keywords

    Returns:
        List of document dicts with: doc_id, score, rank, is_relevant, text
    """
    # Build expanded query
    expanded_query = query
    if keywords:
        expanded_query = f"{query} {' '.join(keywords)}"

    print(f"  [Retrieval] Expanded query: {expanded_query}")

    # Mock documents - in reality these would come from an IR system
    # Simulate different results based on keywords
    if "global warming" in keywords:
        docs = [
            {
                "doc_id": "doc1",
                "score": 0.95,
                "rank": 1,
                "is_relevant": True,
                "text": "Climate change and global warming have significant impacts on agricultural productivity, affecting crop yields worldwide.",
            },
            {
                "doc_id": "doc2",
                "score": 0.87,
                "rank": 2,
                "is_relevant": True,
                "text": "Rising temperatures due to global warming lead to changes in precipitation patterns, affecting water availability for crops.",
            },
            {
                "doc_id": "doc3",
                "score": 0.65,
                "rank": 3,
                "is_relevant": False,
                "text": "The history of thermometers and temperature measurement in the 19th century.",
            },
        ]
    elif "crop yields" in keywords:
        docs = [
            {
                "doc_id": "doc4",
                "score": 0.92,
                "rank": 1,
                "is_relevant": True,
                "text": "Studies show that crop yields are declining in many regions due to changing climate conditions and extreme weather events.",
            },
            {
                "doc_id": "doc5",
                "score": 0.81,
                "rank": 2,
                "is_relevant": True,
                "text": "Agricultural research focuses on developing climate-resilient crops to maintain yields under changing conditions.",
            },
        ]
    else:
        # Initial retrieval without keywords
        docs = [
            {
                "doc_id": "doc6",
                "score": 0.75,
                "rank": 1,
                "is_relevant": False,
                "text": "Weather forecasting methods and their accuracy in predicting climate events.",
            },
            {
                "doc_id": "doc7",
                "score": 0.70,
                "rank": 2,
                "is_relevant": True,
                "text": "Climate change affects agriculture through temperature changes, altered rainfall patterns, and increased extreme weather.",
            },
        ]

    print(f"  [Retrieval] Retrieved {len(docs)} documents")
    relevant_count = sum(1 for d in docs if d["is_relevant"])
    print(f"  [Retrieval] Relevant documents: {relevant_count}/{len(docs)}")

    return docs


def main():
    # Initialize selector in iterative mode with document feedback
    selector = LLMSuggestionSelector(
        model="llama3.1:70b",
        llm_api_url="http://localhost:11434",
        selection_mode="iterative",
        temperature=0.0,
        use_feedback_docs=True,
    )

    # Example query and suggestions
    query = "climate change impacts on agriculture"
    suggestions = [
        "global warming",
        "crop yields",
        "extreme weather",
        "sustainable farming",
        "carbon emissions",
    ]

    print("=" * 60)
    print("Iterative Selection with Document Feedback")
    print("=" * 60)
    print(f"Query: {query}")
    print(f"Available suggestions: {suggestions}")
    print()

    # Select keyphrases iteratively
    print("Starting iterative selection...")
    print()

    selected = selector.select_suggestions(
        query=query,
        suggestions=suggestions,
        user_intent="Find research on how climate affects food production",
        n=3,  # Select up to 3 keyphrases
        get_docs=mock_retrieval_system,
    )

    print()
    print("=" * 60)
    print("Final Results")
    print("=" * 60)
    print(f"Selected keyphrases: {selected}")
    print(f"Total selected: {len(selected)}")
    print()


if __name__ == "__main__":
    main()
