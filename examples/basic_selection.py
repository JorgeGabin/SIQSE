"""
Basic example of using SIQSE for keyphrase selection.

This example demonstrates:
1. Basic batch selection
2. Selection with fixed k
3. Automatic k selection (LLM decides)
"""

from siqse import LLMSuggestionSelector


def main():
    # Initialize the selector
    selector = LLMSuggestionSelector(
        model="llama3.1:70b",
        llm_api_url="http://localhost:11434",
        selection_mode="batch",
        temperature=0.0,
        use_feedback_docs=False,  # No document feedback for this example
    )

    # Example query and suggestions
    query = "climate change impacts on agriculture"
    suggestions = [
        "global warming",
        "crop yields",
        "extreme weather",
        "sustainable farming",
        "carbon emissions",
        "soil degradation",
        "water scarcity",
    ]

    # Example 1: Select top 3 keyphrases
    print("=" * 60)
    print("Example 1: Select top 3 keyphrases")
    print("=" * 60)
    print(f"Query: {query}")
    print(f"Available suggestions: {suggestions}")
    print()

    selected = selector.select_suggestions(query=query, suggestions=suggestions, n=3)

    print(f"Selected keyphrases: {selected}")
    print()

    # Example 2: With user intent
    print("=" * 60)
    print("Example 2: Selection with user intent")
    print("=" * 60)

    user_intent = "I want to understand how rising temperatures affect food production"

    selected_with_intent = selector.select_suggestions(
        query=query, suggestions=suggestions, user_intent=user_intent, n=3
    )

    print(f"Query: {query}")
    print(f"Intent: {user_intent}")
    print(f"Selected keyphrases: {selected_with_intent}")
    print()

    # Example 3: Automatic selection (no fixed k)
    print("=" * 60)
    print("Example 3: Automatic selection (LLM decides count)")
    print("=" * 60)

    selected_auto = selector.select_suggestions(
        query=query, suggestions=suggestions, n=None  # Let LLM decide how many
    )

    print(f"Query: {query}")
    print(f"Selected keyphrases (auto): {selected_auto}")
    print(f"LLM chose {len(selected_auto)} keyphrases")
    print()


if __name__ == "__main__":
    main()
