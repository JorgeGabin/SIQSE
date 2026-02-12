"""
Utility functions for LLM-based keyphrase selection.

Provides formatting, cleaning, and helper functions for working with queries,
documents, and suggestions in the SIQSE system.
"""

from typing import List, Dict


def clean_query_for_pyterrier(query: str) -> str:
    """
    Clean a query string to make it safe for PyTerrier's query parser.

    PyTerrier's query parser is sensitive to special characters like quotes,
    slashes, apostrophes, brackets, and punctuation. This function removes or
    replaces these characters to prevent parsing errors.

    Args:
        query: Raw query string that may contain special characters

    Returns:
        Cleaned query string safe for PyTerrier

    Example:
        >>> clean_query_for_pyterrier("food/drug laws")
        'food drug laws'
        >>> clean_query_for_pyterrier("Parkinson's disease")
        'Parkinsons disease'
    """
    # Remove or replace problematic characters for PyTerrier parser
    cleaned = (
        query.replace('"', "")  # Double quotes
        .replace("'", "")  # Single quotes/apostrophes
        .replace("`", "")  # Backticks
        .replace("(", "")  # Parentheses
        .replace(")", "")
        .replace("[", "")  # Brackets
        .replace("]", "")
        .replace("{", "")  # Braces
        .replace("}", "")
        .replace(":", "")  # Colons
        .replace(";", "")  # Semicolons
        .replace("!", "")  # Exclamation marks
        .replace("?", "")  # Question marks
        .replace("/", " ")  # Replace slash with space to preserve word boundaries
        .strip()
    )

    return cleaned


def format_document_for_display(doc: Dict, snippet_chars: int = 500) -> str:
    """
    Format a document dictionary for display in LLM prompts.

    Args:
        doc: Document dict with keys: rank, doc_id, score, is_relevant, text (optional)
        snippet_chars: Maximum number of characters of document text to show

    Returns:
        Formatted string with metadata and a text snippet
    """
    rank = doc.get("rank", 0)
    doc_id = doc.get("doc_id", "")
    score = doc.get("score", 0.0)
    is_relevant = doc.get("is_relevant", False)
    text = doc.get("text") or ""

    marker = " [RELEVANT]" if is_relevant else ""
    header = f"  {rank}. Document {doc_id} (score: {score:.2f}){marker}"

    if not text:
        return header

    # Clean and truncate text to snippet
    text = " ".join(text.split())
    if len(text) > snippet_chars:
        text = text[:snippet_chars].rsplit(" ", 1)[0] + "..."

    return f"{header}\n      {text}"


def count_relevant_documents(documents: List[Dict]) -> int:
    """
    Count how many documents in a list are marked as relevant.

    Args:
        documents: List of document dicts with 'is_relevant' key

    Returns:
        Count of relevant documents
    """
    return sum(1 for doc in documents if doc.get("is_relevant", False))


def build_prompt_header(query: str, user_intent: str = None) -> List[str]:
    """
    Build the common header parts for LLM prompts.

    Args:
        query: The search query
        user_intent: Optional user intent description

    Returns:
        List of prompt lines
    """
    parts = [f"I am searching for: {query}"]

    if user_intent:
        parts.append(f"\nMy search intent is: {user_intent}")

    return parts


def format_suggestions_list(suggestions: List[str], label: str = None) -> List[str]:
    """
    Format a list of suggestions as numbered items.

    Args:
        suggestions: List of suggestion strings
        label: Optional label to prepend (e.g., "Remaining keyphrase suggestions:")

    Returns:
        List of formatted lines
    """
    parts = []
    if label:
        parts.append(f"\n{label}")

    parts.extend(f"{i}. {s}" for i, s in enumerate(suggestions, 1))
    return parts
