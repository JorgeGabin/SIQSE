"""
SIQSE - Simulation-based Interactive Query Suggestion Evaluation
"""

from .selector import LLMSuggestionSelector, SuggestionSelection
from .utils import (
    clean_query_for_pyterrier,
    format_document_for_display,
    count_relevant_documents,
    build_prompt_header,
    format_suggestions_list,
)

__version__ = "0.1.0"

__all__ = [
    "LLMSuggestionSelector",
    "SuggestionSelection",
    "clean_query_for_pyterrier",
    "format_document_for_display",
    "count_relevant_documents",
    "build_prompt_header",
    "format_suggestions_list",
]
