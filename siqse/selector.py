from typing import List, Optional, Callable, Tuple
from openai import OpenAI
from pydantic import BaseModel, Field
import time

from .utils import format_document_for_display


class SuggestionSelection(BaseModel):
    """Structured output: Pydantic model for LLM response with selected suggestions."""

    selected_suggestions: List[str] = Field(
        description="List of the actual suggestion texts that are most relevant to the query"
    )


class LLMSuggestionSelector:
    """
    A class that uses LLMs (via OpenAI-compatible API) to select relevant suggestions
    based on a query and optional user intent.

    Supports two selection modes:
    - Batch: Select all suggestions at once
    - Iterative: Select suggestions one-by-one with feedback
    """

    def __init__(
        self,
        model: str = "llama3.1:70b",
        llm_api_url: str = "http://localhost:11434",
        llm_api_key: str = "ollama",
        temperature: float = 0.0,
        selection_mode: str = "batch",
        use_feedback_docs: bool = True,
    ):
        """
        Initialize the LLM Suggestion Selector.

        Args:
            model: The LLM model to use (default: "llama3.1:70b")
            llm_api_url: The URL of the LLM service (default: "http://localhost:11434")
            llm_api_key: API key for LLM service (default: "ollama", can be any string)
            temperature: Temperature for LLM generation (default: 0.0 for deterministic results)
            selection_mode: "batch" (select all at once) or "iterative" (select one by one)
            use_feedback_docs: Whether to include retrieved documents in prompts (default: True)
        """
        self.model = model
        self.temperature = temperature
        self.selection_mode = selection_mode
        self.use_feedback_docs = use_feedback_docs
        self.client = OpenAI(base_url=f"{llm_api_url}/v1", api_key=llm_api_key)

    def select_suggestions(
        self,
        query: str,
        suggestions: List[str],
        user_intent: Optional[str] = None,
        n: Optional[int] = None,
        get_docs: Optional[Callable[[str, List[str]], List[dict]]] = None,
    ) -> List[str]:
        """
        Select the most relevant suggestions using the LLM.

        Args:
            query: The user's query
            suggestions: List of suggestions to select from
            user_intent: Optional user intent to guide selection
            n: Number of suggestions to select. If None, LLM decides how many to return.
            get_docs: Optional function to retrieve documents for feedback.
                Signature: func(query: str, selected_suggestions: List[str]) -> List[dict]
                Returns list of dicts with keys: doc_id, score, rank, is_relevant, text (optional)
                Called with empty list initially, then with selected suggestions in iterative mode.

        Returns:
            List of selected suggestions
        """
        if not suggestions:
            return []

        # Batch mode: select all at once
        if self.selection_mode == "batch":
            docs = None
            if self.use_feedback_docs:
                if not get_docs:
                    raise ValueError(
                        "get_docs function must be provided when use_feedback_docs is True"
                    )

                docs = get_docs(query, [])

            system_prompt, user_prompt = self._build_prompt(
                query,
                suggestions,
                user_intent,
                docs,
                n,
                selected=[],
                is_iterative=False,
            )

            response = self._call_llm(system_prompt, user_prompt)

            selected = self._validate_selections(response.selected_suggestions, suggestions)

            return selected[:n] if n is not None else selected

        # Iterative mode: select one by one
        selected = []
        remaining = suggestions.copy()

        while remaining and (n is None or len(selected) < n):
            # Get documents (initial or updated based on selections)
            current_docs = None
            if self.use_feedback_docs:
                if not get_docs:
                    raise ValueError(
                        "get_docs function must be provided when use_feedback_docs is True"
                    )

                current_docs = get_docs(query, selected)

            # Build prompt for next selection
            system_prompt, user_prompt = self._build_prompt(
                query,
                remaining,
                user_intent,
                current_docs,
                n,
                selected=selected,
                is_iterative=True,
            )

            response = self._call_llm(system_prompt, user_prompt)

            if not response or not response.selected_suggestions:
                break

            validated = self._validate_selections(response.selected_suggestions, remaining)
            if not validated:
                break

            next_selection = validated[0]
            selected.append(next_selection)
            remaining.remove(next_selection)

        return selected

    def _build_prompt(
        self,
        query: str,
        suggestions: List[str],
        user_intent: Optional[str],
        documents: Optional[List[dict]],
        n: Optional[int],
        selected: List[str],
        is_iterative: bool,
    ) -> Tuple[str, str]:
        """Build system and user prompts for LLM selection."""
        # System prompt - set the role and context
        system_prompt = (
            "You are a user conducting a search. You are trying to improve your search results by "
            "selecting keyphrase suggestions to add to your query. \n"
            "Think carefully about which suggestions would help you find more relevant documents. "
            "Base your decisions on whether a suggestion would meaningfully improve the search results."
        )

        # User prompt parts
        parts = []

        # What you're searching for
        parts.append(f"You are searching for: {query}")
        if user_intent:
            parts.append(f"Your specific intent is: {user_intent}")

        # Current results (if available)
        if documents:
            parts.append("\nYour current search results:")
            for d in documents:
                parts.append(format_document_for_display(d))

        # Available suggestions
        parts.append("\nKeyphrase suggestions you can add to your query:")
        for i, s in enumerate(suggestions, 1):
            parts.append(f"{i}. {s}")

        # Instructions
        parts.append(self._build_instructions(n, len(selected), is_iterative, bool(documents)))

        return system_prompt, "\n".join(parts)

    def _build_instructions(
        self, n: Optional[int], num_selected: int, is_iterative: bool, has_docs: bool
    ) -> str:
        """Build selection instructions based on mode and constraints."""

        if is_iterative:
            # Iterative mode: select ONE
            if n is not None:
                remaining = n - num_selected
                if remaining <= 0:
                    return "\nYou have selected enough suggestions. Return an empty list."

                if has_docs:
                    return (
                        "\nLooking at your current results, which ONE keyphrase should you add next to get more relevant documents?\n"
                        "Return the exact text of the ONE suggestion to select (copy it exactly from the list above)."
                    )
                else:
                    return (
                        "\nWhich ONE keyphrase should you add next to improve your search?\n"
                        "Return the exact text of the ONE suggestion to select (copy it exactly from the list above)."
                    )
            else:
                if has_docs:
                    return (
                        "\nLooking at your results, which ONE keyphrase should you add next? "
                        "If your results already look good with many relevant documents, or if none of the remaining suggestions would help, "
                        "return an empty list to stop.\n"
                        "Otherwise, return the exact text of the ONE suggestion to select next (or an empty list to stop)."
                    )
                else:
                    return (
                        "\nWhich ONE keyphrase should you add next to improve your search? "
                        "If you think you have enough already or none of the remaining suggestions would help, return an empty list to stop.\n"
                        "Otherwise, return the exact text of the ONE suggestion to select next (or an empty list to stop)."
                    )
        else:
            # Batch mode: select ALL or top N
            if n is not None:
                if has_docs:
                    return (
                        f"\nBased on your current results, select EXACTLY {n} keyphrases that would help you get more relevant documents.\n"
                        f"You MUST select {n} suggestions (or all of them if fewer than {n} are available).\n"
                        "Return the exact text of the suggestions to select (copy them exactly from the list above)."
                    )
                else:
                    return (
                        f"\nSelect EXACTLY {n} keyphrases that would most improve your search.\n"
                        f"You MUST select {n} suggestions (or all of them if fewer than {n} are available).\n"
                        "Return the exact text of the suggestions to select (copy them exactly from the list above)."
                    )
            else:
                if has_docs:
                    return (
                        "\nBased on your current results, which keyphrases would help you get more relevant documents? "
                        "Only pick suggestions that would meaningfully improve your results.\n"
                        "Return the exact text of the suggestions to select (copy them exactly from the list above)."
                    )
                else:
                    return (
                        "\nWhich keyphrases would improve your search? "
                        "Only pick suggestions that are truly relevant and would help.\n"
                        "Return the exact text of the suggestions to select (copy them exactly from the list above)."
                    )

    def _call_llm(
        self, system_prompt: str, user_prompt: str, max_retries: int = 3
    ) -> SuggestionSelection | None:
        """Call the LLM API via OpenAI-compatible interface with structured output."""
        for attempt in range(max_retries):
            try:
                completion = self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.temperature,
                    response_format=SuggestionSelection,
                )

                return completion.choices[0].message.parsed

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2**attempt
                    print(f"Warning: LLM call failed (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"Failed to call LLM after {max_retries} attempts: {e}")

    def _validate_selections(self, selected: List[str], suggestions: List[str]) -> List[str]:
        """Validate that selected suggestions are from the original list."""
        if not selected:
            return []

        # Filter to only include suggestions that are in the original list
        validated = []
        for sel in selected:
            if sel in suggestions:
                validated.append(sel)

        return validated
