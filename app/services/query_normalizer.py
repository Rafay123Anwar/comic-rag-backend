"""
query_normalizer.py -- M-11 Query Normalization

Lightweight, non-semantic query preprocessing:
  - Trim leading/trailing whitespace
  - Collapse repeated internal whitespace to a single space

Does NOT:
  - Rewrite named entities
  - Add or remove keywords
  - Call any LLM
  - Change the semantic meaning of the query
"""

import re


def normalize_query(query: str) -> str:
    """
    Normalize a raw user query for embedding/retrieval.

    Steps
    -----
    1. Strip leading and trailing whitespace.
    2. Collapse any run of internal whitespace (spaces, tabs,
       newlines) down to a single ASCII space.

    The original question string is NOT mutated; a new string
    is returned.  Named entities and all other content are
    preserved exactly.

    Parameters
    ----------
    query : str
        Raw user question, possibly containing extra whitespace.

    Returns
    -------
    str
        Normalized query ready for embedding.  May be an empty
        string if the input contained only whitespace.

    Examples
    --------
    >>> normalize_query("  What   did   Victor   say?  ")
    'What did Victor say?'
    >>> normalize_query("   ")
    ''
    >>> normalize_query("What did Victor say?")
    'What did Victor say?'
    """
    if not isinstance(query, str):
        return ""
    # Step 1 -- strip outer whitespace
    stripped = query.strip()
    # Step 2 -- collapse internal whitespace runs
    normalized = re.sub(r"\s+", " ", stripped)
    return normalized
