def web_search(query: str, max_results: int = 5):
    """Placeholder for Tavily or other web-fallback search."""
    return [f"Result {i+1} for '{query}'" for i in range(max_results)]
