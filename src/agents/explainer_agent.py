import json
import os

from langchain_community.llms import Ollama
from langchain_community.tools.tavily_search import TavilySearchResults

import config
from src.tools.vector_store import store_text_in_local_db, query_vector_store


def _get_local_llm() -> Ollama:
    return Ollama(model=config.OLLAMA_MODEL_NAME, base_url=config.OLLAMA_BASE_URL)


def explain_with_ollama(raw_text: str, enable_web_search: bool = True) -> str:
    """
    Store document context locally, retrieve the most relevant chunks, and ask
    the local Ollama model to generate an explanation.
    """
    print("--- OLLAMA: Initializing explanation process ---")

    store_text_in_local_db(raw_text)
    core_context = query_vector_store(
        "What are the main topics and key data points in this document?"
    )

    web_context = ""
    if enable_web_search and config.TAVILY_API_KEY and config.TAVILY_API_KEY != "your_tavily_key":
        try:
            os.environ["TAVILY_API_KEY"] = config.TAVILY_API_KEY
            web_search_tool = TavilySearchResults(max_results=3)
            web_results = web_search_tool.invoke(
                {"query": "Provide supporting context for the main topics in this document."}
            )
            if web_results:
                web_context = json.dumps(web_results, indent=2)
        except Exception as exc:
            web_context = f"Web search unavailable: {exc}"

    prompt = (
        "You are an explainability assistant. Summarize the document clearly, "
        "identify the main topics, key data points, and notable conclusions.\n\n"
        f"Document context:\n{core_context}\n\n"
        f"Optional web context:\n{web_context or 'Not used.'}\n"
    )

    try:
        return _get_local_llm().invoke(prompt).strip()
    except Exception as exc:
        return f"Local Ollama generation failed: {exc}"
