"""Example Omni Assistant plugin.

Pass this module path to the server:

    server --omni-plugin=kyuteye.omni.example_plugin --kyuteye-config-path=...

Importing the module registers a default :class:`LLMRetriever` (requires
``LLM_BASE_URL`` and an OpenAI-compatible endpoint) and three example
tools that demonstrate the dispatch surface.

Real deployments should put their own plugin module on the import path and
register a domain-specific retriever (vector store, in-memory dict, etc.).
"""

from __future__ import annotations

import datetime
import logging
import os

from kyuteye.omni import LLMRetriever, set_retriever, tool

logger = logging.getLogger(__name__)


# --- Retriever ---------------------------------------------------------------
# Only enable the LLM-based retriever if the user provided an LLM endpoint;
# otherwise leave the registry empty and let the server fall back to "no RAG".
if os.environ.get("LLM_BASE_URL"):
    set_retriever(LLMRetriever())
else:
    logger.warning(
        "[example_plugin] LLM_BASE_URL not set; default retriever disabled. "
        "Set it (and LLM_MODEL_NAME / LLM_API_KEY if needed) or register a "
        "custom retriever via `omni.set_retriever(...)`."
    )


# --- Tools -------------------------------------------------------------------
@tool(name="time", description="Get the current local time")
def current_time() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")


@tool(name="echo", description="Repeat the message back")
def echo(message: str = "") -> str:
    return message


@tool(name="light", description="Toggle a smart-home light on or off")
def light(room: str = "living_room", state: str = "on") -> str:
    # Replace this with an actual smart-home call.
    return f"Set {room} light to {state}."
