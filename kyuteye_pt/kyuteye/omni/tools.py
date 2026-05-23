"""Pluggable tool calling for Omni Assistant.

Tools fire when the model emits a ``[TOOL: name(arg=value, ...)]`` pattern
in its text stream. Tools may be sync or async callables; both are awaited
in a worker task so the conversation loop is never blocked.

There are two ways to register a tool:

* The :func:`tool` decorator wraps a Python function and adds it to a
  global registry. The function's parameter names define the accepted
  kwargs in the ``[TOOL: name(...)]`` syntax. Use this for the common
  fire-and-return-a-string case.

* Subclass :class:`BaseTool` and register an instance with
  :func:`register_tool`. Use this for stateful tools (e.g. ones that hold
  an open connection to a smart-home hub).

The default-arguments parser is intentionally simple: comma-separated
``key=value`` pairs, with bare positional arguments mapped to the
function's first parameters. Values are passed through verbatim as
strings; do your own coercion in the tool body.
"""

from __future__ import annotations

import abc
import inspect
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """Parsed ``[TOOL: name(args)]`` invocation."""

    name: str
    args: list[str] = field(default_factory=list)
    kwargs: dict[str, str] = field(default_factory=dict)
    raw: str = ""


class BaseTool(abc.ABC):
    """Stateful tool. Override :meth:`invoke` (sync or async).

    The name attribute is what ``[TOOL: name(...)]`` matches against. The
    description is informational only -- this implementation does not feed
    tool docs back into the model.
    """

    name: str = ""
    description: str = ""

    @abc.abstractmethod
    def invoke(self, *args: str, **kwargs: str) -> str | Awaitable[str]:
        """Execute the tool. Return a short result string for the UI / LM."""


class _FunctionTool(BaseTool):
    """Adapter that wraps a plain function as a :class:`BaseTool`."""

    def __init__(
        self, fn: Callable[..., Any], *, name: str, description: str = ""
    ) -> None:
        self.fn = fn
        self.name = name
        self.description = description or (fn.__doc__ or "").strip().split("\n", 1)[0]

    def invoke(self, *args: str, **kwargs: str) -> Any:
        return self.fn(*args, **kwargs)


class ToolRegistry:
    """Holds the registered tools and dispatches calls to them."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        if not tool.name:
            raise ValueError(f"Tool {tool!r} has no name")
        if tool.name in self._tools:
            logger.warning("[Tools] overriding existing tool %r", tool.name)
        self._tools[tool.name] = tool
        logger.info("[Tools] registered %r", tool.name)

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    async def dispatch(self, call: ToolCall) -> str:
        """Run the named tool. Returns its result string (or an error message)."""
        tool = self._tools.get(call.name)
        if tool is None:
            return f"[TOOL_ERROR: unknown tool {call.name!r}]"
        try:
            result = tool.invoke(*call.args, **call.kwargs)
            if inspect.isawaitable(result):
                result = await result
            return str(result)
        except TypeError as e:
            return f"[TOOL_ERROR: bad args for {call.name}: {e}]"
        except Exception as e:
            logger.exception("[Tools] %s raised", call.name)
            return f"[TOOL_ERROR: {call.name}: {e}]"


default_registry = ToolRegistry()


def register_tool(tool: BaseTool, *, registry: ToolRegistry | None = None) -> None:
    """Add a :class:`BaseTool` instance to a registry (default = process-wide)."""
    (registry or default_registry).register(tool)


def tool(
    name: str | None = None,
    *,
    description: str = "",
    registry: ToolRegistry | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that registers a plain function as a tool.

    .. code-block:: python

        @tool(name="weather", description="Get the current weather")
        def weather(city: str = "Paris") -> str:
            return f"It is sunny in {city}."
    """

    def wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        tool_name = name or fn.__name__
        (registry or default_registry).register(
            _FunctionTool(fn, name=tool_name, description=description)
        )
        return fn

    return wrap


_TOOL_PATTERN = re.compile(r"\[TOOL:\s*([a-zA-Z_][\w]*)\s*\((.*?)\)\s*\]")


def parse_tool_call(raw: str) -> Optional[ToolCall]:
    """Parse a single ``[TOOL: name(args)]`` string. Returns ``None`` if no match."""
    m = _TOOL_PATTERN.search(raw)
    if m is None:
        return None
    name = m.group(1)
    body = m.group(2).strip()
    args: list[str] = []
    kwargs: dict[str, str] = {}
    for part in (p.strip() for p in body.split(",")):
        if not part:
            continue
        if "=" in part:
            k, _, v = part.partition("=")
            kwargs[k.strip()] = v.strip().strip("\"'")
        else:
            args.append(part.strip("\"'"))
    return ToolCall(name=name, args=args, kwargs=kwargs, raw=m.group(0))
