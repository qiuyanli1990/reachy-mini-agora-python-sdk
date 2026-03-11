from __future__ import annotations
import abc
import sys
import json
import inspect
import logging
import importlib
from typing import Any, Dict, List
from dataclasses import dataclass

from reachy_mini import ReachyMini


logger = logging.getLogger(__name__)


if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s:%(lineno)d | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


ALL_TOOLS: Dict[str, "Tool"] = {}
ALL_TOOL_SPECS: List[Dict[str, Any]] = []
_TOOLS_INITIALIZED = False
DEFAULT_TOOL_MODULES: tuple[str, ...] = (
    "dance",
    "stop_dance",
    "play_emotion",
    "stop_emotion",
    "move_head",
    "head_tracking",
    "camera",
    "do_nothing",
)



def get_concrete_subclasses(base: type[Tool]) -> List[type[Tool]]:
    """Recursively find all concrete (non-abstract) subclasses of a base class."""
    result: List[type[Tool]] = []
    for cls in base.__subclasses__():
        if not inspect.isabstract(cls):
            result.append(cls)
        # recurse into subclasses
        result.extend(get_concrete_subclasses(cls))
    return result


@dataclass
class ToolDependencies:
    """External dependencies injected into tools."""

    reachy_mini: ReachyMini
    movement_manager: Any  # MovementManager from moves.py
    # Optional deps
    camera_worker: Any | None = None  # CameraWorker for frame buffering
    vision_manager: Any | None = None
    head_wobbler: Any | None = None  # HeadWobbler for audio-reactive motion
    motion_duration_s: float = 1.0


# Tool base class
class Tool(abc.ABC):
    """Base abstraction for tools used in function-calling.

    Each tool must define:
      - name: str
      - description: str
      - parameters_schema: Dict[str, Any]  # JSON Schema
    """

    name: str
    description: str
    parameters_schema: Dict[str, Any]

    def spec(self) -> Dict[str, Any]:
        """Return the function spec for LLM consumption."""
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema,
        }

    @abc.abstractmethod
    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Async tool execution entrypoint."""
        raise NotImplementedError


def _format_error(error: Exception) -> str:
    """Format an exception for logging."""
    if isinstance(error, ModuleNotFoundError):
        return f"Missing dependency: {error}"
    if isinstance(error, ImportError):
        return f"Import error: {error}"
    return f"{type(error).__name__}: {error}"


def _load_local_tools() -> None:
    """Load built-in local tools only (agent_config-driven runtime)."""
    logger.info(
        "Loading local tools (fixed set): %s",
        list(DEFAULT_TOOL_MODULES),
    )
    for tool_name in DEFAULT_TOOL_MODULES:
        module_path = f"reachy_mini_agora_python_sdk.tools.{tool_name}"
        try:
            importlib.import_module(module_path)
            logger.info("✓ Loaded local tool: %s", tool_name)
        except Exception as e:
            logger.error(
                "❌ Failed to load local tool '%s' from %s: %s",
                tool_name,
                module_path,
                _format_error(e),
            )
            sys.exit(1)



def _initialize_tools() -> None:
    """Populate registry once, even if module is imported repeatedly."""
    global ALL_TOOLS, ALL_TOOL_SPECS, _TOOLS_INITIALIZED

    if _TOOLS_INITIALIZED:
        logger.debug("Tools already initialized; skipping reinitialization.")
        return

    _load_local_tools()

    ALL_TOOLS = {cls.name: cls() for cls in get_concrete_subclasses(Tool)}  # type: ignore[type-abstract]
    ALL_TOOL_SPECS = [tool.spec() for tool in ALL_TOOLS.values()]

    for tool_name, tool in ALL_TOOLS.items():
        logger.info(f"tool registered: {tool_name} - {tool.description}")

    _TOOLS_INITIALIZED = True


_initialize_tools()


def get_tool_specs(exclusion_list: list[str] = []) -> list[Dict[str, Any]]:
    """Get tool specs, optionally excluding some tools."""
    return [spec for spec in ALL_TOOL_SPECS if spec.get("name") not in exclusion_list]


# Dispatcher
def _safe_load_obj(args_json: str) -> Dict[str, Any]:
    try:
        parsed_args = json.loads(args_json or "{}")
        return parsed_args if isinstance(parsed_args, dict) else {}
    except Exception:
        logger.warning("bad args_json=%r", args_json)
        return {}


async def dispatch_tool_call(tool_name: str, args_json: str, deps: ToolDependencies) -> Dict[str, Any]:
    """Dispatch a tool call by name with JSON args and dependencies."""
    tool = ALL_TOOLS.get(tool_name)

    if not tool:
        return {"error": f"unknown tool: {tool_name}"}

    args = _safe_load_obj(args_json)
    try:
        return await tool(deps, **args)
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        logger.exception("Tool error in %s: %s", tool_name, msg)
        return {"error": msg}
