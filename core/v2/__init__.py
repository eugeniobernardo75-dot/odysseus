"""
Odysseus V2 — Sistema de Agente Principal.

Módulos:
    - api_bridge:       Interface HTTP para o endpoint /api/v2/chat
    - orchestrator:     Agente orquestrador inteligente (planeia, delega, valida)
    - agents:           Catálogo de agentes especializados
    - opencode_agent:   Agente OpenCode para coding assistido via CLI
    - user_agent_config: Agente pessoal configurável do utilizador
    - memory:           Memória temporária (sessão) e permanente (persistente)
    - safety:           Camada de segurança para validação de comandos e ações
    - tool_registry:    Registo central de ferramentas
    - tool_manager:     Motor de execução de ferramentas
    - workspace:        Sistema de espaços de trabalho isolados
    - tools:            Coleção de ferramentas base do sistema
    - tool_chaining:    Sistema de encadeamento automático de ferramentas
    - browser_tool:     Ferramenta de automação web (Playwright/httpx)
"""

from .api_bridge import process_request, process_request_sync, is_opencode_available, get_opencode_info, execute_tool, list_available_tools
from .orchestrator import get_orchestrator
from .safety import get_safety_manager
from .memory import get_memory_manager
from .agents import get_agent_factory
from .llm_client import get_llm_client
from .cache import get_cache, SemanticCache, CacheConfig
from .opencode_agent import OpenCodeAgent, OpenCodeConfig, get_opencode_agent
from .user_agent_config import UserAgent, UserConfig, get_user_agent
from .planner_agent import PlannerAgent, Plan, PlanStep
from .reflection_agent import ReflectionAgent, ReflectionResult
from .tool_registry import ToolRegistry, ToolDefinition, ToolResult, BaseTool, get_tool_registry
from .tool_manager import ToolManager, get_tool_manager, ExecutionRecord, ToolEvent, ToolEventCallback
from .workspace import WorkspaceConfig, SessionWorkspace, WorkspaceManager, get_workspace_manager
from .tools import (
    WebTool, FileTool, ShellTool, PythonTool,
    MemoryTool, SearchTool, CodeTool, GitTool,
    register_base_tools,
)
from .browser_tool import BrowserTool
from .data_routes import setup_data_routes
from .orchestrator_adapter import OrchestratorAdapter, create_adapter_from_config
from .tool_chaining import (
    ChainStep, ChainDefinition, ChainResult, ChainExecutor,
    ChainRegistry, ChainRunner,
    get_chain_registry, reset_chain_registry,
    register_default_chains,
)

from .memory_backend import (
    MemoryBackend,
    MemoryRecord,
    DictMemoryBackend,
    ChromaMemoryBackend,
)

__all__ = [
    "process_request",
    "process_request_sync",
    "is_opencode_available",
    "get_opencode_info",
    "execute_tool",
    "list_available_tools",
    "get_orchestrator",
    "get_safety_manager",
    "get_memory_manager",
    "get_agent_factory",
    "get_llm_client",
    "get_cache",
    "SemanticCache",
    "CacheConfig",
    "OpenCodeAgent",
    "OpenCodeConfig",
    "get_opencode_agent",
    "UserAgent",
    "UserConfig",
    "get_user_agent",
    # Planning & Reflection
    "PlannerAgent",
    "Plan",
    "PlanStep",
    "ReflectionAgent",
    "ReflectionResult",
    # Tool System
    "ToolRegistry",
    "ToolDefinition",
    "ToolResult",
    "BaseTool",
    "get_tool_registry",
    "ToolManager",
    "get_tool_manager",
    "ExecutionRecord",
    "ToolEvent",
    "ToolEventCallback",
    "WorkspaceConfig",
    "SessionWorkspace",
    "WorkspaceManager",
    "get_workspace_manager",
    "WebTool",
    "FileTool",
    "ShellTool",
    "PythonTool",
    "MemoryTool",
    "SearchTool",
    "CodeTool",
    "GitTool",
    "register_base_tools",
    # Browser Tool
    "BrowserTool",
    # Tool Chaining
    "ChainStep",
    "ChainDefinition",
    "ChainResult",
    "ChainExecutor",
    "ChainRegistry",
    "ChainRunner",
    "get_chain_registry",
    "reset_chain_registry",
    "register_default_chains",
    # Data Routes
    "setup_data_routes",
    # OrchestratorAdapter
    "OrchestratorAdapter",
    "create_adapter_from_config",
    # Memory Backend
    "MemoryBackend",
    "MemoryRecord",
    "DictMemoryBackend",
    "ChromaMemoryBackend",
]

from .data_routes import setup_data_routes
from .orchestrator_adapter import OrchestratorAdapter, create_adapter_from_config
