"""strot-ai: compile-time LLM code synthesis with deterministic,
self-routing NLU runtimes.

Implements the STROT framework (Structured Task Reasoning and Output
Transformation, arXiv:2505.01636) with an NLU strategy-pattern runtime:
a local CPU transformer (default ``SupersonicLabs/Julia-1``) or a hosted
cloud API, baked into a standalone compiled production asset.
"""

from .compiler import (
    CompiledAsset,
    StrotCompiler,
    anthropic_llm_client,
    claude_cli_llm_client,
    openai_llm_client,
)
from .exceptions import (
    CompilationError,
    FallbackException,
    RouterConfigurationError,
    SchemaIntrospectionError,
    StrotError,
)
from .router import (
    CONFIDENCE_THRESHOLD,
    DEFAULT_NLU_MODEL,
    MAX_FLAT_OPTIONS,
    HostedApiStrategy,
    LocalEncoderStrategy,
    RouteDecision,
    Router,
    RuntimeStrategy,
    create_router,
    create_strategy,
)
from .schema import DatasetSchema, FieldProfile, SchemaProfiler, introspect_schema

__version__ = "0.2.0"

from .jit import RunResult, run  # noqa: E402  (needs __version__ for cache keys)

__all__ = [
    "run",
    "RunResult",
    "CompiledAsset",
    "StrotCompiler",
    "openai_llm_client",
    "anthropic_llm_client",
    "claude_cli_llm_client",
    "StrotError",
    "CompilationError",
    "FallbackException",
    "RouterConfigurationError",
    "SchemaIntrospectionError",
    "CONFIDENCE_THRESHOLD",
    "DEFAULT_NLU_MODEL",
    "MAX_FLAT_OPTIONS",
    "RuntimeStrategy",
    "LocalEncoderStrategy",
    "HostedApiStrategy",
    "RouteDecision",
    "Router",
    "create_router",
    "create_strategy",
    "DatasetSchema",
    "FieldProfile",
    "SchemaProfiler",
    "introspect_schema",
    "__version__",
]
