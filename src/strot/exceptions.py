"""Exception hierarchy for the STROT NLU pipeline.

Every failure mode in the two lifecycle phases maps to a dedicated
exception so that callers can distinguish compile-time synthesis
failures from runtime routing refusals.
"""

from __future__ import annotations

from typing import Optional, Sequence


class StrotError(Exception):
    """Base class for all errors raised by strot-ai."""


class SchemaIntrospectionError(StrotError):
    """The dataset sample could not be profiled into a structural schema."""


class CompilationError(StrotError):
    """The synthesizer exhausted its self-correction budget without
    producing syntactically valid code that satisfies the output contract."""

    def __init__(self, message: str, attempts: int, last_error: Optional[str] = None) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.last_error = last_error


class RouterConfigurationError(StrotError):
    """The requested ``nlu_model`` could not be resolved to a runtime strategy,
    or the strategy's dependencies are missing from the environment."""


class FallbackException(StrotError):
    """Raised by the runtime router when the calibrated confidence of the
    winning execution path falls below the enforcement threshold.

    Production pipelines register a handler for this exception instead of
    letting low-confidence (potentially corrupt) inputs execute a random path.
    """

    def __init__(
        self,
        message: str,
        confidence: float,
        threshold: float,
        options: Sequence[str] = (),
    ) -> None:
        super().__init__(message)
        self.confidence = confidence
        self.threshold = threshold
        self.options = list(options)
