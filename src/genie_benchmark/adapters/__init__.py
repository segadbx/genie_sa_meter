"""Invocation-path adapters.

Each adapter isolates one customer-specific invocation behind the ``AgentAdapter``
contract so the harness core never hard-codes an endpoint or authentication flow.
"""

from .base import (
    AdapterError,
    AgentAdapter,
    NonRetryableError,
    PrerequisiteNotConfigured,
    TransientError,
)

__all__ = [
    "AdapterError",
    "AgentAdapter",
    "NonRetryableError",
    "PrerequisiteNotConfigured",
    "TransientError",
]
