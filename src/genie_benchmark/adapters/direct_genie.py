"""Direct Genie adapter — the documented Genie Conversation API.

Implements the documented Genie flow: ``start-conversation`` (or ``create-message`` when
reusing a conversation), then poll ``get-message`` through the intermediate states
``SUBMITTED`` / ``FILTERING_CONTEXT`` / ``ASKING_AI`` / ``EXECUTING_QUERY`` until a
terminal state, collecting the generated SQL and response text when available.

This module does not invent an API shape: it depends on a small
:class:`GenieConversationClient` interface. The default implementation wraps
``databricks-sdk``; unit tests inject a fake client driven by checked-in fixtures and
never make live calls. The response-normalization helpers are pure functions so they can
be tested directly.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Protocol

from ..config import Config
from ..models import BenchmarkQuestion, InvocationResult, RunContext, RunStatus
from ..redaction import redact_text
from ..util import content_metadata, utc_now_iso
from .base import NonRetryableError, TransientError

# Terminal message states. Everything else is an intermediate polling state.
TERMINAL_SUCCESS = {"COMPLETED"}
TERMINAL_FAILURE = {"FAILED", "CANCELLED", "QUERY_RESULT_EXPIRED"}
TERMINAL_STATES = TERMINAL_SUCCESS | TERMINAL_FAILURE
INTERMEDIATE_STATES = {
    "SUBMITTED",
    "FILTERING_CONTEXT",
    "ASKING_AI",
    "PENDING_WAREHOUSE",
    "EXECUTING_QUERY",
    "FETCHING_METADATA",
}


class GenieConversationClient(Protocol):
    """Minimal interface over the documented Genie Conversation API.

    Every method returns a plain ``dict`` (the SDK responses are converted with
    ``as_dict()`` in the default implementation).
    """

    def start_conversation(self, space_id: str, content: str) -> dict[str, Any]: ...

    def create_message(self, space_id: str, conversation_id: str, content: str) -> dict[str, Any]: ...

    def get_message(
        self, space_id: str, conversation_id: str, message_id: str
    ) -> dict[str, Any]: ...


def is_terminal(status: str) -> bool:
    return status.upper() in TERMINAL_STATES


def extract_response(message: dict[str, Any]) -> dict[str, Any]:
    """Pull response text and generated SQL out of a Genie message.

    Genie message attachments carry either a ``text`` block (with ``content``) or a
    ``query`` block (with the generated SQL and an optional description). Both shapes are
    handled; missing fields simply yield ``None``.
    """
    response_parts: list[str] = []
    sql_parts: list[str] = []
    attachment_ids: list[str] = []
    for attachment in message.get("attachments", []) or []:
        if not isinstance(attachment, dict):
            continue
        att_id = attachment.get("attachment_id") or attachment.get("id")
        if att_id:
            attachment_ids.append(str(att_id))
        text_block = attachment.get("text")
        if isinstance(text_block, dict) and text_block.get("content"):
            response_parts.append(str(text_block["content"]))
        query_block = attachment.get("query")
        if isinstance(query_block, dict):
            sql = query_block.get("query") or query_block.get("statement")
            if sql:
                sql_parts.append(str(sql))
            description = query_block.get("description")
            if description:
                response_parts.append(str(description))
    # Fall back to a top-level content field if present.
    if not response_parts and message.get("content"):
        response_parts.append(str(message["content"]))
    return {
        "response_text": "\n\n".join(response_parts) if response_parts else None,
        "generated_sql": "\n\n".join(sql_parts) if sql_parts else None,
        "attachment_ids": attachment_ids,
        "sql_count": len(sql_parts),
    }


class DirectGenieAdapter:
    """Adapter for the direct Genie Conversation API path."""

    name = "direct_genie"

    def __init__(
        self,
        client: GenieConversationClient,
        config: Config,
        *,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        max_transient_retries: int = 3,
    ) -> None:
        self._client = client
        self._config = config
        self._sleep = sleep
        self._monotonic = monotonic
        self._max_transient_retries = max_transient_retries

    def invoke(self, question: BenchmarkQuestion, context: RunContext) -> InvocationResult:
        started_at = utc_now_iso()
        start_mono = self._monotonic()
        warnings: list[str] = []
        space_id = self._config.genie_space_id

        try:
            # Start a fresh conversation for independent questions; reuse only on an
            # explicit follow-up (never retried automatically — these create state).
            if context.reuse_conversation_id:
                conversation_id = context.reuse_conversation_id
                initial = self._client.create_message(
                    space_id, conversation_id, question.question
                )
            else:
                initial = self._client.start_conversation(space_id, question.question)
                conversation_id = str(
                    initial.get("conversation_id")
                    or initial.get("conversation", {}).get("id")
                    or ""
                )
            message_id = str(
                initial.get("message_id")
                or initial.get("id")
                or initial.get("message", {}).get("id")
                or ""
            )
            if not conversation_id or not message_id:
                raise NonRetryableError(
                    "Genie did not return a conversation_id/message_id from the initial call."
                )

            message = self._poll_until_terminal(
                space_id, conversation_id, message_id, context, warnings
            )
        except TimeoutError:
            return self._result(
                question,
                context,
                started_at,
                start_mono,
                status=RunStatus.TIMEOUT,
                error="Genie request exceeded the configured timeout.",
                warnings=warnings,
            )
        except NonRetryableError as exc:
            return self._result(
                question,
                context,
                started_at,
                start_mono,
                status=RunStatus.FAILED,
                error=redact_text(str(exc)),
                warnings=warnings,
            )
        except Exception as exc:  # defensive: never crash the whole benchmark
            return self._result(
                question,
                context,
                started_at,
                start_mono,
                status=RunStatus.FAILED,
                error=f"Unexpected Genie error: {redact_text(str(exc))}",
                warnings=warnings,
            )

        status_str = str(message.get("status", "")).upper()
        extracted = extract_response(message)
        run_status = (
            RunStatus.COMPLETED if status_str in TERMINAL_SUCCESS else RunStatus.FAILED
        )
        if run_status is RunStatus.FAILED:
            warnings.append(f"genie_terminal_status: {status_str}")

        return self._result(
            question,
            context,
            started_at,
            start_mono,
            status=run_status,
            conversation_id=str(message.get("conversation_id") or conversation_id),
            trace_id=_opt(message.get("trace_id")),
            raw_ids={
                "message_id": str(message.get("id") or message_id),
                "response_id": _opt(message.get("response_id")),
                "attachment_ids": extracted["attachment_ids"],
            },
            response_text=extracted["response_text"],
            generated_sql=extracted["generated_sql"],
            counts={"sql_count": extracted["sql_count"]},
            error=_opt(message.get("error")),
            warnings=warnings,
        )

    def _poll_until_terminal(
        self,
        space_id: str,
        conversation_id: str,
        message_id: str,
        context: RunContext,
        warnings: list[str],
    ) -> dict[str, Any]:
        deadline = self._monotonic() + context.request_timeout_seconds
        transient_failures = 0
        while True:
            if self._monotonic() >= deadline:
                raise TimeoutError("poll deadline exceeded")
            try:
                message = self._client.get_message(space_id, conversation_id, message_id)
                transient_failures = 0
            except TransientError as exc:
                # Bounded retries only for documented transient poll failures.
                transient_failures += 1
                if transient_failures > self._max_transient_retries:
                    raise NonRetryableError(
                        f"Genie polling failed after {self._max_transient_retries} "
                        f"transient retries: {exc}"
                    ) from exc
                warnings.append(f"transient_poll_retry:{transient_failures}")
                self._sleep(context.poll_interval_seconds)
                continue

            status = str(message.get("status", "")).upper()
            if status in TERMINAL_STATES:
                return message
            if status not in INTERMEDIATE_STATES and status != "":
                # Unknown state: keep polling but record it once.
                warn = f"unknown_genie_state:{status}"
                if warn not in warnings:
                    warnings.append(warn)
            self._sleep(context.poll_interval_seconds)

    def _result(
        self,
        question: BenchmarkQuestion,
        context: RunContext,
        started_at: str,
        start_mono: float,
        *,
        status: RunStatus,
        conversation_id: str | None = None,
        trace_id: str | None = None,
        raw_ids: dict[str, Any] | None = None,
        response_text: str | None = None,
        generated_sql: str | None = None,
        counts: dict[str, Any] | None = None,
        error: str | None = None,
        warnings: list[str] | None = None,
    ) -> InvocationResult:
        latency_ms = int((self._monotonic() - start_mono) * 1000)
        resp_meta = content_metadata(response_text, store_text=context.store_response_text)
        sql_meta = content_metadata(generated_sql, store_text=context.store_response_text)
        if trace_id is None:
            (warnings or []).append("trace_id_unavailable: Genie did not return a trace id.")
        return InvocationResult(
            run_id=context.run_id,
            question_id=question.id,
            variant=self.name,
            repetition=context.repetition,
            started_at=started_at,
            completed_at=utc_now_iso(),
            status=status,
            response_text=resp_meta["text"],
            response_sha256=resp_meta["sha256"],
            generated_sql=sql_meta["text"],
            generated_sql_sha256=sql_meta["sha256"],
            conversation_id=conversation_id,
            trace_id=trace_id,
            raw_ids=raw_ids or {},
            client_metrics={
                "latency_ms": latency_ms,
                "response_chars": resp_meta["chars"],
            },
            counts=counts or {},
            error=error,
            warnings=warnings or [],
        )


def _opt(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(value)


def build_default_client(config: Config) -> GenieConversationClient:
    """Construct the default databricks-sdk-backed Genie client.

    Imported lazily so the offline test suite never requires the SDK or a workspace.
    Authentication is resolved by the SDK from the environment or the named profile;
    no credentials are read from the config file.
    """
    from databricks.sdk import WorkspaceClient  # noqa: PLC0415

    kwargs: dict[str, Any] = {"host": config.workspace_host}
    if config.databricks_profile:
        kwargs["profile"] = config.databricks_profile
    ws = WorkspaceClient(**kwargs)

    class _SdkGenieClient:
        # ``start_conversation``/``create_message`` return a ``Wait[GenieMessage]``
        # long-running-operation wrapper, not the message itself. We do our own
        # polling (see ``_poll_until_terminal``) rather than block on the SDK waiter,
        # so we read the immediate op response off ``.response``. Calling ``as_dict()``
        # directly on the ``Wait`` object would trip its ``__getattr__`` and raise
        # ``KeyError('as_dict')``.
        def start_conversation(self, space_id: str, content: str) -> dict[str, Any]:
            wait = ws.genie.start_conversation(space_id=space_id, content=content)
            return wait.response.as_dict()

        def create_message(
            self, space_id: str, conversation_id: str, content: str
        ) -> dict[str, Any]:
            wait = ws.genie.create_message(
                space_id=space_id, conversation_id=conversation_id, content=content
            )
            return wait.response.as_dict()

        def get_message(
            self, space_id: str, conversation_id: str, message_id: str
        ) -> dict[str, Any]:
            resp = ws.genie.get_message(
                space_id=space_id, conversation_id=conversation_id, message_id=message_id
            )
            return resp.as_dict()

    return _SdkGenieClient()
