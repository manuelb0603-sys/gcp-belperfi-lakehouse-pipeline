"""Compatibility exports for the report harness, which lives in the finance.harness package.

Existing imports (including the private names tests use) keep working; new code should import
from the finance.harness modules directly.
"""

from .model_clients import AnthropicClient, ModelClient, OllamaClient, api_url, build_model_client  # noqa: F401
from .harness.errors import (  # noqa: F401
    RUN_STOP_ERRORS,
    AgentRoundsExhaustedError,
    EmptyModelResponse,
    ModelUnavailableError,
    RunDeadlineExceededError,
    _is_model_call_failure,
    _is_timeout,
)
from .harness.fallback import build_fallback_report  # noqa: F401
from .harness.feedback import (  # noqa: F401
    _SNAPSHOT_FIELD_HINTS,
    FEEDBACK_PRIORITY,
    MAX_RESOLVED_FINDINGS,
    MAX_REVISION_FINDINGS,
    RESOLVED_LABEL_MESSAGE_CHARS,
    STALL_MESSAGE_PREFIX_CHARS,
    CycleProgress,
    _authoritative_values,
    _finding_key,
    _finding_label,
    _update_resolved_findings,
    focused_feedback,
    measure_progress,
)
from .harness.harness import MIN_CALL_TIMEOUT_SECONDS, ReportHarness, _AuthorState, _ValidationState  # noqa: F401
from .harness.html_safety import (  # noqa: F401
    HISTORY_LABEL,
    ReportSanitizationError,
    _plain_text_from_html,
    _UnsafeMarkupParser,
    sanitize_html,
)
from .harness.prompts import (  # noqa: F401
    AUTHOR_PROTOCOL,
    REBUTTAL_PROTOCOL,
    REQUIREMENTS_HEADER,
    VALIDATOR_CHECKLIST_INSTRUCTION,
    requirements_block,
)
from .harness.protocol import (  # noqa: F401
    MAX_REBUTTAL_FIELD_CHARS,
    MAX_REBUTTALS,
    REBUTTALS_END_MARKER,
    REBUTTALS_START_MARKER,
    REPORT_END_MARKER,
    REPORT_START_MARKER,
    VALIDATOR_RESPONSE_SCHEMA,
    ModelReply,
    _extract_rebuttals,
    _first_json_object,
    _parse_author_body,
    _report_result,
    parse_author_response,
    parse_json_response,
)
from .harness.trace import HarnessLimits, RunTrace, ToolFunction, ToolSpec  # noqa: F401
from .harness.transport import QueryFunctionClient, _response_meta, query_ollama  # noqa: F401
from .report_models import ReportSnapshot, ValidationDecision, ValidationFinding  # noqa: F401
