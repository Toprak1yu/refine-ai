from typing import Any, TypedDict


class AgentState(TypedDict):
    raw_file_path: str
    processed_file_path: str | None
    initial_row_count: int
    records: list[dict[str, Any]]
    inferred_schema: dict[str, Any] | None
    schema_method: str | None
    profile: dict[str, Any] | None
    critical_issues: list[dict[str, Any]]
    expert_advice: str | None
    human_resolutions: dict[str, str]
    audit_trail: list[str]
    is_completed: bool
    session_id: str | None
    random_seed: int | None
    start_time: str | None
    end_time: str | None
    execution_duration_sec: float | None
