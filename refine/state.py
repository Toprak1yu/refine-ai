from typing import Any, Dict, List, Optional, TypedDict

class AgentState(TypedDict):
    raw_file_path: str
    processed_file_path: Optional[str]
    initial_row_count: int
    records: List[Dict[str, Any]]
    inferred_schema: Optional[Dict[str, Any]]
    schema_method: Optional[str]
    profile: Optional[Dict[str, Any]]
    critical_issues: List[Dict[str, Any]]
    expert_advice: Optional[str]
    human_resolutions: Dict[str, str]
    audit_trail: List[str]
    is_completed: bool