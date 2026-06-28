from springgraph.rag import query_resolver
from springgraph.rag.agent import planner
from springgraph.rag.agent.planner import (
    apply_entrypoint_lookup_defaults,
    apply_intent_defaults,
    apply_task_planning_defaults,
)
from springgraph.rag.agent.state import QuestionUnderstanding, RetrievalPlan
from springgraph.rag.config.loader import (
    load_execution_trace_config,
    load_intent_configs,
    load_target_trace_config,
    load_task_planning_config,
)
from springgraph.rag.config.models import ToolConfig
from springgraph.rag.intent import infer_query_intent, intent_default_filters
from springgraph.rag.schemas import RagEvidence
from springgraph.rag.target_trace import trace_target_tokens


def test_load_intent_configs_includes_persistence_defaults() -> None:
    intents = load_intent_configs()

    persistence = intents["persistence_location"]

    assert persistence.default_tool == "aggregate_query"
    assert persistence.default_filters == {"group_by": "table_name"}
    assert "\u5b58\u5165" in persistence.chinese_terms
    assert "save" in persistence.english_terms

    task_planning = intents["task_planning"]
    assert task_planning.default_tool == "artifact_search"
    assert "\u5e2e\u6211\u89c4\u5212" in task_planning.chinese_terms


def test_target_trace_heuristics_are_config_driven() -> None:
    config = load_target_trace_config()

    assert "table" in config.generic_terms
    assert "ServiceImpl" in config.class_suffixes
    assert trace_target_tokens("api table where oms_order_operate_history") == [
        "oms_order_operate_history"
    ]
    assert trace_target_tokens("OrderOperateHistoryServiceImpl") == [
        "OrderOperateHistoryServiceImpl"
    ]


def test_execution_trace_heuristics_are_config_driven() -> None:
    config = load_execution_trace_config()

    assert "\u8be6\u7ec6\u6b65\u9aa4" in config.trigger_terms
    assert "\u8be6\u7ec6\u8fc7\u7a0b" in config.trigger_terms
    assert "Handler" in config.entrypoint_suffixes
    assert planner._looks_like_execution_trace_question(
        "tencentMaterialReportHandler"
    )
    assert planner._looks_like_execution_trace_question(
        "syncOrder\u6267\u884c\u8be6\u7ec6\u6b65\u9aa4"
    )
    assert planner._looks_like_execution_trace_question(
        "\u8be6\u7ec6\u8fc7\u7a0b\u662f\u4ec0\u4e48"
    )


def test_query_resolver_carries_recent_job_target(
    monkeypatch: object,
) -> None:
    captured: dict[str, str] = {}

    def fake_invoke_json(prompt: str) -> dict[str, object]:
        captured["prompt"] = prompt
        return {
            "is_follow_up": True,
            "needs_context": True,
            "needs_clarification": False,
            "resolved_target": {
                "type": "job",
                "name": "tencentNineImageMappingHandler",
                "source": "history",
                "confidence": 0.95,
            },
            "rewritten_question": (
                "tencentNineImageMappingHandler "
                "\u8be6\u7ec6\u8fc7\u7a0b\u662f\u4ec0\u4e48"
            ),
            "retrieval_intent": "execution_flow",
            "preferred_tools": ["execution_trace"],
            "reason": "follow-up resolved from history",
        }

    monkeypatch.setattr(query_resolver, "_invoke_json", fake_invoke_json)

    resolution = query_resolver.resolve_contextual_query(
        question=(
            "\u8fd9\u4e24\u4e2ajob\u7684\u7d20\u6750\u62a5\u8868"
            "\u6570\u636e\u90fd\u5165\u4e86\u54ea\u4e2a\u8868\uff1f"
        ),
        conversation_history=[
            {
                "role": "user",
                "content": (
                    "\u8bf7\u95ee\u5e7f\u70b9\u901a\u4e5d\u56fe"
                    "\u7d20\u6750\u6d88\u8017\u6570\u636e\u662f"
                    "\u901a\u8fc7\u54ea\u4e2aJob\u5165\u5e93\u7684"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "\u901a\u8fc7 `tencentNineImageMappingHandler` "
                    "\u8fd9\u4e2a Job \u5165\u5e93\u3002"
                ),
            },
        ],
    )

    assert resolution["rewritten_question"] == (
        "tencentNineImageMappingHandler "
        "\u8be6\u7ec6\u8fc7\u7a0b\u662f\u4ec0\u4e48"
    )
    assert "Current question:" in captured["prompt"]
    assert "Recent conversation:" in captured["prompt"]
    assert "tencentNineImageMappingHandler" in captured["prompt"]
    assert resolution["preferred_tools"] == ["execution_trace"]


def test_query_resolver_supports_multi_target_hard_constraints(
    monkeypatch: object,
) -> None:
    def fake_invoke_json(prompt: str) -> dict[str, object]:
        return {
            "is_follow_up": True,
            "needs_context": True,
            "needs_clarification": False,
            "context_mode": "object_followup",
            "resolved_targets": [
                {
                    "type": "job",
                    "name": "tencentMaterialReportHandler",
                    "source": "history",
                    "confidence": 0.95,
                },
                {
                    "type": "job",
                    "name": "materialReportHandler",
                    "source": "history",
                    "confidence": 0.9,
                },
            ],
            "hard_constraints": {
                "targets": [
                    {
                        "type": "job",
                        "name": "tencentMaterialReportHandler",
                    },
                    {
                        "type": "job",
                        "name": "materialReportHandler",
                    },
                ],
                "scope": "target_call_chain",
                "evidence_must_be_reachable_from_targets": True,
            },
            "rewritten_question": (
                "tencentMaterialReportHandler and materialReportHandler "
                "target material report tables"
            ),
            "retrieval_intent": "table_usage",
            "preferred_tools": ["execution_trace"],
            "reason": "resolved both jobs from history",
        }

    monkeypatch.setattr(query_resolver, "_invoke_json", fake_invoke_json)

    resolution = query_resolver.resolve_contextual_query(
        question=(
            "\u8fd9\u4e24\u4e2ajob\u7684\u7d20\u6750\u62a5\u8868"
            "\u6570\u636e\u90fd\u5165\u4e86\u54ea\u4e2a\u8868\uff1f"
        ),
        conversation_history=[
            {
                "role": "assistant",
                "content": (
                    "through tencentMaterialReportHandler and "
                    "materialReportHandler."
                ),
            }
        ],
    )

    assert resolution["context_mode"] == "object_followup"
    assert [
        target["name"] for target in resolution["hard_constraints"]["targets"]
    ] == ["tencentMaterialReportHandler", "materialReportHandler"]
    assert (
        resolution["hard_constraints"]["evidence_must_be_reachable_from_targets"]
        is True
    )


def test_query_resolver_returns_default_without_history() -> None:
    resolution = query_resolver.resolve_contextual_query(
        "\u8ba2\u5355\u670d\u52a1\u6709\u54ea\u4e9bHTTP\u63a5\u53e3",
        [],
    )

    assert resolution["rewritten_question"] == (
        "\u8ba2\u5355\u670d\u52a1\u6709\u54ea\u4e9bHTTP\u63a5\u53e3"
    )
    assert resolution["is_follow_up"] is False


def test_context_constraints_force_execution_trace_targets(
    monkeypatch: object,
) -> None:
    def fake_invoke_json(prompt: str) -> dict[str, object]:
        return {
            "question_understanding": {
                "task_goal": "find target job tables",
                "intent": "table_usage",
            },
            "retrieval_plan": {
                "task_goal": "find target job tables",
                "steps": [
                    {
                        "tool_name": "aggregate_query",
                        "query": "material report tables",
                        "filters": {
                            "intent": "table_usage",
                            "group_by": "table_name",
                        },
                        "reason": "broad table aggregation",
                    }
                ],
            },
        }

    monkeypatch.setattr(planner, "_invoke_json", fake_invoke_json)

    _, plan = planner.plan_question_retrieval(
        question=(
            "tencentMaterialReportHandler and materialReportHandler target tables"
        ),
        available_tools=[
            ToolConfig(name="execution_trace", description="trace execution"),
            ToolConfig(name="aggregate_query", description="aggregate evidence"),
        ],
        source_available=True,
        memory_observations=[],
        query_resolution={
            "context_mode": "object_followup",
            "hard_constraints": {
                "targets": [
                    {"type": "job", "name": "tencentMaterialReportHandler"},
                    {"type": "job", "name": "materialReportHandler"},
                ],
                "scope": "target_call_chain",
                "evidence_must_be_reachable_from_targets": True,
            },
        },
    )

    steps = plan["steps"]
    assert steps[0]["tool_name"] == "execution_trace"
    assert steps[0]["filters"]["targets"] == [
        "tencentMaterialReportHandler",
        "materialReportHandler",
    ]
    assert steps[0]["filters"]["constraint_scope"] == "hard"
    assert steps[1]["filters"]["constraint_scope"] == "soft_context"


def test_task_planning_config_drives_default_steps() -> None:
    config = load_task_planning_config()

    assert config.intent_name == "task_planning"
    assert "\u5e2e\u6211\u89c4\u5212" in config.planning_terms
    assert [step["tool_name"] for step in config.default_steps] == [
        "artifact_search",
        "relation_search",
        "aggregate_query",
        "source_read",
    ]
    assert "\u4efb\u52a1\u7406\u89e3" in config.output_sections


def test_infer_query_intent_uses_configured_terms() -> None:
    intents = load_intent_configs()

    assert (
        infer_query_intent(
            query="\u8ba2\u5355\u4fe1\u606f\u662f\u5728\u54ea\u91cc\u5b58\u5165\u7684",
            explicit_intent=None,
            group_by=None,
            intent_configs=intents,
        )
        == "persistence_location"
    )
    assert (
        infer_query_intent(
            query="\u8ba2\u5355\u670d\u52a1\u90fd\u7528\u5230\u4e86\u54ea\u4e9b\u8868",
            explicit_intent=None,
            group_by=None,
            intent_configs=intents,
        )
        == "table_usage"
    )
    assert (
        infer_query_intent(
            query="\u8ba2\u5355\u521b\u5efa\u63a5\u53e3\u5728\u54ea\u91cc",
            explicit_intent=None,
            group_by=None,
            intent_configs=intents,
        )
        == "api_entrypoint"
    )
    assert (
        infer_query_intent(
            query=(
                "\u6211\u8981\u7ed9\u4f1a\u5458\u8868\u65b0\u589e\u751f\u65e5"
                "\u5b57\u6bb5\uff0c\u5e2e\u6211\u89c4\u5212\u600e\u4e48\u6539"
            ),
            explicit_intent=None,
            group_by=None,
            intent_configs=intents,
        )
        == "task_planning"
    )
    assert (
        infer_query_intent(
            query="\u8ba2\u5355\u6d41\u7a0b\u662f\u600e\u4e48\u6267\u884c\u7684",
            explicit_intent=None,
            group_by=None,
            intent_configs=intents,
        )
        != "task_planning"
    )


def test_intent_default_filters_returns_copy() -> None:
    intents = load_intent_configs()

    filters = intent_default_filters("persistence_location", intents)
    filters["group_by"] = "changed"

    assert intents["persistence_location"].default_filters == {
        "group_by": "table_name"
    }


def test_apply_intent_defaults_adds_aggregate_filters() -> None:
    intents = load_intent_configs()
    query = "\u8ba2\u5355\u4fe1\u606f\u662f\u5728\u54ea\u91cc\u5b58\u5165\u7684"
    understanding: QuestionUnderstanding = {
        "task_goal": "\u8ffd\u8e2a\u6570\u636e\u5165\u5e93\u4f4d\u7f6e",
        "intent": "persistence_location",
    }
    plan: RetrievalPlan = {
        "task_goal": "\u8ffd\u8e2a\u6570\u636e\u5165\u5e93\u4f4d\u7f6e",
        "steps": [
            {
                "tool_name": "aggregate_query",
                "query": query,
                "filters": {},
            }
        ],
    }

    updated = apply_intent_defaults(plan, understanding, intents)

    assert updated["steps"][0]["filters"] == {
        "group_by": "table_name",
        "intent": "persistence_location",
    }


def test_apply_intent_defaults_promotes_explicit_target_to_trace() -> None:
    intents = load_intent_configs()
    query = "oms_order_operate_history table data source"
    understanding: QuestionUnderstanding = {
        "task_goal": "trace persistence target",
        "intent": "persistence_location",
    }
    plan: RetrievalPlan = {
        "task_goal": "trace persistence target",
        "steps": [
            {
                "tool_name": "aggregate_query",
                "query": query,
                "filters": {},
            }
        ],
    }

    updated = apply_intent_defaults(plan, understanding, intents)

    step = updated["steps"][0]
    assert step["tool_name"] == "target_trace"
    assert step["filters"] == {
        "group_by": "table_name",
        "intent": "persistence_location",
        "target": "oms_order_operate_history",
        "direction": "incoming",
        "edge_kinds": ["writes_table", "defines_contract"],
    }


def test_job_lookup_defaults_preserve_original_question_for_hints() -> None:
    question = (
        "\u8bf7\u95ee\u5e7f\u70b9\u901a\u4e5d\u56fe\u7d20\u6750"
        "\u6d88\u8017\u6570\u636e\u662f\u901a\u8fc7\u54ea\u4e2aJob"
        "\u5165\u5e93\u7684"
    )
    understanding: QuestionUnderstanding = {
        "task_goal": "find job",
        "intent": "persistence_location",
    }
    plan: RetrievalPlan = {
        "task_goal": "find job",
        "steps": [
            {
                "tool_name": "aggregate_query",
                "query": "\u5e7f\u70b9\u901a\u4e5d\u56fe\u7d20\u6750",
                "filters": {},
            }
        ],
    }

    updated = apply_entrypoint_lookup_defaults(plan, understanding, question)

    assert [step["tool_name"] for step in updated["steps"]] == [
        "vector_search",
        "aggregate_query",
    ]
    assert updated["steps"][1]["query"] == question
    assert "\u6d88\u8017\u6570\u636e" in updated["steps"][1]["query"]
    assert updated["steps"][1]["filters"] == {
        "intent": "persistence_location",
        "group_by": "job",
    }


def test_job_lookup_defaults_drop_unneeded_execution_trace() -> None:
    question = (
        "\u8bf7\u95ee\u5e7f\u70b9\u901a\u521b\u610f\u548c\u5e7f\u544a"
        "\u6570\u636e\u662f\u901a\u8fc7\u54ea\u4e2aJob\u5165\u5e93\u7684"
    )
    understanding: QuestionUnderstanding = {
        "task_goal": "find job",
        "intent": "persistence_location",
    }
    plan: RetrievalPlan = {
        "task_goal": "find job",
        "steps": [
            {
                "tool_name": "execution_trace",
                "query": question,
                "filters": {"intent": "execution_flow"},
            },
            {
                "tool_name": "aggregate_query",
                "query": question,
                "filters": {},
            },
        ],
    }

    updated = apply_entrypoint_lookup_defaults(plan, understanding, question)

    assert [step["tool_name"] for step in updated["steps"]] == [
        "vector_search",
        "aggregate_query",
    ]


def test_vector_evidence_summary_is_semantic_context_only() -> None:
    summary = planner._evidence_summary(
        [
            RagEvidence(
                evidence_type="vector_chunk:project_knowledge",
                source="vector",
                file_path="library/keywords.md",
                content_excerpt=(
                    "骞跨偣閫氬箍鍛婃暟鎹叆搴?=> adCreativeInfoV3Handler / "
                    "syncAdV3 / ad_sync"
                ),
            )
        ]
    )

    assert "semantic_context_only=true" in summary
    assert "not as typed artifact candidates" in summary


def test_vector_evidence_summary_omits_hint_details_with_typed_evidence() -> None:
    summary = planner._evidence_summary(
        [
            RagEvidence(
                evidence_type="vector_chunk:project_knowledge",
                source="vector",
                file_path="library/keywords.md",
                content_excerpt=(
                    "骞跨偣閫氬箍鍛婃暟鎹叆搴?=> adCreativeInfoV3Handler / "
                    "syncAdV3 / ad_sync"
                ),
            ),
            RagEvidence(
                evidence_type="job_entrypoint",
                source="aggregate",
                file_path="src/main/java/TencentDataInfoJob.java",
                start_line=137,
                symbol="annotation_usage:XxlJob:adCreativeInfoV3Handler",
                content_excerpt="job=adCreativeInfoV3Handler",
            ),
        ]
    )

    assert "details omitted" in summary
    assert "syncAdV3" not in summary
    assert "job=adCreativeInfoV3Handler" in summary


def test_apply_task_planning_defaults_adds_configured_tool_chain() -> None:
    question = (
        "\u6211\u8981\u7ed9\u4f1a\u5458\u8868\u65b0\u589e\u751f\u65e5"
        "\u5b57\u6bb5\uff0c\u5e2e\u6211\u89c4\u5212\u600e\u4e48\u6539"
    )
    understanding: QuestionUnderstanding = {
        "task_goal": "plan member birthday change",
        "intent": "task_planning",
    }
    plan: RetrievalPlan = {
        "task_goal": "plan member birthday change",
        "steps": [
            {
                "tool_name": "artifact_search",
                "query": question,
                "filters": {"intent": "task_planning"},
            }
        ],
    }

    updated = apply_task_planning_defaults(
        plan=plan,
        understanding=understanding,
        question=question,
        source_available=True,
    )

    assert [step["tool_name"] for step in updated["steps"]] == [
        "artifact_search",
        "relation_search",
        "aggregate_query",
        "source_read",
    ]
    assert updated["steps"][2]["filters"] == {
        "intent": "table_usage",
        "group_by": "table_name",
    }


def test_apply_task_planning_defaults_skips_source_read_when_unavailable() -> None:
    question = "\u5e2e\u6211\u89c4\u5212\u8fd9\u4e2a\u9700\u6c42\u600e\u4e48\u6539"
    understanding: QuestionUnderstanding = {
        "task_goal": "plan change",
        "intent": "task_planning",
    }
    plan: RetrievalPlan = {"task_goal": "plan change", "steps": []}

    updated = apply_task_planning_defaults(
        plan=plan,
        understanding=understanding,
        question=question,
        source_available=False,
    )

    assert [step["tool_name"] for step in updated["steps"]] == [
        "artifact_search",
        "relation_search",
        "aggregate_query",
    ]


def test_apply_task_planning_defaults_moves_source_read_last() -> None:
    question = "\u5e2e\u6211\u89c4\u5212\u8fd9\u4e2a\u9700\u6c42\u600e\u4e48\u6539"
    understanding: QuestionUnderstanding = {
        "task_goal": "plan change",
        "intent": "task_planning",
    }
    plan: RetrievalPlan = {
        "task_goal": "plan change",
        "steps": [
            {"tool_name": "artifact_search", "query": question, "filters": {}},
            {"tool_name": "source_read", "query": question, "filters": {}},
            {"tool_name": "aggregate_query", "query": question, "filters": {}},
        ],
    }

    updated = apply_task_planning_defaults(
        plan=plan,
        understanding=understanding,
        question=question,
        source_available=True,
    )

    assert [step["tool_name"] for step in updated["steps"]] == [
        "artifact_search",
        "relation_search",
        "aggregate_query",
        "source_read",
    ]
