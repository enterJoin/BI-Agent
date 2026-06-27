from springgraph.rag.agent.planner import apply_intent_defaults
from springgraph.rag.agent.state import QuestionUnderstanding, RetrievalPlan
from springgraph.rag.config.loader import (
    load_intent_configs,
    load_target_trace_config,
)
from springgraph.rag.intent import infer_query_intent, intent_default_filters
from springgraph.rag.target_trace import trace_target_tokens


def test_load_intent_configs_includes_persistence_defaults() -> None:
    intents = load_intent_configs()

    persistence = intents["persistence_location"]

    assert persistence.default_tool == "aggregate_query"
    assert persistence.default_filters == {"group_by": "table_name"}
    assert "\u5b58\u5165" in persistence.chinese_terms
    assert "save" in persistence.english_terms


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
