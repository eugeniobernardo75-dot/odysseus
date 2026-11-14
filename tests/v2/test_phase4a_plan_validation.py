"""Testes Fase 4A — validação de planos."""
import pytest

from core.v2.planner_agent import PlannerAgent


@pytest.fixture
def planner():
    return PlannerAgent()


def _base_plan(**overrides):
    plan = {
        "steps": [
            {"step": "analyse", "description": "analisar", "type": "reasoning"},
        ],
    }
    plan.update(overrides)
    return plan


def test_valid_simple_plan(planner):
    assert planner._validate_plan_dict(_base_plan()) == []


def test_empty_steps_rejected(planner):
    errs = planner._validate_plan_dict({"steps": []})
    assert any("não contém steps" in e.lower() for e in errs)


def test_missing_steps_key_rejected(planner):
    errs = planner._validate_plan_dict({"objective": "x"})
    assert any("não contém steps" in e.lower() for e in errs)


def test_missing_description_rejected(planner):
    plan = _base_plan(steps=[{"step": "s1", "type": "reasoning"}])
    errs = planner._validate_plan_dict(plan)
    assert any("description" in e for e in errs)


def test_invalid_type_rejected(planner):
    plan = _base_plan(steps=[{"step": "s1", "description": "x", "type": "git"}])
    errs = planner._validate_plan_dict(plan)
    assert any("tipo inválido" in e for e in errs)


def test_tool_step_without_tool_rejected(planner):
    plan = _base_plan(steps=[{"step": "s1", "description": "x", "type": "tool"}])
    errs = planner._validate_plan_dict(plan)
    assert any("tool" in e.lower() and "sem campo" in e.lower() for e in errs)


def test_tool_params_must_be_dict(planner):
    plan = _base_plan(steps=[
        {"step": "s1", "description": "x", "type": "tool",
         "tool": "shell", "tool_params": "not a dict"},
    ])
    errs = planner._validate_plan_dict(plan)
    assert any("tool_params" in e for e in errs)


def test_duplicate_ids_rejected(planner):
    plan = _base_plan(steps=[
        {"step": "s1", "description": "a", "type": "reasoning"},
        {"step": "s1", "description": "b", "type": "reasoning"},
    ])
    errs = planner._validate_plan_dict(plan)
    assert any("duplicad" in e.lower() for e in errs)


def test_unknown_dependency_rejected(planner):
    plan = _base_plan(steps=[
        {"step": "s1", "description": "a", "type": "reasoning",
         "dependencies": ["naoexiste"]},
    ])
    errs = planner._validate_plan_dict(plan)
    assert any("inexistente" in e.lower() for e in errs)


def test_self_dependency_rejected(planner):
    plan = _base_plan(steps=[
        {"step": "s1", "description": "a", "type": "reasoning",
         "dependencies": ["s1"]},
    ])
    errs = planner._validate_plan_dict(plan)
    assert any("auto-dependência" in e.lower() for e in errs)


def test_two_step_cycle_rejected(planner):
    plan = _base_plan(steps=[
        {"step": "a", "description": "a", "type": "reasoning",
         "dependencies": ["b"]},
        {"step": "b", "description": "b", "type": "reasoning",
         "dependencies": ["a"]},
    ])
    errs = planner._validate_plan_dict(plan)
    assert any("ciclo" in e.lower() for e in errs)


def test_three_step_cycle_rejected(planner):
    plan = _base_plan(steps=[
        {"step": "a", "description": "a", "type": "reasoning",
         "dependencies": ["c"]},
        {"step": "b", "description": "b", "type": "reasoning",
         "dependencies": ["a"]},
        {"step": "c", "description": "c", "type": "reasoning",
         "dependencies": ["b"]},
    ])
    errs = planner._validate_plan_dict(plan)
    assert any("ciclo" in e.lower() for e in errs)


def test_linear_chain_valid(planner):
    plan = _base_plan(steps=[
        {"step": "a", "description": "a", "type": "reasoning"},
        {"step": "b", "description": "b", "type": "reasoning",
         "dependencies": ["a"]},
        {"step": "c", "description": "c", "type": "reasoning",
         "dependencies": ["b"]},
    ])
    assert planner._validate_plan_dict(plan) == []


def test_diamond_valid(planner):
    plan = _base_plan(steps=[
        {"step": "a", "description": "a", "type": "reasoning"},
        {"step": "b", "description": "b", "type": "reasoning",
         "dependencies": ["a"]},
        {"step": "c", "description": "c", "type": "reasoning",
         "dependencies": ["a"]},
        {"step": "d", "description": "d", "type": "reasoning",
         "dependencies": ["b", "c"]},
    ])
    assert planner._validate_plan_dict(plan) == []


def test_parallel_group_internal_dep_rejected(planner):
    plan = _base_plan(steps=[
        {"step": "a", "description": "a", "type": "reasoning",
         "parallel_group": "g1"},
        {"step": "b", "description": "b", "type": "reasoning",
         "parallel_group": "g1", "dependencies": ["a"]},
    ])
    errs = planner._validate_plan_dict(plan)
    assert any("parallel_group" in e for e in errs)


def test_parallel_group_independent_valid(planner):
    plan = _base_plan(steps=[
        {"step": "a", "description": "a", "type": "reasoning",
         "parallel_group": "g1"},
        {"step": "b", "description": "b", "type": "reasoning",
         "parallel_group": "g1"},
        {"step": "c", "description": "c", "type": "reasoning",
         "dependencies": ["a", "b"]},
    ])
    assert planner._validate_plan_dict(plan) == []


def test_dependencies_not_a_list_rejected(planner):
    plan = _base_plan(steps=[
        {"step": "a", "description": "a", "type": "reasoning",
         "dependencies": "not-a-list"},
    ])
    errs = planner._validate_plan_dict(plan)
    assert any("dependencies" in e.lower() for e in errs)


def test_non_dict_step_rejected(planner):
    plan = _base_plan(steps=["not a dict"])
    errs = planner._validate_plan_dict(plan)
    assert any("objecto" in e.lower() for e in errs)