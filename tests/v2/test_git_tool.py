"""
Testes Unitários — GitTool

Cobertura:
    - Definição: nome, capacidades, parâmetros
    - _build_command: construção segura de comandos git
    - _parse_branches: parsing de git branch -a -v
    - _parse_remotes: parsing de git remote -v
    - _parse_shortlog: parsing de git shortlog -sn
    - _parse_tags: parsing de git tag -n
    - _parse_status: parsing de git status --short --branch
    - Edge cases: ação inválida, max_count, args extra

Uso:
    pytest tests/v2/test_git_tool.py -v
"""

from __future__ import annotations

import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


@pytest.fixture
def git_tool():
    from core.v2.tools import GitTool
    return GitTool()


# ======================================================================
#  Tests: Definição
# ======================================================================

class TestGitToolDefinition:
    def test_name(self, git_tool):
        assert git_tool.name == "git"

    def test_capabilities(self, git_tool):
        caps = git_tool.definition.capabilities
        assert "git.status" in caps
        assert "git.log" in caps
        assert "git.diff" in caps
        assert "git.branch" in caps
        assert "git.show" in caps
        assert "git.remotes" in caps
        assert "git.shortlog" in caps
        assert "git.tags" in caps

    def test_actions(self, git_tool):
        actions = git_tool.definition.parameters["properties"]["action"]["enum"]
        assert "status" in actions
        assert "log" in actions
        assert "branch" in actions
        assert "diff" in actions
        assert "show" in actions
        assert "remotes" in actions
        assert "shortlog" in actions
        assert "tags" in actions

    def test_category(self, git_tool):
        assert git_tool.definition.category == "git"

    def test_requires_approval(self, git_tool):
        assert git_tool.definition.requires_approval is False


# ======================================================================
#  Tests: _build_command
# ======================================================================

class TestGitToolBuildCommand:
    def test_basic_command(self, git_tool):
        cmd = git_tool._build_command("status", "", None)
        assert cmd is not None
        assert "git" in cmd
        assert "status" in cmd
        assert "--short" in cmd
        assert "--branch" in cmd

    def test_log_command_default(self, git_tool):
        cmd = git_tool._build_command("log", "", None)
        assert cmd is not None
        assert "log" in cmd
        assert "-30" in cmd

    def test_log_with_max_count(self, git_tool):
        cmd = git_tool._build_command("log", "", 10)
        assert cmd is not None
        assert "-10" in cmd
        assert "-30" not in cmd

    def test_shortlog_with_max_count(self, git_tool):
        cmd = git_tool._build_command("shortlog", "", 5)
        assert cmd is not None
        assert "-5" in cmd

    def test_with_extra_args(self, git_tool):
        cmd = git_tool._build_command("log", "--since=2024-01-01 --author=John", 20)
        assert cmd is not None
        assert "-20" in cmd
        assert "--since=2024-01-01" in cmd
        assert "--author=John" in cmd

    def test_show_with_commit_hash(self, git_tool):
        cmd = git_tool._build_command("show", "abc123", None)
        assert cmd is not None
        assert "abc123" in cmd

    def test_diff_with_args(self, git_tool):
        cmd = git_tool._build_command("diff", "HEAD~1", None)
        assert cmd is not None
        assert "HEAD~1" in cmd

    def test_invalid_action(self, git_tool):
        cmd = git_tool._build_command("invalid_action", "", None)
        assert cmd is None

    def test_empty_action(self, git_tool):
        cmd = git_tool._build_command("", "", None)
        assert cmd is None

    def test_branch_command(self, git_tool):
        cmd = git_tool._build_command("branch", "--merged", None)
        assert cmd is not None
        assert "branch" in cmd
        assert "--merged" in cmd

    def test_remotes_command(self, git_tool):
        cmd = git_tool._build_command("remotes", "", None)
        assert cmd is not None
        assert "remote" in cmd

    def test_stash_list(self, git_tool):
        cmd = git_tool._build_command("stash_list", "", None)
        assert cmd is not None
        assert "stash" in cmd
        assert "list" in cmd


# ======================================================================
#  Tests: _parse_branches
# ======================================================================

class TestGitToolParseBranches:
    def test_parse_current_branch(self, git_tool):
        lines = ["* main abc1234 Latest commit message"]
        result = git_tool._parse_branches(lines)
        assert result["current"] == "main"
        assert result["count"] == 1
        assert result["branches"][0]["current"] is True
        assert result["branches"][0]["commit"] == "abc1234"

    def test_parse_multiple_branches(self, git_tool):
        lines = [
            "* main abc1234 Main branch",
            "  dev def5678 Development",
            "  feat/feature ghi9012 Feature work",
        ]
        result = git_tool._parse_branches(lines)
        assert result["count"] == 3
        assert result["current"] == "main"
        names = {b["name"] for b in result["branches"]}
        assert names == {"main", "dev", "feat/feature"}

    def test_parse_no_branches(self, git_tool):
        result = git_tool._parse_branches([])
        assert result["count"] == 0
        assert result["current"] is None

    def test_parse_detached_head(self, git_tool):
        lines = [
            "  (HEAD detached at abc1234) abc1234 Detached",
            "  main def5678 Main",
        ]
        result = git_tool._parse_branches(lines)
        # Nenhum começa com *
        assert result["current"] is None
        assert result["is_detached"] is True


# ======================================================================
#  Tests: _parse_remotes
# ======================================================================

class TestGitToolParseRemotes:
    def test_parse_single_remote(self, git_tool):
        lines = [
            "origin\thttps://github.com/user/repo.git (fetch)",
            "origin\thttps://github.com/user/repo.git (push)",
        ]
        result = git_tool._parse_remotes(lines)
        assert result["count"] == 1
        assert "origin" in result["remotes"]
        assert result["remotes"]["origin"]["fetch"] == "https://github.com/user/repo.git"
        assert result["remotes"]["origin"]["push"] == "https://github.com/user/repo.git"

    def test_parse_multiple_remotes(self, git_tool):
        lines = [
            "origin\thttps://github.com/user/repo.git (fetch)",
            "upstream\thttps://github.com/other/repo.git (fetch)",
        ]
        result = git_tool._parse_remotes(lines)
        assert result["count"] == 2
        assert "origin" in result["remotes"]
        assert "upstream" in result["remotes"]

    def test_parse_no_remotes(self, git_tool):
        result = git_tool._parse_remotes([])
        assert result["count"] == 0
        assert result["remotes"] == {}


# ======================================================================
#  Tests: _parse_shortlog
# ======================================================================

class TestGitToolParseShortlog:
    def test_parse_multiple_authors(self, git_tool):
        lines = ["  42\tJohn Doe", "  17\tJane Smith", "   5\tBot CI"]
        result = git_tool._parse_shortlog(lines)
        assert result["count"] == 3
        assert result["total_commits"] == 64
        names = {a["name"] for a in result["authors"]}
        assert names == {"John Doe", "Jane Smith", "Bot CI"}

    def test_parse_single_author(self, git_tool):
        lines = ["  100\tSolo Dev"]
        result = git_tool._parse_shortlog(lines)
        assert result["count"] == 1
        assert result["authors"][0]["name"] == "Solo Dev"
        assert result["authors"][0]["commits"] == 100

    def test_parse_empty(self, git_tool):
        result = git_tool._parse_shortlog([])
        assert result["count"] == 0
        assert result["total_commits"] == 0

    def test_parse_malformed_line(self, git_tool):
        # Linha sem número
        result = git_tool._parse_shortlog(["invalid_line"])
        assert result["count"] == 0

    def test_parse_mixed(self, git_tool):
        lines = ["  10\tAlice", "invalid", "  20\tBob"]
        result = git_tool._parse_shortlog(lines)
        assert result["count"] == 2
        assert result["total_commits"] == 30


# ======================================================================
#  Tests: _parse_tags
# ======================================================================

class TestGitToolParseTags:
    def test_parse_multiple_tags(self, git_tool):
        lines = ["v1.0.0\tFirst release", "v1.1.0\tBug fixes"]
        result = git_tool._parse_tags(lines)
        assert result["count"] == 2
        assert result["tags"][0]["name"] == "v1.0.0"
        assert result["tags"][0]["message"] == "First release"

    def test_parse_tag_no_message(self, git_tool):
        lines = ["v1.0.0"]
        result = git_tool._parse_tags(lines)
        assert result["count"] == 1
        assert result["tags"][0]["message"] == ""

    def test_parse_empty(self, git_tool):
        result = git_tool._parse_tags([])
        assert result["count"] == 0


# ======================================================================
#  Tests: _parse_status
# ======================================================================

class TestGitToolParseStatus:
    def test_parse_clean_status(self, git_tool):
        lines = ["## main...origin/main"]
        result = git_tool._parse_status(lines, "## main...origin/main")
        assert result["is_clean"] is True
        assert result["branch"] == "main"
        assert result["summary"]["staged"] == 0
        assert result["summary"]["unstaged"] == 0

    def test_parse_modified(self, git_tool):
        lines = [
            "## main...origin/main",
            " M src/file.py",
            "?? new_file.txt",
            "A  staged.py",
        ]
        result = git_tool._parse_status(lines, "## main...origin/main")
        assert result["is_clean"] is False
        assert result["summary"]["staged"] >= 1
        assert result["summary"]["unstaged"] >= 1
        assert result["summary"]["untracked"] >= 1

    def test_parse_ahead_behind(self, git_tool):
        lines = ["## feature...origin/feature [ahead 3, behind 1]"]
        result = git_tool._parse_status(lines, "## feature...origin/feature [ahead 3, behind 1]")
        assert result["branch"] == "feature"
        assert result["branch_info"]["ahead"] == 3
        assert result["branch_info"]["behind"] == 1


# ======================================================================
#  Tests: _parse_output (dispatch)
# ======================================================================

class TestGitToolParseOutput:
    def test_log_output(self, git_tool):
        output = "* abc1234 first commit\n* def5678 second commit"
        result = git_tool._parse_output("log", output)
        assert result["count"] == 2
        assert "commits" in result

    def test_stash_list_output(self, git_tool):
        output = "stash@{0}: WIP on main\nstash@{1}: WIP on dev"
        result = git_tool._parse_output("stash_list", output)
        assert result["count"] == 2

    def test_default_output(self, git_tool):
        output = "some\ntext"
        result = git_tool._parse_output("diff", output)
        assert result["lines"] == 2


# ======================================================================
#  Tests: Validação de parâmetros
# ======================================================================

class TestGitToolValidation:
    def test_missing_action(self, git_tool):
        errors = git_tool.validate_params({})
        assert len(errors) >= 1

    def test_valid_action(self, git_tool):
        errors = git_tool.validate_params({"action": "status"})
        assert errors == []

    def test_unknown_param(self, git_tool):
        errors = git_tool.validate_params({
            "action": "status",
            "unknown_param": 123,
        })
        unknown = [e for e in errors if "não reconhecido" in e]
        assert len(unknown) >= 1

    def test_bad_max_count_type(self, git_tool):
        errors = git_tool.validate_params({
            "action": "log",
            "max_count": "not_an_int",
        })
        int_errors = [e for e in errors if "deve ser inteiro" in e]
        assert len(int_errors) >= 1
