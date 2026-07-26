import subprocess
from pathlib import Path

import pytest

from webui.unsloth_version import UnslothVersionManager


def run_git(directory: Path, *arguments: str) -> str:
    result = subprocess.run(
        [
            "git", "-C", str(directory),
            "-c", "user.email=test@example.com",
            "-c", "user.name=test",
            "-c", "commit.gpgsign=false",
            *arguments,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


@pytest.fixture()
def repos(tmp_path):
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(upstream)], check=True)
    (upstream / "version.txt").write_text("1\n")
    run_git(upstream, "add", ".")
    run_git(upstream, "commit", "-q", "-m", "release one")
    run_git(upstream, "tag", "2026.1.0")
    (upstream / "version.txt").write_text("2\n")
    run_git(upstream, "commit", "-q", "-am", "release two")
    run_git(upstream, "tag", "2026.2.0")
    (upstream / "version.txt").write_text("3\n")
    run_git(upstream, "commit", "-q", "-am", "head of main")

    clone = tmp_path / "clone"
    subprocess.run(
        ["git", "clone", "-q", str(upstream), str(clone)], check=True
    )
    manager = UnslothVersionManager(
        repo_dir=clone,
        state_file=tmp_path / "state.json",
        remote="origin",
        branch="main",
    )
    return upstream, clone, manager


def test_status_reports_current_commit(repos):
    _, clone, manager = repos
    status = manager.status()
    assert status["available"] is True
    assert status["branch"] == "main"
    assert status["subject"] == "head of main"
    assert status["dirty"] is False
    assert status["rollback_available"] is False
    assert status["commit"] == run_git(clone, "rev-parse", "HEAD")


def test_check_remote_detects_new_upstream_commit(repos):
    upstream, _, manager = repos
    assert manager.status(check_remote=True)["update_available"] is False
    (upstream / "version.txt").write_text("4\n")
    run_git(upstream, "commit", "-q", "-am", "newer upstream")
    assert manager.status(check_remote=True)["update_available"] is True


def test_update_moves_to_latest_and_allows_rollback(repos):
    upstream, clone, manager = repos
    (upstream / "version.txt").write_text("4\n")
    run_git(upstream, "commit", "-q", "-am", "newer upstream")

    result = manager.update()
    assert result["changed"] is True
    assert result["subject"] == "newer upstream"
    assert result["branch"] == "main"
    assert result["rollback_available"] is True
    assert result["update_available"] is False

    rolled = manager.rollback()
    assert rolled["changed"] is True
    assert rolled["subject"] == "head of main"
    assert rolled["branch"] == "main"
    assert rolled["rollback_available"] is False


def test_update_without_new_commits_is_noop(repos):
    _, _, manager = repos
    result = manager.update()
    assert result["changed"] is False
    assert result["rollback_available"] is False


def test_checkout_tag_pins_and_rollback_restores_branch(repos):
    _, _, manager = repos
    result = manager.checkout("2026.1.0")
    assert result["changed"] is True
    assert result["pinned_tag"] == "2026.1.0"
    assert result["branch"] is None
    assert result["subject"] == "release one"

    rolled = manager.rollback()
    assert rolled["branch"] == "main"
    assert rolled["subject"] == "head of main"


def test_checkout_rejects_invalid_and_unknown_refs(repos):
    _, _, manager = repos
    with pytest.raises(RuntimeError):
        manager.checkout("../evil")
    with pytest.raises(RuntimeError):
        manager.checkout("-rf")
    with pytest.raises(RuntimeError):
        manager.checkout("no-such-ref")


def test_dirty_tree_blocks_switching(repos):
    _, clone, manager = repos
    (clone / "version.txt").write_text("modified locally\n")
    status = manager.status()
    assert status["dirty"] is True
    assert "version.txt" in status["dirty_files"]
    with pytest.raises(RuntimeError):
        manager.update()
    with pytest.raises(RuntimeError):
        manager.checkout("2026.1.0")


def test_local_commits_on_branch_block_update(repos):
    upstream, clone, manager = repos
    (clone / "local.txt").write_text("local work\n")
    run_git(clone, "add", ".")
    run_git(clone, "commit", "-q", "-m", "local only commit")
    (upstream / "version.txt").write_text("4\n")
    run_git(upstream, "commit", "-q", "-am", "newer upstream")
    with pytest.raises(RuntimeError):
        manager.update()


def test_remote_tags_lists_newest_first(repos):
    _, _, manager = repos
    tags = manager.remote_tags()
    assert [tag["name"] for tag in tags] == ["2026.2.0", "2026.1.0"]
