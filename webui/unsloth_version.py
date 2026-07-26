"""Git-based version management for the mounted Unsloth source tree.

Unsloth 以 editable install 指向掛載進容器的 git 工作目錄，因此切換
commit / tag 後不需要重建映像檔：下一個訓練任務（獨立 python 程序）
就會使用新版本的原始碼。
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path


REF_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,119}")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class UnslothVersionManager:
    """Inspect and switch the checked-out revision of the Unsloth clone."""

    def __init__(
        self,
        repo_dir: Path,
        state_file: Path,
        remote: str | None = None,
        branch: str | None = None,
    ) -> None:
        self.repo_dir = repo_dir.resolve()
        self.state_file = state_file
        self.remote = remote or os.getenv("UNSLOTH_UPDATE_REMOTE", "origin")
        self.branch = branch or os.getenv("UNSLOTH_UPDATE_BRANCH", "main")
        self.lock = threading.RLock()

    # ---- 基礎工具 -------------------------------------------------------

    def _git(self, *arguments: str, timeout: int = 60) -> str:
        # 容器以 root 執行，掛載目錄屬於宿主機使用者，需標記 safe.directory。
        command = [
            "git",
            "-C", str(self.repo_dir),
            "-c", f"safe.directory={self.repo_dir}",
            *arguments,
        ]
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=timeout, check=False
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"無法執行 git {arguments[0]}：{exc}") from exc
        if result.returncode != 0:
            message = (result.stderr or result.stdout).strip()
            raise RuntimeError(message or f"git {arguments[0]} 結束碼：{result.returncode}")
        return result.stdout.strip()

    def _try_git(self, *arguments: str, timeout: int = 60) -> str | None:
        try:
            return self._git(*arguments, timeout=timeout)
        except RuntimeError:
            return None

    def _ensure_repo(self) -> None:
        if not (self.repo_dir / ".git").exists():
            raise RuntimeError(f"找不到 Unsloth git 原始碼：{self.repo_dir}")

    def _ensure_clean(self) -> None:
        files = self._dirty_files()
        if files:
            raise RuntimeError(
                "Unsloth 原始碼有未提交的修改，為避免遺失請先處理：" + "、".join(files[:5])
            )

    def _dirty_files(self) -> list[str]:
        # _git 會 strip 整段輸出，首行的狀態欄前導空白可能遺失，
        # 因此用 split 取路徑欄位而非固定位移。
        output = self._try_git("status", "--porcelain", "--untracked-files=no") or ""
        files = []
        for line in output.splitlines():
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                files.append(parts[1])
        return files[:20]

    def _commit_info(self, revision: str) -> dict:
        values = self._git("show", "-s", "--format=%H%n%h%n%s%n%cI", revision).splitlines()
        if len(values) < 4:
            raise RuntimeError("無法讀取 Unsloth commit 資訊")
        return {
            "commit": values[0],
            "short_commit": values[1],
            "subject": values[2],
            "committed_at": values[3],
        }

    def _head_branch(self) -> str | None:
        return self._try_git("symbolic-ref", "--quiet", "--short", "HEAD")

    def _pinned_tag(self) -> str | None:
        return self._try_git("describe", "--tags", "--exact-match", "HEAD")

    # ---- 狀態持久化（rollback 資訊） ------------------------------------

    def _load_state(self) -> dict:
        try:
            value = json.loads(self.state_file.read_text(encoding="utf-8"))
            if isinstance(value, dict) and value.get("schema") == 1:
                return value
        except (FileNotFoundError, OSError, ValueError, TypeError):
            pass
        return {}

    def _save_state(self, value: dict) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_file.with_name(f".{self.state_file.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.state_file)

    def _record_previous(self) -> None:
        info = self._commit_info("HEAD")
        self._save_state(
            {
                "schema": 1,
                "previous": {**info, "branch": self._head_branch()},
                "switched_at": utc_now(),
            }
        )

    # ---- 對外操作 -------------------------------------------------------

    def status(self, *, check_remote: bool = False) -> dict:
        with self.lock:
            try:
                self._ensure_repo()
                info = self._commit_info("HEAD")
            except RuntimeError as exc:
                return {"available": False, "error": str(exc)}
            state = self._load_state()
            previous = state.get("previous") or None
            result = {
                "available": True,
                "path": str(self.repo_dir),
                **info,
                "branch": self._head_branch(),
                "pinned_tag": self._pinned_tag(),
                "tracking": f"{self.remote}/{self.branch}",
                "remote_url": self._try_git("remote", "get-url", self.remote),
                "is_shallow": self._try_git("rev-parse", "--is-shallow-repository") == "true",
                "dirty_files": self._dirty_files(),
                "previous_commit": previous.get("commit") if previous else None,
                "previous_short_commit": previous.get("short_commit") if previous else None,
                "previous_subject": previous.get("subject") if previous else None,
                "rollback_available": bool(previous),
                "updated_at": state.get("switched_at"),
            }
            result["dirty"] = bool(result["dirty_files"])
            if check_remote:
                output = self._git(
                    "ls-remote", self.remote, f"refs/heads/{self.branch}", timeout=30
                )
                if not output:
                    raise RuntimeError(f"遠端找不到 branch：{self.branch}")
                latest = output.split()[0]
                result["latest_commit"] = latest
                result["latest_short_commit"] = latest[:9]
                result["update_available"] = latest != info["commit"]
            return result

    def remote_tags(self, limit: int = 40) -> list[dict]:
        with self.lock:
            self._ensure_repo()
            output = self._try_git(
                "-c", "versionsort.suffix=-",
                "ls-remote", "--tags", "--refs", "--sort=-v:refname", self.remote,
                timeout=30,
            )
            if output is None:
                # 舊版 git 的 ls-remote 不支援 --sort，退回預設排序後倒序。
                output = "\n".join(
                    reversed(self._git("ls-remote", "--tags", "--refs", self.remote, timeout=30).splitlines())
                )
            tags = []
            for line in output.splitlines():
                parts = line.split("\t")
                if len(parts) == 2 and parts[1].startswith("refs/tags/"):
                    tags.append({"name": parts[1][len("refs/tags/"):], "commit": parts[0][:9]})
                if len(tags) >= limit:
                    break
            return tags

    def update(self) -> dict:
        with self.lock:
            self._ensure_repo()
            self._ensure_clean()
            self._git("fetch", "--no-tags", self.remote, self.branch, timeout=300)
            target = self._git("rev-parse", "FETCH_HEAD")
            current = self._git("rev-parse", "HEAD")
            latest_info = {
                "latest_commit": target,
                "latest_short_commit": target[:9],
                "update_available": False,
            }
            if target == current:
                return {
                    **self.status(),
                    **latest_info,
                    "changed": False,
                    "message": "目前已是最新版本",
                }
            if self._head_branch() == self.branch:
                ahead = self._try_git("rev-list", "--count", "FETCH_HEAD..HEAD")
                if ahead and int(ahead) > 0:
                    raise RuntimeError(
                        f"本地 {self.branch} 有領先遠端的 commit，為避免遺失請先手動處理"
                    )
            self._record_previous()
            self._git("checkout", "--quiet", "-B", self.branch, target)
            info = self._commit_info("HEAD")
            return {
                **self.status(),
                **latest_info,
                "changed": True,
                "message": f"Unsloth 已更新至 {info['short_commit']}，下一個訓練任務生效",
            }

    def checkout(self, ref: str) -> dict:
        with self.lock:
            self._ensure_repo()
            ref = ref.strip()
            if not REF_PATTERN.fullmatch(ref) or ".." in ref or "@{" in ref:
                raise RuntimeError("無效的版本名稱")
            self._ensure_clean()

            target: str | None = None
            if self._try_git("rev-parse", "--verify", "--quiet", f"refs/tags/{ref}^{{commit}}"):
                target = f"refs/tags/{ref}"
            elif self._try_git("rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"):
                target = ref
            elif self._try_git(
                "fetch", "--no-tags", self.remote,
                f"refs/tags/{ref}:refs/tags/{ref}", timeout=300,
            ) is not None:
                target = f"refs/tags/{ref}"
            elif self._try_git("fetch", "--no-tags", self.remote, ref, timeout=300) is not None:
                target = "FETCH_HEAD"
            if target is None:
                raise RuntimeError(f"本地與 {self.remote} 都找不到版本：{ref}")

            resolved = self._git("rev-parse", f"{target}^{{commit}}")
            if resolved == self._git("rev-parse", "HEAD"):
                return {**self.status(), "changed": False, "message": f"目前已在 {ref}"}
            self._record_previous()
            self._git("checkout", "--quiet", "--detach", resolved)
            return {
                **self.status(),
                "changed": True,
                "message": f"已切換至 {ref}，下一個訓練任務生效",
            }

    def rollback(self) -> dict:
        with self.lock:
            self._ensure_repo()
            state = self._load_state()
            previous = state.get("previous")
            if not previous or not previous.get("commit"):
                raise RuntimeError("沒有可退回的上一個版本")
            self._ensure_clean()
            commit = previous["commit"]
            if not self._try_git("rev-parse", "--verify", "--quiet", f"{commit}^{{commit}}"):
                raise RuntimeError("上一個版本的 commit 已不存在於本地儲存庫")
            if previous.get("branch"):
                self._git("checkout", "--quiet", "-B", previous["branch"], commit)
            else:
                self._git("checkout", "--quiet", "--detach", commit)
            self._save_state({"schema": 1, "previous": None, "switched_at": utc_now()})
            return {
                **self.status(),
                "changed": True,
                "message": f"已退回 {previous.get('short_commit') or commit[:9]}",
            }
