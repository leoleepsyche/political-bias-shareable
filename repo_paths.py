from __future__ import annotations

import os
from pathlib import Path


def candidate_repo_paths(workspace: Path, repo_name: str) -> list[Path]:
    candidates: list[Path] = []

    env_key_base = repo_name.upper().replace("-", "_")
    for env_key in (f"{env_key_base}_DIR", f"{env_key_base}_PATH"):
        value = os.environ.get(env_key)
        if value:
            candidates.append(Path(value).expanduser())

    current = workspace.resolve()
    for parent in [current] + list(current.parents):
        candidates.append(parent / repo_name)
        candidates.append(parent.parent / repo_name)

    home = Path.home()
    candidates.extend(
        [
            home / repo_name,
            home / "Documents" / "Playground" / repo_name,
        ]
    )

    deduped: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.expanduser()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return deduped


def resolve_repo_path(workspace: Path, repo_name: str) -> Path:
    for candidate in candidate_repo_paths(workspace, repo_name):
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not locate dependency repo {repo_name!r}. Checked: "
        + ", ".join(str(path) for path in candidate_repo_paths(workspace, repo_name))
    )
