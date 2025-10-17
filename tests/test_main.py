from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import main


def test_attempt_kaggle_download_includes_auth(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    recorded_command: list[str] = []

    def fake_run_command(command, cwd=None):  # type: ignore[override]
        nonlocal recorded_command
        recorded_command = list(command)
        output_path = Path(command[command.index("-o") + 1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"zip")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setenv("KAGGLE_USERNAME", "test-user")
    monkeypatch.setenv("KAGGLE_KEY", "test-key")
    monkeypatch.setattr(main.shutil, "which", lambda _: "curl")
    monkeypatch.setattr(main, "_run_command", fake_run_command)
    monkeypatch.setattr(main, "_extract_csv_from_zip", lambda *args, **kwargs: True)

    result = main._attempt_kaggle_download(  # pylint: disable=protected-access
        Path("dataset.csv"),
        download_url="https://example.com/dataset.zip",
        downloads_dir=tmp_path,
    )

    assert result is True
    assert "-u" in recorded_command
    assert "test-user:test-key" in recorded_command
    assert "X-Kaggle-User: test-user" in recorded_command
    assert "X-Kaggle-Key: test-key" in recorded_command
    assert "--fail" in recorded_command
    assert "--show-error" in recorded_command
