"""Integration test exercising the full pipeline with mocked LLM calls."""
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from click.testing import CliRunner

from fm.cli import main
from fm.embeddings import EmbeddingResult
from fm.store import TipStore


class TestFullPipeline:
    def test_extract_then_retrieve(self, sample_jsonl: Path, tmp_path: Path) -> None:
        """Full loop: parse → extract → store → retrieve."""
        db_path = tmp_path / "tips.db"

        mock_tips = json.dumps({
            "tips": [
                {
                    "category": "recovery",
                    "content": "When file read fails, check if the path exists before retrying",
                    "purpose": "Prevents repeated failures on missing files",
                    "steps": ["Check file exists", "Read file", "Handle missing file"],
                    "trigger": "When reading files that may not exist",
                    "priority": "high",
                },
                {
                    "category": "strategy",
                    "content": "Always verify login functionality after modifying auth code",
                    "purpose": "Auth changes are high-risk and need immediate verification",
                    "steps": ["Modify auth code", "Run auth tests", "Verify manually"],
                    "trigger": "When modifying authentication code",
                    "priority": "critical",
                },
            ]
        })
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = mock_tips

        runner = CliRunner()
        with patch("fm.extractor.subprocess.run", return_value=mock_result):
            with patch("fm.embeddings._embed_voyage") as mock_voyage:
                mock_voyage.side_effect = lambda text: [0.8, 0.1, 0.1] if "auth" in text.lower() else [0.1, 0.8, 0.1]
                with patch.dict("os.environ", {"VOYAGE_API_KEY": "test"}):
                    result = runner.invoke(
                        main,
                        ["extract", str(sample_jsonl), "--db", str(db_path)],
                    )

        assert result.exit_code == 0
        assert "2 tips" in result.output.lower() or "extracted 2" in result.output.lower()

        store = TipStore(db_path)
        all_tips = store.list_tips()
        assert len(all_tips) == 2

        with patch("fm.retriever.embed_text") as mock_embed:
            mock_embed.return_value = EmbeddingResult(
                vector=[0.75, 0.15, 0.1], provider="voyage"
            )
            result = runner.invoke(
                main,
                ["retrieve", "fix the authentication module", "--db", str(db_path)],
            )

        assert result.exit_code == 0
        assert "auth" in result.output.lower()

    def test_hook_retrieve_integration(self, sample_jsonl: Path, tmp_path: Path) -> None:
        """Test the hook entrypoint reads stdin and returns tips."""
        db_path = tmp_path / "tips.db"

        from fm.models import Tip

        store = TipStore(db_path)
        tip = Tip(
            category="strategy",
            content="Run tests after every code change",
            purpose="Catch regressions early",
            steps=["Make change", "Run tests"],
            trigger="After modifying code",
            priority="high",
            source_session_id="s1",
            source_project="proj",
        )
        store.add_tip(tip, embedding=[0.9, 0.1, 0.0], embedding_provider="voyage")

        hook_input = json.dumps({
            "prompt": "I just changed the validation logic",
            "session_id": "s2",
            "hook_event_name": "UserPromptSubmit",
            "cwd": "/project",
        })

        runner = CliRunner()
        with patch("fm.retriever.embed_text") as mock_embed:
            mock_embed.return_value = EmbeddingResult(
                vector=[0.85, 0.15, 0.0], provider="voyage"
            )
            result = runner.invoke(
                main,
                ["hook-retrieve", "--db", str(db_path)],
                input=hook_input,
            )

        assert result.exit_code == 0
        assert "tests" in result.output.lower()

    def test_extract_all_skips_processed(self, sample_jsonl: Path, tmp_path: Path) -> None:
        """extract-all should skip already-processed sessions."""
        db_path = tmp_path / "tips.db"
        store = TipStore(db_path)

        store.mark_session_processed("test-session-001", str(sample_jsonl), tip_count=1)

        projects_dir = tmp_path / ".claude" / "projects" / "test-project"
        projects_dir.mkdir(parents=True)
        import shutil
        shutil.copy(sample_jsonl, projects_dir / "test-session-001.jsonl")

        runner = CliRunner()
        with patch("fm.cli.Path.home", return_value=tmp_path):
            result = runner.invoke(
                main, ["extract-all", "--db", str(db_path)]
            )

        assert result.exit_code == 0
        assert "0 tips" in result.output.lower() or "processed 0" in result.output.lower()
