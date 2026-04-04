import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from click.testing import CliRunner

from fm.cli import main

_SEGMENTATION_RESPONSE = json.dumps([{
    "subtask_id": "s1",
    "raw_description": "Fix the login bug",
    "generalized_description": "Agent fixes authentication bug",
    "turn_indices": [0, 1],
}])

_INTELLIGENCE_RESPONSE = json.dumps({
    "reasoning_categories": {"analytical": [], "planning": [], "validation": [], "reflection": []},
    "cognitive_patterns": ["error_recognition"],
    "outcome": "recovery",
})

_ATTRIBUTION_RESPONSE = json.dumps({
    "root_causes": ["always returns True"],
    "contributing_factors": [],
    "causal_chain": ["1: auth bypassed", "2: fix applied"],
})


class TestExtractCommand:
    def test_extract_single_session(self, sample_jsonl: Path, tmp_path: Path) -> None:
        db_path = tmp_path / "tips.db"
        mock_tips_json = json.dumps({
            "tips": [
                {
                    "category": "strategy",
                    "content": "Read before editing",
                    "purpose": "Prevents failures",
                    "steps": ["Read", "Edit"],
                    "trigger": "When editing files",
                    "priority": "high",
                }
            ]
        })

        _TASK_LEVEL_SUMMARY = "Agent reads files carefully before editing to prevent failures."
        responses = [
            _SEGMENTATION_RESPONSE,
            _INTELLIGENCE_RESPONSE, _ATTRIBUTION_RESPONSE, mock_tips_json,
            _TASK_LEVEL_SUMMARY,
            _INTELLIGENCE_RESPONSE, _ATTRIBUTION_RESPONSE, json.dumps({"tips": []}),
        ]

        runner = CliRunner()
        with patch("fm.llm.subprocess.run") as mock_run:
            mock_run.side_effect = [
                type("R", (), {"returncode": 0, "stdout": r, "stderr": ""})()
                for r in responses
            ]
            with patch("fm.embeddings._embed_voyage", return_value=[0.1, 0.2]):
                with patch.dict("os.environ", {"VOYAGE_API_KEY": "test"}):
                    result = runner.invoke(
                        main,
                        ["extract", str(sample_jsonl), "--db", str(db_path), "--min-turns=1"],
                    )

        assert result.exit_code == 0
        assert "Extracted" in result.output or "tip" in result.output.lower()

    def test_extract_skips_processed(self, sample_jsonl: Path, tmp_path: Path) -> None:
        db_path = tmp_path / "tips.db"
        from fm.store import TipStore

        store = TipStore(db_path)
        store.mark_session_processed("test-session-001", str(sample_jsonl), tip_count=1)

        runner = CliRunner()
        result = runner.invoke(
            main, ["extract", str(sample_jsonl), "--db", str(db_path)]
        )
        assert result.exit_code == 0
        assert "already processed" in result.output.lower() or "skip" in result.output.lower()


class TestRetrieveCommand:
    def test_retrieve_outputs_tips(self, tmp_path: Path) -> None:
        db_path = tmp_path / "tips.db"
        from fm.models import Tip
        from fm.store import TipStore

        store = TipStore(db_path)
        tip = Tip(
            category="strategy",
            content="Always check prerequisites",
            purpose="Prevents failures",
            steps=["Check step 1"],
            trigger="When deploying",
            priority="high",
            source_session_id="s1",
            source_project="proj",
        )
        store.add_tip(tip, embedding=[1.0, 0.0, 0.0], embedding_provider="voyage")

        runner = CliRunner()
        with patch("fm.retriever.embed_text") as mock_embed:
            from fm.embeddings import EmbeddingResult

            mock_embed.return_value = EmbeddingResult(
                vector=[0.95, 0.05, 0.0], provider="voyage"
            )
            result = runner.invoke(
                main,
                ["retrieve", "deploy the app", "--db", str(db_path)],
            )

        assert result.exit_code == 0
        assert "prerequisites" in result.output


class TestHookRetrieveCommand:
    def test_reads_json_from_stdin(self, tmp_path: Path) -> None:
        db_path = tmp_path / "tips.db"
        from fm.models import Tip
        from fm.store import TipStore

        store = TipStore(db_path)
        tip = Tip(
            category="strategy",
            content="Always check prerequisites",
            purpose="Prevents failures",
            steps=["Check step 1"],
            trigger="When deploying",
            priority="high",
            source_session_id="s1",
            source_project="proj",
        )
        store.add_tip(tip, embedding=[1.0, 0.0, 0.0], embedding_provider="voyage")

        hook_input = json.dumps({
            "session_id": "s1",
            "prompt": "deploy the app",
            "hook_event_name": "UserPromptSubmit",
            "cwd": "/project",
        })

        runner = CliRunner()
        with patch("fm.retriever.embed_text") as mock_embed:
            from fm.embeddings import EmbeddingResult

            mock_embed.return_value = EmbeddingResult(
                vector=[0.95, 0.05, 0.0], provider="voyage"
            )
            result = runner.invoke(
                main,
                ["hook-retrieve", "--db", str(db_path)],
                input=hook_input,
            )

        assert result.exit_code == 0
        assert "prerequisites" in result.output


class TestTipsCommands:
    def test_tips_list(self, tmp_path: Path) -> None:
        db_path = tmp_path / "tips.db"
        from fm.models import Tip
        from fm.store import TipStore

        store = TipStore(db_path)
        tip = Tip(
            category="strategy",
            content="Test tip content",
            purpose="Testing",
            steps=[],
            trigger="Always",
            priority="low",
            source_session_id="s1",
            source_project="proj",
        )
        store.add_tip(tip)

        runner = CliRunner()
        result = runner.invoke(main, ["tips", "list", "--db", str(db_path)])
        assert result.exit_code == 0
        assert "Test tip content" in result.output
