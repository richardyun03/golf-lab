import json
import sqlite3
from pathlib import Path
from app.core.config import Settings
from app.schemas.analysis import AnalysisResponse, SessionTrendPoint, SwingMetrics, SwingPhase


DB_NAME = "golf_lab.db"


class AnalysisStorage:
    """SQLite-backed storage for analysis results."""

    def __init__(self, settings: Settings):
        db_path = settings.data_dir / DB_NAME
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                session_id TEXT PRIMARY KEY,
                result_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def save(self, result: AnalysisResponse) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO analyses (session_id, result_json) VALUES (?, ?)",
            (result.session_id, result.model_dump_json()),
        )
        self.conn.commit()

    @staticmethod
    def _backfill_phase_scores(data: dict) -> dict[str, float]:
        """Compute phase_scores for sessions stored before the feature existed."""
        from app.services.swing_analyzer import SwingAnalyzer
        from app.schemas.analysis import SwingFault
        faults = [SwingFault(**f) for f in data.get("faults", [])]
        return SwingAnalyzer._compute_phase_scores(faults)

    def get(self, session_id: str) -> AnalysisResponse | None:
        row = self.conn.execute(
            "SELECT result_json FROM analyses WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        result = AnalysisResponse.model_validate_json(row["result_json"])
        if not result.phase_scores:
            data = json.loads(row["result_json"])
            result.phase_scores = self._backfill_phase_scores(data)
        return result

    def list_sessions(self, limit: int = 20) -> list[dict]:
        rows = self.conn.execute(
            "SELECT session_id, created_at FROM analyses ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [{"session_id": r["session_id"], "created_at": r["created_at"]} for r in rows]

    def list_sessions_with_metrics(self, limit: int = 50) -> list[SessionTrendPoint]:
        rows = self.conn.execute(
            "SELECT session_id, result_json, created_at FROM analyses ORDER BY created_at ASC LIMIT ?",
            (limit,),
        ).fetchall()
        results = []
        for r in rows:
            data = json.loads(r["result_json"])
            phase_scores = data.get("phase_scores") or self._backfill_phase_scores(data)
            results.append(SessionTrendPoint(
                session_id=r["session_id"],
                created_at=r["created_at"],
                overall_score=data.get("overall_score", 0),
                phase_scores=phase_scores,
                metrics=SwingMetrics(**data.get("metrics", {})),
                fault_count=len(data.get("faults", [])),
                fault_types=[f["fault_type"] for f in data.get("faults", [])],
            ))
        return results
