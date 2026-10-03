"""A small copy of KnowMe's production SQLite retrieval behavior.

This module owns only the isolated training fixture. It keeps the same query
normalization, FTS5/BM25 ordering, CJK substring fallback, and top-k behavior as
the main project, while returning row ids so dataset validation can check the
expected evidence directly.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path


_UNSEGMENTED_SCRIPTS = (
    "\u3040-\u309f"
    "\u30a0-\u30ff"
    "\u3400-\u4dbf"
    "\u4e00-\u9fff"
    "\uf900-\ufaff"
)
_UNSEGMENTED = re.compile(f"[{_UNSEGMENTED_SCRIPTS}]")
_UNSEGMENTED_RUN = re.compile(f"[{_UNSEGMENTED_SCRIPTS}]{{2,}}")


SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY,
    subject TEXT NOT NULL,
    content TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'fixture',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
    subject, content, content=facts, content_rowid=id
);
CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
    INSERT INTO facts_fts(rowid, subject, content)
    VALUES (new.id, new.subject, new.content);
END;
CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, subject, content)
    VALUES ('delete', old.id, old.subject, old.content);
END;
CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, subject, content)
    VALUES ('delete', old.id, old.subject, old.content);
    INSERT INTO facts_fts(rowid, subject, content)
    VALUES (new.id, new.subject, new.content);
END;

CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY,
    happened_at TEXT NOT NULL,
    summary TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
    summary, content=episodes, content_rowid=id
);
CREATE TRIGGER IF NOT EXISTS episodes_ai AFTER INSERT ON episodes BEGIN
    INSERT INTO episodes_fts(rowid, summary)
    VALUES (new.id, new.summary);
END;
CREATE TRIGGER IF NOT EXISTS episodes_ad AFTER DELETE ON episodes BEGIN
    INSERT INTO episodes_fts(episodes_fts, rowid, summary)
    VALUES ('delete', old.id, old.summary);
END;
CREATE TRIGGER IF NOT EXISTS episodes_au AFTER UPDATE ON episodes BEGIN
    INSERT INTO episodes_fts(episodes_fts, rowid, summary)
    VALUES ('delete', old.id, old.summary);
    INSERT INTO episodes_fts(rowid, summary)
    VALUES (new.id, new.summary);
END;
"""


def connect(path: str | Path = ":memory:") -> sqlite3.Connection:
    """Create an isolated fixture database with the production schema."""
    database = sqlite3.connect(path)
    database.row_factory = sqlite3.Row
    database.executescript(SCHEMA)
    return database


def fts_query(text: str) -> str:
    words = re.findall(r"[^\W_]{2,}", text.lower())
    if not words:
        return ""
    return " OR ".join(
        f"{word}*" if _UNSEGMENTED.search(word) else word
        for word in dict.fromkeys(words)
    )


def substring_terms(text: str) -> list[str]:
    terms: list[str] = []
    for run in _UNSEGMENTED_RUN.findall(text.lower()):
        terms.append(run)
        terms.extend(run[index:index + 2] for index in range(len(run) - 1))
    return list(dict.fromkeys(terms))[:12]


def searchable(text: str) -> bool:
    return bool(fts_query(text) or substring_terms(text))


class FactStore:
    def __init__(self, database: sqlite3.Connection):
        self.database = database

    def add(self, subject: str, content: str) -> int:
        cursor = self.database.execute(
            "INSERT INTO facts (subject, content) VALUES (?, ?)",
            (subject.lower().strip(), content),
        )
        self.database.commit()
        return int(cursor.lastrowid)

    def _search_rows(self, query: str, top_k: int) -> list[sqlite3.Row]:
        rows: list[sqlite3.Row] = []
        normalized = fts_query(query)
        if normalized:
            rows = self.database.execute(
                "SELECT f.id, f.subject, f.content "
                "FROM facts_fts JOIN facts f ON f.id = facts_fts.rowid "
                "WHERE facts_fts MATCH ? ORDER BY rank LIMIT ?",
                (normalized, top_k),
            ).fetchall()

        seen = {row["id"] for row in rows}
        for term in substring_terms(query):
            if len(rows) >= top_k:
                break
            candidates = self.database.execute(
                "SELECT id, subject, content FROM facts "
                "WHERE content LIKE ? OR subject LIKE ? "
                "ORDER BY id DESC LIMIT ?",
                (f"%{term}%", f"%{term}%", top_k),
            )
            for row in candidates:
                if row["id"] not in seen:
                    seen.add(row["id"])
                    rows.append(row)
        return rows[:top_k]

    def search(self, query: str, top_k: int = 4) -> list[dict]:
        if not searchable(query):
            return []
        return [dict(row) for row in self._search_rows(query, top_k)]


class EpisodeStore:
    def __init__(self, database: sqlite3.Connection):
        self.database = database

    def add(self, happened_at: str, summary: str) -> int:
        cursor = self.database.execute(
            "INSERT INTO episodes (happened_at, summary) VALUES (?, ?)",
            (happened_at, summary),
        )
        self.database.commit()
        return int(cursor.lastrowid)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        if not searchable(query):
            rows = self.database.execute(
                "SELECT id, happened_at, summary FROM episodes "
                "ORDER BY happened_at DESC LIMIT ?",
                (top_k,),
            )
            return [dict(row) for row in rows]

        rows: list[sqlite3.Row] = []
        normalized = fts_query(query)
        if normalized:
            rows = self.database.execute(
                "SELECT e.id, e.happened_at, e.summary "
                "FROM episodes_fts JOIN episodes e ON e.id = episodes_fts.rowid "
                "WHERE episodes_fts MATCH ? "
                "ORDER BY rank, e.happened_at DESC LIMIT ?",
                (normalized, top_k),
            ).fetchall()

        seen = {row["id"] for row in rows}
        for term in substring_terms(query):
            if len(rows) >= top_k:
                break
            candidates = self.database.execute(
                "SELECT id, happened_at, summary FROM episodes "
                "WHERE summary LIKE ? ORDER BY happened_at DESC LIMIT ?",
                (f"%{term}%", top_k),
            )
            for row in candidates:
                if row["id"] not in seen:
                    seen.add(row["id"])
                    rows.append(row)
        return [dict(row) for row in rows[:top_k]]
