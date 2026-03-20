"""
PostgreSQL data layer for Chainlit 2.x.
Saves threads (chat sessions) and steps (messages) to the shared database.

Tables (managed by Prisma in web/):
  chainlit_threads (id, user_id, name, metadata, tags, created_at, updated_at)
  chainlit_steps   (id, thread_id, type, name, input, output, metadata, is_error, created_at)

Install deps:  pip install asyncpg
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import asyncpg
from chainlit.data.base import BaseDataLayer
from chainlit.types import Feedback, PaginatedResponse, Pagination, ThreadDict, ThreadFilter

if TYPE_CHECKING:
    import chainlit as cl


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PostgreSQLDataLayer(BaseDataLayer):
    """Minimal implementation — persists users, threads, and steps."""

    def __init__(self, db_url: str) -> None:
        self._db_url = db_url
        self._pool: Optional[asyncpg.Pool] = None

    async def _get_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(self._db_url)
        return self._pool

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()

    # ── Users ────────────────────────────────────────────────────────────────

    async def get_user(self, identifier: str):
        import chainlit as cl  # noqa: PLC0415

        pool = await self._get_pool()
        row = await pool.fetchrow("SELECT * FROM users WHERE email = $1", identifier)
        if not row:
            return None
        return cl.PersistedUser(
            id=row["id"],
            identifier=row["email"],
            display_name=row["name"],
            createdAt=row["created_at"].isoformat(),
        )

    async def create_user(self, user) -> Optional[Any]:
        # Users are created via the web app — just look them up
        return await self.get_user(user.identifier)

    # ── Threads ──────────────────────────────────────────────────────────────

    async def update_thread(
        self,
        thread_id: str,
        name: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        pool = await self._get_pool()
        meta_json = json.dumps(metadata) if metadata is not None else None
        await pool.execute(
            """
            INSERT INTO chainlit_threads (id, user_id, name, metadata, tags)
            VALUES ($1, $2, $3, $4::jsonb, $5)
            ON CONFLICT (id) DO UPDATE
              SET name      = COALESCE(EXCLUDED.name, chainlit_threads.name),
                  user_id   = COALESCE(EXCLUDED.user_id, chainlit_threads.user_id),
                  metadata  = COALESCE(EXCLUDED.metadata, chainlit_threads.metadata),
                  tags      = COALESCE(EXCLUDED.tags, chainlit_threads.tags),
                  updated_at = NOW()
            """,
            thread_id,
            user_id,
            name,
            meta_json,
            tags or [],
        )

    async def get_thread(self, thread_id: str) -> Optional[ThreadDict]:
        pool = await self._get_pool()
        row = await pool.fetchrow(
            "SELECT * FROM chainlit_threads WHERE id = $1", thread_id
        )
        if not row:
            return None
        steps = await pool.fetch(
            "SELECT * FROM chainlit_steps WHERE thread_id = $1 ORDER BY created_at",
            thread_id,
        )
        return {
            "id": row["id"],
            "name": row["name"],
            "createdAt": row["created_at"].isoformat(),
            "metadata": row["metadata"] or {},
            "tags": list(row["tags"]),
            "steps": [_step_to_dict(s) for s in steps],
        }  # type: ignore[return-value]

    async def get_thread_author(self, thread_id: str) -> str:
        pool = await self._get_pool()
        row = await pool.fetchrow(
            """
            SELECT u.email FROM chainlit_threads t
            JOIN users u ON u.id = t.user_id
            WHERE t.id = $1
            """,
            thread_id,
        )
        return row["email"] if row else ""

    async def delete_thread(self, thread_id: str) -> None:
        pool = await self._get_pool()
        await pool.execute("DELETE FROM chainlit_threads WHERE id = $1", thread_id)

    async def list_threads(
        self, pagination: Pagination, filters: ThreadFilter
    ) -> PaginatedResponse[ThreadDict]:
        pool = await self._get_pool()
        user_id: Optional[str] = getattr(filters, "userId", None)

        rows = await pool.fetch(
            "SELECT * FROM chainlit_threads WHERE user_id = $1 ORDER BY updated_at DESC LIMIT $2",
            user_id,
            pagination.first or 20,
        )
        threads = []
        for row in rows:
            steps = await pool.fetch(
                "SELECT * FROM chainlit_steps WHERE thread_id = $1 ORDER BY created_at",
                row["id"],
            )
            threads.append(
                {
                    "id": row["id"],
                    "name": row["name"],
                    "createdAt": row["created_at"].isoformat(),
                    "metadata": row["metadata"] or {},
                    "tags": list(row["tags"]),
                    "steps": [_step_to_dict(s) for s in steps],
                }
            )
        return PaginatedResponse(data=threads, pageInfo={"hasNextPage": False, "startCursor": None, "endCursor": None})  # type: ignore[return-value]

    # ── Steps ────────────────────────────────────────────────────────────────

    async def create_step(self, step_dict: Dict[str, Any]) -> None:
        pool = await self._get_pool()
        meta = json.dumps(step_dict.get("metadata")) if step_dict.get("metadata") else None
        created_at = _parse_dt(step_dict.get("createdAt"))
        await pool.execute(
            """
            INSERT INTO chainlit_steps (id, thread_id, type, name, input, output, metadata, is_error, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9)
            ON CONFLICT (id) DO UPDATE
              SET output   = COALESCE(EXCLUDED.output, chainlit_steps.output),
                  metadata = COALESCE(EXCLUDED.metadata, chainlit_steps.metadata)
            """,
            step_dict.get("id"),
            step_dict.get("threadId"),
            step_dict.get("type"),
            step_dict.get("name"),
            step_dict.get("input"),
            step_dict.get("output"),
            meta,
            bool(step_dict.get("isError", False)),
            created_at,
        )

    async def update_step(self, step_dict: Dict[str, Any]) -> None:
        await self.create_step(step_dict)

    async def delete_step(self, step_id: str) -> None:
        pool = await self._get_pool()
        await pool.execute("DELETE FROM chainlit_steps WHERE id = $1", step_id)

    # ── Stubs (not needed for MVP) ────────────────────────────────────────────

    async def create_element(self, element: Any) -> None:
        pass

    async def delete_element(self, element_id: str, thread_id: Optional[str] = None) -> None:
        pass

    async def get_element(self, thread_id: str, element_id: str) -> Optional[Any]:
        return None

    async def upsert_feedback(self, feedback: Feedback) -> str:
        return ""

    async def delete_feedback(self, feedback_id: str) -> bool:
        return True

    async def get_favorite_steps(self, user_id: str) -> List[Any]:
        return []

    async def set_step_favorite(self, step_dict: Any, favorite: bool) -> Any:
        return step_dict

    def build_debug_url(self) -> str:
        return ""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _step_to_dict(row: asyncpg.Record) -> Dict[str, Any]:
    return {
        "id": row["id"],
        "threadId": row["thread_id"],
        "type": row["type"],
        "name": row["name"],
        "input": row["input"],
        "output": row["output"],
        "metadata": row["metadata"] or {},
        "isError": row["is_error"],
        "createdAt": row["created_at"].isoformat(),
    }


def _parse_dt(value: Optional[str]) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return datetime.now(timezone.utc)
