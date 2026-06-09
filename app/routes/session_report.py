import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from scripts.session_report.generate import generate_session_report

router = APIRouter(tags=["session-report"])


class SessionReportRequest(BaseModel):
    session: dict[str, Any] = Field(default_factory=dict)
    turns: list[dict[str, Any]] = Field(default_factory=list)


@router.post("/session/report")
async def create_session_report(body: SessionReportRequest) -> dict:
    if not body.turns:
        raise HTTPException(status_code=400, detail="turns is empty")

    try:
        return await asyncio.to_thread(
            generate_session_report,
            body.model_dump(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {e}") from e
