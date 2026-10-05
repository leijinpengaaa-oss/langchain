from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TicketDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=2, max_length=120)
    description: str = Field(min_length=5, max_length=4000)
    priority: Literal["low", "normal", "high"] = "normal"
    category: Literal["deployment", "account", "api", "other"] = "other"


class ApprovalDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["approve", "reject"]
    draft: TicketDraft | None = None


class MessageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=2000)
    mode: Literal["auto", "answer", "ticket"] = "auto"
    request_id: str = Field(min_length=8, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$")


class QueryPlan(BaseModel):
    intent: Literal["answer", "ticket"]
    search_query: str = Field(min_length=1, max_length=300)


class SupportingQuote(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_id: str = Field(min_length=1, max_length=100)
    quote: str = Field(min_length=8, max_length=760)


class GroundedAnswer(BaseModel):
    answer: str = Field(min_length=1, max_length=4000)
    citation_ids: list[str] = Field(max_length=5)
    supporting_quotes: list[SupportingQuote] = Field(default_factory=list, max_length=8)


class CandidateGrade(BaseModel):
    model_config = ConfigDict(extra="forbid")
    index: int = Field(ge=0, le=23, strict=True)
    grade: int = Field(ge=0, le=3, strict=True)


class RankingDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ratings: list[CandidateGrade] = Field(min_length=1, max_length=24)


class TicketStatus(BaseModel):
    status: Literal["open", "in_progress", "resolved"]


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=2000)
