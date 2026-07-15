"""Strict public API contracts for the one-turn Ask WikiPulse assistant."""

import unicodedata
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _AssistantApiContract(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        hide_input_in_errors=True,
    )


class AudienceQuestionRequest(_AssistantApiContract):
    question: str

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        normalized = unicodedata.normalize("NFC", value)
        if any(unicodedata.category(character).startswith("C") for character in normalized):
            raise ValueError("question contains unsupported characters")
        normalized = " ".join(normalized.split())
        if not 3 <= len(normalized) <= 500:
            raise ValueError("question length is outside the allowed range")
        return normalized


class AssistantCitationResponse(_AssistantApiContract):
    article_title: str
    article_url: str | None = None
    audience_label: str
    relevance: str


class AudienceQuestionResponse(_AssistantApiContract):
    answer: str
    citations: tuple[AssistantCitationResponse, ...]
    evidence_status: Literal["grounded", "insufficient_evidence"]
    suggested_follow_up_questions: tuple[str, ...] = Field(max_length=3)
