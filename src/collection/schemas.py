"""Data validation schemas for raw vacancy payloads using Pydantic."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RawSalary(BaseModel):
    """Salary range representation from raw API payload."""

    model_config = ConfigDict(extra="ignore")

    salary_from: float | None = Field(default=None, alias="from")
    salary_to: float | None = Field(default=None, alias="to")
    currency: str | None = None
    gross: bool | None = None


class RawArea(BaseModel):
    """Area representation from raw API payload."""

    model_config = ConfigDict(extra="ignore")

    id: str
    name: str

    @field_validator("id", mode="before")
    @classmethod
    def coerce_id_to_str(cls, v: Any) -> str:
        return str(v)


class RawKeySkill(BaseModel):
    """Key skill item representation."""

    model_config = ConfigDict(extra="ignore")

    name: str


class RawVacancy(BaseModel):
    """Minimal validated contract for raw vacancy data."""

    model_config = ConfigDict(extra="ignore")

    id: str
    name: str
    published_at: str
    area: RawArea
    salary: RawSalary | None = None
    description: str | None = None
    key_skills: list[RawKeySkill] = Field(default_factory=list)
    professional_roles: list[dict[str, Any]] = Field(default_factory=list)
    experience: dict[str, Any] | None = None
    employment: dict[str, Any] | None = None
    schedule: dict[str, Any] | None = None
    alternate_url: str | None = None

    @field_validator("id", mode="before")
    @classmethod
    def coerce_id_to_str(cls, v: Any) -> str:
        return str(v)

    @property
    def has_salary(self) -> bool:
        """Check whether at least one salary bound is specified."""
        if not self.salary:
            return False
        return self.salary.salary_from is not None or self.salary.salary_to is not None
