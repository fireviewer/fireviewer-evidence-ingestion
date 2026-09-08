"""Area extraction only: no geometry, fabricated tolerance, or incident assignment."""

from typing import Literal

from pydantic import AwareDatetime, Field, model_validator

from fireviewer_contracts.contracts import StrictModel


class ExtractedSurfaceArea(StrictModel):
    component: Literal["affected", "active"]
    scope: Literal["incident", "episode"] = "incident"
    accumulation: Literal["cumulative", "incremental"] = "cumulative"
    qualifier: Literal["exact", "approximate", "minimum", "maximum", "interval"]
    value_ha: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    lower_ha: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    upper_ha: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    valid_from: AwareDatetime
    valid_until: AwareDatetime

    @model_validator(mode="after")
    def validate_area(self) -> "ExtractedSurfaceArea":
        if self.valid_until < self.valid_from:
            raise ValueError("area time interval is reversed")
        if (
            self.lower_ha is not None
            and self.upper_ha is not None
            and self.lower_ha > self.upper_ha
        ):
            raise ValueError("area bounds are reversed")
        if self.qualifier in {"exact", "approximate"} and self.value_ha is None:
            raise ValueError("nominal area is missing")
        if self.qualifier in {"minimum", "interval"} and self.lower_ha is None:
            raise ValueError("lower area bound is missing")
        if self.qualifier in {"maximum", "interval"} and self.upper_ha is None:
            raise ValueError("upper area bound is missing")
        return self
