"""Request models = input validation at the trust boundary + the OpenAPI spec."""
from __future__ import annotations

from typing import Literal

import os

from pydantic import BaseModel, Field, field_validator, model_validator

MAX_DIM = 512
MAX_BATCH = int(os.environ.get("WAFERSCAN_MAX_BATCH", "256"))
MAX_BATCH_DIES = int(os.environ.get("WAFERSCAN_MAX_BATCH_DIES", "2000000"))   # ~256 wafers of 88x88


def _check_map(v: list[list[int]]) -> list[list[int]]:
    if not v or len(v) > MAX_DIM or any(len(r) != len(v[0]) for r in v) or len(v[0]) > MAX_DIM:
        raise ValueError(f"wafer_map must be a non-empty rectangular grid, at most {MAX_DIM}x{MAX_DIM}")
    if any(c not in (0, 1, 2) for r in v for c in r):
        raise ValueError("wafer_map values must be 0 (off-wafer), 1 (pass) or 2 (fail)")
    return v


class Geometry(BaseModel):
    wafer_diameter_mm: float = Field(300.0, gt=0, le=450)
    edge_exclusion_mm: float = Field(3.0, ge=0, le=20)
    die_w_mm: float | None = Field(None, gt=0, le=100)
    die_h_mm: float | None = Field(None, gt=0, le=100)


class PredictRequest(BaseModel):
    wafer_map: list[list[int]] = Field(..., description="WM-811K encoding: 0 off-wafer, 1 pass, 2 fail")
    mode: Literal["auto", "fast", "full"] = "auto"
    tta: bool = False
    mc_samples: int = Field(30, ge=2, le=200)
    geometry: Geometry | None = None

    _v = field_validator("wafer_map")(_check_map)


class BatchWafer(BaseModel):
    wafer_map: list[list[int]]
    wafer_id: str | None = None
    lot_id: str | None = None

    _v = field_validator("wafer_map")(_check_map)


class BatchRequest(BaseModel):
    # limits are checked during validation, before any inference work is queued
    wafers: list[BatchWafer] = Field(..., min_length=1, max_length=MAX_BATCH)
    mode: Literal["auto", "fast", "full"] = "auto"
    geometry: Geometry | None = None
    process: list[dict] | None = Field(None, max_length=MAX_BATCH,
                                       description="optional process log rows keyed by wafer_id "
                                                   "(seq, chamber, etch_time_s, ...) -> adds root-cause Pareto")

    @model_validator(mode="after")
    def _total_dies(self):
        n = sum(len(w.wafer_map) * len(w.wafer_map[0]) for w in self.wafers)
        if n > MAX_BATCH_DIES:
            raise ValueError(f"batch has {n:,} dies; the limit is {MAX_BATCH_DIES:,} (split it into smaller batches)")
        return self


class RootCauseRequest(BaseModel):
    records: list[dict] = Field(..., min_length=10, max_length=200_000,
                                description="one row per wafer: class, seq, equipment columns, process parameters")
