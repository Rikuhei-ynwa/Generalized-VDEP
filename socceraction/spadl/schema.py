"""Schema for SPADL actions."""
from typing import Any, Optional

import pandera as pa
from pandera.typing import Series

from . import config as spadlconfig


GE_LENGTH = -5.0
LE_LENGTH = spadlconfig.FIELD_LENGTH + 5.0

GE_WIDTH = -5.0
LE_WIDTH = spadlconfig.FIELD_WIDTH + 5.0


class SPADLSchema_360(pa.SchemaModel):
    """Definition of a SPADL dataframe including 360 data."""

    game_id: Series[int] = pa.Field()
    original_event_id: Series[Any] = pa.Field(nullable=True)
    action_id: Series[int] = pa.Field()
    period_id: Series[int] = pa.Field(ge=1, le=5)
    time_seconds: Series[float] = pa.Field(ge=0)
    team_id: Series[int] = pa.Field()
    player_id: Series[int] = pa.Field()
    start_x: Series[float] = pa.Field(
        ge=GE_LENGTH, le=LE_LENGTH
        )
    start_y: Series[float] = pa.Field(
        ge=GE_WIDTH, le=LE_WIDTH
        )
    end_x: Series[float] = pa.Field(
        ge=GE_LENGTH, le=LE_LENGTH
    )
    end_y: Series[float] = pa.Field(
        ge=GE_WIDTH, le=LE_WIDTH
    )
    bodypart_id: Series[int] = pa.Field(
        isin=spadlconfig.bodyparts_df().bodypart_id
        )
    bodypart_name: Optional[Series[str]] = pa.Field(
        isin=spadlconfig.bodyparts_df().bodypart_name
        )
    type_id: Series[int] = pa.Field(
        isin=spadlconfig.actiontypes_df().type_id
        )
    type_name: Optional[Series[str]] = pa.Field(
        isin=spadlconfig.actiontypes_df().type_name
        )
    result_id: Series[int] = pa.Field(
        isin=spadlconfig.results_df().result_id
        )
    result_name: Optional[Series[str]] = pa.Field(
        isin=spadlconfig.results_df().result_name
        )
    away_team: Series[int] = pa.Field()
    freeze_frame_360: Series[Any] = pa.Field()
    visible_area_360: Series[Any] = pa.Field()

    class Config:  # noqa: D106
        strict = True
        coerce = True


class SPADLSchema(pa.SchemaModel):
    """Definition of a SPADL dataframe."""

    game_id: Series[Any] = pa.Field()
    original_event_id: Series[Any] = pa.Field(nullable=True)
    action_id: Series[int] = pa.Field()
    period_id: Series[int] = pa.Field(ge=1, le=5)
    time_seconds: Series[float] = pa.Field(ge=0)
    team_id: Series[Any] = pa.Field()
    player_id: Series[Any] = pa.Field()
    start_x: Series[float] = pa.Field(ge=0, le=spadlconfig.FIELD_LENGTH)
    start_y: Series[float] = pa.Field(ge=0, le=spadlconfig.FIELD_WIDTH)
    end_x: Series[float] = pa.Field(ge=0, le=spadlconfig.FIELD_LENGTH)
    end_y: Series[float] = pa.Field(ge=0, le=spadlconfig.FIELD_WIDTH)
    bodypart_id: Series[int] = pa.Field(isin=spadlconfig.bodyparts_df().bodypart_id)
    bodypart_name: Optional[Series[str]] = pa.Field(isin=spadlconfig.bodyparts_df().bodypart_name)
    type_id: Series[int] = pa.Field(isin=spadlconfig.actiontypes_df().type_id)
    type_name: Optional[Series[str]] = pa.Field(isin=spadlconfig.actiontypes_df().type_name)
    result_id: Series[int] = pa.Field(isin=spadlconfig.results_df().result_id)
    result_name: Optional[Series[str]] = pa.Field(isin=spadlconfig.results_df().result_name)

    class Config:  # noqa: D106
        strict = True
        coerce = True
