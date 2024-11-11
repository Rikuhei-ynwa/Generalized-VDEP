"""StatsBomb event stream data to SPADL converter."""

# Rikuhei Umemoto, 2024
# modifying the code https://github.com/ML-KULeuven/socceraction/blob/master/socceraction/spadl/statsbomb.py

import warnings
from typing import Tuple, Dict, Any, Optional, cast

import numpy as np
import numpy.typing as npt
import pandas as pd  # type: ignore
from pandera.typing import DataFrame

from . import config as spadlconfig
from .base import _add_dribbles, _fix_clearances
from .schema import SPADLSchema, SPADLSchema_360

Location = Tuple[float, float]

NUM_PLAYERS = 22
DIMENTION = 2

MIN2SEC = 60
HALF_MIN = 45
EXTRA_HALF_MIN = 15

ATK_ACT = ['Pass','Dribble','Carry','Shot','Miscontrol']
DFD_ACT = ['Interception','Tackle','Clearance','Foul Committed','Goal Keeper']


def convert_to_actions(
    events: pd.DataFrame,
    home_team_id: int,
    xy_fidelity_version: Optional[int] = None,
    shot_fidelity_version: Optional[int] = None,
) -> DataFrame[SPADLSchema]:
    """
    Convert StatsBomb events to SPADL actions.

    Parameters
    ----------
    events : pd.DataFrame
        DataFrame containing StatsBomb events from a single game.
    home_team_id : int
        ID of the home team in the corresponding game.
    xy_fidelity_version : int, optional
        Whether low or high fidelity coordinates are used in the event data.
        If not specified, the fidelity version is inferred from the data.
    shot_fidelity_version : int, optional
        Whether low or high fidelity coordinates are used in the event data
        for shots. If not specified, the fidelity version is inferred from the
        data.

    Returns
    -------
    actions : pd.DataFrame
        DataFrame with corresponding SPADL actions.

    """
    actions = pd.DataFrame()

    # Determine xy_fidelity_version and shot_fidelity_version
    infered_xy_fidelity_version, infered_shot_fidelity_version = _infer_xy_fidelity_versions(
        events
    )
    if xy_fidelity_version is None:
        xy_fidelity_version = infered_xy_fidelity_version
        warnings.warn(
            f'Inferred xy_fidelity_version={infered_xy_fidelity_version}.'
            + ' If this is incorrect, please specify the correct version'
            + ' using the xy_fidelity_version argument'
        )
    else:
        assert xy_fidelity_version in [1, 2], 'xy_fidelity_version must be 1 or 2'
    if shot_fidelity_version is None:
        if xy_fidelity_version == 2:
            shot_fidelity_version = 2
        else:
            shot_fidelity_version = infered_shot_fidelity_version
            warnings.warn(
                f'Inferred shot_fidelity_version={infered_shot_fidelity_version}.'
                + ' If this is incorrect, please specify the correct version'
                + ' using the shot_fidelity_version argument'
            )
    else:
        assert shot_fidelity_version in [1, 2], 'shot_fidelity_version must be 1 or 2'

    events = events.copy()
    events = _insert_interception_passes(events)
    events['extra'].fillna({}, inplace=True)

    actions['game_id'] = events.game_id
    actions['original_event_id'] = events.event_id
    actions['period_id'] = events.period_id
    actions['time_seconds'] = (
        pd.to_timedelta(events.timestamp).dt.total_seconds()
        )
    actions['team_id'] = events.team_id
    actions['player_id'] = events.player_id

    # split (end)location column into x and y columns
    end_location = events[
        ['location', 'extra']
        ].apply(_get_end_location, axis=1)
    # convert StatsBomb coordinates to spadl coordinates
    actions.loc[
        events.type_name == 'Shot', ['start_x', 'start_y']
        ] = _convert_ball_locations(
            events.loc[events.type_name == 'Shot', 'location'],
            shot_fidelity_version,
            )
    actions.loc[
        events.type_name != 'Shot', ['start_x', 'start_y']
        ] = _convert_ball_locations(
            events.loc[events.type_name != 'Shot', 'location'],
            shot_fidelity_version,
            )
    actions.loc[
        events.type_name == 'Shot', ['end_x', 'end_y']
        ] = _convert_ball_locations(
            end_location.loc[events.type_name == 'Shot'],
            shot_fidelity_version,
            )
    actions.loc[
        events.type_name != 'Shot', ['end_x', 'end_y']
        ] = _convert_ball_locations(
            end_location.loc[events.type_name != 'Shot'],
            shot_fidelity_version,
            )

    actions[
        ['type_id', 'result_id', 'bodypart_id']
        ] = (events[['type_name', 'extra']]
             .apply(_parse_event, axis=1, result_type='expand'))

    actions = (
        actions[
            actions.type_id != spadlconfig.actiontypes.index('non_action')
            ].sort_values(
                ['game_id', 'period_id', 'time_seconds'],
                kind='mergesort'
                ).reset_index(drop=True)
    )
    # actions = _fix_direction_of_play(actions, home_team_id)
    actions = _fix_clearances(actions)

    actions['action_id'] = range(len(actions))
    actions = _add_dribbles(actions)

    return cast(DataFrame[SPADLSchema], actions)


def convert_to_actions_360(
    events: pd.DataFrame,
    home_team_id: int,
    xy_fidelity_version: Optional[int] = None,
    shot_fidelity_version: Optional[int] = None,
) -> DataFrame[SPADLSchema]:
    """
    Convert StatsBomb events to SPADL actions with 360 data.

    Parameters
    ----------
    events : pd.DataFrame
        DataFrame containing StatsBomb events from a single game.
    home_team_id : int
        ID of the home team in the corresponding game.
    xy_fidelity_version : int, optional
        Whether low or high fidelity coordinates are used in the event data.
        If not specified, the fidelity version is inferred from the data.
    shot_fidelity_version : int, optional
        Whether low or high fidelity coordinates are used in the event data
        for shots. If not specified, the fidelity version is inferred from the
        data.

    Returns
    -------
    actions : pd.DataFrame
        DataFrame with corresponding SPADL actions.

    """
    actions = pd.DataFrame()

    # Determine xy_fidelity_version and shot_fidelity_version
    infered_xy_fidelity_version, infered_shot_fidelity_version = _infer_xy_fidelity_versions(
        events
    )
    if xy_fidelity_version is None:
        xy_fidelity_version = infered_xy_fidelity_version
        warnings.warn(
            f'Inferred xy_fidelity_version={infered_xy_fidelity_version}.'
            + ' If this is incorrect, please specify the correct version'
            + ' using the xy_fidelity_version argument'
        )
    else:
        assert xy_fidelity_version in [1, 2], 'xy_fidelity_version must be 1 or 2'
    if shot_fidelity_version is None:
        if xy_fidelity_version == 2:
            shot_fidelity_version = 2
        else:
            shot_fidelity_version = infered_shot_fidelity_version
            warnings.warn(
                f'Inferred shot_fidelity_version={infered_shot_fidelity_version}.'
                + ' If this is incorrect, please specify the correct version'
                + ' using the shot_fidelity_version argument'
            )
    else:
        assert (
            shot_fidelity_version in [1, 2], 
            'shot_fidelity_version must be 1 or 2'
            )

    events = events.copy()
    events = _insert_interception_passes(events)
    events['extra'].fillna({}, inplace=True)

    events = events.sort_values(["index"]).reset_index(drop=True)

    actions['game_id'] = events.game_id
    actions['original_event_id'] = events.event_id
    actions['period_id'] = events.period_id
    actions['time_seconds'] = (
        events.minute * MIN2SEC + events.second
        )
    actions['team_id'] = events.team_id
    actions['player_id'] = events.player_id
    # split (end)location column into x and y columns
    end_location = events[
        ['location', 'extra']
        ].apply(_get_end_location, axis=1)
    # convert StatsBomb coordinates to spadl coordinates
    shot_events = events[events.type_name == 'Shot'].index
    other_events = events[events.type_name != 'Shot'].index
    starts = ['start_x', 'start_y']
    ends = ['end_x', 'end_y']
    actions.loc[shot_events, starts] = _convert_ball_locations(
        events.loc[shot_events, 'location'],
        shot_fidelity_version,
    )
    actions.loc[other_events, starts] = _convert_ball_locations(
        events.loc[other_events, 'location'],
        shot_fidelity_version,
    )
    actions.loc[shot_events, ends] = _convert_ball_locations(
        end_location.loc[shot_events],
        shot_fidelity_version,
    )
    actions.loc[other_events, ends] = _convert_ball_locations(
        end_location.loc[other_events],
        shot_fidelity_version,
    )

    ARB = ['type_id', 'result_id', 'bodypart_id']
    actions[ARB] = events[
        ['type_name', 'extra']
        ].apply(_parse_event, axis=1, result_type='expand')
    actions["away_team"] = (
        events.team_id != home_team_id
        ).astype(int)
    
    # 360 data
    locations = pd.DataFrame(
        {"freeze_frame_360": [[] for _ in range(len(events))]}
        )
    visible_areas = pd.DataFrame(
        {"visible_area_360": [[] for _ in range(len(events))]}
    )
    for ev in range(len(events)):
        type_name = events.loc[ev, "type_name"]

        # freeze_frame_360
        location = _get_freeze_frame_360(
            events.loc[
                ev,
                ["freeze_frame_360", "type_name", "extra"]
                ]
        )
        if np.sum(~np.isnan(location)) == 0:
            locations.iat[ev, 0] = np.full(
                (NUM_PLAYERS * DIMENTION,), np.nan
            ).tolist()
        else:
            cie_ball = np.array(
                [actions.loc[ev, "start_x"],
                  actions.loc[ev, "start_y"]]
            )
            location = location.reshape((NUM_PLAYERS, DIMENTION))
            if type_name == "Shot":
                location = _convert_player_locations(
                    location, shot_fidelity_version
                    )
            else:
                location = _convert_player_locations(
                    location, xy_fidelity_version
                    )

            # Correct players' coordinates whether the dribble is
            # successful or unsuccessful.
            if type_name == "Dribble":
                if (events
                    .at[ev, "extra"]["dribble"]["outcome"]["name"]
                    == "Incomplete"):
                    location[:, 0] = (
                        spadlconfig.FIELD_LENGTH - location[:, 0]
                    )
                    location[:, 1] = (
                        spadlconfig.FIELD_WIDTH - location[:, 1]
                    )

            # Arrange players in order of distance to the ball.
            location_copy = location.copy()
            cie_atks = _arrange_players_location(
                location_copy[:11], cie_ball
                )
            cie_dfds = _arrange_players_location(
                location_copy[11:], cie_ball
                )
            location = np.concatenate([cie_atks, cie_dfds])
            
            locations.iat[ev, 0] = location.reshape(
                (NUM_PLAYERS * DIMENTION,)
                ).tolist()

        # visible_area_360
        if type(events.at[ev, "visible_area_360"]) == float:
            pass
        else:
            visible_area = events.at[ev, "visible_area_360"]
            visible_area = np.array(
                visible_area
                ).reshape((-1, DIMENTION))
            visible_area = _convert_visible_area(
                visible_area, xy_fidelity_version
                )
            if type_name == "Dribble":
                if (events
                    .at[ev, "extra"]["dribble"]["outcome"]["name"]
                    == "Incomplete"):
                    visible_area[:, 0] = (
                        spadlconfig.FIELD_LENGTH
                        - visible_area[:, 0]
                    )
                    visible_area[:, 1] = (
                        spadlconfig.FIELD_WIDTH
                        - visible_area[:, 1]
                    )

            visible_areas.iat[ev, 0] = visible_area.reshape(
                (visible_area.shape[0] * DIMENTION,)
                ).tolist()

    actions["freeze_frame_360"] = locations
    actions["visible_area_360"] = visible_areas

    not_non_actions = (
        actions.type_id != spadlconfig.actiontypes.index('non_action')
        )
    actions = (
        actions[not_non_actions]
        .sort_values(['game_id', 'period_id', 'time_seconds'], kind='mergesort')
        .reset_index(drop=True)
    )
    # actions = _fix_direction_of_play(actions, home_team_id)
    actions = _fix_clearances(actions)

    actions['action_id'] = range(len(actions))
    actions = _add_dribbles(actions)

    return cast(DataFrame[SPADLSchema_360], actions)


def _insert_interception_passes(df_events: pd.DataFrame) -> pd.DataFrame:
    """Insert interception actions before passes.

    This function converts passes that are also interceptions (type 64) in the
    StatsBomb event data into two separate events, first an interception and
    then a pass.

    Parameters
    ----------
    df_events : pd.DataFrame
        StatsBomb event dataframe

    Returns
    -------
    pd.DataFrame
        StatsBomb event dataframe in which passes that were also denoted as
        interceptions in the StatsBomb notation are transformed into two events.
    """
    is_interception_pass = (
        lambda x: (
            x
            .get("extra", {})
            .get("pass", {})
            .get("type", {})
            .get("name")
            == "Interception"
            )
    )
    df_events_interceptions = df_events[
        df_events.apply(is_interception_pass, axis=1)
        ].copy()

    if not df_events_interceptions.empty:
        df_events_interceptions["type_name"] = "Interception"
        df_events_interceptions["extra"] = [
            {"interception": 
             {"outcome": 
              {"id": 16, "name": "Success In Play"
               }}}
        ] * len(df_events_interceptions)

        df_events = pd.concat(
            [df_events_interceptions, df_events],
            ignore_index=True
            )
        df_events = df_events.sort_values(
            ["timestamp"], kind="mergesort"
            )
        df_events = df_events.reset_index(drop=True)

    return df_events


def _infer_xy_fidelity_versions(
        events: pd.DataFrame
        ) -> Tuple[int, int]:
    """Find out if x and y are integers disguised as floats."""
    mask_shot = events.type_name == "Shot"
    mask_other = events.type_name != "Shot"
    locations = events.location.apply(pd.Series)
    mask_valid_location = locations.notna().any(axis=1)
    high_fidelity_shots = (
        locations.loc[mask_valid_location & mask_shot] % 1 != 0
        ).any(axis=None)
    high_fidelity_other = (
        locations.loc[mask_valid_location & mask_other] % 1 != 0
        ).any(axis=None)
    xy_fidelity_version = (
        2 if high_fidelity_other else 1
        )
    shot_fidelity_version = (
        2 if high_fidelity_shots else xy_fidelity_version
        )
    return shot_fidelity_version, xy_fidelity_version


def _convert_x_cie(
        x_cie: np.ndarray, adjustment: float
        ) -> npt.NDArray[np.float32]:
    """Convert StatsBomb x-coordinate to SPADL x-coordinate."""
    x_cie = x_cie - adjustment
    return (
        x_cie 
        / spadlconfig.FIELD_LENGTH_STATSBOMB
        * spadlconfig.FIELD_LENGTH
        )


def _convert_y_cie(
        y_cie: Any, adjustment: float
        ) -> npt.NDArray[np.float32]:
    """Convert StatsBomb y-coordinate to SPADL y-coordinate."""
    y_cie = y_cie - adjustment
    return (
        spadlconfig.FIELD_WIDTH 
        - y_cie 
        / spadlconfig.FIELD_WIDTH_STATSBOMB
        * spadlconfig.FIELD_WIDTH
        )


def _convert_ball_locations(
        locations: pd.Series, fidelity_version: int
        ) -> npt.NDArray[np.float32]:
    """
    Convert StatsBomb locations to spadl coordinates.

    StatsBomb coordinates are cell-based, using a 120x80 grid, so 1,1 is the
    top-left square 'yard' of the field (in landscape), even though 0,0 is the
    true coordinate of the corner flag.

    Some matches have metadata like "xy_fidelity_version" : "2", which means
    the grid has higher granularity. In this case 0.1,0.1 is the top left
    cell.

    exempli gratia:
    [1, 120] x [1, 80]
    +-----+-----+
    | 1,1 |  ●  | <= a ball coordinate converted by this function
    +-----+-----◯ <= ball at 2,1 in StatsBomb data
    | 1,2 | 2,2 |
    +-----+-----+

    """
    cell_side = 0.1 if fidelity_version == 2 else 1.0
    cell_relative_center = cell_side / 2
    coordinates = np.empty((len(locations), DIMENTION), dtype=float)
    for i, loc in enumerate(locations):
        if isinstance(loc, list) and len(loc) == DIMENTION:
            coordinates[i, 0] = _convert_x_cie(
                loc[0], cell_relative_center
                )
            coordinates[i, 1] = _convert_y_cie(
                loc[1], cell_relative_center
                )
        elif isinstance(loc, list) and len(loc) == DIMENTION + 1:
            # A coordinate in the goal frame, only used for the end location of
            # Shot events. The y-coordinates and z-coordinates are always detailed
            # to a tenth of a yard.
            CELL_SHOT_SIDE = 0.05
            coordinates[i, 0] = _convert_x_cie(
                loc[0], cell_relative_center
                )
            coordinates[i, 1] = _convert_y_cie(
                loc[1], CELL_SHOT_SIDE
                )
    
    return coordinates


def _convert_player_locations(
        location: np.ndarray, fidelity_version: int
        ) -> npt.NDArray[np.float32]:
    """
    Convert StatsBomb locations to spadl coordinates.

    StatsBomb coordinates are cell-based, using a 120x80 grid, so 1,1 is the
    top-left square 'yard' of the field (in landscape), even though 0,0 is the
    true coordinate of the corner flag.

    Some matches have metadata like "xy_fidelity_version" : "2", which means
    the grid has higher granularity. In this case 0.1,0.1 is the top left
    cell.

    exempli gratia:
    [1, 120] x [1, 80]
    +-----+-----+
    | 1,1 |  ●  | <= a player coordinate converted by this function
    +-----+-----◯ <= player at 2,1 in StatsBomb data
    | 1,2 | 2,2 |
    +-----+-----+

    """
    cell_side = 0.1 if fidelity_version == 2 else 1.0
    cell_relative_center = cell_side / 2
    location[:, 0] = _convert_x_cie(
        location[:, 0], cell_relative_center
        )
    location[:, 1] = _convert_y_cie(
        location[:, 1], cell_relative_center
        )
    return location


def _convert_visible_area(
        visible_area: np.ndarray, fidelity_version: int
        ) -> npt.NDArray[np.float32]:
    """Convert StatsBomb visible area to SPADL coordinates."""
    cell_side = 0.1 if fidelity_version == 2 else 1.0
    cell_relative_center = cell_side / 2
    visible_area[:, 0] = _convert_x_cie(
        visible_area[:, 0], cell_relative_center
        )
    visible_area[:, 1] = _convert_y_cie(
        visible_area[:, 1], cell_relative_center
        )
    return visible_area


def _get_end_location(
        q: Tuple[Location, Dict[str, Any]]
        ) -> Location:
    start_location, extra = q
    for event in ['pass', 'shot', 'carry']:
        if event in extra and 'end_location' in extra[event]:
            return extra[event]['end_location']
    return start_location


def _arrange_players_location(
        players: np.ndarray, ball: np.ndarray
        ) -> npt.NDArray[np.float32]:
    """Arrange players in order of distance to the ball."""
    distances = np.linalg.norm(players - ball, axis=1)
    idx = np.argsort(distances)
    return players[idx]


def _parse_event(
        q: Tuple[str, Dict[str, Any]]
        ) -> Tuple[int, int, int]:
    t, x = q
    events = {
        'Pass': _parse_pass_event,
        'Dribble': _parse_dribble_event,
        'Carry': _parse_carry_event,
        'Foul Committed': _parse_foul_event,
        'Duel': _parse_duel_event,
        'Interception': _parse_interception_event,
        'Shot': _parse_shot_event,
        'Own Goal Against': _parse_own_goal_event,
        'Goal Keeper': _parse_goalkeeper_event,
        'Clearance': _parse_clearance_event,
        'Miscontrol': _parse_miscontrol_event,
    }
    parser = events.get(t, _parse_event_as_non_action)
    a, r, b = parser(x)
    actiontype = spadlconfig.actiontypes.index(a)
    result = spadlconfig.results.index(r)
    bodypart = spadlconfig.bodyparts.index(b)
    return actiontype, result, bodypart


def _parse_event_as_non_action(
        _extra: Dict[str, Any]
        ) -> Tuple[str, str, str]:
    a = 'non_action'
    r = 'success'
    b = 'foot'
    return a, r, b


def _parse_pass_event(
        extra: Dict[str, Any]
        ) -> Tuple[str, str, str]:  # noqa: C901
    a = 'pass'  # default
    b = 'foot'  # default
    p = extra.get('pass', {})
    ptype = p.get('type', {}).get('name')
    height = p.get('height', {}).get('name')
    cross = p.get('cross')
    if ptype == 'Free Kick':
        if height == 'High Pass' or cross:
            a = 'freekick_crossed'
        else:
            a = 'freekick_short'
    elif ptype == 'Corner':
        if height == 'High Pass' or cross:
            a = 'corner_crossed'
        else:
            a = 'corner_short'
    elif ptype == 'Goal Kick':
        a = 'goalkick'
    elif ptype == 'Throw-in':
        a = 'throw_in'
        b = 'other'
    elif cross:
        a = 'cross'
    else:
        a = 'pass'

    pass_outcome = (
        extra
        .get('pass', {})
        .get('outcome', {})
        .get('name')
        )
    if pass_outcome in ['Incomplete', 'Out']:
        r = 'fail'
    elif pass_outcome == 'Pass Offside':
        r = 'offside'
    elif pass_outcome in ['Injury Clearance', 'Unknown']:
        # discard passes that are not part of the play
        a = 'non_action'
        r = 'success'
    else:
        r = 'success'

    bp = extra.get('pass', {}).get('body_part', {}).get('name')
    if bp is not None:
        if 'Head' in bp:
            b = 'head'
        elif bp == 'Left Foot':
            b = 'foot_left'
        elif bp == 'Right Foot':
            b = 'foot_right'
        elif 'Foot' in bp or bp == 'Drop Kick':
            b = 'foot'
        else:
            b = 'other'

    return a, r, b


def _parse_dribble_event(
        extra: Dict[str, Any]
        ) -> Tuple[str, str, str]:
    a = 'take_on'

    dribble_outcome = (
        extra
        .get('dribble', {})
        .get('outcome', {})
        .get('name')
        )
    if dribble_outcome == 'Incomplete':
        r = 'fail'
    elif dribble_outcome == 'Complete':
        r = 'success'
    else:
        r = 'success'

    b = 'foot'

    return a, r, b


def _parse_carry_event(
        _extra: Dict[str, Any]
        ) -> Tuple[str, str, str]:
    a = 'dribble'
    r = 'success'
    b = 'foot'
    return a, r, b


def _parse_foul_event(
        extra: Dict[str, Any]
        ) -> Tuple[str, str, str]:
    a = 'defensive_foul' # default
    r = 'fail' # default
    if (
        extra.get('foul_committed', {}).get('offensive', {})
        == True
        ):
        a = 'offensive_foul'
    else:
        a = 'defensive_foul'

    foul_card = (
        extra
        .get('foul_committed', {})
        .get('card', {})
        .get('name', '')
        )
    if 'Yellow' in foul_card:
        r = 'yellow_card'
    elif 'Red' in foul_card:
        r = 'red_card'
    else:
        r = 'fail'

    b = 'foot'

    return a, r, b


def _parse_duel_event(
        extra: Dict[str, Any]
        ) -> Tuple[str, str, str]:
    if (extra
        .get('duel', {})
        .get('type', {})
        .get('name')
        == 'Tackle'):
        a = 'tackle'
        duel_outcome = (
            extra
            .get('duel', {})
            .get('outcome', {})
            .get('name')
            )
        if duel_outcome in ['Lost In Play', 'Lost Out']:
            r = 'fail'
        elif duel_outcome in ['Success in Play', 'Won']:
            r = 'success'
        else:
            r = 'success'

        b = 'foot'
        return a, r, b
    return _parse_event_as_non_action(extra)


def _parse_interception_event(
        extra: Dict[str, Any]
        ) -> Tuple[str, str, str]:
    a = 'interception'
    interception_outcome = (
        extra
        .get('interception', {})
        .get('outcome', {})
        .get('name')
        )
    if interception_outcome in ['Lost In Play', 'Lost Out']:
        r = 'fail'
    elif interception_outcome == 'Won':
        r = 'success'
    else:
        r = 'success'
    b = 'foot'
    return a, r, b


def _parse_shot_event(
        extra: Dict[str, Any]
        ) -> Tuple[str, str, str]:
    extra_type = extra.get('shot', {}).get('type', {}).get('name')
    if extra_type == 'Free Kick':
        a = 'shot_freekick'
    elif extra_type == 'Penalty':
        a = 'shot_penalty'
    else:
        a = 'shot'

    shot_outcome = (
        extra
        .get('shot', {})
        .get('outcome', {})
        .get('name')
        )
    if shot_outcome == 'Goal':
        r = 'success'
    elif (shot_outcome in [
        'Blocked', 'Off T', 'Post', 'Saved', 'Wayward'
        ]):
        r = 'fail'
    else:
        r = 'fail'

    bp = extra.get('shot', {}).get('body_part', {}).get('name')
    if bp is None:
        b = 'foot'
    elif 'Head' in bp:
        b = 'head'
    elif bp == 'Left Foot':
        b = 'foot_left'
    elif bp == 'Right Foot':
        b = 'foot_right'
    elif 'Foot' in bp:
        b = 'foot'
    else:
        b = 'other'

    return a, r, b


def _parse_own_goal_event(
        _extra: Dict[str, Any]
        ) -> Tuple[str, str, str]:
    a = 'bad_touch'
    r = 'owngoal'
    b = 'foot'
    return a, r, b


def _parse_goalkeeper_event(
        extra: Dict[str, Any]
        ) -> Tuple[str, str, str]:  # noqa: C901
    extra_type = (
        extra
        .get('goalkeeper', {})
        .get('type', {})
        .get('name')
        )
    if extra_type == 'Shot Saved':
        a = 'keeper_save'
    elif extra_type in ('Collected', 'Keeper Sweeper'):
        a = 'keeper_claim'
    elif extra_type == 'Punch':
        a = 'keeper_punch'
    else:
        a = 'non_action'

    goalkeeper_outcome = (
        extra
        .get('goalkeeper', {})
        .get('outcome', {})
        .get('name', 'x')
        )
    if goalkeeper_outcome in [
        'Claim',
        'Clear',
        'Collected Twice',
        'In Play Safe',
        'Success',
        'Touched Out',
    ]:
        r = 'success'
    elif goalkeeper_outcome in ['In Play Danger', 'No Touch']:
        r = 'fail'
    else:
        r = 'success'

    bp = (extra
          .get('goalkeeper', {})
          .get('body_part', {})
          .get('name'))
    if bp is None:
        b = 'other'
    elif 'Head' in bp:
        b = 'head'
    elif bp == 'Left Foot':
        b = 'foot_left'
    elif bp == 'Right Foot':
        b = 'foot_right'
    elif 'Foot' in bp or bp == 'Drop Kick':
        b = 'foot'
    else:
        b = 'other'

    return a, r, b


def _parse_clearance_event(
        extra: Dict[str, Any]
        ) -> Tuple[str, str, str]:
    a = 'clearance'
    r = 'success'
    bp = (extra
          .get('clearance', {})
          .get('body_part', {})
          .get('name'))
    if bp is None:
        b = 'foot'
    elif 'Head' in bp:
        b = 'head'
    elif bp == 'Left Foot':
        b = 'foot_left'
    elif bp == 'Right Foot':
        b = 'foot_right'
    elif 'Foot' in bp:
        b = 'foot'
    else:
        b = 'other'
    return a, r, b


def _parse_miscontrol_event(
        _extra: Dict[str, Any]
        ) -> Tuple[str, str, str]:
    a = 'bad_touch'
    r = 'fail'
    b = 'foot'
    return a, r, b


def _get_freeze_frame_360(
        q: Tuple[Any, str, Dict[str, Any]]
        ) -> np.ndarray:
    """
    Parameters
    q: Tuple[freeze_frame_360, type_name, extra]

    Returns
    locations: np.ndarray
        Locations of players in the freeze frame.
    """
    freeze_frame_360, type_name, extra = q
    locations = np.full((NUM_PLAYERS, DIMENTION), np.nan)
    if type(freeze_frame_360) == float:
        # This means that 360 data is NaN.
        pass 
    elif "location" in freeze_frame_360[0]:
        idx_atk = 0
        idx_dfd = 11

        if (type_name == 'Duel'
            and (extra
                 .get("duel", {})
                 .get("type", {})
                 .get("name") == "Tackle")):
            type_name = "Tackle"

        attacking = type_name in ATK_ACT
        defending = type_name in DFD_ACT
        if attacking:
            for pl in range(len(freeze_frame_360)):
                if freeze_frame_360[pl]["teammate"]:
                    locations[idx_atk] = (
                        freeze_frame_360[pl]["location"]
                        )
                    idx_atk += 1
                else:
                    locations[idx_dfd] = (
                        freeze_frame_360[pl]["location"]
                        )
                    idx_dfd += 1

        elif defending:
            for pl in range(len(freeze_frame_360)):
                if freeze_frame_360[pl]["teammate"]:
                    locations[idx_dfd] = (
                        freeze_frame_360[pl]["location"]
                        )
                    idx_dfd += 1
                else:
                    locations[idx_atk] = (
                        freeze_frame_360[pl]["location"]
                        )
                    idx_atk += 1

    return locations