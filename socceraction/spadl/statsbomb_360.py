import json
import os
import pdb
from typing import Dict, List, Tuple

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import requests

import socceraction.spadl.config as spadlconfig

_free_open_data: str = (
    "https://raw.githubusercontent.com/statsbomb/open-data/master/data/"
)

MIN2SEC: int = 60
HALF2SEC: int = 45 * MIN2SEC

MIN_DRIBBLE_LENGTH: float = 3.0
MAX_DRIBBLE_LENGTH: float = 60.0
MAX_DRIBBLE_DURATION: float = 10.0


def _remoteloadjson(path: str) -> List[Dict]:
    return requests.get(path).json()


def _localloadjson(path: str) -> List[Dict]:
    with open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


class StatsBombLoader:
    """
    Load Statsbomb data either from a remote location
    (e.g., "https://raw.githubusercontent.com/statsbomb/open-data/master/data/")
    or from a local folder.

    This is a temporary class until statsbombpy* becomes compatible with socceraction
    https://github.com/statsbomb/statsbombpy
    """

    def __init__(self, root: str = _free_open_data, getter: str = "remote"):
        """
        Initalize the StatsBombLoader

        :param root: root-path of the data
        :param getter: "remote" or "local"
        """
        self.root = root

        if getter == "remote":
            self.get = _remoteloadjson
        elif getter == "local":
            self.get = _localloadjson
        else:
            raise Exception("invalid getter specified")

    def competitions(self) -> pd.DataFrame:
        return pd.DataFrame(self.get(os.path.join(self.root, "competitions.json")))

    def matches(self, competition_id: int, season_id: int) -> pd.DataFrame:
        path = os.path.join(self.root, f"matches/{competition_id}/{season_id}.json")
        return pd.DataFrame(_flatten(m) for m in self.get(path))

    def _lineups(self, match_id: int) -> List[Dict]:
        path = os.path.join(self.root, f"lineups/{match_id}.json")
        return self.get(path)

    def teams(self, match_id: int) -> pd.DataFrame:
        return pd.DataFrame(self._lineups(match_id))[["team_id", "team_name"]]

    def players(self, match_id: int) -> pd.DataFrame:
        return pd.DataFrame(
            _flatten_id(p)
            for lineup in self._lineups(match_id)
            for p in lineup["lineup"]
        )

    def events(self, match_id: int):
        eventsdf = pd.DataFrame(
            _flatten_id(e)
            for e in self.get(os.path.join(self.root, f"events/{match_id}.json"))
        )
        eventsdf["match_id"] = match_id
        return eventsdf


def _flatten_id(d: dict) -> dict:
    newd = {}
    extra = {}
    for k, v in d.items():
        if isinstance(v, dict):
            if len(v) == 2 and "id" in v and "name" in v:
                newd[k + "_id"] = v["id"]
                newd[k + "_name"] = v["name"]
            else:
                extra[k] = v
        else:
            newd[k] = v
    newd["extra"] = extra
    return newd


def _flatten(d: dict) -> dict:
    newd: dict = {}
    for k, v in d.items():
        if isinstance(v, dict):
            newd = {**newd, **_flatten(v)}
        else:
            newd[k] = v
    return newd


def extract_player_games(events: pd.DataFrame) -> pd.DataFrame:
    """
    Extract player games [player_id,game_id,minutes_played] from statsbomb match events
    """
    game_minutes = max(events[events.type_name == "Half End"].minute)

    game_id = events.match_id.mode().values[0]
    players = {}
    for startxi in events[events.type_name == "Starting XI"].itertuples():
        team_id, team_name = startxi.team_id, startxi.team_name
        for player in startxi.extra["tactics"]["lineup"]:
            player = _flatten_id(player)
            player = {
                **player,
                **{
                    "game_id": game_id,
                    "team_id": team_id,
                    "team_name": team_name,
                    "minutes_played": game_minutes,
                },
            }
            players[player["player_id"]] = player
    for substitution in events[events.type_name == "Substitution"].itertuples():
        replacement = substitution.extra["substitution"]["replacement"]
        replacement = {
            "player_id": replacement["id"],
            "player_name": replacement["name"],
            "minutes_played": game_minutes - substitution.minute,
            "team_id": substitution.team_id,
            "game_id": game_id,
            "team_name": substitution.team_name,
        }
        players[replacement["player_id"]] = replacement
        # minutes_played = substitution.minute
        players[substitution.player_id]["minutes_played"] = substitution.minute
    pg = pd.DataFrame(players.values()).fillna(0)
    for col in pg.columns:
        if "_id" in col:
            pg[col] = pg[col].astype(int)
    return pg


def transform_coordinates(loc: np.array, dim: str):
    CELL_RELATIVE_CENTER = 0.5
    loc = loc - CELL_RELATIVE_CENTER
    if dim == "x":
        return (
            (loc / spadlconfig.field_length_statsbomb) * spadlconfig.field_length
            )
    elif dim == "y":
        return (
            ((spadlconfig.field_width_statsbomb - loc) / spadlconfig.field_width_statsbomb)
              * spadlconfig.field_width
        )


def euclidean_distance(point1, point2) -> float:
    return np.sqrt(np.sum((point1 - point2) ** 2))


def _check_coordinates(loc_pl, loc_ball, type_name):
    """
    The coordinates of players in StatsBomb are team_name attacking 
    from left to right. However, sometimes, this is not the case due to 
    errors in data acquisition. Fortunately, the coordinates of the ball 
    are the same as if team_name were attacking from left to right, 
    so the ones of the players are corrected based on this.
    Also, the coordinates should not be modified when type_name is 'Dribble'.

    Parameters
    ----------
    loc_pl: np.ndarray, shape=(22, 2)
    loc_ball: np.ndarray, shape=(2,)
    type_name: str
    
    Returns
    -------
    need_correction: float
    if need_correction == 0, no correction is needed.
    if need_correction == 0.5, in other words the event was 'Dribble', 
    the correction about players' coordinates is needed.
    if need_correction == 1, the correction is needed.
    """

    THRESHOLD = 3.0
    dist = [
        euclidean_distance(loc_ball, cie_pl) for cie_pl in loc_pl
        ]
    if type_name == 'Dribble':
        return 0.5
    elif min(dist) <= THRESHOLD:
        return 0
    else:
        return 1


def _modify_coordinates(loc_pl, need_correction):
    """
    Parameters
    ----------
    loc_pl: np.ndarray, shape=(22, 2)
    need_correction: float, 0, 0.5 or 1

    Returns
    -------
    loc_pl: np.ndarray, shape=(22, 2)
    """
    if need_correction == 0:
        return loc_pl
    elif need_correction == 0.5 or need_correction == 1:
        loc_pl[:,0] = spadlconfig.field_length - loc_pl[:,0]
        loc_pl[:,1] = spadlconfig.field_width - loc_pl[:,1]

    return loc_pl


def _modify_polygon(visible_area, need_correction):
    """
    Parameters
    ----------
    visible_area: np.ndarray, shape=(n, 2)
    need_correction: float, 0, 0.5 or 1

    Returns
    -------
    visible_area: np.ndarray, shape=(n, 2)
    """
    if need_correction == 0 or need_correction == 0.5:
        return visible_area
    elif need_correction == 1:
        visible_area[:,0] = spadlconfig.field_length - visible_area[:,0]
        visible_area[:,1] = spadlconfig.field_width - visible_area[:,1]

    return visible_area


def _arrange_players(loc_pl, loc_ball):
    """
    Parameters
    ----------
    loc_pl: np.ndarray, shape=(22, 2)
    loc_ball: np.ndarray, shape=(2,)
    team_type: str, 'attack' or 'defend'

    Returns
    -------
    loc_pl: np.ndarray, shape=(22, 2)    
    """
    nans = np.where(np.isnan(loc_pl).any(axis=1))[0]
    nonnans = np.where(~np.isnan(loc_pl).any(axis=1))[0]
    dist = [
        euclidean_distance(loc_ball, cie_pl) for cie_pl in loc_pl[nonnans]
        ]
    idx_nearest = np.concatenate([np.argsort(dist), nans])

    return loc_pl[idx_nearest]


def _get_freeze_frame_360(q):
    """
    Parameters
    q: Tuple[freeze_frame_360, type_name, extra]

    Returns
    locations: np.ndarray
    """
    freeze_frame_360, type_name, extra = q
    locations = np.zeros((44,))
    locations[:] = np.nan
    if type(freeze_frame_360) == int or freeze_frame_360[0]["teammate"] is None:
        return locations
    elif "location" in freeze_frame_360[0]:
        """
        0 ~ 9: attacking players
        10: attacking goalkeeper
        11 ~ 20: defending players
        21: defending goalkeeper
        """
        ATK = 0
        ATK_GK = 10
        DFD = 11
        DFD_GK = 21
        DIMENTION = 2
        if type_name == 'Foul Committed':
            return locations
        elif type_name in [
            'Carry','Dribble','Miscontrol','Pass','Shot',
            ]:
            for pl in range(len(freeze_frame_360)):
                if freeze_frame_360[pl]["teammate"]:
                    if freeze_frame_360[pl]["keeper"]:
                        locations[
                            ATK_GK*DIMENTION : (ATK_GK+1)*DIMENTION
                            ] = freeze_frame_360[pl]["location"]
                    else:
                        locations[
                            ATK*DIMENTION : (ATK+1)*DIMENTION
                            ] = freeze_frame_360[pl]["location"]
                        ATK += 1
                else:
                    if freeze_frame_360[pl]["keeper"]:
                        locations[
                            DFD_GK*DIMENTION : (DFD_GK+1)*DIMENTION
                            ] = freeze_frame_360[pl]["location"]
                    else:
                        locations[
                            DFD*DIMENTION : (DFD+1)*DIMENTION
                            ] = freeze_frame_360[pl]["location"]
                        DFD += 1

        elif (
            type_name in [
            'Clearance','Goal Keeper','Interception','Own Goal Against',
            ] or 
            (type_name == 'Duel' 
                and 
                extra.get("duel", {}).get("type", {}).get("name") == "Tackle")
            ):
            for pl in range(len(freeze_frame_360)):
                import pdb; pdb.set_trace()
                if freeze_frame_360[pl]["teammate"]:
                    if freeze_frame_360[pl]["keeper"]:
                        locations[
                            DFD_GK*DIMENTION : (DFD_GK+1)*DIMENTION
                            ] = freeze_frame_360[pl]["location"]
                    else:
                        locations[
                            DFD*DIMENTION : (DFD+1)*DIMENTION
                            ] = freeze_frame_360[pl]["location"]
                        DFD += 1
                else:
                    if freeze_frame_360[pl]["keeper"]:
                        locations[
                            ATK_GK*DIMENTION : (ATK_GK+1)*DIMENTION
                            ] = freeze_frame_360[pl]["location"]
                    else:
                        locations[
                            ATK*DIMENTION : (ATK+1)*DIMENTION
                            ] = freeze_frame_360[pl]["location"]
                        ATK += 1

        return locations
    pdb.set_trace()
    return freeze_frame_360


# Location = Tuple[float, float]
def _get_end_location(q: Tuple[list, str, dict]) -> list:
    start_location, extra = q
    for event in ["pass", "shot", "carry"]:
        if event in extra and "end_location" in extra[event]:
            return extra[event]["end_location"]
    return start_location


def _get_type_id(q: Tuple[str, dict]) -> int:
    t, extra = q
    a = "non_action"
    if t == "Pass":
        a = "pass"  # default
        p = extra.get("pass", {})
        ptype = p.get("type", {}).get("name")
        height = p.get("height", {}).get("name")
        cross = p.get("cross")
        if ptype == "Free Kick":
            if height == "High Pass" or cross:
                a = "freekick_crossed"
            else:
                a = "freekick_short"
        elif ptype == "Corner":
            if height == "High Pass" or cross:
                a = "corner_crossed"
            else:
                a = "corner_short"
        elif ptype == "Goal Kick":
            a = "goalkick"
        elif ptype == "Throw-in":
            a = "throw_in"
        elif cross:
            a = "cross"
        else:
            a = "pass"
    elif t == "Dribble":
        a = "take_on"
    elif t == "Carry":
        a = "dribble"
    elif t == "Foul Committed":
        a = "foul"
    elif t == "Duel" and extra.get("duel", {}).get("type", {}).get("name") == "Tackle":
        a = "tackle"
    elif t == "Interception":
        a = "interception"
    elif t == "Shot":
        extra_type = extra.get("shot", {}).get("type", {}).get("name")
        if extra_type == "Free Kick":
            a = "shot_freekick"
        elif extra_type == "Penalty":
            a = "shot_penalty"
        else:
            a = "shot"
    elif t == "Own Goal Against":
        a = "shot"
    elif t == "Goal Keeper":
        extra_type = extra.get("goalkeeper", {}).get("type", {}).get("name")
        if extra_type == "Shot Saved":
            a = "keeper_save"
        elif extra_type == "Collected" or extra_type == "Keeper Sweeper":
            a = "keeper_claim"
        elif extra_type == "Punch":
            a = "keeper_punch"
        else:
            a = "non_action"
    elif t == "Clearance":
        a = "clearance"
    elif t == "Miscontrol":
        a = "bad_touch"
    else:
        a = "non_action"
    return spadlconfig.actiontypes.index(a)


def _get_result_id(q: Tuple[str, dict]) -> int:
    t, x = q

    if t == "Pass":
        pass_outcome = x.get("pass", {}).get("outcome", {}).get("name")
        if pass_outcome in ["Incomplete", "Out"]:
            r = "fail"
        elif pass_outcome == "Pass Offside":
            r = "offside"
        else:
            r = "success"
    elif t == "Shot":
        shot_outcome = x.get("shot", {}).get("outcome", {}).get("name")
        if shot_outcome == "Goal":
            r = "success"
        elif shot_outcome in ["Blocked", "Off T", "Post", "Saved", "Wayward"]:
            r = "fail"
        else:
            r = "fail"
    elif t == "Dribble":
        dribble_outcome = x.get("dribble", {}).get("outcome", {}).get("name")
        if dribble_outcome == "Incomplete":
            r = "fail"
        elif dribble_outcome == "Complete":
            r = "success"
        else:
            r = "success"
    elif t == "Foul Committed":
        foul_card = x.get("foul_committed", {}).get("card", {}).get("name", "")
        if "Yellow" in foul_card:
            r = "yellow_card"
        elif "Red" in foul_card:
            r = "red_card"
        else:
            r = "success"
    elif t == "Duel":
        duel_outcome = x.get("duel", {}).get("outcome", {}).get("name")
        if duel_outcome in ["Lost in Play", "Lost Out"]:
            r = "fail"
        elif duel_outcome in ["Success in Play", "Won"]:
            r = "success"
        else:
            r = "success"
    elif t == "Interception":
        interception_outcome = x.get("interception", {}).get("outcome", {}).get("name")
        if interception_outcome in ["Lost In Play", "Lost Out"]:
            r = "fail"
        elif interception_outcome == "Won":
            r = "success"
        else:
            r = "success"
    elif t == "Own Goal Against":
        r = "owngoal"
    elif t == "Goal Keeper":
        goalkeeper_outcome = x.get("goalkeeper", {}).get("outcome", {}).get("name", "x")
        if goalkeeper_outcome in [
            "Claim",
            "Clear",
            "Collected Twice",
            "In Play Safe",
            "Success",
            "Touched Out",
        ]:
            r = "success"
        elif goalkeeper_outcome in ["In Play Danger", "No Touch"]:
            r = "fail"
        else:
            r = "success"
    elif t == "Clearance":
        r = "success"
    elif t == "Miscontrol":
        r = "fail"
    else:
        r = "success"

    return spadlconfig.results.index(r)


def _get_bodypart_id(q: Tuple[str, dict]) -> int:
    t, x = q
    if t == "Shot":
        bp = x.get("shot", {}).get("body_part", {}).get("name")
    elif t == "Pass":
        bp = x.get("pass", {}).get("body_part", {}).get("name")
    elif t == "Goal Keeper":
        bp = x.get("goalkeeper", {}).get("body_part", {}).get("name")
    else:
        bp = None

    if bp is None:
        b = "foot"
    elif "Head" in bp:
        b = "head"
    elif "Foot" in bp or bp == "Drop Kick":
        b = "foot"
    else:
        b = "other"

    return spadlconfig.bodyparts.index(b)


def _fix_clearances(actions: pd.DataFrame) -> pd.DataFrame:
    next_actions = actions.shift(-1)
    next_actions[-1:] = actions[-1:]
    clearance_idx = actions.type_id == spadlconfig.actiontypes.index("clearance")
    actions.loc[clearance_idx, "end_x"] = next_actions[clearance_idx].start_x.values
    actions.loc[clearance_idx, "end_y"] = next_actions[clearance_idx].start_y.values

    return actions


def _fix_direction_of_play(actions: pd.DataFrame) -> pd.DataFrame:
    # The type_id of "take_on" is 7.
    take_on_idx = actions[actions.type_id == 7].index.values
    for idx in take_on_idx:
        players_x = np.asarray(actions.at[idx, "freeze_frame_360"][::2])
        players_y = np.asarray(actions.at[idx, "freeze_frame_360"][1::2])
        actions.at[idx, "freeze_frame_360"][::2] = (
            spadlconfig.field_length - players_x
        ).tolist()
        actions.at[idx, "freeze_frame_360"][1::2] = (
            spadlconfig.field_width - players_y
        ).tolist()
    return actions


def _add_dribbles(actions: pd.DataFrame) -> pd.DataFrame:
    next_actions = actions.shift(-1)

    same_team = actions.team_id == next_actions.team_id
    # not_clearance = actions.type_id != actiontypes.index("clearance")

    dx = actions.end_x - next_actions.start_x
    dy = actions.end_y - next_actions.start_y
    far_enough = dx**2 + dy**2 >= MIN_DRIBBLE_LENGTH**2
    not_too_far = dx**2 + dy**2 <= MAX_DRIBBLE_LENGTH**2

    dt = next_actions.time_seconds - actions.time_seconds
    same_phase = dt < MAX_DRIBBLE_DURATION

    dribble_idx = same_team & far_enough & not_too_far & same_phase

    dribbles = pd.DataFrame()
    prev = actions[dribble_idx]
    nex = next_actions[dribble_idx]
    dribbles["game_id"] = nex.game_id
    dribbles["period_id"] = nex.period_id
    dribbles["action_id"] = prev.action_id + 0.1
    dribbles["time_seconds"] = (prev.time_seconds + nex.time_seconds) / 2
    dribbles["team_id"] = nex.team_id
    dribbles["player_id"] = nex.player_id
    dribbles["start_x"] = prev.end_x
    dribbles["start_y"] = prev.end_y
    dribbles["end_x"] = nex.start_x
    dribbles["end_y"] = nex.start_y
    dribbles["bodypart_id"] = spadlconfig.bodyparts.index("foot")
    dribbles["type_id"] = spadlconfig.actiontypes.index("dribble")
    dribbles["result_id"] = spadlconfig.results.index("success")
    # added
    dribbles["away_team"] = prev.away_team
    dribbles["freeze_frame_360"] = prev.freeze_frame_360
    dribbles["visible_area_360"] = prev.visible_area_360

    actions = pd.concat([actions, dribbles], ignore_index=True, sort=False)
    actions = actions.sort_values(["game_id", "period_id", "action_id"]).reset_index(
        drop=True
    )
    actions["action_id"] = range(len(actions))
    return actions


def convert_to_actions_360(events: pd.DataFrame, home_team_id: int) -> pd.DataFrame:
    """
    Convert StatsBomb events to SPADL actions + 360

    Parameters
    ----------
    events: pd.DataFrame, StatsBomb's format.
    home_team_id: int,
    
    Returns
    -------
    actions: pd.DataFrame, SPADL's format.
    """

    actions = pd.DataFrame()

    events["extra"] = events["extra"].fillna({})
    events = events.fillna(0)

    # if events.at[0, "game_id"] == 3788741:

    #     from matplotlib import pyplot as plt
    #     from matplotlib.animation import FuncAnimation
    #     from mplsoccer import Pitch
    #     pitch = Pitch(pitch_type='statsbomb', pitch_length=120, pitch_width=80)
    #     fig, ax = pitch.draw(figsize=(16, 9))

    #     def anim_statsbomb(idx):
    #         ax.cla()
    #         pitch.draw(ax=ax)
    #         if events.iloc[idx+1000].visible_area_360 == 0:
    #             pass
    #         else:
    #             visible_area = np.array(events.iloc[idx+1000].visible_area_360).reshape(-1, 2)
    #             pitch.polygon([visible_area], color=(1, 0, 0, 0.3), ax=ax)

    #             player_position_data = events.iloc[idx+1000].freeze_frame_360

    #             pitch.scatter(
    #                 events.iloc[idx+1000].location[0], events.iloc[idx+1000].location[1], c='white', s=240, ec='k', ax=ax
    #             )

    #             for player in player_position_data:
    #                 if player["teammate"]:
    #                     pitch.scatter(player["location"][0], player["location"][1], c='orange', s=80, ec='k', ax=ax)
    #                 else:
    #                     pitch.scatter(player["location"][0], player["location"][1], c='dodgerblue', s=80, ec='k', ax=ax)

    #         type_name = events.iloc[idx+1000].type_name
    #         if type_name == 'Duel':
    #             type_name = events.iloc[idx+1000].extra['duel']['type']['name']
            
    #         ax.text(
    #             1.0, 1.5, 
    #             f"Frame: {idx+1000}, Team: {events.iloc[idx+1000].team_name}, Type: {type_name} Player: {events.iloc[idx+1000].player_name}", 
    #             fontsize=12
    #             )

    #     anim = FuncAnimation(fig, anim_statsbomb, frames=1000, interval=200)
    #     os.makedirs("/home/r_umemoto/workspace5/work/Generalized-VDEP/anims", exist_ok=True)
    #     anim.save("/home/r_umemoto/workspace5/work/Generalized-VDEP/anims" + "/sample.mp4", writer="ffmpeg")

    #     import pdb; pdb.set_trace()

    actions["game_id"] = events.game_id  # match_id
    actions["period_id"] = events.period_id  # period

    actions["time_seconds"] = events.minute * MIN2SEC + events.second
    actions["idx_original"] = events.index
    actions["team_id"] = events.team_id
    actions["player_id"] = events.player_id

    actions["start_x"] = events.location.apply(lambda x: x[0] if x else -10)
    actions["start_y"] = events.location.apply(lambda x: x[1] if x else -10)
    actions["start_x"] = transform_coordinates(actions["start_x"].to_numpy(), "x")
    actions["start_y"] = transform_coordinates(actions["start_y"].to_numpy(), "y")

    end_location = events[["location", "extra"]].apply(_get_end_location, axis=1)
    actions["end_x"] = end_location.apply(lambda x: x[0] if x else -10)
    actions["end_y"] = end_location.apply(lambda x: x[1] if x else -10)
    actions["end_x"] = transform_coordinates(actions["end_x"].to_numpy(), "x")
    actions["end_y"] = transform_coordinates(actions["end_y"].to_numpy(), "y")

    actions["type_id"] = events[["type_name", "extra"]].apply(_get_type_id, axis=1)
    actions["result_id"] = events[["type_name", "extra"]].apply(_get_result_id, axis=1)
    actions["bodypart_id"] = events[["type_name", "extra"]].apply(_get_bodypart_id, axis=1)
    actions["away_team"] = (events.team_id != home_team_id).astype(int)
    if actions["away_team"].isna().sum() > 0:
        actions["away_team"] = actions["away_team"].replace(np.nan, False)

    # 360 data
    locations = pd.DataFrame(
        {"freeze_frame_360": [[] for _ in range(len(events))]}
        )
    visible_areas = pd.DataFrame(
        {"visible_area_360": [[] for _ in range(len(events))]}
    )
    for ev in range(len(events)):
        # freeze_frame_360
        location = _get_freeze_frame_360(
            events.loc[ev, ["freeze_frame_360", "type_name", "extra"]]
        )
        cie_ball = np.array(
                [actions.loc[ev, "start_x"], actions.loc[ev, "start_y"]]
            )
        type_name = events.loc[ev, "type_name"]
        if np.sum(~np.isnan(location)) > 0:
            location = location.reshape((22, 2))
            location[:,0] = transform_coordinates(location[:,0], "x")
            location[:,1] = transform_coordinates(location[:,1], "y")
            need_correction = _check_coordinates(location, cie_ball, type_name)
            location = _modify_coordinates(location, need_correction)

            """
            Arrange players in order of distance to the ball.
            """
            location_copy = location.copy()
            cie_atks = _arrange_players(location_copy[:11], cie_ball)
            cie_dfds = _arrange_players(location_copy[11:], cie_ball)
            location = np.concatenate([cie_atks, cie_dfds])

            # visible_area_360
            if events.at[ev, "visible_area_360"] == 0:
                pass
            else:
                visible_area = events.at[ev, "visible_area_360"]
                visible_area = np.array(visible_area).reshape((-1, 2))
                visible_area[:,0] = transform_coordinates(visible_area[:,0], "x")
                visible_area[:,1] = transform_coordinates(visible_area[:,1], "y")
                visible_area = _modify_polygon(visible_area, need_correction)
                visible_areas.iat[ev, 0] = visible_area.flatten().tolist()

        locations.iat[ev, 0] = location.reshape((44,)).tolist()

    actions["freeze_frame_360"] = locations
    actions["visible_area_360"] = visible_areas

    actions = (
        actions[actions.type_id != spadlconfig.actiontypes.index("non_action")]
        .sort_values(["idx_original"])
        .reset_index(drop=True)
    )
    actions = actions.drop(["idx_original"], axis=1)
    actions = _fix_direction_of_play(actions)
    actions = _fix_clearances(actions)

    actions["action_id"] = range(len(actions))
    actions = _add_dribbles(actions)

    for col in actions.columns:
        if "_id" in col:
            actions[col] = actions[col].astype(int)
        elif "away_team" in col:
            actions[col] = actions[col].astype(int)

    """for ev in range(len(actions)):
        if np.sum(np.isnan(actions.loc[ev,"freeze_frame_360"]))>0:
            import pdb; pdb.set_trace()
            error('freeze_frame_360 includes nan')"""
    
    if events.at[0, "game_id"] == 3794688:
        
        from matplotlib.animation import FuncAnimation
        from mplsoccer import Pitch
        pitch = Pitch(pitch_type='custom', pitch_length=105, pitch_width=68)
        fig, ax = pitch.draw(figsize=(16, 9))

        # animation
        def update(frame):
            ax.cla()
            pitch.draw(ax=ax)

            # the visible area
            visible_area = np.array(actions.iloc[frame+1500].visible_area_360).reshape(-1, 2)
            pitch.polygon([visible_area], color=(1, 0, 0, 0.3), ax=ax)

            # the ball's location
            cie_ball = np.array([actions.iloc[frame+1500].start_x, actions.iloc[frame+1500].start_y])
            pitch.scatter(
                cie_ball[0], cie_ball[1], c='white', s=240, ec='k', ax=ax
            )

            # the players' location
            cie_pl = np.array(actions.loc[frame+1500, "freeze_frame_360"]).reshape((22, 2))
            cie_atk = cie_pl[:11,:]
            cie_dfd = cie_pl[11:,:]

            # the players' location
            cie_pl = np.array(actions.loc[frame+1500, "freeze_frame_360"]).reshape((22, 2))
            cie_atk = cie_pl[:11]
            cie_dfd = cie_pl[11:]
            pitch.scatter(
                cie_atk[:, 0], cie_atk[:, 1], c='orange', s=80, ec='k', ax=ax
                )
            pitch.scatter(
                cie_dfd[:, 0], cie_dfd[:, 1], c='dodgerblue', s=80, ec='k', ax=ax
                )
            
            ax.text(
                1.0, 1.5, f"Frame: {frame+1500}, Type: {actions.iloc[frame+1500].type_id}, Player: {actions.iloc[frame+1500].player_id}", fontsize=12
                )

        anim = FuncAnimation(fig, update, frames=500, interval=200)
        os.makedirs("/home/r_umemoto/workspace5/work/Generalized-VDEP/anims", exist_ok=True)
        anim.save("/home/r_umemoto/workspace5/work/Generalized-VDEP/anims" + "/converted_sample.mp4", writer="ffmpeg")

        import pdb; pdb.set_trace()

    return actions

