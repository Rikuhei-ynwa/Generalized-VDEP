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


def convert_to_actions(events: pd.DataFrame, home_team_id: int) -> pd.DataFrame:
    events["extra"] = events["extra"].fillna({})
    events = events.fillna(0)

    actions = pd.DataFrame()
    actions["game_id"] = events.game_id  # match_id
    actions["period_id"] = events.period_id  # period
    actions["time_seconds"] = (
        60 * events.minute - ((actions["period_id"] == 2) * 45 * 60) + events.second
    )
    actions["timestamp"] = events.timestamp
    actions["team_id"] = events.team_id
    actions["player_id"] = events.player_id

    actions["start_x"] = events.location.apply(lambda x: x[0] if x else 1)
    actions["start_y"] = events.location.apply(lambda x: x[1] if x else 1)
    actions["start_x"] = ((actions["start_x"] - 1) / 119) * spadlconfig.field_length
    actions["start_y"] = 68 - ((actions["start_y"] - 1) / 79) * spadlconfig.field_width

    end_location = events[["location", "extra"]].apply(_get_end_location, axis=1)
    actions["end_x"] = end_location.apply(lambda x: x[0] if x else 1)
    actions["end_y"] = end_location.apply(lambda x: x[1] if x else 1)
    actions["end_x"] = ((actions["end_x"] - 1) / 119) * spadlconfig.field_length
    actions["end_y"] = 68 - ((actions["end_y"] - 1) / 79) * spadlconfig.field_width

    actions["type_id"] = events[["type_name", "extra"]].apply(_get_type_id, axis=1)
    actions["result_id"] = events[["type_name", "extra"]].apply(_get_result_id, axis=1)
    actions["bodypart_id"] = events[["type_name", "extra"]].apply(
        _get_bodypart_id, axis=1
    )

    actions = (
        actions[actions.type_id != spadlconfig.actiontypes.index("non_action")]
        .sort_values(["game_id", "period_id", "time_seconds", "timestamp"])
        .reset_index(drop=True)
    )
    actions = _fix_direction_of_play(actions, home_team_id)
    actions = _fix_clearances(actions)

    actions["action_id"] = range(len(actions))
    actions = _add_dribbles(actions)

    for col in actions.columns:
        if "_id" in col:
            actions[col] = actions[col].astype(int)
    return actions


def convert_to_actions_360(events: pd.DataFrame, home_team_id: int) -> pd.DataFrame:
    """
    Convert StatsBomb events to SPADL actions + 360

    Parameters
    events: pd.DataFrame, StatsBomb's format.
    home_team_id: int,
    ---
    Returns
    """
    actions = pd.DataFrame()

    events["extra"] = events["extra"].fillna({})
    events = events.fillna(0)

    actions["game_id"] = events.game_id  # match_id
    actions["period_id"] = events.period_id  # period

    actions["time_seconds"] = (
        60 * events.minute - ((actions["period_id"] == 2) * 45 * 60) + events.second
    )
    actions["timestamp"] = events.timestamp
    actions["team_id"] = events.team_id
    actions["player_id"] = events.player_id

    actions["start_x"] = events.location.apply(lambda x: x[0] if x else 1)
    actions["start_y"] = events.location.apply(lambda x: x[1] if x else 1)
    actions["start_x"] = ((actions["start_x"] - 1) / 119) * spadlconfig.field_length
    actions["start_y"] = 68 - ((actions["start_y"] - 1) / 79) * spadlconfig.field_width

    end_location = events[["location", "extra"]].apply(_get_end_location, axis=1)
    actions["end_x"] = end_location.apply(lambda x: x[0] if x else 1)
    actions["end_y"] = end_location.apply(lambda x: x[1] if x else 1)
    actions["end_x"] = ((actions["end_x"] - 1) / 119) * spadlconfig.field_length
    actions["end_y"] = 68 - ((actions["end_y"] - 1) / 79) * spadlconfig.field_width

    actions["type_id"] = events[["type_name", "extra"]].apply(_get_type_id, axis=1)
    actions["result_id"] = events[["type_name", "extra"]].apply(_get_result_id, axis=1)
    actions["bodypart_id"] = events[["type_name", "extra"]].apply(
        _get_bodypart_id, axis=1
    )
    # if you add new variable, please add it to _add_dribbles
    actions["away_team"] = (events.team_id != home_team_id).astype(int)
    # NaN is replaced with False
    if actions["away_team"].isna().sum() > 0:
        actions["away_team"] = actions["away_team"].replace(
            np.nan, False
        )  # .fillna(False, inplace=True) does not work

    # 360 data
    actions["visible_area_360"] = events["visible_area_360"]
    locations = pd.DataFrame({"freeze_frame_360": [[] for _ in range(len(events))]})
    for ev in range(len(events)):
        location = _get_freeze_frame_360(
            events.loc[ev, ["freeze_frame_360", "type_name"]]
        )
        location = location.reshape((22, 2))
        if np.sum(np.isnan(location)) > 0:
            nonnanindex = np.where(~np.isnan(location[:, 0]))
            location[nonnanindex, 0] = (
                (location[nonnanindex, 0] - 1) / 119
            ) * spadlconfig.field_length
            location[nonnanindex, 1] = (
                68 - ((location[nonnanindex, 1] - 1) / 79) * spadlconfig.field_width
            )
            location_copy = location.copy()
            ball_location = np.array(
                [actions.loc[ev, "start_x"], actions.loc[ev, "start_y"]]
            )

            # attackers
            nonnan_atk = np.where(~np.isnan(location[:11, 0]))
            nan_atk = np.where(np.isnan(location[:11, 0]))
            ball2_atk = location[nonnan_atk] - np.repeat(
                ball_location[:, np.newaxis].T, len(nonnan_atk[0]), axis=0
            )
            dist2_atk = np.sum((ball2_atk) ** 2, axis=1)
            atk_nearest = np.concatenate([np.argsort(dist2_atk), nan_atk[0]])
            location[:11] = location_copy[atk_nearest]

            # defenders
            nonnan_dfd = np.where(~np.isnan(location[11:-1, 0]))  # goalkeeper first
            nan_dfd = np.where(np.isnan(location[11:-1, 0]))
            ball2_dfd = location[11 + nonnan_dfd[0]] - np.repeat(
                ball_location[:, np.newaxis].T, len(nonnan_dfd[0]), axis=0
            )
            dist2_dfd = np.sum((ball2_dfd) ** 2, axis=1)
            dfd_nearest = np.concatenate([11 + np.argsort(dist2_dfd), 11 + nan_dfd[0]])
            location[12:] = location_copy[dfd_nearest]
            location[11] = location_copy[-1]  # goalkeeper
        locations.iat[ev, 0] = location.reshape((44,)).tolist()

        # visible_area_360
        if actions.at[ev, "visible_area_360"] == []:
            actions.at[ev, "visible_area_360"] = 0

    actions["freeze_frame_360"] = locations

    # end of 360 data processing
    actions = (
        actions[actions.type_id != spadlconfig.actiontypes.index("non_action")]
        .sort_values(["game_id", "period_id", "time_seconds", "timestamp"])
        .reset_index(drop=True)
    )
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

    return actions


Location = Tuple[float, float]


def _get_freeze_frame_360(q):
    freeze_frame_360, type_name = q
    locations = np.zeros(
        (44,)
    )  # attacking players, attacking goalkeeper, defensive players, defensive goal keeper
    locations[:] = np.nan
    for event in [
        "Pass",
        "Dribble",
        "Carry",
        "Foul Committed",
        "Duel",
        "Interception",
        "Shot",
        "Own Goal Against",
        "Miscontrol",
    ]:
        if type(freeze_frame_360) == int or freeze_frame_360[0]["teammate"] is None:
            return locations
        elif "location" in freeze_frame_360[0]:  # event in type_name and
            atk = 0
            dfd = 0
            if event in ["Foul Committed", "Duel", "Miscontrol"]:
                pdb.set_trace()
            elif event in ["Pass", "Dribble", "Carry", "Duel", "Shot"]:
                for pl in range(len(freeze_frame_360)):
                    if freeze_frame_360[pl]["teammate"]:
                        if freeze_frame_360[pl]["keeper"]:
                            locations[(10) * 2 : (11) * 2] = freeze_frame_360[pl][
                                "location"
                            ]
                        else:
                            locations[(atk) * 2 : (atk + 1) * 2] = freeze_frame_360[pl][
                                "location"
                            ]
                            atk += 1
                    else:
                        if freeze_frame_360[pl]["keeper"]:
                            locations[(21) * 2 : (22) * 2] = freeze_frame_360[pl][
                                "location"
                            ]
                        else:
                            locations[
                                (dfd + 11) * 2 : (dfd + 12) * 2
                            ] = freeze_frame_360[pl]["location"]
                            dfd += 1

            elif event in ["Interception", "Own Goal Against"]:
                for pl in range(len(freeze_frame_360)):
                    if freeze_frame_360[pl]["teammate"]:
                        if freeze_frame_360[pl]["keeper"]:
                            locations[(21) * 2 : (22) * 2] = freeze_frame_360[pl][
                                "location"
                            ]
                        else:
                            locations[
                                (dfd + 11) * 2 : (dfd + 12) * 2
                            ] = freeze_frame_360[pl]["location"]
                            dfd += 1
                    else:
                        if freeze_frame_360[pl]["keeper"]:
                            locations[(10) * 2 : (11) * 2] = freeze_frame_360[pl][
                                "location"
                            ]
                        else:
                            locations[(atk) * 2 : (atk + 1) * 2] = freeze_frame_360[pl][
                                "location"
                            ]
                            atk += 1
            return locations
    pdb.set_trace()
    return freeze_frame_360


def _get_end_location(q: Tuple[Location, dict]) -> Location:
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


min_dribble_length: float = 3.0
max_dribble_length: float = 60.0
max_dribble_duration: float = 10.0


def _add_dribbles(actions: pd.DataFrame) -> pd.DataFrame:
    next_actions = actions.shift(-1)

    same_team = actions.team_id == next_actions.team_id
    # not_clearance = actions.type_id != actiontypes.index("clearance")

    dx = actions.end_x - next_actions.start_x
    dy = actions.end_y - next_actions.start_y
    far_enough = dx**2 + dy**2 >= min_dribble_length**2
    not_too_far = dx**2 + dy**2 <= max_dribble_length**2

    dt = next_actions.time_seconds - actions.time_seconds
    same_phase = dt < max_dribble_duration

    dribble_idx = same_team & far_enough & not_too_far & same_phase

    dribbles = pd.DataFrame()
    prev = actions[dribble_idx]
    nex = next_actions[dribble_idx]
    dribbles["game_id"] = nex.game_id
    dribbles["period_id"] = nex.period_id
    dribbles["action_id"] = prev.action_id + 0.1
    dribbles["time_seconds"] = (prev.time_seconds + nex.time_seconds) / 2
    dribbles["timestamp"] = nex.timestamp
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
