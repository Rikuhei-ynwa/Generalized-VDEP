from typing import Callable, List

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

import socceraction.spadl.config as spadlconfig

_spadlcolumns = [
    "game_id",
    "period_id",
    "time_seconds",
    "timestamp",
    "team_id",
    "player_id",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
    "result_id",
    "result_name",
    "bodypart_id",
    "bodypart_name",
    "type_id",
    "type_name",
    "away_team",  # add
    "freeze_frame_360",  # add
]
_dummy_actions = pd.DataFrame(np.zeros((10, len(_spadlcolumns))), columns=_spadlcolumns)
for c in _spadlcolumns:
    if "name" in c:
        _dummy_actions[c] = _dummy_actions[c].astype(str)
    elif "away_team" in c:
        _dummy_actions[c] = _dummy_actions[c].astype(bool)
    elif "freeze_frame_360" in c:
        _dummy_actions[c] = [np.zeros((44,)).tolist() for _ in range(10)]


def feature_column_names(fs: List[Callable], nb_prev_actions: int = 3) -> List[str]:
    gs = gamestates(_dummy_actions, nb_prev_actions)
    tmp = list(pd.concat([f(gs) for f in fs], axis=1).columns)
    return tmp


def gamestates(actions: pd.DataFrame, nb_prev_actions: int = 3) -> List[pd.DataFrame]:
    """This function take a dataframe <actions> and outputs gamestates.
    Each gamestate is represented as the <nb_prev_actions> previous actions.

    The list of gamestates is internally represented as a list of actions dataframes [a_0,a_1,..]
    where each row in the a_i dataframe contains the previous action of
    the action in the same row in the a_i-1 dataframe.
    """
    states = [actions]
    for i in range(1, nb_prev_actions):
        prev_actions = actions.copy().shift(i, fill_value=0)
        prev_actions.loc[: i - 1, :] = pd.concat([actions[:1]] * i, ignore_index=True)
        states.append(prev_actions)
    return states


def simple(actionfn):
    "Function decorator to apply actionfeatures to gamestates"

    def wrapper(gamestates):
        if not isinstance(gamestates, (list,)):
            gamestates = [gamestates]
        X = []
        for i, a in enumerate(gamestates):
            Xi = actionfn(a)
            Xi.columns = [c + "_a" + str(i) for c in Xi.columns]
            X.append(Xi)
        return pd.concat(X, axis=1)

    return wrapper


# SIMPLE FEATURES


@simple
def actiontype(actions):
    return actions[["type_id"]]


@simple
def actiontype_onehot(actions):
    X = pd.DataFrame()
    for type_name in spadlconfig.actiontypes:
        col = "type_" + type_name
        X[col] = actions["type_name"] == type_name
    return X


@simple
def result(actions):
    return actions[["result_id"]]


@simple
def result_onehot(actions):
    X = pd.DataFrame()
    for result_name in spadlconfig.results:
        col = "result_" + result_name
        X[col] = actions["result_name"] == result_name
    return X


@simple
def actiontype_result_onehot(actions):
    res = result_onehot(actions)
    tys = actiontype_onehot(actions)
    df = pd.DataFrame()
    for tyscol in list(tys.columns):
        for rescol in list(res.columns):
            df[tyscol + "_" + rescol] = tys[tyscol] & res[rescol]
    return df


@simple
def bodypart(actions):
    return actions[["bodypart_id"]]


@simple
def bodypart_onehot(actions):
    X = pd.DataFrame()
    for bodypart_name in spadlconfig.bodyparts:
        col = "bodypart_" + bodypart_name
        X[col] = actions["bodypart_name"] == bodypart_name
    return X


@simple
def time(actions):
    timedf = actions[["period_id", "time_seconds"]].copy()
    timedf["time_seconds_overall"] = (
        (timedf.period_id - 1) * 45 * 60
    ) + timedf.time_seconds
    return timedf


@simple
def startlocation(actions):
    return actions[["start_x", "start_y"]]


@simple
def endlocation(actions):
    return actions[["end_x", "end_y"]]


_goal_x: float = spadlconfig.field_length
_goal_y: float = spadlconfig.field_width / 2


@simple
def startpolar(actions):
    polardf = pd.DataFrame()
    dx = abs(_goal_x - actions["start_x"])
    dy = abs(_goal_y - actions["start_y"])
    polardf["start_dist_to_goal"] = (
        dx**2 + dy**2 + 1e-6
    ) ** 0.5  # np.sqrt(dx ** 2 + dy ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        polardf["start_angle_to_goal"] = np.nan_to_num(np.arctan(dy / dx))
    return polardf


@simple
def endpolar(actions):
    polardf = pd.DataFrame()
    dx = abs(_goal_x - actions["end_x"])
    dy = abs(_goal_y - actions["end_y"])
    polardf["end_dist_to_goal"] = (
        dx**2 + dy**2 + 1e-6
    ) ** 0.5  # np.sqrt(dx ** 2 + dy ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        polardf["end_angle_to_goal"] = np.nan_to_num(np.arctan(dy / dx))
    return polardf


@simple
def movement(actions):
    mov = pd.DataFrame()
    mov["dx"] = actions.end_x - actions.start_x
    mov["dy"] = actions.end_y - actions.start_y
    mov["movement"] = (
        mov.dx**2 + mov.dy**2 + 1e-6
    ) ** 0.5  # np.sqrt(mov.dx ** 2 + mov.dy ** 2 + 1e-6)

    return mov


# STATE FEATURES


def team(gamestates):
    a0 = gamestates[0]
    teamdf = pd.DataFrame()
    for i, a in enumerate(gamestates[1:]):
        teamdf["team_" + (str(i + 1))] = a.team_id == a0.team_id
    return teamdf


def time_delta(gamestates):
    a0 = gamestates[0]
    dt = pd.DataFrame()
    for i, a in enumerate(gamestates[1:]):
        dt["time_delta_" + (str(i + 1))] = a0.time_seconds - a.time_seconds
    return dt


def space_delta(gamestates):
    a0 = gamestates[0]
    spaced = pd.DataFrame()
    for i, a in enumerate(gamestates[1:]):
        dx = a.end_x - a0.start_x
        spaced["dx_a0" + (str(i + 1))] = dx
        dy = a.end_y - a0.start_y
        spaced["dy_a0" + (str(i + 1))] = dy
        spaced["mov_a0" + (str(i + 1))] = (
            dx**2 + dy**2 + 1e-6
        ) ** 0.5  # np.sqrt(dx ** 2 + dy ** 2)
    return spaced


# CONTEXT FEATURES


def goalscore(gamestates):
    """
    This function determines the nr of goals scored by each team after the
    action
    """
    actions = gamestates[0]
    teamA = actions["team_id"].values[0]
    goals = actions["type_name"].str.contains("shot") & (
        actions["result_id"] == spadlconfig.results.index("success")
    )
    owngoals = actions["type_name"].str.contains("shot") & (
        actions["result_id"] == spadlconfig.results.index("owngoal")
    )
    teamisA = actions["team_id"] == teamA
    teamisB = ~teamisA
    goalsteamA = (goals & teamisA) | (owngoals & teamisB)
    goalsteamB = (goals & teamisB) | (owngoals & teamisA)
    goalscoreteamA = goalsteamA.cumsum() - goalsteamA
    goalscoreteamB = goalsteamB.cumsum() - goalsteamB

    scoredf = pd.DataFrame()
    scoredf["goalscore_team"] = (goalscoreteamA * teamisA) + (goalscoreteamB * teamisB)
    scoredf["goalscore_opponent"] = (goalscoreteamB * teamisA) + (
        goalscoreteamA * teamisB
    )
    scoredf["goalscore_diff"] = (
        scoredf["goalscore_team"] - scoredf["goalscore_opponent"]
    )
    return scoredf


# ADDED FOR VDEP


@simple
def away_team(actions):
    return actions[["away_team"]]


@simple
def player_loc_dist(actions):
    df = pd.DataFrame()
    locations = np.zeros((len(actions), 44))
    for ev in range(len(actions)):
        locations[ev] = np.array(actions.loc[ev].freeze_frame_360)

    ball_location = np.concatenate(
        [
            np.array(actions["start_x"])[:, np.newaxis],
            np.array(actions["start_y"])[:, np.newaxis],
        ],
        1,
    )
    ball2_players = locations.reshape((-1, 22, 2)) - np.repeat(
        ball_location[:, np.newaxis, :], 22, 1
    )
    distances_ = np.sqrt(np.sum(ball2_players**2, 2))
    min_dist = 0.1
    # Feature importance
    distances = distances_.copy()
    distances[distances_ > 0] = 1 / distances_[distances_ > 0]
    distances[distances_ < min_dist] = 1 / min_dist  # to avoid zero division

    goal_location = np.array([_goal_x, _goal_y])
    goal_vector = np.repeat(
        np.repeat(goal_location[np.newaxis, np.newaxis, :], locations.shape[0], 0),
        22,
        1,
    ) - locations.reshape((-1, 22, 2))

    cols = []
    data = np.empty((len(actions), 88))
    for pl in range(11):
        cols.extend(
            [
                f"at{pl}_x",
                f"at{pl}_y",
                f"df{pl}_x",
                f"df{pl}_y",
                f"dist_at{pl}",
                f"dist_df{pl}",
                f"angle_at{pl}",
                f"angle_df{pl}",
            ]
        )

    for action in range(len(actions)):
        dist_at = distances[action, :11]
        dist_df = distances[action, 11:]
        atk_x = locations[action, :22:2]
        atk_y = locations[action, 1:22:2]
        dfd_x = locations[action, 22::2]
        dfd_y = locations[action, 23::2]
        with np.errstate(divide="ignore", invalid="ignore"):
            angle_at = np.arctan(
                goal_vector[action, :11, 1] / goal_vector[action, :11, 0]
            )
            angle_df = np.arctan(
                goal_vector[action, 11:, 1] / goal_vector[action, 11:, 0]
            )

        # sort by "distance"
        # standard
        std_at = np.argsort(np.nan_to_num(dist_at))
        std_df = np.argsort(np.nan_to_num(dist_df))
        # sort
        atk_x = atk_x[std_at][::-1]
        atk_y = atk_y[std_at][::-1]
        dfd_x = dfd_x[std_df][::-1]
        dfd_y = dfd_y[std_df][::-1]
        dist_at = dist_at[std_at][::-1]
        dist_df = dist_df[std_df][::-1]
        angle_at = angle_at[std_at][::-1]
        angle_df = angle_df[std_df][::-1]
        # rearrangement
        data[action] = np.stack(
            [atk_x, atk_y, dfd_x, dfd_y, dist_at, dist_df, angle_at, angle_df], 1
        ).reshape(-1)

    df = pd.DataFrame(data=data, columns=cols)

    return df


@simple
def gain(actions):
    df = pd.DataFrame()
    offside = actions["result_name"] == "offside"
    regains = (
        (actions["type_name"] == "tackle") | (actions["type_name"] == "interception")
    ) & (actions["result_id"] == 1)
    change_period_id = (actions["period_id"] - actions["period_id"].shift(1)).fillna(
        0
    ) != 0
    on_penalty_id = actions["period_id"] == 5

    gains = (offside | regains) & (~change_period_id & ~on_penalty_id)
    df["regain"] = gains

    return df


@simple
def penetration(actions):
    df = pd.DataFrame()
    end_nan = actions["end_x"].isna()
    x = actions["start_x"]
    y = actions["start_y"]
    x_e = actions["end_x"]
    y_e = actions["end_y"]

    penalty_left = spadlconfig.field_length - 16.5
    penalty_right = spadlconfig.field_length
    penalty_top = spadlconfig.field_width / 2 - 20.16
    penalty_bottom = spadlconfig.field_width / 2 + 20.16

    penetrate = (
        (x_e >= penalty_left)
        & (x_e <= penalty_right)
        & (y_e >= penalty_top)
        & (y_e <= penalty_bottom)
        & (actions["period_id"] < 5)
    )
    penetrate.loc[end_nan] = (
        (x.loc[end_nan] >= penalty_left)
        & (x.loc[end_nan] <= penalty_right)
        & (y.loc[end_nan] >= penalty_top)
        & (y.loc[end_nan] <= penalty_bottom)
        & (actions["period_id"] < 5)
    )
    df["penetration"] = penetrate
    return df
