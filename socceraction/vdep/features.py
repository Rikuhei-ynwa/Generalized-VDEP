from typing import Callable, List

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

import socceraction.spadl.config as spadlconfig

NUM_PLAYERS = 22
DIMENTION = 2
NUM_FEATURES_PLAYER = 4 # x, y, dist, angle

SPADLCOLUMNS = [
    'game_id',
    'original_event_id',
    'action_id',
    'period_id',
    'time_seconds',
    'team_id',
    'player_id',
    'start_x',
    'start_y',
    'end_x',
    'end_y',
    'result_id',
    'result_name',
    'bodypart_id',
    'bodypart_name',
    'type_id',
    'type_name',
    "away_team",  # add
    "freeze_frame_360",  # add
]

GOAL_X: float = spadlconfig.FIELD_LENGTH
GOAL_Y: float = spadlconfig.FIELD_WIDTH / 2

# Penalty area in statsbomb coordinates
PENALTY_LEFT_STATSBOMB = 102.0
PENALTY_TOP_STATSBOMB = 18.0
PENALTY_BOTTOM_STATSBOMB = 62.0

# Penalty area in spadl coordinates
PENALTY_LEFT = (
    PENALTY_LEFT_STATSBOMB 
    / spadlconfig.FIELD_LENGTH_STATSBOMB 
    * spadlconfig.FIELD_LENGTH
) # 102 / 120 * 105 = 89.25

PENALTY_TOP = (
    spadlconfig.FIELD_WIDTH
    - (PENALTY_TOP_STATSBOMB
    / spadlconfig.FIELD_WIDTH_STATSBOMB
    * spadlconfig.FIELD_WIDTH)
) # 68 - 18 / 120 * 105 = 68 - 15.75 = 52.25

PENALTY_BOTTOM = (
    spadlconfig.FIELD_WIDTH
    - (PENALTY_BOTTOM_STATSBOMB
    / spadlconfig.FIELD_WIDTH_STATSBOMB
    * spadlconfig.FIELD_WIDTH)
) # 68 - 62 / 120 * 105 = 68 - 54.75 = 15.25


_dummy_actions = pd.DataFrame(
    np.zeros((10, len(SPADLCOLUMNS))), columns=SPADLCOLUMNS
    )
for c in SPADLCOLUMNS:
    if "name" in c:
        _dummy_actions[c] = _dummy_actions[c].astype(str)
    elif "away_team" in c:
        _dummy_actions[c] = _dummy_actions[c].astype(bool)
    elif "freeze_frame_360" in c:
        _dummy_actions[c] = [np.zeros((NUM_PLAYERS * DIMENTION,)).tolist() for _ in range(10)]


def feature_column_names(
        fs: List[Callable], nb_prev_actions: int = 3
        ) -> List[str]:
    gs = gamestates(_dummy_actions, nb_prev_actions)
    tmp = list(pd.concat([f(gs) for f in fs], axis=1).columns)
    return tmp


def gamestates(
        actions: pd.DataFrame, nb_prev_actions: int = 3
        ) -> List[pd.DataFrame]:
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
            df[tyscol + "_" + rescol] = (
                tys[tyscol] & res[rescol]
                )
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
    return timedf


@simple
def startlocation(actions):
    return actions[["start_x", "start_y"]]


@simple
def endlocation(actions):
    return actions[["end_x", "end_y"]]


@simple
def startpolar(actions):
    polardf = pd.DataFrame()
    dx = abs(GOAL_X - actions["start_x"])
    dy = abs(GOAL_Y - actions["start_y"])
    polardf["start_dist_to_goal"] = (
        dx**2 + dy**2 + 1e-6
    ) ** 0.5  # np.sqrt(dx ** 2 + dy ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        polardf["start_angle_to_goal"] = np.nan_to_num(
            np.arctan(dy / dx)
            )
    return polardf


@simple
def endpolar(actions):
    polardf = pd.DataFrame()
    dx = abs(GOAL_X - actions["end_x"])
    dy = abs(GOAL_Y - actions["end_y"])
    polardf["end_dist_to_goal"] = (
        dx**2 + dy**2 + 1e-6
    ) ** 0.5  # np.sqrt(dx ** 2 + dy ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        polardf["end_angle_to_goal"] = np.nan_to_num(
            np.arctan(dy / dx)
            )
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
        teamdf["team_" + (str(i + 1))] = (
            a.team_id == a0.team_id
            )
    return teamdf


def time_delta(gamestates):
    a0 = gamestates[0]
    dt = pd.DataFrame()
    for i, a in enumerate(gamestates[1:]):
        dt["time_delta_" + (str(i + 1))] = (
            a0.time_seconds - a.time_seconds
            )
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
    goals = (
        actions["type_name"].str.contains("shot")
        & (actions["result_id"] == spadlconfig.results.index("success"))
        )
    owngoals = (
        actions["type_name"].str.contains("shot")
        & (actions["result_id"] == spadlconfig.results.index("owngoal"))
        )
    teamisA = actions["team_id"] == teamA
    teamisB = ~teamisA
    goalsteamA = (goals & teamisA) | (owngoals & teamisB)
    goalsteamB = (goals & teamisB) | (owngoals & teamisA)
    goalscoreteamA = goalsteamA.cumsum() - goalsteamA
    goalscoreteamB = goalsteamB.cumsum() - goalsteamB

    scoredf = pd.DataFrame()
    scoredf["goalscore_team"] = (
        (goalscoreteamA * teamisA)
        + (goalscoreteamB * teamisB)
        )
    scoredf["goalscore_opponent"] = (
        (goalscoreteamB * teamisA)
        + (goalscoreteamA * teamisB)
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
    NUM_PLAYERS_TEAM = int(NUM_PLAYERS / 2)
    df = pd.DataFrame()
    locations = np.zeros(
        (len(actions), NUM_PLAYERS * DIMENTION)
        )
    for ev in range(len(actions)):
        locations[ev] = np.array(
            actions.loc[ev].freeze_frame_360
            )

    ball_location = np.concatenate(
        [
            np.array(actions["start_x"])[:, np.newaxis],
            np.array(actions["start_y"])[:, np.newaxis],
        ],
        1,
    )
    ball2_players = (
        locations.reshape((-1, NUM_PLAYERS, DIMENTION))
        - np.repeat(
            ball_location[:, np.newaxis, :], NUM_PLAYERS, 1
            )
        )
    distances_ = np.sqrt(np.sum(ball2_players**2, 2))
    min_dist = 0.1
    # Feature importance
    distances = distances_.copy()
    distances[distances_ > 0] = 1 / distances_[distances_ > 0]
    distances[distances_ < min_dist] = 1 / min_dist  # to avoid zero division

    goal_location = np.array([GOAL_X, GOAL_Y])
    vec2goal = (
        np.repeat(
            np.repeat(
                goal_location[np.newaxis, np.newaxis, :],
                locations.shape[0],
                0
                ), NUM_PLAYERS, 1,
            ) 
        - locations.reshape((-1, NUM_PLAYERS, DIMENTION))
        )

    cols = []
    data = np.empty(
        (len(actions), NUM_PLAYERS * NUM_FEATURES_PLAYER)
        )
    for pl in range(NUM_PLAYERS_TEAM):
        # atk: attacker, dfd: defender
        cols.extend(
            [
                f"atk{pl}_x",
                f"atk{pl}_y",
                f"dist_atk{pl}",
                f"angle_atk{pl}",
                f"dfd{pl}_x",
                f"dfd{pl}_y",
                f"dist_dfd{pl}",
                f"angle_dfd{pl}",
            ]
        )
        idx_atk = pl
        idx_dfd = pl + NUM_PLAYERS_TEAM

        # Sort features in the above order
        # atk_x
        data[:, idx_atk * NUM_FEATURES_PLAYER] = np.nan_to_num(
            locations[:, idx_atk * DIMENTION],
            nan = -spadlconfig.FIELD_LENGTH,
            )
        # atk_y
        data[:, idx_atk * NUM_FEATURES_PLAYER + 1] = np.nan_to_num(
            locations[:, idx_atk * DIMENTION + 1],
            nan = -spadlconfig.FIELD_WIDTH,
            )
        # atk_dist
        data[:, idx_atk * NUM_FEATURES_PLAYER + 2] = np.nan_to_num(
            distances[:, idx_atk],
            nan = 0,
        )
        # atk_angle
        data[:, idx_atk * NUM_FEATURES_PLAYER + 3] = np.nan_to_num(
            np.arctan(
                (vec2goal[:, idx_atk, 1] / vec2goal[:, idx_atk, 0])
            ),
            nan = 0,
        )
        # dfd_x
        data[:, idx_dfd * NUM_FEATURES_PLAYER] = np.nan_to_num(
            locations[:, idx_dfd * DIMENTION],
            nan = -spadlconfig.FIELD_LENGTH,
        )
        # dfd_y
        data[:, idx_dfd * NUM_FEATURES_PLAYER + 1] = np.nan_to_num(
            locations[:, idx_dfd * DIMENTION + 1],
            nan = -spadlconfig.FIELD_WIDTH,
        )
        # dfd_dist
        data[:, idx_dfd * NUM_FEATURES_PLAYER + 2] = np.nan_to_num(
            distances[:, idx_dfd],
            nan = 0,
        )
        # dfd_angle
        data[:, idx_dfd * NUM_FEATURES_PLAYER + 3] = np.nan_to_num(
            np.arctan(
                (vec2goal[:, idx_dfd, 1] / vec2goal[:, idx_dfd, 0])
            ),
            nan = 0,
        )
    if np.isnan(data).any():
        print("NAN in player_loc_dist")
    df = pd.DataFrame(data=data, columns=cols)

    return df


@simple
def gain(actions):
    df = pd.DataFrame()
    offside = actions["result_name"] == "offside"
    regains = (
        (actions["type_name"] == "tackle") | (actions["type_name"] == "interception")
    ) & (actions["result_id"] == 1)
    change_period_id = (
        actions["period_id"] - actions["period_id"].shift(1)
        ).fillna(0) != 0
    on_penalty_id = actions["period_id"] == spadlconfig.ON_PENALTY_PERIOD_ID

    gains = (
        (offside | regains) & (~change_period_id & ~on_penalty_id)
        )
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

    penetrate = (
        (x_e >= PENALTY_LEFT)
        & (x_e <= spadlconfig.FIELD_LENGTH)
        & (y_e >= PENALTY_BOTTOM)
        & (y_e <= PENALTY_TOP)
        & (actions["period_id"] < spadlconfig.ON_PENALTY_PERIOD_ID)
    )
    penetrate.loc[end_nan] = (
        (x.loc[end_nan] >= PENALTY_LEFT)
        & (x.loc[end_nan] <= spadlconfig.FIELD_LENGTH)
        & (y.loc[end_nan] >= PENALTY_BOTTOM)
        & (y.loc[end_nan] <= PENALTY_TOP)
        & (actions["period_id"] < spadlconfig.ON_PENALTY_PERIOD_ID)
    )
    df["penetration"] = penetrate
    return df
