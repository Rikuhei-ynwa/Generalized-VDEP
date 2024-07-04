import pandas as pd  # type: ignore

import socceraction.spadl.config as spadlcfg
import socceraction.vdep.features as fs


def gains(actions: pd.DataFrame, nr_actions: int = 5) -> pd.DataFrame:
    """
    This function determines whether a defending team regained the ball
    within the next x actions
    """

    offside = actions["result_name"] == "offside"
    regains = (
        (actions["type_name"] == "tackle") | (actions["type_name"] == "interception")
    ) & (actions["result_id"] == 1)
    change_period_id = (actions["period_id"] - actions["period_id"].shift(1)).fillna(
        0
    ) != 0
    on_penalty_id = actions["period_id"] == spadlcfg.ON_PENALTY_PERIOD_ID

    gains = (offside | regains) & (~change_period_id & ~on_penalty_id)

    y = pd.concat([gains, actions["team_id"]], axis=1)
    y.columns = ["gains", "team_id"]

    # adding future results
    for i in range(1, nr_actions):
        for c in ["gains", "team_id"]:
            shifted = y[c].shift(-i)
            shifted[-i:] = y[c][len(y) - 1]
            y["%s+%d" % (c, i)] = shifted

    res = y["gains"]
    for i in range(1, nr_actions):
        gi = y["gains+%d" % i] & (y["team_id+%d" % i] == y["team_id"])
        res = res | gi

    return pd.DataFrame(res, columns=["gains"])


def effective_attack(actions: pd.DataFrame, nr_actions=5) -> pd.DataFrame:
    """
    This function determines whether a team possessing the ball penetrated into
    the penalty area within the next x actions
    """

    end_nan = actions["end_x"].isna()
    x = actions["start_x"]
    y = actions["start_y"]
    x_e = actions["end_x"]
    y_e = actions["end_y"]

    penalty_left = fs.PENALTY_LEFT
    penalty_right = spadlcfg.FIELD_LENGTH
    penalty_top = fs.PENALTY_TOP
    penalty_bottom = fs.PENALTY_BOTTOM

    penetrate = (
        (x_e >= penalty_left)
        & (x_e <= penalty_right)
        & (y_e <= penalty_top)
        & (y_e >= penalty_bottom)
        & (actions["period_id"] < spadlcfg.ON_PENALTY_PERIOD_ID)
    )
    penetrate.loc[end_nan] = (
        (x.loc[end_nan] >= penalty_left)
        & (x.loc[end_nan] <= penalty_right)
        & (y.loc[end_nan] <= penalty_top)
        & (y.loc[end_nan] >= penalty_bottom)
        & (actions["period_id"] < spadlcfg.ON_PENALTY_PERIOD_ID)
    )

    # merging shots, owngoals and team_ids
    shots = (actions["type_name"].str.contains("shot")) & (actions["period_id"] < 5)
    owngoals = actions["result_name"] == "owngoal"
    y = pd.concat([shots, owngoals, actions["team_id"], penetrate], axis=1)
    y.columns = ["shot", "owngoal", "team_id", "penetrate"]

    # adding future results
    for i in range(1, nr_actions):
        for c in ["team_id", "shot", "owngoal", "penetrate"]:
            shifted = y[c].shift(-i)
            shifted[-i:] = y[c][len(y) - 1]
            y["%s+%d" % (c, i)] = shifted

    res = y["shot"]
    for i in range(1, nr_actions):
        si = y["shot+%d" % i] & (y["team_id+%d" % i] != y["team_id"])
        ogi = y["owngoal+%d" % i] & (y["team_id+%d" % i] == y["team_id"])
        pni = y["penetrate+%d" % i] & (y["team_id+%d" % i] == y["team_id"])
        res = res | (si & ~ogi) | pni

    return pd.DataFrame(res, columns=["effective_attack"])


def scores(
        actions: pd.DataFrame, nr_actions: int = 10
        ) -> pd.DataFrame:
    """
    This function determines whether a goal was scored by the team possessing
    the ball within the next x actions
    """
    # merging goals, owngoals and team_ids

    goals = (
        actions["type_name"].str.contains("shot")
        & (actions["result_id"] == spadlcfg.results.index("success"))
        & (actions["period_id"] < spadlcfg.ON_PENALTY_PERIOD_ID)
    )
    owngoals = actions["result_name"] == "owngoal"
    y = pd.concat([goals, owngoals, actions["team_id"]], axis=1)
    y.columns = ["goal", "owngoal", "team_id"]

    # adding future results
    for i in range(1, nr_actions):
        for c in ["team_id", "goal", "owngoal"]:
            shifted = y[c].shift(-i)
            shifted[-i:] = y[c][len(y) - 1]
            y["%s+%d" % (c, i)] = shifted

    res = y["goal"]
    for i in range(1, nr_actions):
        gi = (
            y["goal+%d" % i]
            & (y["team_id+%d" % i] == y["team_id"])
            )
        ogi = (
            y["owngoal+%d" % i]
            & (y["team_id+%d" % i] != y["team_id"])
            )
        res = res | gi | ogi

    return pd.DataFrame(res, columns=["scores"])


def concedes(
        actions: pd.DataFrame, nr_actions=10
        ) -> pd.DataFrame:
    """
    This function determines whether a goal was scored by the team not
    possessing the ball within the next x actions
    """
    # merging goals,owngoals and team_ids
    goals = (
        actions["type_name"].str.contains("shot")
        & (actions["result_id"] == spadlcfg.results.index("success"))
        & (actions["period_id"] < spadlcfg.ON_PENALTY_PERIOD_ID)
    )
    owngoals = actions["result_name"] == "owngoal"
    y = pd.concat([goals, owngoals, actions["team_id"]], axis=1)
    y.columns = ["goal", "owngoal", "team_id"]

    # adding future results
    for i in range(1, nr_actions):
        for c in ["team_id", "goal", "owngoal"]:
            shifted = y[c].shift(-i)
            shifted[-i:] = y[c][len(y) - 1]
            y["%s+%d" % (c, i)] = shifted

    res = y["owngoal"]
    for i in range(1, nr_actions):
        gi = (
            y["goal+%d" % i]
            & (y["team_id+%d" % i] != y["team_id"])
            )
        ogi = (
            y["owngoal+%d" % i]
            & (y["team_id+%d" % i] == y["team_id"])
            )
        res = res | gi | ogi

    return pd.DataFrame(res, columns=["concedes"])


def goal_from_shot(actions: pd.DataFrame) -> pd.DataFrame:
    goals = (
        actions["type_name"].str.contains("shot")
        & (actions["result_id"] == spadlcfg.results.index("success"))
        & (actions["period_id"] < 5)
    )

    return pd.DataFrame(goals, columns=["goal_from_shot"])
