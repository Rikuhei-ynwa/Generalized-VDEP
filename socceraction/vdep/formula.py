# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd  # type: ignore


def _prev(x: pd.Series) -> pd.Series:
    prev_x = x.shift(1)
    prev_x[:1] = x.values[0]
    return prev_x


_samephase_nb: int = 10


def gain_value(
    actions: pd.DataFrame,
    gains: pd.Series,
) -> pd.Series:
    sameteam = _prev(actions.team_id) == actions.team_id
    prev_gains = _prev(gains) * sameteam  # + _prev(scores) * (~sameteam)

    # if the previous action was too long ago, the odds of prev_gains are now 0
    toolong_idx = (
        abs(actions.time_seconds - _prev(actions.time_seconds)) > _samephase_nb
    )
    prev_gains[toolong_idx] = 0

    # if the previous action was a goal, the odds of prev_gains are now 0
    prevgoal_idx = (
        _prev(actions.type_name).isin(["shot", "shot_freekick", "shot_penalty"])
    ) & (_prev(actions.result_name) == "success")
    prev_gains[prevgoal_idx] = 0

    return gains - prev_gains


def effective_attack_value(
    actions: pd.DataFrame,
    effective_attack: pd.Series,
) -> pd.Series:

    sameteam = _prev(actions.team_id) == actions.team_id
    prev_effective_attack = (
        _prev(effective_attack) * sameteam
    )  # + _prev(scores) * (~sameteam)

    # if the previous action was too long ago, the odds of prev_effective_attack are now 0
    toolong_idx = (
        abs(actions.time_seconds - _prev(actions.time_seconds)) > _samephase_nb
    )
    prev_effective_attack[toolong_idx] = 0

    # if the previous action was a goal, the odds of prev_effective_attack are now 0
    prevgoal_idx = (
        _prev(actions.type_name).isin(["shot", "shot_freekick", "shot_penalty"])
    ) & (_prev(actions.result_name) == "success")
    prev_effective_attack[prevgoal_idx] = 0

    return -(effective_attack - prev_effective_attack)


def offensive_value(
    actions: pd.DataFrame,
    scores: pd.Series,
    concedes: pd.Series,
) -> pd.Series:

    sameteam = _prev(actions.team_id) == actions.team_id
    prev_scores = _prev(scores) * sameteam + _prev(concedes) * (~sameteam)

    # if the previous action was too long ago, the odds of scoring are now 0
    toolong_idx = (
        abs(actions.time_seconds - _prev(actions.time_seconds)) > _samephase_nb
    )
    prev_scores[toolong_idx] = 0

    # if the previous action was a goal, the odds of scoring are now 0
    prevgoal_idx = (
        _prev(actions.type_name).isin(["shot", "shot_freekick", "shot_penalty"])
    ) & (_prev(actions.result_name) == "success")
    prev_scores[prevgoal_idx] = 0

    # fixed odds of scoring when penalty
    penalty_idx = actions.type_name == "shot_penalty"
    prev_scores[penalty_idx] = 0.792453

    # fixed odds of scoring when corner
    corner_idx = actions.type_name.isin(["cornerkick", "corner_short"])
    prev_scores[corner_idx] = 0.046500

    return scores - prev_scores


def defensive_value(
    actions: pd.DataFrame,
    scores: pd.Series,
    concedes: pd.Series,
) -> pd.Series:

    sameteam = _prev(actions.team_id) == actions.team_id
    prev_concedes = _prev(concedes) * sameteam + _prev(scores) * (~sameteam)

    toolong_idx = (
        abs(actions.time_seconds - _prev(actions.time_seconds)) > _samephase_nb
    )
    prev_concedes[toolong_idx] = 0

    # if the previous action was a goal, the odds of conceding are now 0
    prevgoal_idx = (
        _prev(actions.type_name).isin(["shot", "shot_freekick", "shot_penalty"])
    ) & (_prev(actions.result_name) == "success")
    prev_concedes[prevgoal_idx] = 0

    return -(concedes - prev_concedes)


def value(
    actions: pd.DataFrame,
    features: pd.DataFrame,
    preds: pd.DataFrame,
    drops: list,
    C_vdep_v0=3.9,
) -> pd.DataFrame:
    # insert np.nan into the index where all elements in freeze_frame_360 are NaNs.
    preds = preds.values
    for index in drops:
        preds = np.insert(preds, index, np.nan, axis=0)
    preds = pd.DataFrame(
        data=preds, columns=["scores", "concedes", "gains", "effective_attack"]
    )

    # which team is attacking or defending?
    teams = actions["team_name"].unique()
    attack_team = actions["team_name"].values.tolist()
    defense_team = actions["team_name"].values.tolist()
    for i, team in enumerate(actions["team_name"]):
        if (
            (
                actions.at[actions.index[i], "type_name"]
                in [
                    "interception",
                    "tackle",
                    "clearance",
                    "keeper_punch",
                    "keeper_claim",
                    "keeper_save",
                ]
            )
            or (  # which side commit foul?
                (0 < i < len(actions) - 1)
                & (actions.at[actions.index[i], "type_name"] == "foul")
                & (actions.at[actions.index[i - 1], "team_name"] != team)
            )
            or (actions.at[actions.index[i], "result_name"] == "owngoal")
        ):
            attack_team[i] = teams[teams != team][0]
        else:
            defense_team[i] = teams[teams != team][0]

    v = pd.DataFrame()
    v["offensive_value"] = offensive_value(actions, preds.scores, preds.concedes)
    v["defensive_value"] = defensive_value(actions, preds.scores, preds.concedes)
    v["vaep_value"] = v["offensive_value"] + v["defensive_value"]
    d = pd.DataFrame()
    d["vdep_value"] = preds.gains - C_vdep_v0 * preds.effective_attack
    d["gain_value"] = gain_value(actions, preds.gains)
    d["attacked_value"] = effective_attack_value(actions, preds.effective_attack)
    ids = pd.DataFrame()
    ids["attack_team"] = pd.Series(attack_team)
    ids["defense_team"] = pd.Series(defense_team)
    # ids["gains_id"] = features["regain_a0"] | features["result_offside_a0"]
    ids["gains_id"] = features["regain_a0"]
    ids["effective_id"] = features["penetration_a0"] | (
        (actions["type_name"].str.contains("shot")) & (actions["period_id"] < 5)
    )

    return v, d, ids
