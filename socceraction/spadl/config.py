# -*- coding: utf-8 -*-
"""Configuration of the SPADL language.

Attributes
----------
FIELD_LENGTH : float
    The length of a pitch (in meters).
FIELD_WIDTH : float
    The width of a pitch (in meters).
bodyparts : list(str)
    The bodyparts used in the SPADL language.
results : list(str)
    The action results used in the SPADL language.
actiontypes : list(str)
    The action types used in the SPADL language.

"""

# Rikuhei Umemoto, 2024
# modifying the code https://github.com/ML-KULeuven/socceraction/blob/master/socceraction/spadl/config.py

from typing import List

import pandas as pd  # type: ignore

FIELD_LENGTH_STATSBOMB: float = 120.0
FIELD_WIDTH_STATSBOMB: float = 80.0

FIELD_LENGTH: float = 105.0  # unit: meters
FIELD_WIDTH: float = 68.0  # unit: meters

ON_PENALTY_PERIOD_ID: int = 5

bodyparts: List[str] = [
    'foot', 'foot_right', 'foot_left', 'head', 'other', 'head/other'
    ]

results: List[str] = [
    'fail',
    'success',
    'offside',
    'owngoal',
    'yellow_card',
    'red_card',
]

actiontypes: List[str] = [
    'pass',
    'cross',
    'throw_in',
    'freekick_crossed',
    'freekick_short',
    'corner_crossed',
    'corner_short',
    'take_on',
    'offensive_foul',
    'defensive_foul',
    'tackle',
    'interception',
    'shot',
    'shot_penalty',
    'shot_freekick',
    'keeper_save',
    'keeper_claim',
    'keeper_punch',
    'keeper_pick_up',
    'clearance',
    'bad_touch',
    'non_action',
    'dribble',
    'goalkick',
]


def actiontypes_df() -> pd.DataFrame:
    """Return a dataframe with the type id and type name of each SPADL action type.

    Returns
    -------
    pd.DataFrame
        The 'type_id' and 'type_name' of each SPADL action type.
    """
    return pd.DataFrame(list(enumerate(actiontypes)), columns=['type_id', 'type_name'])


def results_df() -> pd.DataFrame:
    """Return a dataframe with the result id and result name of each SPADL action type.

    Returns
    -------
    pd.DataFrame
        The 'result_id' and 'result_name' of each SPADL action type.
    """
    return pd.DataFrame(list(enumerate(results)), columns=['result_id', 'result_name'])


def bodyparts_df() -> pd.DataFrame:
    """Return a dataframe with the bodypart id and bodypart name of each SPADL action type.

    Returns
    -------
    pd.DataFrame
        The 'bodypart_id' and 'bodypart_name' of each SPADL action type.
    """
    return pd.DataFrame(list(enumerate(bodyparts)), columns=['bodypart_id', 'bodypart_name'])
