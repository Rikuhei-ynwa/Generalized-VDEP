import tqdm
import pandas as pd

import socceraction.spadl as spadl
import socceraction.vdep.features as fs
import socceraction.vdep.labels as lab


def compute_features(spadl_h5, features_h5, nb_prev_actions=1):
    """
    Compute features for each game and store them in a HDF5 file.

    Parameters
    ----------
    spadl_h5: str
        Path to a h5 file with spadl data.
    features_h5: str
        Path to the output h5 file with features.
    nb_prev_actions: int
        Number of previous actions to take into account for features.
        Attention: nb_prev_actions=1 means only the action at this time is considered.
    """
    games = pd.read_hdf(spadl_h5, "games")
    
    xfns = [
        fs.actiontype,
        fs.actiontype_onehot,
        fs.bodypart,
        fs.bodypart_onehot,
        fs.result,
        fs.result_onehot,
        fs.goalscore,
        fs.startlocation,
        fs.endlocation,
        fs.movement,
        fs.space_delta,
        fs.startpolar,
        fs.endpolar,
        fs.team,
        fs.time,
        fs.time_delta,
        fs.away_team,  # add
        fs.player_loc_dist,  # add
        fs.gain,  # add
        fs.penetration,  # add
    ]

    for game in tqdm.tqdm(
        list(games.itertuples()),
        desc=f"Generating and storing features in {features_h5}",
        ):
        actions = pd.read_hdf(
            spadl_h5, f"actions/game_{game.game_id}"
            )
        # Remove actions on penalty shootouts
        actions = actions[actions.period_id <= 4]
        gamestates = fs.gamestates(
            spadl.add_names_360(actions), nb_prev_actions
            )
        X = pd.concat(
            [fn(gamestates) for fn in xfns], axis=1
            )
        X.to_hdf(features_h5, f"game_{game.game_id}")


def compute_labels(spadl_h5, labels_h5):
    games = pd.read_hdf(spadl_h5, "games")

    yfns = [
        lab.gains,
        lab.effective_attack,
        lab.scores,
        lab.concedes
        ]
    
    for game in tqdm.tqdm(
        list(games.itertuples()),
        desc=f"Computing and storing labels in {labels_h5}"
        ):
        actions = pd.read_hdf(
            spadl_h5, f"actions/game_{game.game_id}"
            )
        actions = actions[actions.period_id <= 4]
        Y = pd.concat(
            [fn(spadl.add_names_360(actions)) for fn in yfns],
            axis=1
            )
        Y.to_hdf(labels_h5, f"game_{game.game_id}")