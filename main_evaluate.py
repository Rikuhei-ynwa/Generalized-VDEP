import argparse
import os
import pickle
import random
import time
import warnings

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tqdm

import analysis
import convert_rawdata as crd
import set_variables as sv
import train_models as tm

import socceraction.spadl as spadl
import socceraction.vdep.formula as vdepformula

pd.set_option("display.max_columns", None)
warnings.simplefilter(
    action="ignore",
    category=pd.errors.PerformanceWarning
    )
warnings.filterwarnings(
    action="ignore", 
    message="credentials were not supplied. open data access only"
    )

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

PLAYER_NUM_PER_TEAM = 11
DIMENTION = 2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=str, default="statsbomb"
        )
    parser.add_argument(
        "--game", type=str, default="all"
        )
    parser.add_argument(
        "--feature", type=str, default="all"
        )
    parser.add_argument(
        "--nb_prev_actions", type=int, default=1
        )
    parser.add_argument(
        "--date_opendata", type=int, default=20231002, required=True, 
        help="Since statsbomb data is constantly being updated, it is important to indicate in advance when the data was downloaded."
        )
    parser.add_argument(
        "--date_experiment", type=int, default=0, required=True,
        help="The date of the experiment is important because the data is constantly being updated."
        )
    parser.add_argument(
        "--no_games", type=int, default=0
        )
    parser.add_argument(
        "--seed", type=int, default=0
        )
    parser.add_argument(
        "--n_nearest", type=int, default=11
        )
    parser.add_argument(
        "--numProcess", type=int, default=16
        )
    parser.add_argument(
        "--skip_convert_rawdata", action="store_true"
        )
    parser.add_argument(
        "--skip_preprocess", action="store_true"
        )
    parser.add_argument(
        "--predict_actions", action="store_true"
        )
    parser.add_argument(
        "--model", type=str, default="xgboost", required=True,
        help="Please select the model to be used for training. Now the options is only 'xgboost'.",
        )
    parser.add_argument(
        "--grid_search", action="store_true"
        )
    parser.add_argument(
        "--skip_train", action="store_true"
        )
    parser.add_argument(
        "--test", action="store_true"
        )
    parser.add_argument(
        "--teamView", type=str, default="",
        help="Please set the name of the country, not the name of the team",
        )
    parser.add_argument(
        "--sample_events", type=int, default=4
        )
    parser.add_argument(
        "--pickle", type=int, default=4
        )
    args, _ = parser.parse_known_args()

    return args


def convert_rawdata(args, datafolder):
    DLoader = crd.set_DLoader(args)
    games, selected_competitions = crd.select_games(DLoader, args)
    crd.store_converted_data(
        DLoader, datafolder, games, selected_competitions,
        )


def compute_features_and_labels(
        spadl_h5, features_h5, labels_h5, nb_prev_actions
        ):
    sv.compute_features(spadl_h5, features_h5, nb_prev_actions)
    sv.compute_labels(spadl_h5, labels_h5)


def compute_gvdep_values(
        spadl_h5, features_h5, predictions_h5, datafolder, model_str 
        ):
    with pd.HDFStore(spadl_h5, pickle_protocol=4) as spadlstore:
        games = (
            spadlstore["games"]
            .merge(spadlstore["competitions"], how="left")
            .merge(spadlstore["teams"].add_prefix("home_"), how="left")
            .merge(spadlstore["teams"].add_prefix("away_"), how="left")
        )
        players = spadlstore["players"]
        teams = spadlstore["teams"]
    # Merge for home team names
    games = games.merge(teams, how='left', left_on='home_team_id', right_on='team_id')
    games = games.drop(columns=['team_id', 'team_name'])


    print("nb of games:", len(games))

    with open(
        os.path.join(datafolder, "static_" + model_str + ".pkl"), "rb"
        ) as f:
        _, C_vdep_v0, _,  = pickle.load(f)

    A = []
    for game in tqdm.tqdm(list(games.itertuples()), desc="Rating actions"):
        actions = pd.read_hdf(spadl_h5,f"actions/game_{game.game_id}")
        actions = (
            spadl.add_names_360(actions)
            .merge(players, how="left")
            .merge(teams, how="left")
            .sort_values(["game_id", "period_id", "action_id"])
            .reset_index(drop=True)
        )
        preds = pd.read_hdf(predictions_h5, f"game_{game.game_id}")
        features = pd.read_hdf(features_h5, f"game_{game.game_id}")
        vaep, vdep, ids = vdepformula.value(actions, features, preds, C_vdep_v0)
        A.append(
            pd.concat([actions, preds, vaep, vdep, ids], axis=1)
            )
    A = pd.concat(A).sort_values(
        ["game_id", "period_id", "time_seconds"]
        ).reset_index(drop=True)

    # Gvdep calculated by vaep, gain_value and attacked_value
    num_gains = (A["gains_id"]).sum()
    num_attacked = (A["effective_id"]).sum()
    weight_gains = (
        (A["vaep_value"] * A["gains_id"]).sum() / num_gains
        )
    weight_attacked = (
        ((-A["vaep_value"]) * (A["effective_id"])).sum() / num_attacked
        )
    A["gvdep_value"] = (
        weight_gains * A["gain_value"] - weight_attacked * A["attacked_value"]
        )

    return A, games


def main():
    args = parse_args()

    pickle.HIGHEST_PROTOCOL = args.pickle  # 4 is the highest in an older version of pickle
    numProcess = args.numProcess
    os.environ["OMP_NUM_THREADS"] = str(numProcess)

    start = time.time()
    NB_PREV_ACTIONS = args.nb_prev_actions

    random.seed(args.seed)
    np.random.seed(args.seed)

    datafolder = (
        f"../GVDEP_data/data-{args.data}"
        + f"/{args.game}"
        + f"/{args.date_experiment}"
        )
    os.makedirs(datafolder, exist_ok=True)
    figuredir = datafolder + "/figures/"
    os.makedirs(figuredir, exist_ok=True)
    args.figuredir = figuredir

    # Configure file and folder names
    spadl_h5 = os.path.join(datafolder, "spadl-statsbomb.h5")
    features_h5 = os.path.join(datafolder, "features.h5")
    labels_h5 = os.path.join(datafolder, "labels.h5")
    predictions_h5 = os.path.join(datafolder, "predictions.h5")

    # 1. load and convert statsbomb data
    if args.skip_convert_rawdata:
        print("loading rawdata is skipped")
    else:
        convert_rawdata(args, datafolder)
    print(pd.read_hdf(spadl_h5, "competitions")) # checking
    games = pd.read_hdf(spadl_h5, "games")


    # 2. compute features and labels
    if args.skip_preprocess:
        print("preprocessing is skipped")
    else:
        compute_features_and_labels(
            spadl_h5, features_h5, labels_h5, NB_PREV_ACTIONS
            )

    # 3. estimate scoring and conceding probabilities
    _, _, model_str = tm.set_conditions(args, games)
    if args.skip_train:
        print("model training is skipped")
    else:
        tm.save_predictions_for_evaluation(
            args, games, datafolder,
            spadl_h5, features_h5, labels_h5, predictions_h5,
            use_location = False, use_polar = False
            )

    # 4. compute gvdep values and top players
    A, games = compute_gvdep_values(
        spadl_h5, features_h5, predictions_h5, datafolder, model_str
        )

    # (optional) inspect country's top 5 most valuable "tackle" and "interception" actions
    if args.test:
        import visualize_sample as vs
        vs.visualise_sample(args, A, games)


    # 5. Analyze team defense
    analysis.analyze_team_defense(args, A, games)
    analysis.plot_nan_hist(args, A)
    

    import pdb; pdb.set_trace()


if __name__ == "__main__":
    main()
