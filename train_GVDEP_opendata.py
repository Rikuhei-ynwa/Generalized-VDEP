import argparse
import itertools
import os
import pdb
import pickle
import random
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import tqdm
from adjustText import adjust_text

import socceraction.spadl as spadl
import socceraction.vdep.features as fs
import socceraction.vdep.labels as lab
from socceraction.data.statsbomb import StatsBombLoader
from socceraction.data.wyscout import PublicWyscoutLoader, WyscoutLoader
from socceraction.spadl.statsbomb_360 import convert_to_actions_360
import socceraction.vdep.formula as vdepformula

pd.set_option("display.max_columns", None)
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings(
    action="ignore", message="credentials were not supplied. open data access only"
)

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="statsbomb")
parser.add_argument("--game", type=str, default="all")
parser.add_argument("--feature", type=str, default="all")
parser.add_argument("--datadir", type=str, default="")
parser.add_argument("--no_games", type=int, default=0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_nearest", type=int, default=11)
parser.add_argument("--numProcess", type=int, default=16)
parser.add_argument("--skip_load_rawdata", action="store_true")
parser.add_argument("--skip_preprocess", action="store_true")
parser.add_argument("--predict_actions", action="store_true")
parser.add_argument("--grid_search", action="store_true")
parser.add_argument("--split_train_eval", action="store_true")
parser.add_argument("--skip_train", action="store_true")
parser.add_argument("--test", action="store_true")
parser.add_argument(
    "--teamView",
    type=str,
    default="",
    help="Please set the name of the country, not the name of the team",
)
parser.add_argument(
    "--pickle", type=int, default=4, help="if a higher python version, please use 5"
)
args, _ = parser.parse_known_args()

pickle.HIGHEST_PROTOCOL = args.pickle  # 4 is the highest in an older version of pickle
numProcess = args.numProcess
os.environ["OMP_NUM_THREADS"] = str(numProcess)

start = time.time()
data_dir = args.datadir

random.seed(args.seed)
np.random.seed(args.seed)

datafolder = "../data-statsbomb/vdep/" + args.game
os.makedirs(datafolder, exist_ok=True)

"""1. load and convert statsbomb data"""
if args.skip_load_rawdata:
    print("loading rawdata is skipped")
else:
    """Set up the Loader"""
    # https://socceraction.readthedocs.io/en/latest/documentation/providers.html

    if args.data == "statsbomb":
        # Use this if you want to use the free public statsbomb data
        # or provide credentials to access the API
        # DLoader = StatsBombLoader(getter="remote", creds={"user": None, "passwd": None})

        # Uncomment the code below if you have a local folder on your computer with statsbomb data
        data_folder = "../statsbomb-open-rawdata/data"  # Example of local folder with statsbomb data
        DLoader = StatsBombLoader(root=data_folder, getter="local")

    elif args.data == "wyscout":
        # Wyscout
        # https://www.nature.com/articles/s41597-019-0247-7
        # https://socceraction.readthedocs.io/en/latest/modules/generated/socceraction.data.wyscout.PublicWyscoutLoader.html
        data_folder = "../Wyscout-open-data/"
        if not os.path.isdir(data_folder):
            os.makedirs(data_folder)
            DLoader = PublicWyscoutLoader(root=data_folder, download=True)
        else:
            DLoader = WyscoutLoader(
                root=data_folder, getter="local"
            )  # (root='https://apirest.wyscout.com/v2/', getter='remote') #

    if args.data == "statsbomb":  # No competition and season information in wyscout!!!!
        # View all available competitions
        competitions = DLoader.competitions()
        set(competitions.competition_name)

        if args.game == "fifa":
            selected_competitions = competitions[
                competitions.competition_name == "FIFA World Cup"
            ]
        elif args.game == "male":
            selected_competitions = competitions[
                competitions.competition_gender == "male"
            ]
        elif args.game == "euro2020":
            selected_competitions = competitions[
                competitions.competition_name == "UEFA Euro"
            ]
        elif args.game == "euro2022":
            selected_competitions = competitions[
                competitions.competition_name == "UEFA Women's Euro"
            ]
        else:  # All data
            selected_competitions = competitions

    # Get games from all selected competitions
    games = pd.concat(
        [
            DLoader.games(row.competition_id, row.season_id)
            for row in selected_competitions.itertuples()
        ]
    )

    # Store converted spadl data in a h5-file
    games_verbose = tqdm.tqdm(list(games.itertuples()), desc="Loading game data")
    teams, players, games_360 = [], [], []
    actions = {}
    for game in games_verbose:
        events = DLoader.events(game.game_id, load_360=True)
        # convert data
        actions[game.game_id] = convert_to_actions_360(
            events, game.home_team_id
        )  # spadl.statsbomb.convert_to_actions

        # load data
        teams.append(DLoader.teams(game.game_id))
        players.append(DLoader.players(game.game_id))
        games_360.append(game.game_id)

    teams = pd.concat(teams).drop_duplicates(subset="team_id")
    players = pd.concat(players)

    spadl_h5 = os.path.join(datafolder, "spadl-statsbomb.h5")
    # Store all spadl data in h5-file
    with pd.HDFStore(spadl_h5, pickle_protocol=4) as spadlstore:
        spadlstore["competitions"] = selected_competitions
        spadlstore["games"] = games
        spadlstore["teams"] = teams
        spadlstore["players"] = players[
            ["player_id", "player_name", "nickname"]
        ].drop_duplicates(subset="player_id")
        spadlstore["player_games"] = players[
            [
                "player_id",
                "game_id",
                "team_id",
                "is_starter",
                "starting_position_id",
                "starting_position_name",
                "minutes_played",
            ]
        ]
        for game_id in actions.keys():
            spadlstore[f"actions/game_{game_id}"] = actions[game_id]

# Configure file and folder names
spadl_h5 = os.path.join(datafolder, "spadl-statsbomb.h5")
features_h5 = os.path.join(datafolder, "features.h5")
labels_h5 = os.path.join(datafolder, "labels.h5")

print(pd.read_hdf(spadl_h5, "competitions"))


"""2. compute features and labels"""
nb_prev_actions = 1
if args.skip_preprocess:
    print("preprocessing is skipped")
else:
    games = pd.read_hdf(spadl_h5, "games")
    print("nb of games:", len(games))
    # Compute features
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
        actions = pd.read_hdf(spadl_h5, f"actions/game_{game.game_id}")
        gamestates = fs.gamestates(spadl.add_names_360(actions), nb_prev_actions)
        # gamestates = fs.play_left_to_right(gamestates, game.home_team_id)
        X = pd.concat([fn(gamestates) for fn in xfns], axis=1)
        X.to_hdf(features_h5, f"game_{game.game_id}")

    # Compute labels
    yfns = [lab.gains, lab.effective_attack, lab.scores, lab.concedes]
    g = 0
    for game in tqdm.tqdm(
        list(games.itertuples()), desc=f"Computing and storing labels in {labels_h5}"
    ):
        actions = pd.read_hdf(spadl_h5, f"actions/game_{game.game_id}")
        Y = pd.concat([fn(spadl.add_names_360(actions)) for fn in yfns], axis=1)
        Y.to_hdf(labels_h5, f"game_{game.game_id}")


"""3. estimate scoring and conceding probabilities"""
random.seed(args.seed)
np.random.seed(args.seed)
if args.predict_actions:
    datafolder += "/vaep_framework"
    os.makedirs(datafolder, exist_ok=True)
figuredir = datafolder + "/figures"
os.makedirs(figuredir, exist_ok=True)
predictions_h5 = os.path.join(datafolder, "predictions.h5")
games = pd.read_hdf(spadl_h5, "games")
print("nb of games:", len(games))

"""note: only for the purpose of this example and due to the small dataset,
        we use the same data for training and evaluation"""
if args.no_games < 10 and args.no_games > 0:
    traingames = games[: args.no_games - 1]
    testgames = games[args.no_games - 1 : args.no_games]
    model_str = str(args.no_games) + "games"
elif args.no_games >= 10 or args.no_games == 0:
    split_traintest = int(9 * len(games) / 10)
    traingames = games[:split_traintest]
    testgames = games[split_traintest:]
    model_str = (
        str(args.no_games) + "games" if args.no_games >= 10 else "_traindata_all"
    )
elif args.no_games == -1:
    traingames = games
    testgames = games
    model_str = "all"

if args.n_nearest <= 11:
    model_str += "_" + str(args.n_nearest) + "_nearest"
print("train/test of games:", len(traingames), "/", len(testgames))

if args.skip_train:
    print("model training is skipped")
else:
    xfns = [
        fs.actiontype,
        fs.actiontype_onehot,
        # fs.bodypart,
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
        # fs.time,…
        fs.time_delta,
        # fs.actiontype_result_onehot
        fs.away_team,  # add
        fs.gain,  # add
        fs.penetration,  # add
        fs.player_loc_dist,
    ]

    Xcols = fs.feature_column_names(xfns, nb_prev_actions)

    def getXY(games: pd.DataFrame, Xcols: list):
        """
        Parameters
        games: pd.DataFrame, this contains games' information.
        Xcols: list, this is features' list we want to use.
        ---
        Returns
        X: pd.DataFrame, this is the dataframe about the features.
        Y: pd.DataFrame, this is the dataframe about the labels.
        drop_index: list, the numbers in this list mean the index of the rows that doesn't have players' coodinates.
        """
        # generate the columns of the selected features and labels
        X = []
        for game_id in tqdm.tqdm(games.game_id, desc="Selecting features"):
            Xi = pd.read_hdf(features_h5, f"game_{game_id}")
            X.append(Xi[Xcols])
        X = pd.concat(X).reset_index(drop=True)

        Ycols = ["scores", "concedes", "gains", "effective_attack"]
        Y = []
        for game_id in tqdm.tqdm(games.game_id, desc="Selecting label"):
            Yi = pd.read_hdf(labels_h5, f"game_{game_id}")
            Y.append(Yi[Ycols])
        Y = pd.concat(Y).reset_index(drop=True)

        # Drop the rows if their "dist_at0_a0" and "dist_df0_a0" are NaN
        if "dist_at0_a0" in X.columns.values and "dist_df0_a0" in X.columns.values:
            at_nan = X[X["dist_at0_a0"].isnull()]
            at_df_nan = at_nan[at_nan["dist_df0_a0"].isnull()]
            drop_index = at_df_nan.index.values.tolist()
            X = X.drop(index=drop_index)
            Y = Y.drop(index=drop_index)

        return X, Y, drop_index

    # remove input variables according to n_nearest (<= 11)
    # About vdep
    # if not setting args.predict_actions, the following features should be removed.
    # [0:23] 'type_id_a0' and 23 actions
    # [28:33] 'result_id_a0', 'result_fail_a0', 'result_success_a0', 'result_offside_a0', 'result_owngoal_a0'
    # [49:52] 'away_team_a0', 'regain_a0', 'penetration_a0'

    # About vaep, the following features also should be removed if so.
    # [28:33] 'result_id_a0', 'result_fail_a0', 'result_success_a0', 'result_offside_a0', 'result_owngoal_a0'
    if args.n_nearest == 0:
        if args.predict_actions:
            Xcols_vaep = Xcols[:52]
            Xcols = Xcols[:52]
        else:
            Xcols_vaep = Xcols[:28] + Xcols[33:49]
            Xcols = Xcols[24:28] + Xcols[31:49]
    else:
        if args.predict_actions:
            Xcols_vaep = Xcols[: 52 + 8 * args.n_nearest]
            Xcols = Xcols[: 52 + 8 * args.n_nearest]
        else:
            Xcols_vaep = Xcols[:28] + Xcols[33:49] + Xcols[52 : 52 + 8 * args.n_nearest]
            Xcols = Xcols[24:28] + Xcols[33:49] + Xcols[52 : 52 + 8 * args.n_nearest]

    X, Y, drop_index = getXY(traingames, Xcols)
    trainX_vaep, _, _ = getXY(traingames, Xcols_vaep)
    print("X:", list(X.columns), ", length: ", str(len(X.columns)))
    print("Y:", list(Y.columns))

    testX, testY, _ = getXY(testgames, Xcols)
    testX_vaep, _, _ = getXY(testgames, Xcols_vaep)

    # Train classifiers
    import xgboost

    Y_hat = pd.DataFrame()
    if args.grid_search:
        from sklearn.model_selection import GridSearchCV

        param_grid = {
            "max_depth": [3, 5, 7],  #
        }

    models = {}
    n_jobs = 1

    from sklearn.model_selection import train_test_split

    for col in list(Y.columns):
        print(
            f"training {col} positive: {Y[col].sum()} negative: {len(Y[col]) - Y[col].sum()}"
        )
        model = xgboost.XGBClassifier(
            n_estimators=50,
            max_depth=5,
            n_jobs=-3,
            verbosity=0,
            random_state=args.seed,
            objective="binary:logistic",
            eval_metric="logloss",
            use_label_encoder=False,
        )

        if args.grid_search:
            xgb_scikit = xgboost.XGBClassifier(  # verbose=0,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=args.seed,
                n_jobs=n_jobs,
                use_label_encoder=False,
            )

            model = GridSearchCV(
                xgb_scikit,
                param_grid,
                verbose=3,
                cv=3,
                n_jobs=n_jobs,
            )

        if col == "scores" or col == "concedes":
            if args.split_train_eval:
                split_X, split_X_eval, split_Y, split_Y_eval = train_test_split(
                    trainX_vaep, Y[col], random_state=args.seed, stratify=Y[col]
                )
                model.fit(
                    trainX_vaep,
                    Y[col],
                    eval_set=[(split_X, split_Y), (split_X_eval, split_Y_eval)],
                    verbose=True,
                )
            else:
                model.fit(trainX_vaep, Y[col])
                # setting directories
                feature_importances = figuredir + "/feature_importances"
                os.makedirs(feature_importances, exist_ok=True)
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(trainX_vaep)
                # shap_summary
                shap.summary_plot(
                    shap_values=shap_values,
                    features=trainX_vaep,
                    feature_names=trainX_vaep.columns,
                    show=False,
                )
                plt.savefig(
                    os.path.join(feature_importances, f"model_{col}_summary.png")
                )
                plt.clf()
                plt.close()
        elif col == "gains" or col == "effective_attack":
            if args.split_train_eval:
                split_X, split_X_eval, split_Y, split_Y_eval = train_test_split(
                    X, Y[col], random_state=args.seed, stratify=Y[col]
                )
                model.fit(
                    X,
                    Y[col],
                    eval_set=[(split_X, split_Y), (split_X_eval, split_Y_eval)],
                    verbose=True,
                )
            else:
                model.fit(X, Y[col])
                # setting directories
                feature_importances = figuredir + "/feature_importances"
                os.makedirs(feature_importances, exist_ok=True)
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
                # shap_summary
                shap.summary_plot(
                    shap_values=shap_values,
                    features=X,
                    feature_names=X.columns,
                    show=False,
                )
                plt.savefig(
                    os.path.join(feature_importances, f"model_{col}_summary.png")
                )
                plt.clf()
                plt.close()
        models[col] = model

    with open(os.path.join(datafolder, "VDEPmodel_" + model_str + ".pkl"), "wb") as f:
        pickle.dump(models, f, protocol=4)
    print(os.path.join(datafolder, "VDEPmodel_" + model_str + ".pkl") + " is saved")

    # Evaluate the model
    from sklearn.metrics import (
        brier_score_loss,
        confusion_matrix,
        f1_score,
        log_loss,
        roc_auc_score,
    )

    def evaluate_metrics(y: pd.Series, y_hat: pd.Series) -> float:
        """
        Parameters
        y: pd.Series, ground truth (correct) target values.
        y_hat: pd.Series, estimated targets as returned by a classifier.
        ---
        Return
        f1_score(y, y_hat_bi): float, f1 score of the positive class in binary classification.
        """
        p = sum(y) / len(y)
        base = [p] * len(y)
        brier = brier_score_loss(y, y_hat)
        print(f"  Brier score: {brier:.5f}, {(brier / brier_score_loss(y, base)):.5f}")
        ll = log_loss(y, y_hat)
        print(f"  log loss score: {ll:.5f}, {(ll / log_loss(y, base)):.5f}")
        print(f"  ROC AUC: {roc_auc_score(y, y_hat):.5f}")

        y_hat_bi = y_hat.round()
        print(f"F1 score:{f1_score(y, y_hat_bi)}")
        print(confusion_matrix(y, y_hat_bi))
        return f1_score(y, y_hat_bi)

    f1scores = np.empty(4)
    for i, col in enumerate(testY.columns):
        if col == "scores" or col == "concedes":
            Y_hat[col] = [p[1] for p in models[col].predict_proba(testX_vaep)]
        else:
            Y_hat[col] = [p[1] for p in models[col].predict_proba(testX)]
        print(
            f"### Y: {col} ###\npositive: {testY[col].sum()} negative: {len(testY[col]) - testY[col].sum()}"
        )
        f1scores[i] = evaluate_metrics(testY[col], Y_hat[col])

    # Save predictions
    A = []
    for game_id in tqdm.tqdm(games.game_id, "Loading game ids"):
        Ai = pd.read_hdf(spadl_h5, f"actions/game_{game_id}")
        A.append(Ai[["game_id"]])
    A = pd.concat(A)
    A = A.reset_index(drop=True).drop(index=drop_index)
    A = A.reset_index(drop=True)

    # concatenate action game id rows with predictions and save per game
    grouped_predictions = pd.concat([A, Y_hat], axis=1).groupby("game_id")
    for k, df in tqdm.tqdm(grouped_predictions, desc="Saving predictions per game"):
        df = df.reset_index(drop=True)
        df[Y_hat.columns].to_hdf(predictions_h5, f"game_{int(k)}")

    # constants for VDEP
    C_vdep_v0 = Y["gains"].sum() / Y["effective_attack"].sum()
    # gain_concedes = sum((Y['gains']==True) & (Y['concedes']==True))/len(Y['gains'])
    # attacked_scores = sum((Y['effective_attack']==True) & (Y['scores']==True))/len(Y['gains'])
    # C_vdep_v1 = attacked_scores/gain_concedes

    # save static variables
    static = [args, C_vdep_v0, f1scores, drop_index]
    with open(os.path.join(datafolder, "static_" + model_str + ".pkl"), "wb") as f:
        pickle.dump(static, f, protocol=4)


"""4. compute gvdep values and top players"""
# Select data
with pd.HDFStore(spadl_h5, pickle_protocol=4) as spadlstore:
    games = (
        spadlstore["games"]
        .merge(spadlstore["competitions"], how="left")
        .merge(spadlstore["teams"].add_prefix("home_"), how="left")
        .merge(spadlstore["teams"].add_prefix("away_"), how="left")
    )
    players = spadlstore["players"]
    teams = spadlstore["teams"]
print("nb of games:", len(games))

with open(os.path.join(datafolder, "static_" + model_str + ".pkl"), "rb") as f:
    _, C_vdep_v0, _, drop_index = pickle.load(f)

# Compute each values
drop_index = np.asarray(drop_index)
A = []
for game in tqdm.tqdm(list(games.itertuples()), desc="Rating actions"):
    actions = pd.read_hdf(spadl_h5, f"actions/game_{game.game_id}")
    actions = (
        spadl.add_names_360(actions)
        .merge(players, how="left")
        .merge(teams, how="left")
        .sort_values(["game_id", "period_id", "action_id"])
        .reset_index(drop=True)
    )
    preds = pd.read_hdf(predictions_h5, f"game_{game.game_id}")
    features = pd.read_hdf(features_h5, f"game_{game.game_id}")
    drop_array = np.asarray([index for index in drop_index if index < len(actions)])
    vaep, vdep, ids = vdepformula.value(actions, features, preds, drop_array, C_vdep_v0)
    A.append(pd.concat([actions, preds, vaep, vdep, ids], axis=1))
    drop_index = np.asarray(sorted(list(set(drop_index) - set(drop_array)))) - len(
        actions
    )
A = (
    pd.concat(A)
    .sort_values(["game_id", "period_id", "time_seconds"])
    .reset_index(drop=True)
)

# Gvdep calculated by vaep, gain_value and attacked_value
num_gains = (A["gains_id"]).sum()
num_attacked = (A["effective_id"]).sum()
weight_gains = (A["vaep_value"] * A["gains_id"]).sum() / num_gains
weight_attacked = ((-A["vaep_value"]) * (A["effective_id"])).sum() / num_attacked
A["g_vdep_value"] = (
    weight_gains * A["gain_value"] - weight_attacked * A["attacked_value"]
)

# (optional) inspect country's top 5 most valuable non-shot actions
if args.test:
    import matplotsoccer

    def get_time(period_id: int, time_seconds: int) -> str:
        """
        Parameters
        period_id: int, it takes values from 1 to 5.
        time_seconds: int, seconds per half-time.
        ---
        Return
        f"{m}m{s}s": str, "m" and "s" are minutes and seconds respectively.
        """
        m = int((period_id - 1) * 45 + time_seconds // 60)
        s = time_seconds % 60
        if s == int(s):
            s = int(s)
        return f"{m}m{s}s"

    sorted_A = A.sort_values("g_vdep_value", ascending=False)
    sorted_A = sorted_A[sorted_A.team_name.str.contains(args.teamView)]
    sorted_A = sorted_A[(~sorted_A.type_name.str.contains("shot"))]  # eliminate shots

    for j in range(10):
        row = list(sorted_A[j : j + 1].itertuples())[0]
        i = row.Index
        a = A[i - 2 : i + 1].copy()

        a["player_name"] = a[["nickname", "player_name"]].apply(
            lambda x: x[0] if x[0] else x[1], axis=1
        )

        g = list(games[games.game_id == a.game_id.values[0]].itertuples())[0]
        game_info = f"{g.game_date} {g.home_team_name} {g.home_score}-{g.away_score} {g.away_team_name}"
        minute = int((row.period_id - 1) * 45 + row.time_seconds // 60)

        a["gain_value"] = a.gain_value.apply(lambda x: "%.3f" % x)
        a["attacked_value"] = a.attacked_value.apply(lambda x: "%.3f" % x)
        a["g_vdep_value"] = a.g_vdep_value.apply(lambda x: "%.3f" % x)
        a["time"] = a[["period_id", "time_seconds"]].apply(
            lambda x: get_time(*x), axis=1
        )
        cols = [
            "freeze_frame_360",
            "visible_area_360",
            "result_name",
            "start_x",
            "start_y",
            "end_x",
            "end_y",
            "time",
            "type_name",
            "player_name",
            "team_name",
            "gain_value",
            "attacked_value",
            "g_vdep_value",
        ]
        a = a.reindex(columns=cols)

        # make folders to store example figures
        os.makedirs(figuredir + "/example", exist_ok=True)
        figure_eg_dir = os.path.join(
            figuredir + "/example",
            f"{g.home_team_name} {g.home_score}-{g.away_score} {g.away_team_name}/",
        )  # data_id
        os.makedirs(figure_eg_dir, exist_ok=True)
        filename_fig = f"{minute}' {row.type_name} {row.player_name}"

        # Creating example figures.
        ref_team = args.teamView
        if args.game == "euro2022":
            if ref_team == "Finland":
                ref_team = "WNT Finland"
            elif ref_team == "Northern Ireland":
                continue
            else:
                ref_team += " Women's"
        matplotsoccer.actions_individual(
            data=a,
            teamView=ref_team,
            label=a[cols[7:]],
            labeltitle=cols[7:],
            show=False,
            visible_area=a["visible_area_360"],
            save_dir=figure_eg_dir,
            filename=filename_fig,
        )
        j += 1


"""Analyze team defense"""
# To consider teams advancing to the konckout stage, select the teams
if args.game == "euro2020":
    # Group Stage and Round of 16
    analysis_games_list = []
    for id, stage in zip(games["game_id"], games["competition_stage"]):
        if (stage == "Group Stage") | (stage == "Round of 16"):
            analysis_games_list.append(id)
    A = A[A["game_id"].isin(analysis_games_list)]
    knockout_teams_list = (
        games[games["competition_stage"] == "Round of 16"]
        .filter(like="team_name", axis=1)
        .values.ravel()
        .tolist()
    )

elif args.game == "euro2022":
    # Group Stage and Semi-finals
    analysis_games_list = []
    for id, stage in zip(games["game_id"], games["competition_stage"]):
        if (stage == "Group Stage") | (stage == "Quarter-finals"):
            analysis_games_list.append(id)
    A = A[A["game_id"].isin(analysis_games_list)]
    knockout_teams_list = (
        games[games["competition_stage"] == "Quarter-finals"]
        .filter(like="team_name", axis=1)
        .values.ravel()
        .tolist()
    )

analysis_games_df = games[(games["game_id"].isin(analysis_games_list))]
# counting concedes by each team that reached the knockout stage
concedes_array = np.empty(len(knockout_teams_list))
for i, team in enumerate(knockout_teams_list):
    team_homeside = analysis_games_df[analysis_games_df["home_team_name"] == team]
    team_awayside = analysis_games_df[analysis_games_df["away_team_name"] == team]
    concede = team_homeside["away_score"].sum() + team_awayside["home_score"].sum()
    concedes_array[i] = concede
concedes_ser = pd.Series(
    data=concedes_array,
    index=knockout_teams_list,
    name="concedes",
)

team_values_df = (
    A[["defense_team", "gain_value", "attacked_value", "g_vdep_value"]]
    .groupby(["defense_team"])
    .mean()
    .reset_index()
)

result_df = team_values_df[
    team_values_df["defense_team"].isin(knockout_teams_list).values
].reset_index(drop=True)
result_df = pd.merge(result_df, concedes_ser, left_on="defense_team", right_index=True)
result_df = result_df.reindex(
    columns=[
        "defense_team",
        "concedes",
        "gain_value",
        "attacked_value",
        "g_vdep_value",
    ]
)

# Plot figures about defense values of the teams in Euro 2020 or 2022
for v in itertools.combinations(result_df.columns.values.tolist()[1:], 2):
    # columns
    x = v[0]
    y = v[1]

    # Calculate PCC amd P_value
    from scipy.stats import pearsonr

    r, p_value = pearsonr(result_df[x].values, result_df[y].values)
    print(f"### {x} - {y} ###")
    print(f"r : {r:.5f}")
    print(f"P-value : {p_value:.5f}")

    # Plot a figure
    fig_team = plt.figure(figsize=(8, 6))
    ax_team = fig_team.add_subplot(111)
    ax_team.scatter(result_df[x], result_df[y], c="blue")
    ax_team.set_xlabel(result_df[x].name, size=16)
    ax_team.set_ylabel(result_df[y].name, size=16)
    text_team = [
        ax_team.text(
            result_df.at[index, x],
            result_df.at[index, y],
            result_df.at[index, "defense_team"],
            fontsize=12,
            color="black",
            zorder=500,
        )
        for index in result_df.index
    ]
    adjust_text(text_team)
    ax_team.tick_params(
        axis="both",
        labelsize=12,
        grid_color="lightgray",
        grid_alpha=0.5,
    )
    ax_team.grid()
    ax_team.axvline(result_df[x].mean(), 0, 1, c="silver")
    ax_team.axhline(result_df[y].mean(), 0, 1, c="silver")
    plt.tight_layout()
    fig_team.savefig(os.path.join(figuredir, f"teams_{x}_{y}.png"))
    print(os.path.join(figuredir, f"teams_{x}_{y}.png") + " is saved")
    plt.clf()
    plt.close()

# Plot bars about the number of players in each freeze_frame_360
nan_array = np.empty(len(A["freeze_frame_360"]))
for i, coordinates in enumerate(A["freeze_frame_360"]):
    count_nan = int(np.count_nonzero(np.isnan(np.asarray(coordinates))))
    nan_array[i] = count_nan
nan_array = nan_array / 2
nan_array = 22 - nan_array

fig_ff360 = plt.figure(figsize=(8, 6))
ax_ff360 = fig_ff360.add_subplot(111, xticks=range(0, 23, 1))
ax_ff360.set_ylim(0, 32000)
n, bins, _ = ax_ff360.hist(nan_array, bins=range(0, 24, 1), align="left")

# Display the degrees at the head of the bar
text_degree = [
    ax_ff360.text(
        bin + 0.1,  # Adjustments
        num + 120,  # Adjustments
        int(num),
        fontsize=10,
        rotation=70,
        horizontalalignment="center",
    )
    for num, bin in zip(n, bins)
    if num
]
fig_ff360.savefig(os.path.join(figuredir, f"the_number_of_nan_in_{args.game}.png"))
print(os.path.join(figuredir, f"the_number_of_nan_in_{args.game}.png"))
print(time.time() - start)

pdb.set_trace()
