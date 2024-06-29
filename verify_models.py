import argparse
import itertools
import os
import pdb
import pickle
import random
import time
import warnings

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import shap
import tqdm
from adjustText import adjust_text

import socceraction.spadl as spadl
import socceraction.vdep.formula as vdepformula

pd.set_option("display.max_columns", None)
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings(action="ignore", message="credentials were not supplied. open data access only")

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="statsbomb")
parser.add_argument("--game", type=str, default="all")
parser.add_argument("--feature", type=str, default="all")
parser.add_argument(
    "--date_opendata", type=int, default=20231002, required=True, 
    help="Since statsbomb data is constantly being updated, it is important to indicate in advance when the data was downloaded."
    )
parser.add_argument(
    "--date_experiment", type=int, default=0, required=True,
    help="The date of the experiment is important because the data is constantly being updated."
    )
parser.add_argument("--no_games", type=int, default=0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_nearest", type=int, default=11)
parser.add_argument("--numProcess", type=int, default=16)
parser.add_argument("--skip_convert_rawdata", action="store_true")
parser.add_argument("--skip_preprocess", action="store_true")
parser.add_argument("--predict_actions", action="store_true")
parser.add_argument(
    "--model", type=str, default="xgboost", required=True,
    help="Please select the model to be used for training. Now the options is only 'xgboost'.",
    )
parser.add_argument("--grid_search", action="store_true")
parser.add_argument("--skip_train", action="store_true")
parser.add_argument("--test", action="store_true")
parser.add_argument("--teamView", type=str, default="", help="Please set the name of the country, not the name of the team",)
parser.add_argument("--pickle", type=int, default=4)
args, _ = parser.parse_known_args()

pickle.HIGHEST_PROTOCOL = args.pickle  # 4 is the highest in an older version of pickle
numProcess = args.numProcess
os.environ["OMP_NUM_THREADS"] = str(numProcess)

start = time.time()

random.seed(args.seed)
np.random.seed(args.seed)

datafolder = f"../GVDEP_data/data-{args.data}/{args.game}/{args.date_experiment}"
os.makedirs(datafolder, exist_ok=True)

# Preparation
if "euro" in args.game or "wc" in args.game:
    datafolder = "../GVDEP_data/data-statsbomb/" + args.game
    os.makedirs(datafolder, exist_ok=True)
    spadl_h5 = os.path.join(datafolder, "spadl-statsbomb.h5")

features_h5 = os.path.join(datafolder, "features.h5")
labels_h5 = os.path.join(datafolder, "labels.h5")

nb_prev_actions = 1

# Cross Validation by n_nearest
random.seed(args.seed)
np.random.seed(args.seed)
if args.predict_actions:
    modelfolder = datafolder + "/vaep_framework/models"
    os.makedirs(modelfolder, exist_ok=True)
    predictions_h5 = os.path.join(datafolder + "/vaep_framework", "predictions.h5")
else:
    modelfolder = datafolder + "/models"
    os.makedirs(modelfolder, exist_ok=True)
    predictions_h5 = os.path.join(datafolder, "predictions.h5")
games = pd.read_hdf(spadl_h5, "games")
print("nb of games:", len(games))

# note: only for the purpose of this example and due to the small dataset,
# we use the same data for training and evaluation
if args.calculate_f1scores:
    for n_nearest in range(12):
        for cv in range(1, 11, 1):
            k_fold = 10
            model_str = f"_CV_{cv}_{k_fold}_all"

            # Split games by k-fold for cross validation
            split_traineval = int(len(games) / k_fold)
            start_point = (cv - 1) * split_traineval
            end_point = start_point + split_traineval
            if "euro" in args.game:
                # Due to the number of games in the competitions, 46:5 or 45:6
                if cv == k_fold:
                    split_traineval += 1
                    start_point = len(games) - (k_fold - cv + 1) * split_traineval
                    end_point = start_point + split_traineval
            elif "wc" in args.game:
                # Due to the number of games in data, 6:58 or 7:57
                if cv > 6:
                    split_traineval += 1
                    start_point = len(games) - (k_fold - cv + 1) * split_traineval
                    end_point = start_point + split_traineval
            evalgames_k = games[start_point:end_point]
            traingames_k = games.drop(games.index[start_point:end_point])
            print("train/eval of games:", len(traingames_k), "/", len(evalgames_k))

            if n_nearest <= 11:
                model_str += "_" + str(n_nearest) + "_nearest"

            xfns = [
                fs.actiontype,
                fs.actiontype_onehot,
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
                fs.time_delta,
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
                # generate the columns of the selected feature
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

                # Drop the rows not including players' coodinates.
                # Hence, consider whether the rows' "dist_at0_a0" and "dist_df0_a0" are NaN.
                if "dist_at0_a0" in X.columns.values and "dist_df0_a0" in X.columns.values:
                    at_nan = X[X["dist_at0_a0"].isnull()]
                    at_df_nan = at_nan[at_nan["dist_df0_a0"].isnull()]
                    drop_index = at_df_nan.index.values.tolist()
                    X = X.drop(index=drop_index).reset_index(drop=True)
                    Y = Y.drop(index=drop_index).reset_index(drop=True)
                else:
                    drop_index = []

                return X, Y, drop_index

            # remove input variables according to n_nearest (<= 11)
            # About vdep
            # if not setting args.predict_actions, the following features should be removed.
            # [0:23] 'type_id_a0' and 23 actions should also be removed
            # [28:33] 'result_id_a0', 'result_fail_a0', 'result_success_a0', 'result_offside_a0', 'result_owngoal_a0'
            # [49:52] 'away_team_a0', 'regain_a0', 'penetration_a0'

            # About vaep, the following features also should be removed if so.
            # [28:33] 'result_id_a0', 'result_fail_a0', 'result_success_a0', 'result_offside_a0', 'result_owngoal_a0'
            if n_nearest == 0:
                if args.predict_actions:
                    Xcols_vaep = Xcols[:52]
                    Xcols = Xcols[:52]
                else:
                    Xcols_vaep = Xcols[:28] + Xcols[33:49]
                    Xcols = Xcols[24:28] + Xcols[31:49]
            else:
                if args.predict_actions:
                    Xcols_vaep = Xcols[: 52 + 8 * n_nearest]
                    Xcols = Xcols[: 52 + 8 * n_nearest]
                else:
                    Xcols_vaep = Xcols[:28] + Xcols[33:49] + Xcols[52 : 52 + 8 * n_nearest]
                    Xcols = Xcols[24:28] + Xcols[33:49] + Xcols[52 : 52 + 8 * n_nearest]

            #  train classifiers F(X) = Y (for CV and Test)
            import xgboost

            Y_hat = pd.DataFrame()
            if args.grid_search:
                from sklearn.model_selection import GridSearchCV

                param_grid = {
                    "max_depth": [3, 5, 7],  #
                }

            models = {}
            n_jobs = 1

            trainX, trainY, drop_index = getXY(traingames_k, Xcols)
            evalX, evalY, _ = getXY(evalgames_k, Xcols)
            trainX_vaep, _, _ = getXY(traingames_k, Xcols_vaep)
            evalX_vaep, _, _ = getXY(evalgames_k, Xcols_vaep)
            print(
                "trainX_vaep:",
                trainX_vaep.columns.values.tolist(),
                ", length: ",
                str(len(trainX_vaep.columns.values.tolist())),
            )
            print(
                "trainX:",
                trainX.columns.values.tolist(),
                ", length: ",
                str(len(trainX.columns.values.tolist())),
            )

            f1scores = np.empty(len(trainY.columns))
            for col in list(trainY.columns):
                print(f"training {col}\n positive: {trainY[col].sum()} negative: {len(trainY[col]) - trainY[col].sum()}")

                model = xgboost.XGBClassifier(
                    n_estimators=50,
                    max_depth=5,
                    n_jobs=-3,
                    verbosity=1,
                    random_state=args.seed,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    use_label_encoder=False,
                )

                if args.grid_search:
                    xgb_scikit = xgboost.XGBClassifier(
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
                    model.fit(trainX_vaep, trainY[col])
                elif col == "gains" or col == "effective_attack":
                    model.fit(trainX, trainY[col])
                models[col] = model

            vdep_model_pkl = os.path.join(modelfolder, "VDEPmodel" + model_str + ".pkl")
            with open(vdep_model_pkl, "wb") as f:
                pickle.dump(models, f, protocol=4)
                print(vdep_model_pkl + " is saved")

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

            for i, col in enumerate(evalY.columns):
                if col == "scores" or col == "concedes":
                    Y_hat[col] = [p[1] for p in models[col].predict_proba(evalX_vaep)]
                elif col == "gains" or col == "effective_attack":
                    Y_hat[col] = [p[1] for p in models[col].predict_proba(evalX)]
                print(f"### Y: {col} ###\n positive: {evalY[col].sum()} negative: {len(evalY[col]) - evalY[col].sum()}")
                f1scores[i] = evaluate_metrics(evalY[col], Y_hat[col])

            A = []
            for game_id in tqdm.tqdm(games.game_id, "Loading game ids"):
                Ai = pd.read_hdf(spadl_h5, f"actions/game_{game_id}")
                A.append(Ai[["game_id"]])
            A = pd.concat(A).reset_index(drop=True)
            A = A.drop(index=drop_index).reset_index(drop=True)

            # concatenate action game id rows with predictions and save per game
            grouped_predictions = pd.concat([A, Y_hat], axis=1).groupby("game_id")
            for k, df in tqdm.tqdm(grouped_predictions, desc="Saving predictions per game"):
                df = df.reset_index(drop=True)
                df[Y_hat.columns].to_hdf(predictions_h5, f"game_{int(k)}")

            # save static variables
            static_pkl = os.path.join(modelfolder, "static" + model_str + ".pkl")
            static = [args, f1scores, drop_index]
            with open(static_pkl, "wb") as f:
                pickle.dump(static, f, protocol=4)
                print(static_pkl + " is saved")

            print(f"Evaluation time of {model_str} : ", time.time() - start)

# Show the graphs about F1scores of each probabilities
if args.show_f1scores:
    if args.predict_actions:
        figuredir = datafolder + "/vaep_framework/figures"
        os.makedirs(figuredir, exist_ok=True)
    else:
        figuredir = datafolder + "/figures"
        os.makedirs(figuredir, exist_ok=True)

    cols = ["scores", "concedes", "gains", "effective_attack", "n_nearest"]

    # Make the dataframe about F1-scores grouped by 'n_nearest'
    for n_nearest in range(12):
        for cv in range(1, 11, 1):
            model_str = f"static_CV_{cv}_10_all_{n_nearest}_nearest"
            with open(os.path.join(modelfolder, model_str + ".pkl"), "rb") as f:
                args, f1scores, drop_index = pickle.load(f)
                print(f1scores)
            if cv == 1:
                f1cv_array = np.insert(f1scores, 4, n_nearest)
            else:
                f1cv_array = np.vstack([f1cv_array, np.insert(f1scores, 4, n_nearest)])

        if n_nearest == 0:
            f1cv_arrays = f1cv_array
        else:
            f1cv_arrays = np.vstack([f1cv_arrays, f1cv_array])

    df_f1cv = pd.DataFrame(f1cv_arrays, columns=cols)
    df_f1 = df_f1cv.groupby("n_nearest")

    # Show figures by Prediction of the Probabilities
    subplots = ["(a)", "(b)", "(c)", "(d)"]
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(16, 9),
        sharex="col",
        sharey=True,
    )
    # plt.subplots_adjust(hspace=0.2)
    for prob in range(4):
        ax = axes[prob // 2, prob % 2]
        # ax.set_title(f'{cols[prob]}')
        ax.boxplot(
            [df_f1.get_group(n).iloc[:, prob] for n in range(12)],
            labels=[f"{n}" for n in range(12)],
            showmeans=True,
            zorder=10,
            patch_artist=True,
            boxprops=dict(facecolor="white"),
            medianprops=dict(color="blue"),
            flierprops=dict(marker="x", markeredgecolor="black"),
        )
        ax.set_title(subplots[prob], fontsize=24, loc="left")

        # set details
        ax.set_ylim(0, 0.9)
        ax.tick_params(axis="both", labelsize=20)
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.1))

    fig.supxlabel("the number of nearest attacker/defender to the ball (n_nearest)", fontsize=32)
    fig.supylabel("F1-scores", fontsize=32)
    plt.tight_layout()
    fig.savefig(os.path.join(figuredir, "prob_all.png"))
    print(os.path.join(figuredir, "prob_all.png") + " is saved")
    plt.show()

pdb.set_trace()
