import os
import pickle
import tqdm

import numpy as np
import pandas as pd
import shap
import xgboost
from matplotlib import pyplot as plt
from sklearn.metrics import brier_score_loss, confusion_matrix, f1_score, log_loss, roc_auc_score

import socceraction.vdep.features as fs


NB_PREV_ACTIONS = 1


def set_conditions(args, games):
    if args.no_games < 10 and args.no_games > 0:
        traingames = games[: args.no_games - 1]
        testgames = games[args.no_games - 1 : args.no_games]
        model_str = str(args.no_games) + "games"
    elif args.no_games >= 10 or args.no_games == 0:
        split_traintest = int(9 * len(games) / 10)
        traingames = games[:split_traintest]
        testgames = games[split_traintest:]
        model_str = str(args.no_games) + "games" if args.no_games >= 10 else "_traindata_all"
    elif args.no_games == -1:
        traingames = games
        testgames = games
        model_str = "all"

    if args.n_nearest <= int(fs.NUM_PLAYERS / 2):
        model_str += "_" + str(args.n_nearest) + "_nearest"

    return traingames, testgames, model_str


def getXY(games, features_h5, labels_h5, Xcols, vaep=False):
    """
    Parameters
    ----------
    games: pd.DataFrame, 
        the games that you want to select.
    features_h5: str, 
        the path of the h5 file that contains the features.
    labels_h5: str, 
        the path of the h5 file that contains the labels.
    Xcols: list, 
        the columns of the features that you want to select.


    Returns
    -------
    X: pd.DataFrame, 
        the selected features.
    Y: pd.DataFrame, 
        the selected labels.
    drop_index: list, 
        the index of the rows that are dropped.
    """
    # generate the columns of the selected features and labels
    X = []
    for game_id in tqdm.tqdm(
        games.game_id, desc="Selecting features"
        ):
        Xi = pd.read_hdf(features_h5, f"game_{game_id}")
        X.append(Xi[Xcols])
    X = pd.concat(X).reset_index(drop=True)

    if vaep:
        Ycols = ["scores", "concedes"]
    else:
        Ycols = ["gains", "effective_attack"]
    Y = []
    for game_id in tqdm.tqdm(
        games.game_id, desc="Selecting label"
        ):
        Yi = pd.read_hdf(labels_h5, f"game_{game_id}")
        Y.append(Yi[Ycols])
    Y = pd.concat(Y).reset_index(drop=True)

    return X, Y


def set_xcols(
    nb_prev_actions=1, 
    use_location=False, 
    use_polar=False
    ):
    X_FNS = [
        fs.actiontype,
        fs.actiontype_onehot,
        fs.bodypart_onehot,
        fs.movement,
        fs.space_delta,
        fs.team,
        fs.time_delta,
    ]
    if use_location:
        X_FNS += [fs.startlocation, fs.endlocation]
    if use_polar:
        X_FNS += [fs.startpolar, fs.endpolar]

    # Player features
    X_FNS += [fs.player_loc_dist]

    return fs.feature_column_names(X_FNS, nb_prev_actions)


def choose_input_variables(args, Xcols):
    """
    Choose input variables according to n_nearest (<= 11).
    The features about results should be removed 
    regarding vaep and vdep if not setting args.predict_actions.

    'result_id', 'result_fail', 'result_success', 
    'result_offside', 'result_owngoal'
    'goalscore_team', 'goalscore_opponent', 'goalscore_diff',

    However, the following features should also be removed 
    regarding vdep.

    'type_id' and 23 actions

    Parameters
    ----------
    args: argparse.Namespace, 
        the arguments.
    Xcols: list, 
        the columns of the features.

    Returns
    -------
    Xcols: list, 
        the columns of the features after removing.
    """
    # Decide the number of unused players features
    num_unused_pl_team = int(fs.NUM_PLAYERS / 2 - args.n_nearest)
    num_unused_pl_team_feat = num_unused_pl_team * 4 * 2

    Xcols = Xcols[: len(Xcols) - num_unused_pl_team_feat]

    # Remove the features
    remove_cols = ["result", "goalscore"]
    Xcols_vaep = [
        s for s in Xcols if not any(word in s for word in remove_cols)
        ]
    if args.predict_actions:
        Xcols_vdep = Xcols_vaep
    else:
        Xcols_vdep = [
            col for col in Xcols_vaep if "type" not in col
            ]

    return Xcols_vdep, Xcols_vaep


def create_model(args):
    n_jobs = 1
    if args.model == "xgboost":
        if not args.grid_search:
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
        elif args.grid_search:
            from sklearn.model_selection import GridSearchCV
            param_grid = {"max_depth": [3, 5, 7]}

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
    else:
        raise ValueError(f"Unknown model: {args.model}")

    return model


def illustrate_shap(model, trainX):
    explainer = shap.TreeExplainer(model)
    # shap_summary
    shap.summary_plot(
        shap_values=explainer.shap_values(trainX),
        features=trainX,
        feature_names=trainX.columns,
        show=False,
    )


def train(args, trainX, trainY, do="evaluation"):
    figuredir_features = (
        args.figuredir
        + f"{do}_feature_importances"
        )
    os.makedirs(figuredir_features, exist_ok=True)
    print(
        f"trainX: {list(trainX.columns)}\n"
        + f"length: {len(trainX.columns)}\n"
        + f"trainY: {list(trainY.columns)}"
        )

    # Train classifiers
    models = {}
    for col in list(trainY.columns):
        print(
            f"training {col} "
            + f"positive: {trainY[col].sum()} "
            + f"negative: {len(trainY[col]) - trainY[col].sum()}"
            )
        model = create_model(args)
        model.fit(trainX, trainY[col])
        illustrate_shap(model, trainX)
        plt.savefig(
            os.path.join(
                figuredir_features, 
                f"{args.model_str}_{col}_summary.png"
                )
            )
        plt.clf()
        plt.close()
        models[col] = model

    return models


def inference_for_evaluation(models, testX, testY):
    Y_hat = pd.DataFrame()
    f1scores = np.empty(len(testY.columns))
    for i, col in enumerate(testY.columns):
        Y_hat[col] = [
            probability[1] for probability in models[col].predict_proba(testX)
            ]
        print(
            f"### Y: {col} ### "
            + f"positive: {testY[col].sum()} "
            + f"negative: {len(testY[col]) - testY[col].sum()}"
            )
        _, _, _, f1scores[i] = evaluate_metrics(
            testY[col], Y_hat[col]
            )

    return Y_hat, f1scores


def inference_for_verification(models, testX, testY):
    Y_hat = pd.DataFrame()
    briers = np.empty(len(testY.columns))
    lls = np.empty(len(testY.columns))
    roc_aucs = np.empty(len(testY.columns))
    f1scores = np.empty(len(testY.columns))
    for i, col in enumerate(testY.columns):
        Y_hat[col] = [
            probability[1] for probability in models[col].predict_proba(testX)
            ]
        print(
            f"### Y: {col} ### "
            + f"positive: {testY[col].sum()} "
            + f"negative: {len(testY[col]) - testY[col].sum()}"
            )
        briers[i], lls[i], roc_aucs[i], f1scores[i] = evaluate_metrics(
            testY[col], Y_hat[col]
            )

    return Y_hat, briers, lls, roc_aucs, f1scores


def save_predictions(predictions_h5, A, Y_hat):
    # concatenate action game id rows with predictions and save per game
    grouped_predictions = pd.concat([A, Y_hat], axis=1).groupby("game_id")
    for k, df in tqdm.tqdm(
        grouped_predictions,
        desc="Saving predictions per game"
        ):
        df = df.reset_index(drop=True)
        df[Y_hat.columns].to_hdf(predictions_h5, f"game_{int(k)}")

    
def save_static_variables_for_evaluation(
        args, datafolder, model_str, testY, f1scores
        ):
    C_vdep_v0 = testY["gains"].sum() / testY["effective_attack"].sum()
    static = [args, C_vdep_v0, f1scores, ]
    with open(
        os.path.join(datafolder, "static_" + model_str + ".pkl"),
        "wb") as f:
        pickle.dump(static, f, protocol=4)


def save_static_variables_for_verification(
        args, briers, lls, roc_aucs, f1scores
        ):
    static = {
        "args": args,
        "briers": briers,
        "lls": lls,
        "roc_aucs": roc_aucs,
        "f1scores": f1scores,
    }
    with open(
        os.path.join(args.modelfolder, "static_" + args.model_str + ".pkl"),
        "wb") as f:
        pickle.dump(static, f, protocol=4)


# Evaluate the model
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
    print(
        f"  Brier score: {brier:.5f}, "
          + f"{(brier / brier_score_loss(y, base)):.5f}"
          )
    
    ll = log_loss(y, y_hat)
    roc_auc = roc_auc_score(y, y_hat)
    print(
        f"  log loss score: {ll:.5f}, "
        + f"{(ll / log_loss(y, base)):.5f}"
        )
    print(f"  ROC AUC: {roc_auc:.5f}")

    y_hat_bi = y_hat.round()
    f1score = f1_score(y, y_hat_bi)
    print(f" F1 score: {f1score}")
    print(confusion_matrix(y, y_hat_bi))

    return brier, ll, roc_auc, f1score


def save_predictions_for_evaluation(
        args, games, datafolder,
        spadl_h5, features_h5, labels_h5, predictions_h5,
        use_location=False, use_polar=False,
        ):
    if args.predict_actions:
        datafolder += "/vaep_framework"
        os.makedirs(datafolder, exist_ok=True)
    
    # note: only for the purpose of this example and due to the small dataset,
    # we use the same data for training and evaluation
    traingames, testgames, model_str = set_conditions(
        args, games
        )
    args.model_str = model_str
    print(
        f"train/test of games: {len(traingames)}" 
        + f"/{len(testgames)}"
        )
    
    Xcols = set_xcols(
        NB_PREV_ACTIONS, use_location=use_location, use_polar=use_polar,
        )
    Xcols_vdep, Xcols_vaep = choose_input_variables(args, Xcols)

    # Train models
    trainX_vaep, trainY_vaep = getXY(
        traingames, features_h5, labels_h5, Xcols_vaep, vaep=True
        )
    trainX_vdep, trainY_vdep = getXY(
        traingames, features_h5, labels_h5, Xcols_vdep, vaep=False
        )
    models_vaep = train(
        args, trainX_vaep, trainY_vaep, do="evaluation"
        )
    models_vdep = train(
        args, trainX_vdep, trainY_vdep, do="evaluation"
        )
    models = {**models_vaep, **models_vdep}
    
    with open(
        os.path.join(datafolder, "VDEPmodel_" + model_str + ".pkl"),"wb"
        ) as f:
        pickle.dump(models, f, protocol=4)
    print(
        os.path.join(datafolder,"VDEPmodel_" + model_str + ".pkl") + " is saved."
        )

    # inference
    testX_vaep, testY_vaep = getXY(
        testgames, features_h5, labels_h5, Xcols_vaep, vaep=True
        )
    testX_vdep, testY_vdep = getXY(
        testgames, features_h5, labels_h5, Xcols_vdep, vaep=False
        )
    Y_hat_vaep, f1scores_vaep = inference_for_evaluation(
        models_vaep, testX_vaep, testY_vaep
        )
    Y_hat_vdep, f1scores_vdep = inference_for_evaluation(
        models_vdep, testX_vdep, testY_vdep
        )

    testY = pd.concat([testY_vaep, testY_vdep], axis=1)
    Y_hat = pd.concat([Y_hat_vaep, Y_hat_vdep], axis=1)
    f1scores = np.concatenate([f1scores_vaep, f1scores_vdep])

    # Save predictions
    A = []
    for game_id in tqdm.tqdm(games.game_id, "Loading game ids"):
        Ai = pd.read_hdf(spadl_h5, f"actions/game_{game_id}")
        Ai = Ai[Ai["period_id"] <= 4]
        A.append(Ai[["game_id"]])
    A = pd.concat(A).reset_index(drop=True)
    save_predictions(predictions_h5, A, Y_hat)
    save_static_variables_for_evaluation(
        args, datafolder, model_str, testY, f1scores
        )


def save_prediction_for_verification(
        args, games,
        spadl_h5, features_h5, labels_h5, predictions_h5,
        use_location=False, use_polar=False,
        ):
    train_set, test_set = args.sets
    train_games = games.iloc[sorted(train_set)]
    test_games = games.iloc[sorted(test_set)]
    print(
        "train/eval of games:", len(train_games), "/", len(test_games)
        )
    
    Xcols = set_xcols(
        NB_PREV_ACTIONS, use_location=use_location, use_polar=use_polar
        )
    Xcols_vdep, Xcols_vaep = choose_input_variables(args, Xcols)

    # Train models
    trainX_vaep, trainY_vaep = getXY(
        train_games, features_h5, labels_h5, Xcols_vaep, vaep=True
        )
    trainX_vdep, trainY_vdep = getXY(
        train_games, features_h5, labels_h5, Xcols_vdep, vaep=False
        )
    models_vaep = train(
        args, trainX_vaep, trainY_vaep, do="verification"
        )
    models_vdep = train(
        args, trainX_vdep, trainY_vdep, do="verification"
        )
    models = {**models_vaep, **models_vdep}

    with open(
        os.path.join(
            args.modelfolder, "VDEPmodel_" + args.model_str + ".pkl"
            ),"wb"
        ) as f:
        pickle.dump(models, f, protocol=4)
    print(
        os.path.join(
            args.modelfolder, "VDEPmodel_" + args.model_str + ".pkl"
            ) + " is saved."
        )

    # inference
    testX_vaep, testY_vaep = getXY(
        test_games, features_h5, labels_h5, Xcols_vaep, vaep=True
        )
    testX_vdep, testY_vdep = getXY(
        test_games, features_h5, labels_h5, Xcols_vdep, vaep=False
        )
    Y_hat_vaep, briers_vaep, lls_vaep, roc_aucs_vaep, f1scores_vaep = (
        inference_for_verification(models_vaep, testX_vaep, testY_vaep)
        )
    Y_hat_vdep, briers_vdep, lls_vdep, roc_aucs_vdep, f1scores_vdep = (
        inference_for_verification(models_vdep, testX_vdep, testY_vdep)
        )

    testY = pd.concat([testY_vaep, testY_vdep], axis=1)
    Y_hat = pd.concat([Y_hat_vaep, Y_hat_vdep], axis=1)
    briers = np.concatenate([briers_vaep, briers_vdep])
    lls = np.concatenate([lls_vaep, lls_vdep])
    roc_aucs = np.concatenate([roc_aucs_vaep, roc_aucs_vdep])
    f1scores = np.concatenate([f1scores_vaep, f1scores_vdep])

    # Save predictions
    A = []
    for game_id in tqdm.tqdm(test_games.game_id, "Loading game ids"):
        Ai = pd.read_hdf(spadl_h5, f"actions/game_{game_id}")
        Ai = Ai[Ai["period_id"] <= 4]
        A.append(Ai[["game_id"]])
    A = pd.concat(A).reset_index(drop=True)
    save_predictions(predictions_h5, A, Y_hat)
    save_static_variables_for_verification(
        args, briers, lls, roc_aucs, f1scores
        )



# ------------------------------
# Appendix
# ------------------------------
"""
Xcols
['type_id_a0', 'type_pass_a0', 'type_cross_a0', 'type_throw_in_a0', 
'type_freekick_crossed_a0', 'type_freekick_short_a0', 'type_corner_crossed_a0', 
'type_corner_short_a0', 'type_take_on_a0', 'type_foul_a0', 'type_tackle_a0', 
'type_interception_a0', 'type_shot_a0', 'type_shot_penalty_a0', 
'type_shot_freekick_a0', 'type_keeper_save_a0', 'type_keeper_claim_a0', 
'type_keeper_punch_a0', 'type_keeper_pick_up_a0', 'type_clearance_a0', 
'type_bad_touch_a0', 'type_non_action_a0', 'type_dribble_a0', 'type_goalkick_a0', 
'bodypart_foot_a0', 'bodypart_foot_right_a0', 'bodypart_foot_left_a0', 
'bodypart_head_a0', 'bodypart_other_a0', 'bodypart_head/other_a0', 
'result_id_a0', 'result_fail_a0', 'result_success_a0', 'result_offside_a0', 
'result_owngoal_a0', 'result_yellow_card_a0', 'result_red_card_a0', 
'goalscore_team', 'goalscore_opponent', 'goalscore_diff', 
'start_x_a0', 'start_y_a0', 'end_x_a0', 'end_y_a0', 'dx_a0', 'dy_a0', 
'movement_a0', 'start_dist_to_goal_a0', 'start_angle_to_goal_a0', 
'end_dist_to_goal_a0', 'end_angle_to_goal_a0', 
'away_team_a0', 'regain_a0', 'penetration_a0', 
'atk0_x_a0', 'atk0_y_a0', 'dist_atk0_a0', 'angle_atk0_a0', 
'dfd0_x_a0', 'dfd0_y_a0', 'dist_dfd0_a0', 'angle_dfd0_a0', 
'atk1_x_a0', 'atk1_y_a0', 'dist_atk1_a0', 'angle_atk1_a0', 
'dfd1_x_a0', 'dfd1_y_a0', 'dist_dfd1_a0', 'angle_dfd1_a0', 
'atk2_x_a0', 'atk2_y_a0', 'dist_atk2_a0', 'angle_atk2_a0', 
'dfd2_x_a0', 'dfd2_y_a0', 'dist_dfd2_a0', 'angle_dfd2_a0', 
'atk3_x_a0', 'atk3_y_a0', 'dist_atk3_a0', 'angle_atk3_a0', 
'dfd3_x_a0', 'dfd3_y_a0', 'dist_dfd3_a0', 'angle_dfd3_a0', 
'atk4_x_a0', 'atk4_y_a0', 'dist_atk4_a0', 'angle_atk4_a0', 
'dfd4_x_a0', 'dfd4_y_a0', 'dist_dfd4_a0', 'angle_dfd4_a0', 
'atk5_x_a0', 'atk5_y_a0', 'dist_atk5_a0', 'angle_atk5_a0', 
'dfd5_x_a0', 'dfd5_y_a0', 'dist_dfd5_a0', 'angle_dfd5_a0', 
'atk6_x_a0', 'atk6_y_a0', 'dist_atk6_a0', 'angle_atk6_a0', 
'dfd6_x_a0', 'dfd6_y_a0', 'dist_dfd6_a0', 'angle_dfd6_a0', 
'atk7_x_a0', 'atk7_y_a0', 'dist_atk7_a0', 'angle_atk7_a0', 
'dfd7_x_a0', 'dfd7_y_a0', 'dist_dfd7_a0', 'angle_dfd7_a0', 
'atk8_x_a0', 'atk8_y_a0', 'dist_atk8_a0', 'angle_atk8_a0', 
'dfd8_x_a0', 'dfd8_y_a0', 'dist_dfd8_a0', 'angle_dfd8_a0', 
'atk9_x_a0', 'atk9_y_a0', 'dist_atk9_a0', 'angle_atk9_a0', 
'dfd9_x_a0', 'dfd9_y_a0', 'dist_dfd9_a0', 'angle_dfd9_a0', 
'atk10_x_a0', 'atk10_y_a0', 'dist_atk10_a0', 'angle_atk10_a0', 
'dfd10_x_a0', 'dfd10_y_a0', 'dist_dfd10_a0', 'angle_dfd10_a0']
"""