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
import tqdm
from adjustText import adjust_text

import convert_rawdata as crd
import set_variables as sv
import train_models as tm

import socceraction.spadl as spadl
import socceraction.spadl.config as spadlconfig
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


def train(args, datafolder, trainX, trainY):
    figuredir = datafolder + "/figures/feature_importances"
    os.makedirs(figuredir, exist_ok=True)
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
        model = tm.create_model(args)
        model.fit(trainX, trainY[col])
        tm.illustrate_shap(model, trainX)
        plt.savefig(os.path.join(figuredir, f"model_{col}_summary.png"))
        plt.clf()
        plt.close()
        models[col] = model

    return models


def inference(models, testX, testY):
    Y_hat = pd.DataFrame()
    f1scores = np.empty(len(testY.columns))
    for i, col in enumerate(testY.columns):
        Y_hat[col] = [p[1] for p in models[col].predict_proba(testX)]
        print(
            f"### Y: {col} ### "
            + f"positive: {testY[col].sum()} "
            + f"negative: {len(testY[col]) - testY[col].sum()}"
            )
        f1scores[i] = tm.evaluate_metrics(testY[col], Y_hat[col])

    return Y_hat, f1scores


def save_predictions(predictions_h5, A, Y_hat):
    # concatenate action game id rows with predictions and save per game
    grouped_predictions = pd.concat([A, Y_hat], axis=1).groupby("game_id")
    for k, df in tqdm.tqdm(
        grouped_predictions,
        desc="Saving predictions per game"
        ):
        df = df.reset_index(drop=True)
        df[Y_hat.columns].to_hdf(predictions_h5, f"game_{int(k)}")

    
def save_static_variables(args, datafolder, model_str, testY, f1scores):    
    C_vdep_v0 = testY["gains"].sum() / testY["effective_attack"].sum()
    static = [args, C_vdep_v0, f1scores, ]
    with open(
        os.path.join(datafolder, "static_" + model_str + ".pkl"),
        "wb") as f:
        pickle.dump(static, f, protocol=4)


def compute_gvdep_values(
        spadl_h5, features_h5, predictions_h5, 
        games, players, teams, C_vdep_v0, 
        ):
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
        (A["vaep_value"] * A["gains_id"]).sum()
        / num_gains
        )
    weight_attacked = (
        ((-A["vaep_value"]) * (A["effective_id"])).sum()
        / num_attacked
        )
    A["g_vdep_value"] = (
        weight_gains * A["gain_value"]
        - weight_attacked * A["attacked_value"]
        )

    return A


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
    if args.skip_train:
        print("model training is skipped")
        _, _, model_str = tm.set_conditions(args, games)
    else:
        if args.predict_actions:
            datafolder += "/vaep_framework"
            os.makedirs(datafolder, exist_ok=True)
        
        # note: only for the purpose of this example and due to the small dataset,
        # we use the same data for training and evaluation
        traingames, testgames, model_str = tm.set_conditions(
            args, games
            )
        print(
            f"train/test of games: {len(traingames)}" 
            + f"/{len(testgames)}"
            )
        
        Xcols = tm.set_xcols(NB_PREV_ACTIONS, use_polar=False,)
        Xcols_vdep, Xcols_vaep = tm.choose_input_variables(args, Xcols)

        # Train models
        trainX_vaep, trainY_vaep = tm.getXY(
            traingames, features_h5, labels_h5, Xcols_vaep, vaep=True
            )
        trainX_vdep, trainY_vdep = tm.getXY(
            traingames, features_h5, labels_h5, Xcols_vdep, vaep=False
            )
        models_vaep = train(
            args, datafolder, trainX_vaep, trainY_vaep, 
            )
        models_vdep = train(
            args, datafolder, trainX_vdep, trainY_vdep, 
            )
        models = {**models_vaep, **models_vdep}
        
        with open(os.path.join(datafolder, "VDEPmodel_" + model_str + ".pkl"),"wb") as f:
            pickle.dump(models, f, protocol=4)
        print(os.path.join(datafolder,"VDEPmodel_" + model_str + ".pkl") + " is saved")

        # inference
        testX_vaep, testY_vaep = tm.getXY(
            testgames, features_h5, labels_h5, Xcols_vaep, vaep=True
            )
        testX_vdep, testY_vdep = tm.getXY(
            testgames, features_h5, labels_h5, Xcols_vdep, vaep=False
            )
        Y_hat_vaep, f1scores_vaep = inference(models_vaep, testX_vaep, testY_vaep)
        Y_hat_vdep, f1scores_vdep = inference(models_vdep, testX_vdep, testY_vdep)

        testY = pd.concat([testY_vaep, testY_vdep], axis=1)
        Y_hat = pd.concat([Y_hat_vaep, Y_hat_vdep], axis=1)
        f1scores = np.concatenate([f1scores_vaep, f1scores_vdep])

        # Save predictions
        A = []
        for game_id in tqdm.tqdm(games.game_id, "Loading game ids"):
            Ai = pd.read_hdf(spadl_h5, f"actions/game_{game_id}")
            A.append(Ai[["game_id"]])
        A = pd.concat(A).reset_index(drop=True)
        save_predictions(predictions_h5, A, Y_hat)
        save_static_variables(args, datafolder, model_str, testY, f1scores)

    # 4. compute gvdep values and top players
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

    with open(
        os.path.join(datafolder, "static_" + model_str + ".pkl"), "rb"
        ) as f:
        _, C_vdep_v0, _,  = pickle.load(f)

    # Compute each values
    A = compute_gvdep_values(
        spadl_h5, features_h5, predictions_h5, 
        games, players, teams, C_vdep_v0, 
        )

    # (optional) inspect country's top 5 most valuable non-shot actions
    if args.test:
        from mplsoccer import Pitch

        sorted_A = A.sort_values("g_vdep_value", ascending=False)
        sorted_A = sorted_A[
            sorted_A.defense_team.str.contains(args.teamView)
            ]
        sorted_A = sorted_A[
            # (sorted_A.type_name.str.contains("interception")) &
            (~sorted_A.type_name.str.contains("shot"))
            ]

        for j in range(10):
            row = list(sorted_A[j : j + 1].itertuples())[0]
            i = row.Index
            sample = A[i - 2 : i + 1].copy()

            sample["player_name"] = sample[
                ["nickname", "player_name"]
                ].apply(lambda x: x[0] if x[0] else x[1], axis=1)

            g = list(
                games[
                    games.game_id == sample.game_id.values[0]
                    ].itertuples()
                )[0]
            game_info = (
                f"{g.game_date}"
                + f"{g.home_team_name}"
                + f"{g.home_score}-{g.away_score}"
                + f"{g.away_team_name}"
                )

            sample["gain_value"] = sample.gain_value.apply(lambda x: "%.3f" % x)
            sample["attacked_value"] = sample.attacked_value.apply(lambda x: "%.3f" % x)
            sample["g_vdep_value"] = sample.g_vdep_value.apply(lambda x: "%.3f" % x)
            sample["time"] = sample[
                ["period_id", "time_seconds"]
                ]
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
                "defense_team",
                "gain_value",
                "attacked_value",
                "g_vdep_value",
            ]
            sample = sample.reindex(columns=cols)
            print(sample)

            # make folders to store example figures
            os.makedirs(figuredir + "/example", exist_ok=True)
            figure_eg_dir = os.path.join(
                figuredir + "/example",
                f"{g.home_team_name} {g.home_score}-{g.away_score} {g.away_team_name}/",
            )  # data_id
            os.makedirs(figure_eg_dir, exist_ok=True)
            filename_fig = f"{row.type_name} {row.player_name}"

            # Creating example figures.
            ref_team = args.teamView
            if args.game == "euro2022":
                if ref_team == "Finland":
                    ref_team = "WNT Finland"
                elif ref_team == "Northern Ireland":
                    continue
                else:
                    ref_team += " Women's"
            
            p = Pitch(
                pitch_type="custom",
                pitch_length=105, 
                pitch_width=68
                )
            fig, ax = p.draw(figsize=(16, 9))

            # visible area
            visible_area = np.array(
                sample.at[i,"visible_area_360"]
                ).reshape(-1, 2)
            p.polygon(
                [visible_area],
                color=(0.5, 0.5, 0.5, 0.3),
                ax=ax
                )

            # players' positions
            cie_pl = sample.at[i,"freeze_frame_360"]
            cie_atk = cie_pl[:PLAYER_NUM_PER_TEAM * DIMENTION]
            cie_dfd = cie_pl[PLAYER_NUM_PER_TEAM * DIMENTION:]
            p.scatter(
                cie_atk[::2], cie_atk[1::2],
                c='orange', s=120, ec='k', ax=ax, zorder=100
                )
            p.scatter(
                cie_dfd[::2], cie_dfd[1::2],
                c='dodgerblue', s=120, ec='k', ax=ax, zorder=100
                )

            # the ball's position
            # texts = []
            order = 0
            for i, row in sample.iterrows():
                order += 1
                p.scatter(
                    row.start_x, row.start_y,
                    c='yellow', s=300, ec='k', ax=ax, zorder=50
                    )
                p.arrows(
                    row.start_x, row.start_y, row.end_x, row.end_y, 
                    ax=ax, color='black', width=1.0
                )

            plt.savefig(
                os.path.join(figure_eg_dir, f"{filename_fig}.png")
                )


    # 5. Analyze team defense
    # To consider teams advancing to the knockout stage, select the teams
    # if args.game == "euro2020" or args.game == "wc2022":
    #     # Group Stage and Round of 16
    #     analysis_games_list = []
    #     for id, stage in zip(games["game_id"], games["competition_stage"]):
    #         if (stage == "Group Stage") | (stage == "Round of 16"):
    #             analysis_games_list.append(id)
    #     A = A[A["game_id"].isin(analysis_games_list)]
    #     knockout_teams_list = (
    #         games[games["competition_stage"] == "Round of 16"].filter(like="team_name", axis=1).values.ravel().tolist()
    #     )

    # elif args.game == "euro2022":
    #     # Group Stage and Semi-finals
    #     analysis_games_list = []
    #     for id, stage in zip(games["game_id"], games["competition_stage"]):
    #         if (stage == "Group Stage") | (stage == "Quarter-finals"):
    #             analysis_games_list.append(id)
    #     A = A[A["game_id"].isin(analysis_games_list)]
    #     knockout_teams_list = (
    #         games[games["competition_stage"] == "Quarter-finals"].filter(like="team_name", axis=1).values.ravel().tolist()
    #     )

    # analysis_games_df = games[(games["game_id"].isin(analysis_games_list))]
    # # counting concedes by each team that reached the knockout stage
    # concedes_array = np.empty(len(knockout_teams_list))
    # for i, team in enumerate(knockout_teams_list):
    #     team_homeside = analysis_games_df[analysis_games_df["home_team_name"] == team]
    #     team_awayside = analysis_games_df[analysis_games_df["away_team_name"] == team]
    #     concede = team_homeside["away_score"].sum() + team_awayside["home_score"].sum()
    #     concedes_array[i] = concede
    # concedes_ser = pd.Series(
    #     data=concedes_array,
    #     index=knockout_teams_list,
    #     name="concedes",
    # )

    # team_values_df = (
    #     A[["defense_team", "gain_value", "attacked_value", "g_vdep_value"]].groupby(["defense_team"]).mean().reset_index()
    # )

    # result_df = team_values_df[team_values_df["defense_team"].isin(knockout_teams_list).values].reset_index(drop=True)
    # result_df = pd.merge(result_df, concedes_ser, left_on="defense_team", right_index=True)
    # result_df = result_df.reindex(
    #     columns=["defense_team","concedes","gain_value","attacked_value","g_vdep_value",]
    #     )

    # # Plot figures about defense values of the teams in Euro 2020 or 2022
    # for v in itertools.combinations(result_df.columns.values.tolist()[1:], 2):
    #     # columns
    #     x = v[0]
    #     y = v[1]

    #     # Calculate PCC amd P_value
    #     from scipy.stats import pearsonr

    #     r, p_value = pearsonr(result_df[x].values, result_df[y].values)
    #     print(f"### {x} - {y} ###")
    #     print(f"r : {r}")
    #     print(f"P-value : {p_value}")

    #     # Plot sample figure
    #     fig_team = plt.figure(figsize=(16, 9))
    #     ax_team = fig_team.add_subplot(111)
    #     ax_team.scatter(result_df[x], result_df[y], c="blue")
    #     if x == "concedes":
    #         ax_team.set_xlabel("Total " + result_df[x].name + " per team", size=32)
    #     else:
    #         ax_team.set_xlabel(result_df[x].name + " averaged per team", size=32)
    #     ax_team.set_ylabel(result_df[y].name + " averaged per team", size=32)
    #     text_team = [
    #         ax_team.text(
    #             result_df.at[index, x],
    #             result_df.at[index, y],
    #             result_df.at[index, "defense_team"],
    #             fontsize=24,
    #             color="black",
    #             zorder=500,
    #         )
    #         for index in result_df.index
    #     ]
    #     adjust_text(text_team)
    #     ax_team.tick_params(
    #         axis="both",
    #         labelsize=24,
    #         grid_color="lightgray",
    #         grid_alpha=0.5,
    #     )
    #     ax_team.grid()
    #     ax_team.axvline(result_df[x].mean(), 0, 1, c="silver")
    #     ax_team.axhline(result_df[y].mean(), 0, 1, c="silver")
    #     plt.tight_layout()
    #     fig_team.savefig(os.path.join(figuredir, f"teams_{x}_{y}.png"))
    #     print(os.path.join(figuredir, f"teams_{x}_{y}.png") + " is saved")
    #     plt.clf()
    #     plt.close()

    # # Plot bars about the number of players in each freeze_frame_360
    # nan_array = np.empty(len(A["freeze_frame_360"]))
    # for i, coordinates in enumerate(A["freeze_frame_360"]):
    #     count_nan = int(np.count_nonzero(np.isnan(np.asarray(coordinates))))
    #     nan_array[i] = count_nan
    # nan_array = nan_array / 2
    # nan_array = 22 - nan_array

    # fig_ff360 = plt.figure(figsize=(8, 6))
    # ax_ff360 = fig_ff360.add_subplot(111, xticks=range(0, 23, 1))
    # ax_ff360.set_ylim(0, 32000)
    # n, bins, _ = ax_ff360.hist(nan_array, bins=range(0, 24, 1), align="left")

    # # Display the degrees at the head of the bar
    # text_degree = [
    #     ax_ff360.text(
    #         bin + 0.1,  # Adjustments
    #         num + 120,  # Adjustments
    #         int(num),
    #         fontsize=10,
    #         rotation=70,
    #         horizontalalignment="center",
    #     )
    #     for num, bin in zip(n, bins)
    #     if num
    # ]
    # fig_ff360.savefig(os.path.join(figuredir, f"the_number_of_nan_in_{args.game}.png"))
    # print(os.path.join(figuredir, f"the_number_of_nan_in_{args.game}.png"))
    # print(time.time() - start)

    # pdb.set_trace()

if __name__ == "__main__":
    main()
