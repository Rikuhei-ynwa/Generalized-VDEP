import itertools
import os

import numpy as np
import pandas as pd
from adjustText import adjust_text
from matplotlib import pyplot as plt
from scipy import stats


PLAYER_NUM_PER_TEAM = 11
DIMENTION = 2


def select_games_to_analyze(args, A, games):
    if args.game == "euro2020" or args.game == "wc2022":
        # Group Stage and Round of 16
        analysis_games_list = []
        for id, stage in zip(games["game_id"], games["competition_stage"]):
            if (stage == "Group Stage") or (stage == "Round of 16"):
                analysis_games_list.append(id)
        A_target = A[A["game_id"].isin(analysis_games_list)]
        knockout_teams_list = (
            games[games["competition_stage"] == "Round of 16"]
            .filter(like="team_name", axis=1)
            .values
            .ravel()
            .tolist()
        )

    elif args.game == "euro2022":
        # Group Stage and Semi-finals
        analysis_games_list = []
        for id, stage in zip(games["game_id"], games["competition_stage"]):
            if (stage == "Group Stage") | (stage == "Quarter-finals"):
                analysis_games_list.append(id)
        A_target = A[A["game_id"].isin(analysis_games_list)]
        knockout_teams_list = (
            games[games["competition_stage"] == "Quarter-finals"]
            .filter(like="team_name", axis=1)
            .values
            .ravel()
            .tolist()
        )

    analysis_games_df = games[(games["game_id"].isin(analysis_games_list))]

    return A_target, analysis_games_df, knockout_teams_list


def count_goal_conceded_per_team(knockout_teams_list, analysis_games_df):
    concedes_array = np.empty(len(knockout_teams_list))
    for i, team in enumerate(knockout_teams_list):
        team_homeside = analysis_games_df[
            analysis_games_df["home_team_name"] == team
            ]
        team_awayside = analysis_games_df[
            analysis_games_df["away_team_name"] == team
            ]
        concede = team_homeside["away_score"].sum() + team_awayside["home_score"].sum()
        concedes_array[i] = concede
    concedes_ser = pd.Series(
        data=concedes_array,
        index=knockout_teams_list,
        name="concedes",
    )

    return concedes_ser


def extract_df_to_analyze(A_target, knockout_teams_list, concedes_ser):
    team_values_df = (
        A_target[[
            "defense_team", 
            "gain_value", 
            "attacked_value", 
            "vdep_value", 
            "gvdep_value"]]
        .groupby(["defense_team"])
        .mean()
        .reset_index()
    )

    result_df = team_values_df[
        team_values_df["defense_team"].isin(knockout_teams_list).values
        ].reset_index(drop=True)
    result_df = pd.merge(
        result_df, concedes_ser, left_on="defense_team", right_index=True
        )
    result_df = result_df.reindex(
        columns=[
            "defense_team",
            "concedes",
            "gain_value",
            "attacked_value",
            "vdep_value",
            "gvdep_value",
            ]
        )
    
    return result_df


def evaluate_team_defense(args, A, games):
    figuredir_analysis = args.figuredir + "/analysis"
    os.makedirs(figuredir_analysis, exist_ok=True)

    A_target, analysis_games_df, knockout_teams_list = (
        select_games_to_analyze(args, A, games)
        )
    concedes_ser = count_goal_conceded_per_team(
        knockout_teams_list, analysis_games_df
        )
    result_df = extract_df_to_analyze(
        A_target, knockout_teams_list, concedes_ser
        )

    for v in itertools.combinations(result_df.columns.values.tolist()[1:], 2):
        # columns
        x = v[0]
        y = v[1]

        # Calculate PCC amd P_value
        print(f"### {x} - {y} ###")
        r_pearson, p_value_pearson = stats.pearsonr(
            result_df[x].values, result_df[y].values
            )
        print("Pearson | " + f"r: {r_pearson} | p-value: {p_value_pearson}")
        r_spearman, p_value_spearman = stats.spearmanr(
            result_df[x].values, result_df[y].values
            )
        print("Spearman | " + f"r: {r_spearman} | p-value: {p_value_spearman}")

        # Plot sample figure
        fig_team = plt.figure(figsize=(16, 9))
        ax_team = fig_team.add_subplot(111)
        ax_team.scatter(result_df[x], result_df[y], c="blue")
        if x == "concedes":
            ax_team.set_xlabel("Total number of goals conceded per team", size=32)
        else:
            ax_team.set_xlabel(result_df[x].name + " per team", size=32)
        ax_team.set_ylabel(result_df[y].name + " per team", size=32)
        text_team = [
            ax_team.text(
                result_df.at[index, x],
                result_df.at[index, y],
                result_df.at[index, "defense_team"],
                fontsize=24,
                color="black",
                zorder=500,
            )
            for index in result_df.index
        ]
        adjust_text(text_team)
        ax_team.tick_params(
            axis="both",
            labelsize=24,
            grid_color="lightgray",
            grid_alpha=0.5,
        )
        # ax_team.grid()
        ax_team.axvline(result_df[x].mean(), 0, 1, c="silver")
        ax_team.axhline(result_df[y].mean(), 0, 1, c="silver")
        plt.tight_layout()
        fig_team.savefig(os.path.join(figuredir_analysis, f"teams_{x}_{y}.png"))
        print(os.path.join(figuredir_analysis, f"teams_{x}_{y}.png") + " is saved")
        plt.clf()
        plt.close()


def plot_nan_hist(args, A):
    figuredir_analysis = args.figuredir + "/analysis"
    os.makedirs(figuredir_analysis, exist_ok=True)

    num_nan_atks = np.empty(len(A["freeze_frame_360"]))
    num_nan_dfds = np.empty(len(A["freeze_frame_360"]))
    for i, cies in enumerate(A["freeze_frame_360"]):
        cies_atk = cies[0:PLAYER_NUM_PER_TEAM * DIMENTION]
        cies_dfd = cies[PLAYER_NUM_PER_TEAM * DIMENTION:]
        num_nan_atk = int(
            np.count_nonzero(np.isnan(np.array(cies_atk)))
            / DIMENTION)
        num_nan_dfd = int(
            np.count_nonzero(np.isnan(np.array(cies_dfd)))
            / DIMENTION)
        num_nan_atks[i] = num_nan_atk
        num_nan_dfds[i] = num_nan_dfd

    num_nan_atks_hist = np.bincount(
        num_nan_atks.astype(int)
        )
    num_nan_dfds_hist = np.bincount(
        num_nan_dfds.astype(int)
        )
    
    print(
        "Statistics of the number of missing attackers and defenders\n"
        + f"Attackers: "
        + f"mean: {np.mean(num_nan_atks)} | "
        + f"std: {np.std(num_nan_atks)} | "
        + f"median: {np.median(num_nan_atks)} |"
        + f"mode: {stats.mode(num_nan_atks)}\n"
        + f"Defenders: "
        + f"mean: {np.mean(num_nan_dfds)} | "
        + f"std: {np.std(num_nan_dfds)} | "
        + f"median: {np.median(num_nan_dfds)} |"
        + f"mode: {stats.mode(num_nan_dfds)}"
    )

    plt.figure(figsize=(10, 6))
    # First bar graph
    plt.bar(
        np.arange(0,12,1),
        num_nan_atks_hist,
        color='orange',
        width=0.3, 
        label='Attackers',
        align="center",
        zorder=2
        )
    # Second bar graph
    plt.bar(
        np.arange(0,12,1),
        num_nan_dfds_hist + num_nan_atks_hist,
        color='dodgerblue',
        width=0.3, 
        label='Defenders',
        align="center",
        zorder=1
        )
    # Set the title and labels
    plt.title(
        'Histogram of the number of missing attackers and defenders.',
        fontsize=16
        )
    plt.xlabel(
        'Number of missing attackers/defenders in freeze_frame_360',
        fontsize=12
        )
    plt.ylabel('Frequency', fontsize=12)
    plt.xticks(np.arange(0,12,1), fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            figuredir_analysis, 
            "the_number_of_nan_in_freeze_frame_360.png"
            )
            )
    print(
        os.path.join(
            figuredir_analysis, 
            "the_number_of_nan_in_freeze_frame_360.png"
            ) + " is saved"
        )