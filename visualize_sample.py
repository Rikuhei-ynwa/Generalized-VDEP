import os
import numpy as np
import matplotlib.pyplot as plt

from adjustText import adjust_text
from mplsoccer import Pitch

import socceraction.spadl.config as spadlcfg

PLAYER_NUM_PER_TEAM = 11
DIMENTION = 2


def sorted_data(A, args):
    sorted_A = A.sort_values("gvdep_value", ascending=False)
    sorted_A = sorted_A[
        sorted_A.defense_team.str.contains(args.teamView)
        ]
    sorted_A = sorted_A[
        sorted_A.type_name.str.contains("tackle") 
            | sorted_A.type_name.str.contains("interception")
        ]
    return sorted_A


def set_figuredir_sample(args):
    figuredir_sample = os.path.join(
        args.figuredir, "sample"
        )
    os.makedirs(figuredir_sample, exist_ok=True)
    return figuredir_sample


def set_figuredir_game(args, games, sample, i, row):
    game = list(
            games[
                games.game_id == sample.game_id.values[0]
                ].itertuples()
            )[0]
    game_info = (
        f"{game.home_team_name} "
        + f"{game.home_score}-{game.away_score} "
        + f"{game.away_team_name}"
        )
    
    figuredir_game = os.path.join(
        args.figuredir_sample + f"/{game_info}",
    )  # data_id
    os.makedirs(figuredir_game, exist_ok=True)
    time_at_event = sample.at[i, "time"]
    filename_fig = (
        f"{time_at_event}" 
        + f" {row.type_name}"
        + f" {row.player_name}"
        )

    return figuredir_game, filename_fig


def set_sample_events(sample):
    sample["player_name"] = sample[
        ["nickname", "player_name"]
        ].apply(lambda x: x[0] if x[0] else x[1], axis=1)
    sample["gain_value"] = (
        sample.gain_value.apply(lambda x: "%.6f" % x)
        )
    sample["attacked_value"] = (
        sample.attacked_value.apply(lambda x: "%.6f" % x)
        )
    sample["gvdep_value"] = (
        sample.gvdep_value.apply(lambda x: "%.6f" % x)
        )
    sample["time"] = sample.time_seconds.apply(
        lambda x: f"{int(x // 60)}m{int(x % 60):02d}s"
        )
    cols = [
        "game_id",
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
        "gvdep_value",
    ]
    sample = sample.reindex(columns=cols)

    return sample


def visualise_sample(args, A, games):
    figuredir_sample = set_figuredir_sample(args)
    args.figuredir_sample = figuredir_sample
    sorted_A = sorted_data(A, args)

    for j in range(10):
        row = list(sorted_A[j : j + 1].itertuples())[0]
        i = row.Index
        sample = A[i - (args.sample_events - 1) : i + 1].copy()

        sample = set_sample_events(sample)
        figuredir_game, filename_fig = set_figuredir_game(
            args, games, sample, i, row
            )

        # Creating example figures.
        ref_team = args.teamView
        if args.game == "euro2022":
            if ref_team == "Finland":
                ref_team = "WNT Finland"
            elif ref_team == "Northern Ireland":
                continue
            else:
                ref_team += " Women's"
        
        pitch = Pitch(
            pitch_type="custom",
            pitch_length=105, 
            pitch_width=68
            )
        fig, ax = pitch.draw(figsize=(16, 9))

        # visible area
        visible_area = np.array(
            sample.at[i,"visible_area_360"]
            ).reshape(-1, 2)
        pitch.polygon(
            [visible_area],
            color="lightgray",
            ax=ax,
            )

        # players' positions
        cie_pl = sample.at[i,"freeze_frame_360"]
        cie_atk = cie_pl[:PLAYER_NUM_PER_TEAM * DIMENTION]
        cie_dfd = cie_pl[PLAYER_NUM_PER_TEAM * DIMENTION:]
        pitch.scatter(
            cie_atk[::2], cie_atk[1::2],
            c='orange', s=120, ec='k', ax=ax
            )
        pitch.scatter(
            cie_dfd[::2], cie_dfd[1::2],
            c='dodgerblue', s=120, ec='k', ax=ax
            )

        # the ball's position
        ev_start = sample.iloc[0].name
        texts = []
        for i, row in sample.iterrows():
            order = i - ev_start + 1
            if row.team_name == ref_team:
                start_x = row.start_x
                start_y = row.start_y
                end_x = row.end_x
                end_y = row.end_y
            else:
                start_x = spadlcfg.FIELD_LENGTH - row.start_x
                start_y = spadlcfg.FIELD_WIDTH - row.start_y
                end_x = spadlcfg.FIELD_LENGTH - row.end_x
                end_y = spadlcfg.FIELD_WIDTH - row.end_y
            
            if row.result_name == 'success':
                color = 'black'
            else:
                color = 'red'
            if row.type_name in ['take_on', 'dribble',]:
                pitch.arrows(
                    start_x, start_y, end_x, end_y,
                    color=color,
                    ax=ax,
                    linestyles="--",
                    linewidths=3.5,
                    )
            else:
                pitch.scatter(
                    start_x, start_y, 
                    facecolors='none',
                    edgecolors=color,
                    s=150, 
                    ax=ax,
                    )
                pitch.lines(
                    start_x, start_y, end_x, end_y,
                    ax=ax,
                    color=color,
                    lw=5,
                    transparent=True,
                    comet=True,
                    )
            text = ax.text(
                start_x, start_y, 
                f"{order}: {row.type_name}\n {row.gvdep_value}", 
                fontsize=16, 
                color='black',
                )
            texts.append(text)

        adjust_text(texts, arrowprops=dict(arrowstyle='->', color='black'))
            
        plt.savefig(
            os.path.join(figuredir_game, f"{filename_fig}.png")
            )
        print(
            os.path.join(figuredir_game, f"{filename_fig}.png")
            + " is saved"
            )