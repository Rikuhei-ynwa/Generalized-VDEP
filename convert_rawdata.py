import os

import pandas as pd
from tqdm import tqdm

from socceraction.data.statsbomb import StatsBombLoader
# from socceraction.spadl.statsbomb_360 import convert_to_actions_360
from socceraction.spadl.statsbomb import convert_to_actions_360


def set_DLoader(args):
    """Set up the Loader"""
    # https://socceraction.readthedocs.io/en/latest/documentation/providers.html

    if args.data == "statsbomb":
        # Use this if you want to use the free public statsbomb data
        # or provide credentials to access the API
        # DLoader = StatsBombLoader(getter="remote", creds={"user": None, "passwd": None})

        # Uncomment the code below if you have a local folder on your computer with statsbomb data
        # Caution: the data is constantly updated, so make sure you have the version with the date you want
        data_folder = f"../open-data-{args.date_opendata}/data"  # Example of local folder with statsbomb data
        DLoader = StatsBombLoader(root=data_folder, getter="local")
    else:
        raise ValueError("Please set the data to 'statsbomb'.")

    return DLoader


def select_games(DLoader, args):
    if args.data == "statsbomb":  # No competition and season information in wyscout!!!!
        # View all available competitions
        competitions = DLoader.competitions()
        set(competitions.competition_name)
        if args.game == "wc2022":
            selected_competitions = competitions[
                (competitions.competition_name == "FIFA World Cup") 
                & (competitions.season_name == "2022")
            ]
        elif args.game == "euro2020":
            selected_competitions = competitions[
                (competitions.competition_name == "UEFA Euro") 
                & (competitions.season_name == "2020")
            ]
        elif args.game == "euro2022":
            selected_competitions = competitions[
                (competitions.competition_name == "UEFA Women's Euro") 
                & (competitions.season_name == "2022")
            ]
        else:  # All data
            selected_competitions = competitions

    else:
        raise ValueError("Please set the data to 'statsbomb'.")
    
    # Get games from all selected competitions
    games = pd.concat(
        [DLoader.games(row.competition_id, row.season_id) for row in selected_competitions.itertuples()]
        )
    games = games.sort_values(["game_id", "game_day"]).reset_index(drop=True)

    return games, selected_competitions


def store_converted_data(
        DLoader, 
        datafolder, 
        games, 
        selected_competitions,
        plot_sample=False
        ):
    # Store converted spadl data in a h5-file
    games_verbose = tqdm(list(games.itertuples()), desc="Loading game data")
    teams, players, games_360 = [], [], []
    actions = {}
    for game in games_verbose:
        events = DLoader.events(game.game_id, load_360=True)
        # convert data
        actions[game.game_id] = convert_to_actions_360(
            events,
            game.home_team_id,
            xy_fidelity_version = 1,
            shot_fidelity_version = 1,
            )  # spadl.statsbomb.convert_to_actions

        if plot_sample:
            # Plot a sample animation
            _plot_sample_animation(actions[game.game_id], datafolder)

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
            ["player_id",
             "game_id",
             "team_id",
             "is_starter",
             "starting_position_id",
             "starting_position_name",
             "minutes_played"]
            ]
        for game_id in actions.keys():
            spadlstore[f"actions/game_{game_id}"] = actions[game_id]
    

def _plot_sample_animation(samples, datafolder):
    import numpy as np
    from matplotlib.animation import FuncAnimation
    from mplsoccer import Pitch

    pitch = Pitch(pitch_type='custom', pitch_length=105, pitch_width=68)
    fig, ax = pitch.draw(figsize=(16, 9))

    def anim_statsbomb(idx):
        ax.cla()
        pitch.draw(ax=ax)
        if len(samples.iloc[idx+1000].freeze_frame_360) == 0:
            # This means that 360 data is NaN.
            pass
        else:
            if len(samples.iloc[idx+1000].visible_area_360) == 0:
                pass
            else:
                visible_area = np.array(
                    samples.iloc[idx+1000].visible_area_360
                    ).reshape(-1, 2)
                pitch.polygon([visible_area], color=(1, 0, 0, 0.3), ax=ax)

            player_position_data = samples.iloc[idx+1000].freeze_frame_360
            player_position_data = np.array(player_position_data).reshape(-1, 2)

            pitch.scatter(
                samples.iloc[idx+1000].start_x, 
                samples.iloc[idx+1000].start_y, 
                c='white', s=240, ec='k', ax=ax
            )

            for i, position in enumerate(player_position_data):
                if 0 <= i < 11:
                    pitch.scatter(
                        position[0],
                        position[1],
                        c='orange', s=80, ec='k', ax=ax
                        )
                else:
                    pitch.scatter(
                        position[0],
                        position[1],
                        c='dodgerblue', s=80, ec='k', ax=ax
                        )

        type_id = samples.iloc[idx+1000].type_id
        
        ax.text(
            1.0, 1.5, 
            (f"Frame: {idx+1000}, "
                + f"Team: {samples.iloc[idx+1000].team_id}, "
                + f"Type: {type_id}, "
                + f"Player: {samples.iloc[idx+1000].player_id}"), 
            fontsize=12
            )
        
    # Create a progress bar using tqdm
    progress_bar = tqdm(total=1000)

    # Wrapping the update function to include the progress bar update
    def update_with_progress(frame):
        progress_bar.update()
        return anim_statsbomb(frame)

    anim = FuncAnimation(
        fig, update_with_progress, frames=1000, interval=200
        )
    anim_dir = os.path.join(
        os.path.abspath(os.path.join(__file__, "../../")),
        "anims"
        )
    os.makedirs(anim_dir,exist_ok=True)
    anim.save(anim_dir + "/converted_sample.mp4", writer="ffmpeg")