"""Implements serializers for RDD data."""
import ast
import glob
import os
import warnings
from typing import Any, Dict, List, Optional

import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore
import pysrt
from pandera.typing import DataFrame

try:
    from statsbombpy import api_client, sb

    def my_has_auth(creds: Dict[str, str]) -> bool:
        """Monkeypatch to hide the repeated print messages."""
        if creds.get("user") in [None, ""] or creds.get("passwd") in [None, ""]:
            warnings.warn("credentials were not supplied. open data access only")
            return False
        return True

    api_client.has_auth = my_has_auth
except ImportError:
    sb = None

from socceraction.data.base import EventDataLoader, ParseError, _localloadjson

from .schema import (RDDCompetitionSchema, RDDEventSchema, RDDGameSchema,
                     RDDPlayerSchema, RDDTeamSchema)


class RDDLoader(EventDataLoader):
    """Load RDD data either from a remote location or from a local folder.

    To load local data, point ``root`` to the root folder of the data. This folder
    should use the same directory structure as used in the Open Data GitHub repo.

    Parameters
    ----------
    getter : str
        "remote" or "local"
    root : str, optional
        Root-path of the data. Only used when getter is "local".
    creds: dict, optional
        Login credentials in the format {"user": "", "passwd": ""}. Only used
        when getter is "remote".
    """

    def __init__(
        self,
        getter: str = "remote",
        root: Optional[str] = None,
        creds: Optional[Dict[str, str]] = None,
    ) -> None:
        if getter == "remote":
            import pdb; pdb.set_trace()
            if sb is None:
                raise ImportError(
                    """The 'statsbombpy' package is required. Install with 'pip install statsbombpy'."""
                )
            self._creds = creds or sb.DEFAULT_CREDS
            self._local = False
        elif getter == "local":
            if root is None:
                raise ValueError("""The 'root' parameter is required when loading local data.""")
            self._local = True
            self._root = root
        else:
            raise ValueError("Invalid getter specified")

    def competitions(self) -> DataFrame[RDDCompetitionSchema]:
        """Return a dataframe with all available competitions and seasons.

        Raises
        ------
        ParseError
            When the raw data does not adhere to the expected format.

        Returns
        -------
        pd.DataFrame
            A dataframe containing all available competitions and seasons. See
            :class:`~socceraction.spadl.RDD.RDDCompetitionSchema` for the schema.
        """
        cols = [
            "competition_id",
            "competition_name",
            "season_id",
            "season_name",
            "country_name",
            "competition_gender",
        ]
        obj = _localloadjson(str(os.path.join(self._root, "competitions.json")))
        if not isinstance(obj, list):
            raise ParseError("The retrieved data should contain a list of competitions")
        if len(obj) == 0:
            return pd.DataFrame(columns=cols).pipe(DataFrame[RDDCompetitionSchema])
        try: return pd.DataFrame(obj)[cols].pipe(DataFrame[RDDCompetitionSchema])
        except: import pdb; pdb.set_trace()

    def games(self, competition_id: int, season_id: int) -> DataFrame[RDDGameSchema]:
        """Return a dataframe with all available games in a season.

        Parameters
        ----------
        competition_id : int
            The ID of the competition.
        season_id : int
            The ID of the season.

        Raises
        ------
        ParseError
            When the raw data does not adhere to the expected format.

        Returns
        -------
        pd.DataFrame
            A dataframe containing all available games. See
            :class:`~socceraction.spadl.RDD.RDDGameSchema` for the schema.
        """
        cols = [
            "game_id",
            "season_id",
            "competition_id",
            "competition_stage",
            "game_day",
            "game_date",
            "home_team_id",
            "away_team_id",
            "home_score",
            "away_score",
            "venue",
            "referee",
        ]
        if self._local:
            try: 
                obj = _localloadjson(
                    str(os.path.join(self._root, f"matches/{competition_id}/{season_id}.json"))
                )
            except: import pdb; pdb.set_trace()
        else:
            obj = list(
                sb.matches(competition_id, season_id, fmt="dict", creds=self._creds).values()
            )
        if not isinstance(obj, list):
            raise ParseError("The retrieved data should contain a list of games")
        if len(obj) == 0:
            return pd.DataFrame(columns=cols).pipe(DataFrame[RDDGameSchema])
        gamesdf = pd.DataFrame(_flatten(m) for m in obj)
        gamesdf["kick_off"] = gamesdf["kick_off"].fillna("12:00:00.000")
        gamesdf["match_date"] = pd.to_datetime(
            gamesdf[["match_date", "kick_off"]].agg(" ".join, axis=1)
        )
        gamesdf.rename(
            columns={
                "match_id": "game_id",
                "match_date": "game_date",
                "match_week": "game_day",
                "stadium_name": "venue",
                "referee_name": "referee",
                "competition_stage_name": "competition_stage",
            },
            inplace=True,
        )
        if "venue" not in gamesdf:
            gamesdf["venue"] = None
        if "referee" not in gamesdf:
            gamesdf["referee"] = None
        try: return gamesdf[cols].pipe(DataFrame[RDDGameSchema])
        except: import pdb; pdb.set_trace()
    def _lineups(self, game_id: int) -> List[Dict[str, Any]]:
        if self._local:
            obj = _localloadjson(str(os.path.join(self._root, f"lineups/{game_id}.json")))
        else:
            obj = list(sb.lineups(game_id, fmt="dict", creds=self._creds).values())
        if not isinstance(obj, list):
            raise ParseError("The retrieved data should contain a list of teams")
        if len(obj) != 2:
            raise ParseError("The retrieved data should contain two teams")
        return obj

    def teams(self, game_id: int) -> DataFrame[RDDTeamSchema]:
        """Return a dataframe with both teams that participated in a game.

        Parameters
        ----------
        game_id : int
            The ID of the game.

        Raises
        ------
        ParseError  # noqa: DAR402
            When the raw data does not adhere to the expected format.

        Returns
        -------
        pd.DataFrame
            A dataframe containing both teams. See
            :class:`~socceraction.spadl.RDD.RDDTeamSchema` for the schema.
        """
        cols = ["team_id", "team_name"]
        obj = self._lineups(game_id)
        return pd.DataFrame(obj)[cols].pipe(DataFrame[RDDTeamSchema])

    def players(self, game_id: int) -> DataFrame[RDDPlayerSchema]:
        """Return a dataframe with all players that participated in a game.

        Parameters
        ----------
        game_id : int
            The ID of the game.

        Raises
        ------
        ParseError  # noqa: DAR402
            When the raw data does not adhere to the expected format.

        Returns
        -------
        pd.DataFrame
            A dataframe containing all players. See
            :class:`~socceraction.spadl.RDD.RDDPlayerSchema` for the schema.
        """
        cols = [
            "game_id",
            "team_id",
            "player_id",
            "player_name",
            "nickname",
            "jersey_number",
            "is_starter",
            "starting_position_id",
            "starting_position_name",
            "minutes_played",
        ]

        obj = self._lineups(game_id)
        playersdf = pd.DataFrame(_flatten_id(p) for lineup in obj for p in lineup["lineup"])
        playergamesdf = extract_player_games(self.events(game_id))
        playersdf = pd.merge(
            playersdf,
            playergamesdf[
                ["player_id", "team_id", "position_id", "position_name", "minutes_played"]
            ],
            on="player_id",
        )
        playersdf["game_id"] = game_id
        playersdf["position_name"] = playersdf["position_name"].replace(0, "Substitute")
        playersdf["position_id"] = playersdf["position_id"].fillna(0).astype(int)
        playersdf["is_starter"] = playersdf["position_id"] != 0
        playersdf.rename(
            columns={
                "player_nickname": "nickname",
                "country_name": "country",
                "position_id": "starting_position_id",
                "position_name": "starting_position_name",
            },
            inplace=True,
        )
        return playersdf[cols].pipe(DataFrame[RDDPlayerSchema])

    def events_trajectories(self, game_id, competition_id, trajectories, meta_trajs, load_360=False, Plot_raw=False):
        """Return a dataframe with the event stream of a game.

        Parameters
        ----------
        game_id : int
            The ID of the game.
        load_360 : bool
            Whether to load the 360 data.

        Raises
        ------
        ParseError
            When the raw data does not adhere to the expected format.

        Returns
        -------
        pd.DataFrame
            A dataframe containing the event stream. See
            :class:`~socceraction.spadl.RDD.RDDEventSchema` for the schema.
        """
        cols = [
            "game_id",
            "event_id",
            "period_id",
            "team_id",
            "player_id",
            "type_id",
            "type_name",
            "index", # "timestamp",            
            "minute",
            "second",
            "millisecond",
            "possession",
            "possession_team_id",
            "possession_team_name",
            "play_pattern_id",
            "play_pattern_name",
            "team_name", # "duration",
            "extra",
            "related_events",
            "player_name",
            "position_id",
            "position_name", # 
            "location",
            "under_pressure",
            "counterpress",
        ]
        player_names,all_data_id,_ = meta_trajs 
        # all_data_id = [all_data_id[0]]
        # Load the events
        if False: # json
            if self._local:
                obj = _localloadjson(str(os.path.join(self._root, f"events/{game_id}.json")))
            else:
                obj = list(sb.events(game_id, fmt="dict", creds=self._creds).values())

            if not isinstance(obj, list):
                raise ParseError("The retrieved data should contain a list of events")
            if len(obj) == 0:
                return pd.DataFrame(columns=cols).pipe(DataFrame[RDDEventSchema])

            eventsdf = pd.DataFrame(_flatten_id(e) for e in obj)
            eventsdf["match_id"] = game_id
            # eventsdf["timestamp"] = pd.to_datetime(eventsdf["timestamp"], format="%H:%M:%S.%f")
            #eventsdf["related_events"] = eventsdf["related_events"].apply(
            #    lambda d: d if isinstance(d, list) else []
            #)
            eventsdf["under_pressure"] = eventsdf["under_pressure"].fillna(False).astype(bool)
            eventsdf["counterpress"] = eventsdf["counterpress"].fillna(False).astype(bool)
            eventsdf.rename(
                columns={"id": "event_id", "period": "period_id", "match_id": "game_id"},
                inplace=True,
            )
            # eventsdf.to_csv('events.csv',header=True, index=False)
        else: # from csv
            for d, data_id in enumerate(all_data_id):
                eventsdf = pd.read_csv(str(os.path.join(self._root, f"{competition_id}/"+data_id+"/events.csv")))
                # eventsdf.assign(location=0)
                # eventsdf[ev:ev+1]["location"] = np.zeros((2,)).tolist()
                if load_360:
                    eventsdf = eventsdf.assign(visible_area_360=0)
                    eventsdf = eventsdf.assign(freeze_frame_360=[[] for _ in range(len(eventsdf))]) # 0)# 
                locations = trajectories[d][:,1:].reshape((-1,22,2))
                locations[:,:,0] *= 105
                locations[:,:,1] *= 68
                
                for ev, event in eventsdf.iterrows(): 
                    for ballevent in ["pass", "shot", "carry"]:
                        
                        if ballevent in event["extra"]:
                            time_event = event["minute"]*60+event["second"]+event["millisecond"]/1000
                            idx = np.abs(trajectories[d][:,0] - time_event).argmin()
                            player = event["player_name"]
                            ind_player = player_names.index(player)
                            #if "pass" in event["extra"]: 
                            #    extra.get("duel", {}).get("type", {}).get("name")
                            #p_x = location[idx,ind_player,0] # trajectories[d][idx,ind_player*2+1:ind_player*2+2]*105
                            # p_y = location[idx,ind_player,1] # trajectories[d][idx,ind_player*2+2:ind_player*2+3]*68
                            # eventsdf.at[ev, 'location'] = np.concatenate([p_x,p_y],0).tolist()
                            # eventsdf[ev:ev+1]["location"] = pd.DataFrame(trajectories[d][idx,ind_player*2+1:ind_player*2+3]).str.split(',')
                            eventsdf.at[ev, 'location'] = locations[idx,ind_player].tolist()   
                            if "pass" in event["extra"] or "carry" in event["extra"]: 
                                # next event　df = df
                                time_event2 = (eventsdf[ev+1:ev+2]["minute"]*60+eventsdf[ev+1:ev+2]["second"]+eventsdf[ev+1:ev+2]["millisecond"]/1000).values[0]
                                idx2 = np.abs(trajectories[d][:,0] - time_event2).argmin()
                                #player2 = eventsdf[ev+1:ev+2]["player_name"].str
                                player2 = eventsdf.at[ev+1, 'player_name']
                                ind_player2 = player_names.index(player2)
                                #p_x2 = trajectories[d][idx2,ind_player2*2+1:ind_player2*2+2]*105
                                #p_y2 = trajectories[d][idx2,ind_player2*2+2:ind_player2*2+3]*68
                                #eventsdf.at[ev+1, 'location'] = np.concatenate([p_x2,p_y2],0).tolist()
                                eventsdf.at[ev+1, 'location'] = locations[idx2,ind_player2].tolist()

                                # end location
                                extra_str = str(eventsdf.at[ev, 'extra']).replace("[]",str(eventsdf.at[ev+1, 'location']))
                                eventsdf.at[ev, 'extra'] = ast.literal_eval(extra_str)    

                            if load_360:
                                try: eventsdf.at[ev,"freeze_frame_360"] = locations[idx].reshape((44,)).tolist()
                                except: import pdb; pdb.set_trace()
                                '''ball_location = locations[idx,ind_player]
                                location_ = locations[idx].copy()
                                # attackers
                                #nonnan_at = np.where(~np.isnan(location[:11,0]))
                                # nan_at = np.where(np.isnan(location[:11,0]))
                                #nonzero_at = np.where(np.abs(location[:11,0])>0.0001)
                                #zero_at = np.where(np.abs(location[:11,0])<0.0001)
                                #dist2_at = np.sum((location[nonnan_at] - np.repeat(ball_location[:,np.newaxis].T,len(nonnan_at[0]),axis=0))**2,axis=1)
                                # at_nearest = np.concatenate([np.argsort(dist2_at),nan_at[0]])
                                dist2_at = np.sum((location_[:11] - np.repeat(ball_location[np.newaxis,:],11,axis=0))**2,axis=1)
                                at_nearest = np.argsort(dist2_at)
                                location[idx,:11] = location_[at_nearest]
                                
                                # defenders
                                #nonnan_df = np.where(~np.isnan(location[11:,0]))
                                #nan_df = np.where(np.isnan(location[11:,0]))
                                #nonzero_df = np.where(np.abs(location[11:,0])>0.0001)
                                #zero_df = np.where(np.abs(location[11:,0])<0.0001)
                                #dist2_df = np.sum((location[11+nonnan_df[0]] - np.repeat(ball_location[:,np.newaxis].T,len(nonnan_df[0]),axis=0))**2,axis=1)
                                # df_nearest = np.concatenate([11+np.argsort(dist2_df),11+nan_df[0]])
                                # goalkeeper first
                                dist2_df = np.sum((location_[11:-1] - np.repeat(ball_location[np.newaxis,:],10,axis=0))**2,axis=1)
                                df_nearest = np.argsort(dist2_df)
                                try: location[idx,12:] = location_[df_nearest]
                                except: import pdb; pdb.set_trace()
                                location[idx,11] = location_[-1]
                                eventsdf.at[ev,"freeze_frame_360"] = location[idx]'''

                            else:
                                if type(eventsdf.at[ev, 'extra']) != dict:
                                    eventsdf.at[ev, 'extra'] = ast.literal_eval(eventsdf.at[ev, 'extra'])  
                        else:
                            if type(eventsdf.at[ev, 'extra']) != dict:
                                eventsdf.at[ev, 'extra'] = ast.literal_eval(eventsdf.at[ev, 'extra']) 


                                    
                if Plot_raw:
                    figure_dir = os.path.join("../data-RDD/figure", f"{competition_id}/"+data_id+"/")
                    if not os.path.isdir(figure_dir):
                        os.makedirs(figure_dir)    

                    markersize = 3
                    TextCircle = 1
                    for ev, event in eventsdf.iterrows(): 
                        for ballevent in ["pass", "shot", "carry"]:
                            if ballevent in event["extra"]:
                                fig = plt.figure(figsize=(24, 12))
                                self.plotCourt()
                                time_event = event["minute"]*60+event["second"]+event["millisecond"]/1000
                                idx = np.abs(trajectories[d][:,0] - time_event).argmin()
                                p_x = trajectories[d][idx,1::2]*105
                                p_y = trajectories[d][idx,2::2]*68
                                team_A = np.concatenate([p_x[np.newaxis,:11,np.newaxis],p_y[np.newaxis,:11,np.newaxis]],2)
                                team_B = np.concatenate([p_x[np.newaxis,11:,np.newaxis],p_y[np.newaxis,11:,np.newaxis]],2)
                                self.plotPosition3(team_A,team_B,0,markersize,TextCircle)
                                timestr = str(event["minute"])+"_"+str(event["second"])+"_"+str(event["millisecond"])
                                plt.savefig(figure_dir+"plot_"+timestr+".png", bbox_inches='tight')
                                plt.close()
                                print(figure_dir+"plot_"+timestr+".png was plotted")

                if d == 0:
                    eventsdfs = eventsdf.copy()
                else:
                    eventsdfs = pd.concat([eventsdfs, eventsdf.copy()], axis=0)

        # if load_360:
        # Load the 360 data
        # cols_360 = ["visible_area_360", "freeze_frame_360"]
        # eventsdf["visible_area_360"] = None
        # if self._local:
        #    obj = _localloadjson(str(os.path.join(self._root, f"three-sixty/{game_id}.json")))
        #else:
        #    obj = sb.frames(game_id, fmt="dict", creds=self._creds)
        #if not isinstance(obj, list):
        #    raise ParseError("The retrieved data should contain a list of frames")
        #if len(obj) == 0:
        #    eventsdf["freeze_frame_360"] = None
        #    return eventsdf[cols + cols_360].pipe(DataFrame[RDDEventSchema])
        #framesdf = pd.DataFrame(obj).rename(
        #    columns={
        #        "event_uuid": "event_id",
        #        "visible_area": "visible_area_360",
        #        "freeze_frame": "freeze_frame_360",
        #    },
        #)[["event_id", "visible_area_360", "freeze_frame_360"]]
        # return pd.merge(eventsdf, framesdf, on="event_id", how="left")[cols + cols_360].pipe(
        #    DataFrame[RDDEventSchema]
        # )
        eventsdfs = eventsdfs.reset_index(drop=True)
        return eventsdfs

    def plotPosition3(self,team_A,team_B,i,markersize,TextCircle,allPlayers=False,ball=None,K=11):
        if allPlayers:
            start_pl = 0  
            Tm = 2
        else:
            start_pl = 0  if K == 5 else 1
            Tm = 2

        n_all_agent = 10 if K == 5 else 22
        n_team_agent = 5 if K == 5 else 11

        ax = plt.gca() 
        im_team = [[],[]] 

        for tm in range(Tm): 
            pos = team_A if tm == 0 else team_B
            clr = 'b' if tm == 0 else 'r'
            for j in range(start_pl,n_team_agent):
                xx = pos[i, j, 0]
                yy = pos[i, j, 1]         

                if TextCircle == 1: # player circle

                    if tm == 1: # offense
                        im_team[tm] = patches.Circle((xx,yy), radius=markersize-0.15,
                                        fc=clr,ec='k')
                    else:
                        im_team[tm] = patches.CirclePolygon((xx,yy), radius = markersize,
                            resolution = 3, fc = clr, ec = "k") # triangle

                    ax.add_patch(im_team[tm])

                    # player jersey # (text)
                    ax.text(xx,yy,str(j+1),color='w',ha='center',va='center')
                elif TextCircle == 2:
                    ax.text(xx,yy,str(j+1),color='k',ha='center',va='center')

                else: 
                    im_team[tm] = plt.scatter(xx,yy, marker=".", s=markersize, ec=clr, color=clr)

        im_teamA, im_teamB = im_team
        return im_teamA, im_teamB 

    def plotCourt(self):
        plt.xlim(0,52.5*2)  #  -52.5~52.5, -34~34 10, 40
        plt.ylim(0,34*2) # -34,34 -21, 21

        plt.vlines(52.5, 34-34, 34+34, linestyles="solid") # center line
        plt.vlines(52.5+36, 34-20.16, 34+20.16, linestyles="solid") # penalty area
        plt.hlines(-20.16+34, 52.5+36, 52.5+52.5, linestyles="solid")
        plt.hlines(20.16+34, 52.5+36, 52.5+52.5, linestyles="solid")
        plt.vlines(52.5-36, -20.16+34, 20.16+34, linestyles="solid")
        plt.hlines(-20.16+34, 52.5-36, 52.5-52.5, linestyles="solid")
        plt.hlines(20.16+34, 52.5-36, 52.5-52.5, linestyles="solid")
        plt.vlines(52.5+47, -9.16+34, 9.16+34, linestyles="solid") # goal area
        plt.hlines(9.16+34, 52.5+47, 52.5+52.5, linestyles="solid")
        plt.hlines(-9.16+34, 52.5+47, 52.5+52.5, linestyles="solid")
        plt.vlines(52.5-47, -9.16+34, 9.16+34, linestyles="solid")
        plt.hlines(9.16+34, 52.5-47, 52.5-52.5, linestyles="solid")
        plt.hlines(-9.16+34, 52.5-47, 52.5-52.5, linestyles="solid")

    def trajectories(self, competition_id: int):
        """Return a list of arrays with the trajectories of a game.

        Returns
        -------
        pd.DataFrame
            A list of arrays containing the trajectories. 
        """
        cols = [
            "at1",
            "at2",
            "at3",
            "at4",
            "at5",
            "at6",
            "at7",
            "at8",
            "at9",
            "at10",
            "at_goal",
            "df1",
            "df2",
            "df3",
            "df4",
            "df5",
            "df6",
            "df7",
            "df8",
            "df9",
            "df10",
            "df_goal",
        ]
        # Load the trajectory
        TowardRight = []
        if self._local:
            max_length = 10000
            path = os.path.join(self._root, f"{competition_id}")
            files = os.listdir(str(path))
            all_data_id = [f for f in files if os.path.isdir(os.path.join(path, f))]
            # all_data_id = os.listdir(str(os.path.join(self._root, f"{competition_id}")))
            trajectories = [np.zeros((max_length,22*2+1)) for _ in range(len(all_data_id))]
            for d, data_id in enumerate(all_data_id):
                all_players = [i.split(os.sep)[-1].split('.')[0] for i in glob.glob(str(os.path.join(self._root, f"{competition_id}/"+data_id+'/*.srt')))]
                for player in all_players:
                    ind_player = cols.index(player)
                    trajectory = np.zeros((max_length,3))
                    subs = pysrt.open(str(os.path.join(self._root, f"{competition_id}/"+data_id+"/"+player+".srt")))
                    length_ = len(subs)
                    for t in range(length_):
                        trajectory[t,0] = subs[t].start.hours*3600+subs[t].start.minutes*60+subs[t].start.seconds+subs[t].start.milliseconds/1000
                        dic = ast.literal_eval(subs[t].text.replace('[','').replace(']','').replace('false','False').replace('true','True'))
                        trajectory[t,1] = dic["p_x"]
                        trajectory[t,2] = dic["p_y"]
                    
                    time_index = int(np.floor(trajectory[0,0]/(1/30)))
                    if trajectory[0,0] < 0.001:
                        trajectories[d][:,ind_player*2+1:ind_player*2+3] = trajectory[:,1:]
                    else:
                        trajectories[d][time_index:,ind_player*2+1:ind_player*2+3] = trajectory[:-time_index,1:]

                    zero_index = np.min(np.where(trajectories[d][:,0]==0)[0][1:])
                    if zero_index == 1 and trajectory[0,0] < 0.001:
                        trajectories[d][:,0] = trajectory[:,0]
                    elif trajectory[0,0] < 0.001:
                        trajectories[d][zero_index:,0] = trajectory[zero_index:,0]
                    elif zero_index > 1 and zero_index-time_index >= 0:
                        try: trajectories[d][zero_index:,0] = trajectory[zero_index-time_index:-time_index,0]
                        except: import pdb; pdb.set_trace()
                zero_index = np.min(np.where(trajectories[d][:,0]==0)[0][1:])
                trajectories[d] = trajectories[d][:zero_index]

                if np.mean(trajectories[d][:,1::2],axis=(0,1)) < 0.5: # home always attacks toward right
                    trajectories[d][:,1::2] = 1 - trajectories[d][:,1::2]
                    trajectories[d][:,2::2] = 1 - trajectories[d][:,2::2]
                    TowardRight.append(False)
                else:
                    TowardRight.append(True)
        else:
            error('remote is TBD')
            obj = list(sb.events(game_id, fmt="dict", creds=self._creds).values())
        
        meta_trajs = [cols, all_data_id, TowardRight]
        return trajectories, meta_trajs

    def events(self, game_id: int, load_360: bool = False) -> DataFrame[RDDEventSchema]:
        """Return a dataframe with the event stream of a game.

        Parameters
        ----------
        game_id : int
            The ID of the game.
        load_360 : bool
            Whether to load the 360 data.

        Raises
        ------
        ParseError
            When the raw data does not adhere to the expected format.

        Returns
        -------
        pd.DataFrame
            A dataframe containing the event stream. See
            :class:`~socceraction.spadl.RDD.RDDEventSchema` for the schema.
        """
        cols = [
            "game_id",
            "event_id",
            "period_id",
            "team_id",
            "player_id",
            "type_id",
            "type_name",
            "index",
            "timestamp",
            "minute",
            "second",
            "possession",
            "possession_team_id",
            "possession_team_name",
            "play_pattern_id",
            "play_pattern_name",
            "team_name",
            "duration",
            "extra",
            "related_events",
            "player_name",
            "position_id",
            "position_name",
            "location",
            "under_pressure",
            "counterpress",
        ]
        # Load the events
        if self._local:
            obj = _localloadjson(str(os.path.join(self._root, f"events/{game_id}.json")))
        else:
            obj = list(sb.events(game_id, fmt="dict", creds=self._creds).values())

        if not isinstance(obj, list):
            raise ParseError("The retrieved data should contain a list of events")
        if len(obj) == 0:
            return pd.DataFrame(columns=cols).pipe(DataFrame[RDDEventSchema])

        eventsdf = pd.DataFrame(_flatten_id(e) for e in obj)
        eventsdf["match_id"] = game_id
        eventsdf["timestamp"] = pd.to_datetime(eventsdf["timestamp"], format="%H:%M:%S.%f")
        eventsdf["related_events"] = eventsdf["related_events"].apply(
            lambda d: d if isinstance(d, list) else []
        )
        eventsdf["under_pressure"] = eventsdf["under_pressure"].fillna(False).astype(bool)
        eventsdf["counterpress"] = eventsdf["counterpress"].fillna(False).astype(bool)
        eventsdf.rename(
            columns={"id": "event_id", "period": "period_id", "match_id": "game_id"},
            inplace=True,
        )
        if not load_360:
            return eventsdf[cols].pipe(DataFrame[RDDEventSchema])

        # Load the 360 data
        cols_360 = ["visible_area_360", "freeze_frame_360"]
        if self._local:
            obj = _localloadjson(str(os.path.join(self._root, f"three-sixty/{game_id}.json")))
        else:
            obj = sb.frames(game_id, fmt="dict", creds=self._creds)
        if not isinstance(obj, list):
            raise ParseError("The retrieved data should contain a list of frames")
        if len(obj) == 0:
            eventsdf["visible_area_360"] = None
            eventsdf["freeze_frame_360"] = None
            return eventsdf[cols + cols_360].pipe(DataFrame[RDDEventSchema])
        framesdf = pd.DataFrame(obj).rename(
            columns={
                "event_uuid": "event_id",
                "visible_area": "visible_area_360",
                "freeze_frame": "freeze_frame_360",
            },
        )[["event_id", "visible_area_360", "freeze_frame_360"]]
        return pd.merge(eventsdf, framesdf, on="event_id", how="left")[cols + cols_360].pipe(
            DataFrame[RDDEventSchema]
        )


def extract_player_games(events: pd.DataFrame) -> pd.DataFrame:
    """Extract player games [player_id, game_id, minutes_played] from RDD match events.

    Parameters
    ----------
    events : pd.DataFrame
        DataFrame containing RDD events of a single game.

    Returns
    -------
    player_games : pd.DataFrame
        A DataFrame with the number of minutes played by each player during the game.
    """
    game_minutes = max(events[events.type_name == "Half End"].minute)

    game_id = events.game_id.mode().values[0]
    players = {}
    for startxi in events[events.type_name == "Starting XI"].itertuples():
        team_id, team_name = startxi.team_id, startxi.team_name
        for player in startxi.extra["tactics"]["lineup"]:
            player = _flatten_id(player)
            player = {
                **player,
                **{
                    "game_id": game_id,
                    "team_id": team_id,
                    "team_name": team_name,
                    "minutes_played": game_minutes,
                },
            }
            players[player["player_id"]] = player
    for substitution in events[events.type_name == "Substitution"].itertuples():
        replacement = substitution.extra["substitution"]["replacement"]
        replacement = {
            "player_id": replacement["id"],
            "player_name": replacement["name"],
            "minutes_played": game_minutes - substitution.minute,
            "team_id": substitution.team_id,
            "game_id": game_id,
            "team_name": substitution.team_name,
        }
        players[replacement["player_id"]] = replacement
        # minutes_played = substitution.minute
        players[substitution.player_id]["minutes_played"] = substitution.minute
    pg = pd.DataFrame(players.values()).fillna(0)
    for col in pg.columns:
        if "_id" in col:
            pg[col] = pg[col].astype(int)  # pylint: disable=E1136,E1137
    return pg


def _flatten_id(d: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    newd = {}
    extra = {}
    for k, v in d.items():
        if isinstance(v, dict):
            if "id" in v and "name" in v:
                newd[k + "_id"] = v["id"]
                newd[k + "_name"] = v["name"]
            else:
                extra[k] = v
        else:
            newd[k] = v
    newd["extra"] = extra
    return newd


def _flatten(d: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    newd = {}
    for k, v in d.items():
        if isinstance(v, dict):
            if "id" in v and "name" in v:
                newd[k + "_id"] = v["id"]
                newd[k + "_name"] = v["name"]
                newd[k + "_extra"] = {l: w for (l, w) in v.items() if l in ("id", "name")}
            else:
                newd = {**newd, **_flatten(v)}
        else:
            newd[k] = v
    return newd
