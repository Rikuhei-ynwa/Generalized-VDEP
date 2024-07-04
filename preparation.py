import argparse
import os
import pickle
import random
import time
import warnings

import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings(action="ignore", message="credentials were not supplied. open data access only")

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="statsbomb")
parser.add_argument("--game", type=str, default="all")
parser.add_argument("--feature", type=str, default="all")
parser.add_argument(
    "--date_opendata", type=int, default=0, required=True, 
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


# 1. load and convert statsbomb data
if args.skip_convert_rawdata:
    print("loading rawdata is skipped")
else:
    import convert_rawdata as crd
    DLoader = crd.set_DLoader(args)
    games, selected_competitions = crd.select_games(DLoader, args)
    actions = crd.store_converted_data(DLoader, datafolder, games, selected_competitions)


# Configure file and folder names
spadl_h5 = os.path.join(datafolder, "spadl-statsbomb.h5")
features_h5 = os.path.join(datafolder, "features.h5")
labels_h5 = os.path.join(datafolder, "labels.h5")

print(pd.read_hdf(spadl_h5, "competitions"))

NB_PREV_ACTIONS = 1

# 2. compute features and labels
if args.skip_preprocess:
    print("preprocessing is skipped")
else:
    import set_variables as sv
    games = pd.read_hdf(spadl_h5, "games")
    print("nb of games:", len(games))

    X = sv.compute_features(spadl_h5, features_h5, NB_PREV_ACTIONS)
    Y = sv.compute_labels(spadl_h5, labels_h5, NB_PREV_ACTIONS)