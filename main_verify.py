import argparse
import os
import pickle
import random
import time
import warnings

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text

import train_models as tm

pd.set_option("display.max_columns", None)
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings(action="ignore", message="credentials were not supplied. open data access only")

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
        "--seed", type=int, default=0
        )
    parser.add_argument(
        "--n_nearest", type=int, default=11
        )
    parser.add_argument(
        "--numProcess", type=int, default=16
        )
    parser.add_argument(
        "--skip_calculate_f1scores", action="store_true"
        )
    parser.add_argument(
        "--predict_actions", action="store_true"
        )
    parser.add_argument(
        "--model", type=str, default="xgboost", required=True,
        help="Please select the model to be used for training. Now the options is only 'xgboost'.",
        )
    parser.add_argument(
        "--k_fold", type=int, default=5
        )
    parser.add_argument(
        "--grid_search", action="store_true"
        )
    parser.add_argument(
        "--pickle", type=int, default=4
        )
    args, _ = parser.parse_known_args()

    return args


def create_train_test_sets(games, k_fold):
    num_games = len(games)
    idx_games = np.arange(num_games)
    np.random.shuffle(idx_games)
    test_sets = np.array_split(idx_games, k_fold)
    train_sets = []
    for i, _ in enumerate(test_sets):
        train_set = np.concatenate([
            x for j, x in enumerate(test_sets) if j != i
            ])
        train_sets.append(train_set)

    return train_sets, test_sets


def boxplot_per_metric(arrays, probs_cols, figuredir, metric):
    cols = probs_cols + ["n_nearest"]
    df = pd.DataFrame(arrays, columns=cols)
    df = df.groupby("n_nearest")

    # Show figures by Prediction of the Probabilities
    # alphabet = ["(a)", "(b)", "(c)", "(d)"]
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(16, 12),
        sharex="col",
        sharey=True,
    )
    # plt.subplots_adjust(hspace=0.2)
    for i in range(len(probs_cols)):
        ax = axes[i // 2, i % 2]
        # ax.set_title(f'{cols[prob]}')
        ax.boxplot(
            [df.get_group(n).iloc[:, i] for n in range(12)],
            labels=[f"{n}" for n in range(12)],
            showmeans=True,
            showfliers=True,
            medianprops=dict(color="blue"),
            flierprops=dict(marker="x", markeredgecolor="black"),
        )
        ax.set_title(
            cols[i], fontsize=32, loc="left"
            )

        # set details
        ax.set_ylim(0.0, 1.0)
        ax.tick_params(axis="both", labelsize=24)

    fig.supxlabel(
        "The number of nearest attacker/defender to the ball (n_nearest)", 
        fontsize=32
        )
    fig.supylabel(
        metric, fontsize=32
        )
    plt.tight_layout()
    fig.savefig(
        os.path.join(figuredir, f"{metric}.png")
        )
    print(
        os.path.join(figuredir, f"{metric}.png") + " is saved"
        )


def main():
    args = parse_args()

    pickle.HIGHEST_PROTOCOL = args.pickle  # 4 is the highest in an older version of pickle
    numProcess = args.numProcess
    os.environ["OMP_NUM_THREADS"] = str(numProcess)

    start = time.time()

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

    # Preparation
    if "euro" in args.game or "wc" in args.game:
        spadl_h5 = os.path.join(datafolder, "spadl-statsbomb.h5")
        features_h5 = os.path.join(datafolder, "features.h5")
        labels_h5 = os.path.join(datafolder, "labels.h5")
    else:
        raise ValueError("Please select the game from 'euro' or 'wc'.")

    n_nearest = args.n_nearest

    if args.predict_actions:
        modelfolder = datafolder + "/vaep_framework/models"
    else:
        modelfolder = datafolder + "/models"
    os.makedirs(modelfolder, exist_ok=True)
    args.modelfolder = modelfolder
    predictions_h5 = os.path.join(modelfolder, "predictions.h5")
    games = pd.read_hdf(spadl_h5, "games")
    print("nb of games:", len(games))

    # note: only for the purpose of this example and due to the small dataset,
    # we use the same data for training and evaluation
    if args.skip_calculate_f1scores:
        print("Calculating f1scores is skipped")
    else:
        # Create train/test sets
        train_sets, test_sets = create_train_test_sets(games, args.k_fold)

        # Split games randomly by the number of k-fold for cross validation
        for nearest in range(0, n_nearest+1, 1):
            args.n_nearest = nearest
            for cv, sets in enumerate(zip(train_sets, test_sets), 1):
                model_str = f"CV_{cv}_{args.k_fold}_all_{nearest}_nearest"
                args.sets = sets
                args.model_str = model_str

                # Train the model
                tm.save_prediction_for_verification(
                    args, games,
                    spadl_h5, features_h5, labels_h5, predictions_h5,
                    use_location = True, use_polar = True
                    )

                print(f"Evaluation time of {model_str} : ", time.time() - start)

    # Show the graphs about F1scores of each probabilities
    probs_cols = ["scores", "concedes", "gains", "effective_attacks"]

    n_nearest_array = np.repeat(np.arange(args.n_nearest+1), args.k_fold)
    briers, lls, roc_aucs, f1scores = (
        np.empty((args.k_fold * (args.n_nearest + 1), len(probs_cols))),
        np.empty((args.k_fold * (args.n_nearest + 1), len(probs_cols))),
        np.empty((args.k_fold * (args.n_nearest + 1), len(probs_cols))),
        np.empty((args.k_fold * (args.n_nearest + 1), len(probs_cols))),
    )
    for nearest in range(0, args.n_nearest+1, 1):
        for cv in range(1, args.k_fold+1, 1):
            model_str = f"static_CV_{cv}_{args.k_fold}_all_{nearest}_nearest"
            with open(os.path.join(modelfolder, model_str + ".pkl"), "rb") as f:
                result_stats = pickle.load(f)
            briers[(cv-1) + (nearest * args.k_fold), :] = result_stats["briers"]
            lls[(cv-1) + (nearest * args.k_fold), :] = result_stats["lls"]
            roc_aucs[(cv-1) + (nearest * args.k_fold), :] = result_stats["roc_aucs"]
            f1scores[(cv-1) + (nearest * args.k_fold), :] = result_stats["f1scores"]

    briers = np.hstack([briers, n_nearest_array.reshape(-1, 1)])
    lls = np.hstack([lls, n_nearest_array.reshape(-1, 1)])
    roc_aucs = np.hstack([roc_aucs, n_nearest_array.reshape(-1, 1)])
    f1scores = np.hstack([f1scores, n_nearest_array.reshape(-1, 1)])
    
    # Show figures by Prediction of the Probabilities
    boxplot_per_metric(
        briers, probs_cols, figuredir, "Brier scores"
        )
    boxplot_per_metric(
        lls, probs_cols, figuredir, "Log-likelihoods"
        )
    boxplot_per_metric(
        roc_aucs, probs_cols, figuredir, "ROC AUCs"
        )
    boxplot_per_metric(
        f1scores, probs_cols, figuredir, "F1 scores"
        )

    import pdb; pdb.set_trace()


if __name__ == "__main__":
    main()
