#!/bin/bash

date_experiment=$(date +"%Y%m%d%H%M")
$HOME/.local/bin/poetry install

# poetry install

# GVDEP (TBD, but uefa 360 data can be used)
# data360=(euro2020 euro2022)
# # training with uefa statsbomb data
# for info in ${data360[@]};do
#         python train_GVDEP_opendata.py --data statsbomb --game ${info} --n_nearest 11 --no_games -1 --test --teamView England #--skip_convert_rawdata --skip_preprocess --skip_train
# done
# # Storing, calculating and showing F1-scores by each "n_nearest" and "CV".
# for info in ${data360[@]};do
#         python calc_f1scores.py --game ${info} --no_games -1 --calculate_f1scores --show_f1scores
# done

data360=(euro2020 euro2022)

# training with uefa statsbomb data
for info in ${data360[@]};do
        # poetry run python main_evaluate.py --data statsbomb --game ${info} --n_nearest 11 --no_games -1 --date_opendata 20231002 --date_experiment ${date_experiment} --model xgboost --test --teamView England
        poetry run python main_evaluate.py --data statsbomb --game ${info} --n_nearest 11 --no_games -1 --date_opendata 20231002 --date_experiment ${date_experiment} --model xgboost --test --teamView England
        poetry run python main_verify.py --data statsbomb --game ${info} --date_opendata 20231002 --date_experiment ${date_experiment} --model xgboost --k_fold 5
done


# FIFA World Cup
# python train_GVDEP_opendata.py --data statsbomb --game wc2022 --n_nearest 11 --no_games -1 # --test --teamView Argentina --skip_convert_rawdata --skip_preprocess --skip_train
# python calc_f1scores.py --game wc2022 --no_games -1 --calculate_f1scores --show_f1scores