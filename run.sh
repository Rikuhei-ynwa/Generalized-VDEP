#!/bin/bash

# GVDEP (TBD, but uefa 360 data can be used)
data360=(euro2020 euro2022)
# training with uefa statsbomb data
for info in ${data360[@]};do
        python train_GVDEP_opendata.py --data statsbomb --game ${info} --date 20221003 --n_nearest 11 --no_games -1 --test --teamView England #--skip_load_rawdata --skip_preprocess --skip_train
        python train_GVDEP_opendata.py --data statsbomb --game ${info} --date 20230708 --n_nearest 11 --no_games -1 --test --teamView England #--skip_load_rawdata --skip_preprocess --skip_train
done
# Storing, calculating and showing F1-scores by each "n_nearest" and "CV".
for info in ${data360[@]};do
        python calc_f1scores.py --game ${info} --date statsbomb --date 20221003 --no_games -1 --calculate_f1scores --show_f1scores
        python calc_f1scores.py --game ${info} --date statsbomb --date 20230708 --no_games -1 --calculate_f1scores --show_f1scores
done
