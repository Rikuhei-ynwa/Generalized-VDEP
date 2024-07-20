#!/bin/bash

date_experiment=$(date +"%Y%m%d%H%M")
$HOME/.local/bin/poetry install

# poetry install

# GVDEP 
# UEFA Euro 2020
poetry run python main_evaluate.py --data statsbomb --game euro2020 --n_nearest 11 --no_games -1 --date_opendata 20240702 --date_experiment ${date_experiment} --model xgboost --test --teamView England
poetry run python main_verify.py --data statsbomb --game euro2020 --date_opendata 20240702 --date_experiment ${date_experiment} --model xgboost --k_fold 5


# UEFA Women's Euro 2022 (TBD)


# 2022 FIFA World Cup (TBD)


# 2023 FIFA Women's World Cup (TBD)
