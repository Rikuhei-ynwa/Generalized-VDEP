# [Generalized-VDEP](https://arxiv.org/pdf/2212.00021)
This is an official implementation of "[Generalized-VDEP](https://arxiv.org/pdf/2212.00021)". (Submitted to IJCSS)

### Installation
0. Prerequisites
Before running this project, make sure you have the following requirements
- Python 3.7.
- Poetry: This project uses Poetry to manage dependencies. Install Poetry before using this repository.

1. Clone this repository:
```
$ git clone https://github.com/Rikuhei-ynwa/Generalized-VDEP.git
```

2. Install dependencies:
```
$ cd Generalized-VDEP
$ poetry install
```

3. Install StatsBomb data
```
$ git clone https://github.com/statsbomb/open-data.git
```

This open data is often updated, so if you want to reproduce the results of the paper, you should checkout the commit hash of the URL mentioned in the paper.
```
$ cd ./open-data-master
$ git checkout 533862946a73608c134d18b78226b6371ce7173c
```
Also, to run the code, you need to rename the directory to `open-data-20240702`.

### Evaluation
You can set the country name in the `teamView` option. The following command evaluates the performance of the England team in the UEFA EURO 2020 tournament.
```
$ cd ./Generalized-VDEP
$ poetry run python main_evaluate.py --data statsbomb --game euro2020 --n_nearest 11 --no_games -1 --date_opendata 20240702 --date_experiment ${date_experiment} --model xgboost --test --teamView England
```

### Verification
Attention: This verification process must be performed after the evaluation process.
``` 
$ poetry run python main_verify.py --data statsbomb --game euro2020 --date_opendata 20240702 --date_experiment ${date_experiment} --model xgboost --k_fold 5
```

### Results folder
The results of the evaluation and verification are saved in the `./GVDEP_data` directory. The resulting directory structure will be as follows:
```
📁 Generalized-VDEP/
📁 open-data-2024-0702/
📁 GVDEP_data/
└─📁 data-statsbomb/
  └─📁 euro2020/
    └─📁 ${date_experiment}/
      └─(The results of running the code are summarized)
```

## Citation
If our work is useful for your project, please consider citing the paper:
```tex
@article{umemoto2022location,
  title={Location analysis of players in UEFA EURO 2020 and 2022 using generalized valuation of defense by estimating probabilities},
  author={Umemoto, Rikuhei and Tsutsui, Kazushi and Fujii, Keisuke},
  journal={arXiv preprint arXiv:2212.00021},
  year={2022}
}
```

## Acknowledgements
We appreciate the following repositories:
- [ML-KULeuven/socceraction](https://github.com/ML-KULeuven/socceraction)
- [statsbomb/open-data](https://github.com/statsbomb/open-data)

## License
- This software is created under MIT License same as [ML-KULeuven/socceraction](https://github.com/ML-KULeuven/socceraction)

## Contact
If you have any questions, please contact author:
- Rikuhei Umemoto (umemoto.rikuhei[at]g.sp.m.is.nagoya-u.ac.jp)
