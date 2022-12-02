# Generalized-VDEP
Generalized VDEP is a generalisation of an existing indicator (VDEP).
We hope that this indicator will allow a quantitative assessment of defensive teams in football.

## Requirements

* python 3.7
* To install requirements:

```shell
python3 -m pip install -r requirements.txt
```
## Data
* StatsBomb opendata: https://github.com/statsbomb/open-data/ (I modified the filename (``open-data-master``) to ``statsbomb-open-rawdata``)
* SPADL format representation: https://socceraction.readthedocs.io/en/latest/documentation/SPADL.html

## Usage

* See `run.sh` 

## Reference
* [VDEP (previous work)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0263051)
```tex
@article{toda2022evaluation,
  title={Evaluation of soccer team defense based on prediction models of ball recovery and being attacked: A pilot study},
  author={Toda, Kosuke and Teranishi, Masakiyo and Kushiro, Keisuke and Fujii, Keisuke},
  journal={Plos one},
  volume={17},
  number={1},
  pages={e0263051},
  year={2022},
  publisher={Public Library of Science San Francisco, CA USA}
}
```

* [Generalized VDEP (this work)](https://arxiv.org/abs/2212.00021)
```tex
@article{umemoto2022location,
  title={Location analysis of players in UEFA EURO 2020 and 2022 using generalized valuation of defense by estimating probabilities},
  author={Umemoto, Rikuhei and Tsutsui, Kazushi and Keisuke,Fujii},
  journal={arXiv preprint arXiv:2212.00021},
  year={2022}
}
``` 
