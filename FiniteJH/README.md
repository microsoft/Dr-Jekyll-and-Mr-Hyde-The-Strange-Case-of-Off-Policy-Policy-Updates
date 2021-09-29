# Implementation of Dr Jekyll and Mr Hyde in finite MDPs

## Prerequisites

The project is implemented in Python 3.5 and requires *numpy* and *scipy*.

## Usage

This relies on two main files:
* `main_config.py` to run experiments in a fixed setting and report average/quantile performance across time,
* `main_hyperparam.py` to run experiments where a hyperparameter varies, and report the time to reach a fixed normalized performance target.

They both use the same configuration file determining the setting the experiment (the hyperparameter search overwrites the setting of this hyperparameter during the run): which setting (planning or RL), which environment (chain or random MDPs), which algorithms, with which hyperparameters, for how many steps, and for how many runs. Such configuration files may be found in the ``expes`` folder, where the settings of all reported experiments may be retrieved as well as their figures (and some more). Nevertheless, we deleted the experiment result files because they are voluminous (10s of Go), but we have them and can provide them on demand, or better: their random seeds.

The two environments are implemented in python files:
* `chain.py`: the chain environment described in Appendix C.1,
* `garnets.py`: the random MDPs environment described in Appendix C.2.

Two utility python files are used:
* `utils.py` contains all utility files,
* `proj_simplex.py` implementing the projection on the simplex, that has been isolated in order to account for their authors (see reference Blondel et al below).

Finally, all the algorithmic innovation that we claim belongs to `gradient_ascent.py`, including the implementation of Dr Jekyll and Mr Hyde.

## Reference

Please consider citing us if you use this code:

```
@inproceedings{laroche2021,
      title={Dr Jekyll and Mr Hyde: The Strange Case of Off-Policy Policy Updates},
      author={Laroche, Romain and Tachet des Combes, Remi},
      year={2021},
      booktitle={Advances in Neural Information Processing Systems}
}
```

The projection implementation stems from:

```
@inproceedings{Blondel2014,
  title={Large-scale multiclass support vector machine training via euclidean projection onto the simplex},
  author={Blondel, Mathieu and Fujino, Akinori and Ueda, Naonori},
  booktitle={Proceedings of the 22nd International Conference on Pattern Recognition (ICPR)},
  pages={1289--1294},
  year={2014},
  organization={IEEE}
}
```

## License

This project is MIT-licensed.
