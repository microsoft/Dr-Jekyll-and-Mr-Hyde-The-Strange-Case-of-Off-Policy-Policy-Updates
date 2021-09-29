# Deep implementation of Dr Jekyll and Mr Hyde

## Prerequisites

Check the ``requirements.txt`` file for dependencies.

## Usage

To reproduce our results, run:

    python results/Four_Rooms_Static.py --agent {agent} --level 1

with agent in: ``[JH_Discrete, SAC_Discrete, SAC_DiscreteRND, DDQN, DDQNRND]``.

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

This part of the code was built on top of the repo: https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch published under the MIT License. Its reference is:

```
@misc{pchrist,
	Author = {Petros Christodoulou},
	Howpublished = {\url{https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch}},
	Journal = {GitHub repository},
	Publisher = {GitHub},
	Title = {Deep Reinforcement Learning Algorithms with PyTorch},
	Year = {2019}
}
```