# Developing Artificial Intelligence Agents to Manipulate Quantum Entanglement

This repository provides all the code needed to run the
experiments that support the claims in the text of the
master thesis titled <i>`Developing Artificial Intelligence
Agents to Manipulate Quantum Entanglement'</i>.


## How to run the code
Download the repository locally:
```
git clone https://github.com/cacao-macao/entanglement-control.git
```

The code was written using `Python 3.10`. All of the dependencies
can be found in the `requirements.txt` package and installed using:
```
pip3 install -r requirements.txt
```

To start a training procedure run the following script from inside
the `scripts` folder:
```
cd path-to-repo/entanglement-control/scripts
python3 pg_train_agent.py -q 5 --env_batch 1024 --steps 40 -i 10001 --ereg 0.01
```

The results from the training procedure will be stored inside the
`logs` folder.
Note that running 10001 iterations on a 5-qubit quantum system
takes around 19 hours on a TeslaT4 GPU.


## How to read the code
The code in the repository has to the following structure:
```bash
  |--- scripts
  |   |--- ...      # scripts for running different simulations
  |--- src
  |   |--- agents   # implementations of different RL agents
  |   |--- envs     # implementation of a quantum simulator
  |   |--- infrastructure # logging and utilities
  |   |--- policies # implementations of policies using deep learning
```


The following agents for controlling entanglement are implemented:
  * `base_agent.py` provides an interface for the agent object
  * `ac_agent.py`   implements an actor-critic agent following the
advantage actor-critic algorithm
  * `pg_agent.py`   implements a policy gradient agent following
the vanilla policy gradient algorithm
  * `il_agent.py`   implements an imitation learning agent following
the behaviour cloning algorithm
  * `expert.py`     implements an agent using beam search

For implementing the agents the following API is used:
  * agents have a `policy` and an `environment` object,
  * every agent has a `rollout()` method that starts an
    agent-environment interaction and produces an episode,
  * every agent has a `train()` method that starts producing
    rollouts and uses the experiences from those rollouts to
    update the policy parameters.

The main policy model used in the experiments is the `fcnn_policy.py`.
This package implements a fully-connected neural network using
`PyTorch`.

