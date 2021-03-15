# Offline Reinforcement Learning on Atari

This project work was executed as part of the masters curriculum "Artifical Intelligence"
at Johannes Kepler University Linz (Institute of Machine Learning) for the course
"Practical work in AI - . This work aims to compare
two different recent publication in the realm of Offline Reinforcement Learning,
that reported very different results for standard off-policy algorithms on the 
Atari Environment ([Bellemare et al., 2012](https://arxiv.org/abs/1207.4708)).

While [Agarwal et al., 2020](https://arxiv.org/abs/1907.04543) reported very optimistic resutls,
[Fujimoto et al., 2019](https://arxiv.org/abs/1910.01708) reported very pessimistic results.
I hypothesized, that the main difference between the two works is how they generated the dataset,
the offline agent is trained on. For further information, please refer to my [report]().

### Prerequisites

Install the conda environment via the [environment](./environment.yml) file, to install the necessary
python version as well as the necessary dependencies. Note that while most things should be updateable
to the newest versions, atari_py which is needed to run the Atari games, requires python <= 3.7.

### Train the Agents

Basically all of the training is done via calls of [train.py](./train.py), which takes the following
command line arguments:

    --online            Agent gets trained online
    --offline           Agent gets trained offline
    --agent <agent>     Defines which algorithm to train, default=dqn
    --runs #            How many runs will be executed, default=3
    --config <config>   Configuration file to load for the experiment, default=experiment
    --seed #            Seed to use, default=42

Note that <action> can be the following:

* dqn
* rem
* qrdqn
* bcq
* random

Which correspond to the algorithms DQN ([Mnih et al., 2014](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)), 
REM ([Agarwal et al., 2020](https://arxiv.org/abs/1907.04543)), QR-DQN ([Dabney et al., 2017](https://arxiv.org/abs/1710.10044))
and BCQ ([Fujimoto et al., 2019](https://arxiv.org/abs/1910.01708)) as well as a Random policy.

The experimental details themselves are defined in the configuration file, the example file
would be [experiment](./config/experiment.cfg) and the other experiments covered in the report are
executed with the other respective config files. Note that for every configuration, an online
run has to be executed prior to the offline run, otherwise no dataset is available for offline training.

This would be the minimal example, where an online DQN agent gets trained and an offline DQN agent is trained
on the dataset created by the online DQN agent.

    python train.py --online

This would be the trivial case, here the experiment config file gets loaded and dqn is trained on seed 42. runs does not
affect the number of runs in the online mode though! After training you obtain two models, labelled with run 1 and 2,
which correspond to the model at the end of training and the best model during training. After training one would execute

    python train.py --offline --agent dqn

which trains an offline DQN on the dataset created through the prior comamnd, for 3 runs. The seed is only the basement for the 3 runs and does actually
get incremented after each run.

### Test the Agents

Testing the agent is pretty simple, the agent is simply visualized for a single episode. Further,
a coverage estimate for the state space can be done. Both is handled by the following parameters:

    --online            Use the online version of the agent
    --config <config>   Which experimental configuration shall be used, default=experiment
    --agent <agent>     Which agent shall be used, default=dqn
    --seed #            Seed to use, default=42
    --run #             The agent obtained from which run shall be used, default=1
    --coverage          Estimate state coverage

So basically three usage modes are possible:

    python test.py 

Show the best policy of offline DQN in the "experiment" configuration of run 1

    python test.py --online

Show the last policy of online DQN in the "experiment" configuration, --run 2 would give the best policy during training.

    python test.py --coverage

Estimates the coverage of the dataset, plots are found in [results](results) folder.
    