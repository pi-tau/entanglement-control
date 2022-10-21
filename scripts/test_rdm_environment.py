# Author: Stefan
# Date:   23.09.2022

import itertools
import numpy as np
import sys
import os
import pdb

print(os.getcwd())
sys.path.append("..")

from src.envs.rdm_environment import QubitsEnvironment

np.random.seed(14)
np.set_printoptions(precision=4, suppress=True)
env = QubitsEnvironment(num_qubits=3, batch_size=1)


for _ in range(1):
    env.set_random_states()
    PSI = env.states.copy()
    action_set = list(env.actions.keys())
    for a1, a2 in itertools.combinations(action_set, 2):
        print((a1, a2))
        env.states = PSI
        trajectory1 = [a1, a2]
        trajectory2 = [a1, a2, a2]
        # pdb.set_trace()
        states1, Us1, rdms1 = [], [], []
        for a in trajectory1:
            s, _, _ = env.step([a])
            states1.append(s)
            Us1.append(env.last_U.copy())
            rdms1.append(env.last_rdms.copy())
        final1 = env.states.copy()
        env.states = PSI
        states2, Us2, rdms2 = [], [], []
        for a in trajectory2:
            s, _, _ = env.step([a])
            states2.append(s)
            Us2.append(env.last_U.copy())
            rdms2.append(env.last_rdms.copy())
        final2 = env.states.copy()
        assert np.all(np.isclose(final1, final2))
