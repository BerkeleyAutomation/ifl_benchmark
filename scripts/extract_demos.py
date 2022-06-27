"""
This script extracts task demos from a fully supervised run (num robots=1, num humans=1, free_humans=False)
"""
import pickle
import sys
import numpy as np

p = pickle.load(open(sys.argv[1], 'rb')) # raw data log file
obs, act, rew, done, next_obs = list(), list(), list(), list(), list()
for i in range(len(p)):
    if p[i]['real_act'][0] is None:
        continue
    obs.append(p[i]['state'][0])
    act.append(p[i]['real_act'][0])
    rew.append(p[i]['reward'][0])
    done.append(p[i]['done'][0])
    if i < len(p) - 1:
        next_obs.append(p[i+1]['state'][0])
next_obs.append(np.zeros(len(obs[0])))
pickle.dump({'obs': np.stack(obs), 'act': np.stack(act), 'obs2': np.stack(next_obs), 
    'rew': np.array(rew), 'done': np.array(done)}, open('demos.pkl', 'wb'))
