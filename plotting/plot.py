"""
Plot given a parent logdir where each subdirectory is an experiment logdir (and each is replicated 3x for 3 seeds)
and a statistic (e.g., cumulative_successes, cumulative_viols, cumulative_idle_time)
"""
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
import os

if len(sys.argv) < 3:
    assert False, "usage: python plot.py [logdir] [key]"
directory = sys.argv[1]
KEY = sys.argv[2] # e.g. 'cumulative_successes'
files = sorted(os.listdir(directory), key=lambda x:x[::-1])
filedatas = [pickle.load(open(directory+'/'+f+'/run_stats.pkl','rb'))[KEY] for f in files]
minlen = min([len(fd) for fd in filedatas])
filedatas = [fd[:minlen] for fd in filedatas]
if len(sys.argv) == 4:
    KEY2 = sys.argv[3]
    filedatas2 = [pickle.load(open(directory+'/'+f+'/run_stats.pkl', 'rb'))[KEY2] for f in files]
    filedatas2 = [fd[:minlen] for fd in filedatas2]
# load data
colors = ['#4daf4a', '#ff7f00','#377eb8', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']
for i in range(0,len(files),3):
    label = '{}'.format(files[i][files[i].rindex('_')+1:])
    data = np.array(filedatas[i:i+3])
    if len(sys.argv) == 4:
        data2 = np.array(filedatas2[i:i+3])
        LAMBDA=0.01
        data = data-LAMBDA*data2
    plt.plot(data.mean(axis=0), label=label, color=colors[i//3])
    plt.fill_between(np.arange(minlen), data.mean(axis=0)-data.std(axis=0), data.mean(axis=0)+data.std(axis=0), alpha=0.2, color=colors[i//3])
plt.legend()
if len(sys.argv) == 4:
    plt.title('{}-{}'.format(KEY, KEY2))
    plt.savefig('cumulative_diff.jpg'.format(KEY, KEY2), bbox_inches='tight')
else:
    plt.title('{}'.format(KEY))
    plt.savefig('{}.jpg'.format(KEY), bbox_inches='tight')
        
