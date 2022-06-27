import argparse
import os
import pickle
import numpy as np

def compute_stats(logdir):
    raw_data = pickle.load(open(os.path.join(logdir, 'raw_data.pkl'), 'rb'))
    exp_cfg = pickle.load(open(os.path.join(logdir, 'args.pkl'), 'rb'))
    exp_stats = {
        'total_successes': 0,
        'total_viols': 0,
        'total_hard_resets': 0,
        'total_switches': 0,
        'total_human_actions': 0,
        'total_idle_time': 0,
        'total_reward': 0,
        'cumulative_successes': [0],
        'cumulative_viols': [0],
        'cumulative_hard_resets': [0],
        'cumulative_switches': [0],
        'cumulative_human_actions': [0],
        'cumulative_idle_time': [0],
        'cumulative_reward': [0]
    }
    for i in range(exp_cfg.num_envs):
        exp_stats[i] = {
            'num_successes': 0,
            'num_viols': 0,
            'num_hard_resets': 0,
            'num_human_actions': 0,
            'idle_time': 0,
            'cumulative_successes': [0],
            'cumulative_viols': [0],
            'cumulative_hard_resets': [0],
            'cumulative_human_actions': [0],
            'cumulative_idle_time': [0]
        }
    assignments = np.zeros((exp_cfg.num_envs, exp_cfg.num_humans))
    for t in range(len(raw_data)):
        num_switches = np.sum(np.abs(assignments - raw_data[t]['assignments']).sum(1) > 0)
        exp_stats['total_switches'] += num_switches
        exp_stats['cumulative_switches'].append(exp_stats['cumulative_switches'][-1] + num_switches)
        assignments = np.copy(raw_data[t]['assignments'])
        exp_stats['cumulative_viols'].append(exp_stats['cumulative_viols'][-1])
        exp_stats['cumulative_successes'].append(exp_stats['cumulative_successes'][-1])
        exp_stats['cumulative_hard_resets'].append(exp_stats['cumulative_hard_resets'][-1])
        exp_stats['cumulative_human_actions'].append(exp_stats['cumulative_human_actions'][-1])
        exp_stats['cumulative_idle_time'].append(exp_stats['cumulative_idle_time'][-1])
        exp_stats['cumulative_reward'].append(exp_stats['cumulative_reward'][-1])
        for i in range(exp_cfg.num_envs):
            exp_stats[i]['cumulative_hard_resets'].append(exp_stats[i]['cumulative_hard_resets'][-1])
            exp_stats[i]['cumulative_human_actions'].append(exp_stats[i]['cumulative_human_actions'][-1])
            exp_stats[i]['cumulative_idle_time'].append(exp_stats[i]['cumulative_idle_time'][-1])
            exp_stats[i]['cumulative_viols'].append(exp_stats[i]['cumulative_viols'][-1])
            exp_stats[i]['cumulative_successes'].append(exp_stats[i]['cumulative_successes'][-1])
            use_human_ac = np.sum(assignments[i]) == 1
            if raw_data[t]['reward'][i]:
                exp_stats['total_reward'] += raw_data[t]['reward'][i]
                exp_stats['cumulative_reward'][-1] += raw_data[t]['reward'][i]
            if use_human_ac and raw_data[t]['constraint'][i]:
                exp_stats[i]['num_hard_resets'] += 1
                exp_stats[i]['cumulative_hard_resets'][-1] += 1
                exp_stats['total_hard_resets'] += 1
                exp_stats['cumulative_hard_resets'][-1] += 1
            if use_human_ac:
                exp_stats['total_human_actions'] += 1
                exp_stats['cumulative_human_actions'][-1] += 1
                exp_stats[i]['num_human_actions'] += 1
                exp_stats[i]['cumulative_human_actions'][-1] += 1
            if raw_data[t]['idle'][i]:
                exp_stats[i]['idle_time'] += 1
                exp_stats[i]['cumulative_idle_time'][-1] += 1
                exp_stats['total_idle_time'] += 1
                exp_stats['cumulative_idle_time'][-1] += 1
            if raw_data[t]['info'][i] and raw_data[t]['info'][i]['constraint']:
                exp_stats[i]['num_viols'] += 1
                exp_stats['total_viols'] += 1
                exp_stats[i]['cumulative_viols'][-1] += 1
                exp_stats['cumulative_viols'][-1] += 1
            if raw_data[t]['info'][i] and raw_data[t]['info'][i]['success']:
                exp_stats[i]['num_successes'] += 1
                exp_stats['total_successes'] += 1
                exp_stats[i]['cumulative_successes'][-1] += 1
                exp_stats['cumulative_successes'][-1] += 1
    pickle.dump(exp_stats, open(os.path.join(logdir, 'run_stats.pkl'), 'wb'))
    return len(raw_data), exp_stats['total_reward'], exp_stats['total_successes'], exp_stats['total_viols'], \
        exp_stats['total_switches'], exp_stats['total_human_actions'], exp_stats['total_idle_time']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', type=str, help='filepath to log directory (parent of raw_data.pkl)')
    args = parser.parse_args()
    t, rew, succ, viol, switch, human, idle = compute_stats(args.logdir)
    print("Steps: %d Successes: %d Violations: %d Switches: %d Human Acts: %d Idle Time: %d"%(
        t, succ, viol, switch, human, idle))
