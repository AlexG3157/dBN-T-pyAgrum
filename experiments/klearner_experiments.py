import pandas as pd

import csv
import os

import time

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../ktbn')))

from KTBN import KTBN
from KLearner import KLearner
from pyagrum.lib.discreteTypeProcessor import DiscreteTypeProcessor


# Log functions
def already_done(config_dict, path="data/klearner_results.csv"):
    config_id = str(tuple(config_dict.values()))

    if not os.path.exists(path):
        return False
    
    with open(path, newline='') as file:
        reader = csv.DictReader(file)
        return any(row.get('config_id') == config_id for row in reader)


def log_result(config_dict, metrics, path="data/klearner_results.csv"):

    file_exists = os.path.exists(path)

    config_id = str(tuple(config_dict.values()))  
    result = {'config_id': config_id}
    result.update(config_dict)
    result.update(metrics)

    with open(path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=result.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


file_path = "data/klearner_results.csv"

# Search space for KLearner experiments
k_true_range = [2, 3, 5, 7, 10]  # True k to be discovered
max_k_range = [8, 12, 15]  # Maximum k tested by KLearner
n_traj_range = [500, 1500, 2500]  # More data needed for k learning
traj_length_range = [20, 35, 50]  # Longer trajectories for better k estimation

n_vars_range = [4]
n_mods_range = [3]
density_range = [0.5]

repetitions = 10  # Fewer repetitions as KLearner is slower

for mods in n_mods_range:
    for density in density_range:
        for n_vars in n_vars_range:
            for k_true in k_true_range:
                for max_k in max_k_range:
                    if max_k < k_true:  # Skip if max_k is less than true k
                        continue
                    for n_traj in n_traj_range:
                        for traj_length in traj_length_range:

                            config_dict = {
                                      'mods': mods,
                                      'density': density, 
                                      'n_vars': n_vars, 
                                      'k_true': k_true,
                                      'max_k': max_k,
                                      'n_traj': n_traj, 
                                      'traj_length': traj_length
                                      }
                            
                            print("Starting config: ", config_dict)
                            
                            if already_done(config_dict, path=file_path):
                                print("Config found on file, skipping.")
                                continue

                            for i in range(repetitions):
                                
                                n_arcs = int(density*(k_true/2)*((k_true-1)*n_vars**2 + n_vars*(n_vars-1)))

                                true_ktbn = KTBN.random(k_true, n_vars, mods, n_arcs, delimiter='_')
                                trajs = true_ktbn.sample(n_traj, traj_length)

                                # Learning k using KLearner
                                klearner = KLearner(trajs, DiscreteTypeProcessor(), delimiter='_')

                                learning_start_time = time.perf_counter()
                                learned_ktbn = klearner.learn(max_k)
                                learning_end_time = time.perf_counter()

                                k_learning_time = learning_end_time - learning_start_time

                                learned_k = klearner.get_best_k()
                                best_bic_score = klearner.get_best_bic_score()

                                # KLearner specific metrics
                                data = {
                                    'k_learning_time': k_learning_time,
                                    'learned_k': learned_k,
                                    'best_bic_score': best_bic_score,
                                    'k_accuracy': 1 if learned_k == k_true else 0,
                                    'k_error': abs(learned_k - k_true) if learned_k is not None else max_k
                                }

                                # Optional: all BIC scores tested
                                bic_scores = klearner.get_bic_scores()
                                for k_test, bic_score in bic_scores.items():
                                    data[f'bic_k_{k_test}'] = bic_score

                                print(data)
                                log_result(config_dict, data, path=file_path)

                            print("Config done!")
