import pyAgrum as gum
import pyAgrum.lib.bn_vs_bn as gcm
from pyAgrum.lib.discretizer import Discretizer

import pandas as pd

import csv
import os

import time

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../ktbn')))

from KTBN import KTBN
from KLearner import KLearner
from Learner import Learner


# Log functions
def already_done(config_dict, path="grid_results.csv"):
    config_id = str(tuple(config_dict.values()))

    if not os.path.exists(path):
        return False
    
    with open(path, newline='') as file:
        reader = csv.DictReader(file)
        return any(row.get('config_id') == config_id for row in reader)


def log_result(config_dict, metrics, path="grid_results.csv"):

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


file_path = "data/learner_ntraj_length.csv"

#Search space
k_range = [2,5,7,10]
n_traj_range = [100,500,1500,2500]
traj_length_range = [11,20,35,50]

n_vars_range = [4]
n_mods_range =[3]
density_range = [0.5]

repetitions = 20

do_learner = True
do_KLearner = False

for mods in n_mods_range:
    for density in density_range:
        for n_vars in n_vars_range:
            for k in k_range:
                for n_traj in n_traj_range:
                    for traj_length in traj_length_range:

                        config_dict = {
                                  'mods': mods,
                                  'density':density, 
                                  'n_vars': n_vars, 
                                  'k':k, 
                                  'n_traj':n_traj, 
                                  'traj_length':traj_length,
                                  'do_learner' : do_learner,
                                  'do_KLearner' : do_KLearner
                                  }
                        
                        print("Starting config: " , config_dict)
                        
                        if already_done(config_dict, path=file_path):
                            print("Config found on file, skipping.")
                            continue

                        for i in range(repetitions):
                            
                            n_arcs = int(density*(k/2)*((k-1)*n_vars**2 + n_vars*(n_vars-1)))

                            true_ktbn = KTBN.random(k,n_vars,mods,n_arcs, delimiter='_')
                            trajs = true_ktbn.sample(n_traj, traj_length)

                            data = dict()

                            if do_learner:

                                learner = Learner(trajs, Discretizer(),delimiter='_', k=k)

                                learning_start_time = time.perf_counter()
                                learned_ktbn = learner.learn_ktbn()
                                learning_end_time = time.perf_counter()

                                learning_time = learning_end_time-learning_start_time

                                true_bn = true_ktbn.to_bn()
                                learned_bn = learned_ktbn.to_bn()

                                g = gum.GibbsBNdistance(true_bn, learned_bn)
                                g.setVerbosity(True)
                                g.setMaxTime(120)
                                g.setBurnIn(5000)
                                g.setEpsilon(1e-7)
                                g.setPeriodSize(500)
                                data.update(g.compute())

                                cm = gcm.GraphicalBNComparator(true_bn, learned_bn)

                                data.update(cm.scores())
            
                                skeleton_scores =cm.skeletonScores()

                                for key in skeleton_scores:
                                    data["skeleton_"+key] = skeleton_scores[key]

                                data.update(cm.hamming())
                                data.update({'learning_time' : learning_time})

                            if do_KLearner:

                                klearner = KLearner(trajs, Discretizer(), delimiter='_')

                                learning_start_time = time.perf_counter()
                                klearner.learn(10)
                                learning_end_time = time.perf_counter()

                                k_learning_time = learning_end_time-learning_start_time
        
                                data.update({
                                    'k_learning_time' : k_learning_time,
                                    'BIC_Score' : klearner.get_best_bic_score(),
                                    'learned_k' : klearner.get_best_k()
                                    })

                            print(data)
                            log_result(config_dict, data, path=file_path)

                        print("Config done!")
                         



                        






