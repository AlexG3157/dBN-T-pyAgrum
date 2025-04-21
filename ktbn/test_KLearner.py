import unittest
import pandas as pd
import numpy as np
import pyAgrum as gum
from pyAgrum.lib.discretizer import Discretizer
import random

from KTBN import KTBN
from KLearner import KLearner

class TestKLearner(unittest.TestCase):
    """
    Tests for the KLearner class.
    """

    def setUp(self):
        """
        Preparation of data and objects for testing.
        """
        # Create a simple discretizer
        self.discretizer = Discretizer()
        
        # Create a more complex KTBN with k=3
        k = 3
        self.ktbn = KTBN(k=k, delimiter='_')
        
        # Add temporal variables
        self.ktbn.addVariable("A", temporal=True)
        self.ktbn.addVariable("B", temporal=True)
        self.ktbn.addVariable("C", temporal=True)
        self.ktbn.addVariable("D", temporal=True)
        
        # Add atemporal variables
        self.ktbn.addVariable("E", temporal=False)
        self.ktbn.addVariable("F", temporal=False)
        
        # Add intra-slice arcs (within the same time-slice)
        for t in range(k):
            self.ktbn.addArc(("A", t), ("B", t))
            self.ktbn.addArc(("B", t), ("C", t))
            self.ktbn.addArc(("C", t), ("D", t))
            self.ktbn.addArc(("A", t), ("D", t))
        
        # Add inter-slice arcs (between consecutive time-slices)
        for t in range(k-1):
            self.ktbn.addArc(("A", t), ("A", t+1))
            self.ktbn.addArc(("B", t), ("B", t+1))
            self.ktbn.addArc(("C", t), ("C", t+1))
            self.ktbn.addArc(("D", t), ("D", t+1))
        
        # Add arcs from atemporal variables
        for t in range(k):
            self.ktbn.addArc(("E", -1), ("A", t))
            self.ktbn.addArc(("F", -1), ("C", t))
        
        # Add cross-time-slice arcs
        for t in range(k-1):
            self.ktbn.addArc(("B", t), ("A", t+1))
            self.ktbn.addArc(("C", t), ("D", t+1))
            self.ktbn.addArc(("D", t), ("B", t+1))
        
        # Random generation of CPTs
        self.ktbn._bn.generateCPTs()
        
        # Generate trajectories from this KTBN
        self.trajectories = self.ktbn.sample(n_trajectories=5, trajectory_len=10)
        
        # Diagnostic information to understand potential errors
        print("\n===== DIAGNOSTIC INFORMATION =====")
        print(f"Number of generated trajectories: {len(self.trajectories)}")

        if len(self.trajectories) > 0:
            first_traj = self.trajectories[0]
            print(f"\nDimensions of the first trajectory: {first_traj.shape}")
            print(f"Available columns in the first trajectory:\n{list(first_traj.columns)}")
            print("\nPreview of the first trajectory:")
            print(first_traj.head(3))

        print(f"\nTemporal variables in the KTBN:\n{self.ktbn._temporal_variables}")
        print(f"Atemporal variables in the KTBN:\n{self.ktbn._atemporal_variables}")

        print("\nNames in the underlying BN:")
        print(sorted(self.ktbn._bn.names()))

    def test_learn_and_print_results(self):
        """
        Test learning and display results.
        """
        # 1. Get trajectories from KTBN for testing
        trajectories = self.ktbn.sample(n_trajectories=5, trajectory_len=10)
        
        # 2. Pass trajectories directly to KLearner
        # KLearner will call prepare_for_learner internally
        
        klearner = KLearner(trajectories, self.discretizer, delimiter='_')
        
        # Learn the KTBN with optimal k (testing up to k=10)
        ktbn = klearner.learn(max_k=10)
        
        # Display the optimal k found
        print(f"\nOptimal k found: {klearner.get_best_k()}")
        
        # Display log-likelihoods for each k
        log_likelihoods = klearner.get_log_likelihoods()
        print("\nLog-likelihoods by k:")
        for k, ll in sorted(log_likelihoods.items()):
            print(f"  k={k}: {ll}")
        
        # Verify that optimal k is defined
        self.assertIsNotNone(klearner.get_best_k())
        
        # Verify that log-likelihoods are consistent
        best_k = klearner.get_best_k()
        best_ll = log_likelihoods[best_k]
        for k, ll in log_likelihoods.items():
            # The optimal k must have the highest log-likelihood
            self.assertLessEqual(ll, best_ll)

if __name__ == '__main__':
    unittest.main(verbosity=2)
