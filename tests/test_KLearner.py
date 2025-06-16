import unittest
import pandas as pd
import numpy as np
import pyagrum as gum
from pyagrum.lib.discreteTypeProcessor import DiscreteTypeProcessor
import random


import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../ktbn')))

from KLearner import KLearner
from KTBN import KTBN

class TestKLearner(unittest.TestCase):
    """
    Tests for the KLearner class.
    """

    def setUp(self):
        """
        Preparation of data and objects for testing.
        """
        # Create a simple discreteTypeProcessor
        self.discreteTypeProcessor = DiscreteTypeProcessor()
        
        # Create a more complex KTBN with k=3
        k = 3
        self.ktbn = KTBN(k=k, delimiter='_')
        
        # Add temporal variables with explicit domains (5 variables)
        var_A = gum.LabelizedVariable("A", "Variable A", ["0", "1"])
        var_B = gum.LabelizedVariable("B", "Variable B", ["0", "1"])
        var_C = gum.LabelizedVariable("C", "Variable C", ["0", "1"])
        var_D = gum.LabelizedVariable("D", "Variable D", ["0", "1"])
        var_G = gum.LabelizedVariable("G", "Variable G", ["0", "1"])
        
        self.ktbn.addVariable(var_A, temporal=True)
        self.ktbn.addVariable(var_B, temporal=True)
        self.ktbn.addVariable(var_C, temporal=True)
        self.ktbn.addVariable(var_D, temporal=True)
        self.ktbn.addVariable(var_G, temporal=True)
        
        # Add atemporal variables with explicit domains (2 variables)
        var_E = gum.LabelizedVariable("E", "Variable E", ["0", "1"])
        var_F = gum.LabelizedVariable("F", "Variable F", ["0", "1"])
        
        self.ktbn.addVariable(var_E, temporal=False)
        self.ktbn.addVariable(var_F, temporal=False)
        
        # Add intra-slice arcs (within the same time-slice)
        for t in range(k):
            # Main causal chain: A→B→C→D
            self.ktbn.addArc(("A", t), ("B", t))
            self.ktbn.addArc(("B", t), ("C", t))
            self.ktbn.addArc(("C", t), ("D", t))
            self.ktbn.addArc(("A", t), ("D", t))
            
            # Cross-influences within same time slice
            self.ktbn.addArc(("A", t), ("G", t))
            self.ktbn.addArc(("G", t), ("D", t))
        
        # Add inter-slice arcs (between consecutive time-slices)
        for t in range(k-1):
            # Auto-correlations (each variable influences itself)
            self.ktbn.addArc(("A", t), ("A", t+1))
            self.ktbn.addArc(("B", t), ("B", t+1))
            self.ktbn.addArc(("C", t), ("C", t+1))
            self.ktbn.addArc(("D", t), ("D", t+1))
            self.ktbn.addArc(("G", t), ("G", t+1))
        
        # Add arcs from atemporal variables
        for t in range(k):
            self.ktbn.addArc(("E", -1), ("A", t))
            self.ktbn.addArc(("F", -1), ("C", t))
        
        # Add some cross-time-slice arcs (simpler temporal dependencies)
        for t in range(k-1):
            self.ktbn.addArc(("B", t), ("A", t+1))
            self.ktbn.addArc(("C", t), ("D", t+1))
            self.ktbn.addArc(("G", t), ("C", t+1))
        
        # Random generation of CPTs
        self.ktbn._bn.generateCPTs()
        
        # Generate trajectories from this KTBN
        self.trajectories = self.ktbn.sample(n_trajectories=5, trajectory_len=10)

    def test_learn_and_print_results(self):
        """
        Test learning and display results.
        """
        # 1. Get trajectories from KTBN for testing
        trajectories = self.ktbn.sample(n_trajectories=5, trajectory_len=10)
        
        # 2. Pass trajectories directly to KLearner
        
        klearner = KLearner(trajectories, self.discreteTypeProcessor, delimiter='_')
        
        # Learn the KTBN with optimal k (testing up to k=10)
        ktbn = klearner.learn(max_k=10)
        
        # Display the optimal k found
        print(f"\nOptimal k found: {klearner.get_best_k()}")
        
        # Display bic scores for each k
        bic_scores = klearner.get_bic_scores()
        print("\nBIC scores by k:")
        for k, ll in sorted(bic_scores.items()):
            print(f"  k={k}: {ll}")
        
        # Verify that optimal k is defined
        self.assertIsNotNone(klearner.get_best_k())
        
        # Verify that the BIC scores are consistent
        best_k = klearner.get_best_k()
        best_bic = bic_scores[best_k]
        for k, ll in bic_scores.items():
            # The optimal k must have the minimum BIC score.
            self.assertGreaterEqual(ll, best_bic)

if __name__ == '__main__':
    unittest.main(verbosity=2)
