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
    Tests for the KLearner class with BIC scoring.
    """

    def setUp(self):
        """
        Preparation of data and objects for testing using random KTBN generation.
        """
        # Set random seed for reproducibility
        random.seed(42)
        np.random.seed(42)
        
        # Create a simple discretizer
        self.discretizer = Discretizer()
        
        # Generate a random KTBN using the random generator
        self.k = 3
        self.n_vars = 5
        self.n_mods = 2
        self.n_arcs = 12
        self.delimiter = '_'
        
        # Generate random KTBN
        self.ktbn = KTBN.random(
            k=self.k,
            n_vars=self.n_vars,
            n_mods=self.n_mods,
            n_arcs=self.n_arcs,
            delimiter=self.delimiter
        )
        
        # Generate trajectories from this random KTBN
        self.trajectories = self.ktbn.sample(n_trajectories=2000, trajectory_len=20)

    def test_bic_learning_basic(self):
        """
        Test basic BIC learning functionality.
        """
        # Create KLearner
        klearner = KLearner(self.trajectories, self.discretizer, delimiter='_')
        
        learned_ktbn = klearner.learn()
        
        # Verify that optimal k is defined
        best_k = klearner.get_best_k()
        self.assertIsNotNone(best_k)
        self.assertGreaterEqual(best_k, 2)
        self.assertLessEqual(best_k, 6)
        
        # Verify that learned KTBN is returned
        self.assertIsNotNone(learned_ktbn)
        self.assertIsInstance(learned_ktbn, KTBN)

    def test_bic_scores_calculation(self):
        """
        Test that BIC scores are correctly calculated and stored.
        """
        klearner = KLearner(self.trajectories, self.discretizer, delimiter='_')
        klearner.learn(max_k=5)
        
        # Get BIC scores
        bic_scores = klearner.get_bic_scores()
        
        # Verify that BIC scores are calculated for each k
        self.assertEqual(len(bic_scores), 4)  # k from 2 to 5
        for k in range(2, 6):
            self.assertIn(k, bic_scores)
            self.assertIsInstance(bic_scores[k], float)
        
        # Verify that best BIC score is defined
        best_bic = klearner.get_best_bic_score()
        self.assertIsNotNone(best_bic)
        self.assertIsInstance(best_bic, float)

    def test_bic_minimization(self):
        """
        Test that the optimal k indeed minimizes the BIC score.
        """
        klearner = KLearner(self.trajectories, self.discretizer, delimiter='_')
        klearner.learn(max_k=5)
        
        bic_scores = klearner.get_bic_scores()
        best_k = klearner.get_best_k()
        best_bic = klearner.get_best_bic_score()
        
        # Verify that the best k has the minimum BIC score
        min_bic = min(bic_scores.values())
        self.assertEqual(best_bic, min_bic)
        self.assertEqual(bic_scores[best_k], min_bic)
        
        # Verify that all other k values have BIC >= best_bic
        for k, bic in bic_scores.items():
            self.assertGreaterEqual(bic, best_bic)

    def test_log_likelihood_compatibility(self):
        """
        Test that log-likelihood methods still work for backward compatibility.
        """
        klearner = KLearner(self.trajectories, self.discretizer, delimiter='_')
        klearner.learn(max_k=4)
        
        # Get log-likelihoods
        log_likelihoods = klearner.get_log_likelihoods()
        
        # Verify that log-likelihoods are calculated for each k
        self.assertEqual(len(log_likelihoods), 3)  # k from 2 to 4
        for k in range(2, 5):
            self.assertIn(k, log_likelihoods)
            self.assertIsInstance(log_likelihoods[k], float)

    def test_bic_formula_validation(self):
        """
        Test that BIC formula is correctly applied.
        """
        klearner = KLearner(self.trajectories, self.discretizer, delimiter='_')
        klearner.learn(max_k=3)
        
        bic_scores = klearner.get_bic_scores()
        log_likelihoods = klearner.get_log_likelihoods()
        
        # Calculate expected n_data_points
        expected_n = len(self.trajectories) * len(self.trajectories[0])
        
        # For one value of k, manually verify BIC calculation
        k_test = 2
        if k_test in bic_scores:
            # Get the learned network for this k
            from Learner import Learner
            learner = Learner(klearner._filtered_trajectories, self.discretizer, delimiter='_', k=k_test)
            bn = learner.learn_ktbn()
            
            # Calculate expected BIC
            k_params = bn.dim()
            log_likelihood = log_likelihoods[k_test]
            expected_bic = k_params * np.log(expected_n) - 2 * log_likelihood
            
            # Compare with stored BIC (allowing for small floating point differences)
            self.assertAlmostEqual(bic_scores[k_test], expected_bic, places=20)

    def test_learn_and_print_results(self):
        """
        Test learning and display comprehensive results (BIC vs log-likelihood).
        """
        # Generate fresh trajectories for testing
        trajectories = self.ktbn.sample(n_trajectories=2000, trajectory_len=20)
        
        # Create KLearner
        klearner = KLearner(trajectories, self.discretizer, delimiter='_')
        
        # Learn the KTBN with optimal k (testing up to k=8)
        ktbn = klearner.learn(max_k=8)
        
        # Display the optimal k found
        print(f"\n=== BIC-based Learning Results ===")
        print(f"Optimal k found (BIC): {klearner.get_best_k()}")
        print(f"Best BIC score: {klearner.get_best_bic_score():.2f}")
        
        # Display BIC scores and log-likelihoods for each k
        bic_scores = klearner.get_bic_scores()
        log_likelihoods = klearner.get_log_likelihoods()
        
        print("\nComparison BIC vs Log-likelihood by k:")
        for k in sorted(bic_scores.keys()):
            print(f"  k={k}: BIC={bic_scores[k]:.2f}, log-likelihood={log_likelihoods[k]:.2f}")
        
        # Find what would be optimal with log-likelihood only
        best_ll_k = max(log_likelihoods.keys(), key=lambda k: log_likelihoods[k])
        print(f"\nBest k by log-likelihood only: {best_ll_k}")
        print(f"Best k by BIC: {klearner.get_best_k()}")
        
        if best_ll_k != klearner.get_best_k():
            print("BIC and log-likelihood give different optimal k values!")
        else:
            print("BIC and log-likelihood agree on optimal k.")
        
        # Verify that optimal k is defined
        self.assertIsNotNone(klearner.get_best_k())
        
        # Verify that BIC scores are consistent
        best_k = klearner.get_best_k()
        best_bic = klearner.get_best_bic_score()
        for k, bic in bic_scores.items():
            # The optimal k must have the lowest BIC score
            self.assertGreaterEqual(bic, best_bic)

    def test_empty_trajectories(self):
        """
        Test behavior with empty trajectories.
        """
        empty_trajectories = []
        klearner = KLearner(empty_trajectories, self.discretizer, delimiter='_')
        
        # This should handle gracefully without crashing
        with self.assertRaises(Exception):
            klearner.learn(max_k=3)

if __name__ == '__main__':
    unittest.main(verbosity=2)
