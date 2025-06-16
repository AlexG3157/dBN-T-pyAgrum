import random
import unittest
import pandas as pd
import numpy as np
from pyagrum.lib.discreteTypeProcessor import DiscreteTypeProcessor

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../ktbn')))


import Learner
from KTBN import KTBN

class TestLearner(unittest.TestCase):

    def setUp(self):

        self.df1 = pd.DataFrame(
            {'A': np.ones(20), 
             'B': range(20), 
             'C' : np.ones(20)*2,
             'D' : range(50,70)
             })
        self.df2 = pd.DataFrame(
            {'A': np.ones(20)*5, 
             'B': np.ones(20), 
             'C' : np.ones(20)*48,
             'D' : range(20)
             })

        self.dfs = [self.df1, self.df2]
        self.discreteTypeProcessor = DiscreteTypeProcessor()
        self.delimiter = '_'
        self.k = 4  
        self.learner = Learner.Learner(self.dfs, self.discreteTypeProcessor, self.delimiter, self.k)

    def test_init_learner_valid(self):
        """
        Test that a Learner initializes correctly 
        """
        self.assertEqual(['A', 'C'], self.learner._atemporal_vars)
        self.assertEqual(['B', 'D'], self.learner._temporal_vars)

    def test_get_delimiter(self):
        """
        Ensure get_delimiter works correctly.
        """
        self.assertEqual(self.learner.get_delimiter(), self.delimiter)

    def test_verify_timeslice_valid(self):
        """
        Verify that a valid time slice order does not raise an error.
        """
        try:
            Learner._verify_timeslice(1, 2)
        except ValueError:
            self.fail("verify_timeslice raised ValueError unexpectedly for valid input.")

    def test_verify_timeslice_invalid(self):
        """
        Verify that an invalid time slice (arc from future to past) raises ValueError.
        """
        with self.assertRaises(ValueError):
            Learner._verify_timeslice(2, 1)

    def test_is_atemporal(self):
        """
        Verifies that Learner can correctly classify temporal and atemporal variables.
        """

        self.assertTrue(Learner._is_atemporal(self.dfs, 'A'))
        self.assertFalse(Learner._is_atemporal(self.dfs, 'B'))

    def test_create_sequences(self):
        """
        Verifies the structure of the database created from the list of trajectories. 
        """

        lags, first = Learner._create_sequences(self.dfs, self.k, self.delimiter, 
                                               self.learner._temporal_vars, self.learner._atemporal_vars)
        
        # Check that the atemporal variables are in the sequences. 
        self.assertIn('A', lags.columns)
        self.assertIn('A', first.columns)
        self.assertIn('C', lags.columns)
        self.assertIn('C', first.columns)

        # Check that all of the lagged columns are in the sequences and that no atemporal variable
        # was treated as temporal.
        for i in range(self.k):
            
            self.assertIn(KTBN.encode_name_static('B',i,self.delimiter), lags.columns)
            self.assertIn(KTBN.encode_name_static('B',i,self.delimiter), first.columns)
            self.assertIn(KTBN.encode_name_static('D',i,self.delimiter), lags.columns)
            self.assertIn(KTBN.encode_name_static('D',i,self.delimiter), first.columns)
            
            self.assertNotIn(KTBN.encode_name_static('A',i,self.delimiter), lags.columns)
            self.assertNotIn(KTBN.encode_name_static('A',i,self.delimiter), first.columns)
            self.assertNotIn(KTBN.encode_name_static('C',i,self.delimiter), lags.columns)
            self.assertNotIn(KTBN.encode_name_static('C',i,self.delimiter), first.columns)


        # Ensure that no extra columns are created

        self.assertNotIn(KTBN.encode_name_static('B',self.k,self.delimiter), lags.columns)
        self.assertNotIn(KTBN.encode_name_static('B',self.k,self.delimiter), first.columns)
        self.assertNotIn(KTBN.encode_name_static('D',self.k,self.delimiter), lags.columns)
        self.assertNotIn(KTBN.encode_name_static('D',self.k,self.delimiter), first.columns)

        # Ensure that the length of the first df is 2 (only two trajectories were provided).
        self.assertEqual(len(first), 2)

    def test_learn_ktbn_structure(self):
        """
        Verifies that the learning algorithm yields a KTBN with the expected variables. 
        """
        ktbn = self.learner.learn_ktbn().to_bn()

        # Check that the KTBN contains temporal variables from all time slices.
        for col in self.learner._temporal_vars:
            for i in range(self.k):

                self.assertIn(KTBN.encode_name_static(col, i, self.delimiter), ktbn.names())
        
        # Ensure no extra time slices are added.
                self.assertNotIn(KTBN.encode_name_static(col, self.k, self.delimiter), ktbn.names())

        # Check that the KTBN contains atemporal variables and that they were not treated as temporal.
        for col in self.learner._atemporal_vars:
            self.assertIn(col, ktbn.names())

            for i in range(self.k):
                self.assertNotIn(KTBN.encode_name_static(col, i, self.delimiter), ktbn.names())

    def test_no_k_learning(self):
        """
        Verifies that the K-selection algorithm is not executed when K is specified.
        """
        self.assertIsNone(self.learner._log_likelihoods)
        self.assertIsNone(self.learner._bic_scores)
        self.assertIsNone(self.learner._best_bic_score)

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
        
        # Create a simple discreteTypeProcessor
        self.discreteTypeProcessor = DiscreteTypeProcessor()
        
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
        self.learner = Learner.Learner(self.trajectories, self.discreteTypeProcessor, delimiter=self.delimiter)

    def test_bic_learning_basic(self):
        """
        Test basic BIC learning functionality.
        """
        # Create KLearner
        
        learned_ktbn = self.learner.learn_ktbn(max_k=6)
        
        # Verify that optimal k is defined
        best_k = self.learner.get_k()
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
        self.learner.learn_ktbn(max_k=5)
        
        # Get BIC scores
        bic_scores = self.learner.get_bic_scores()
        
        # Verify that BIC scores are calculated for each k
        self.assertEqual(len(bic_scores), 4)  # k from 2 to 5
        for k in range(2, 6):
            self.assertIn(k, bic_scores)
            self.assertIsInstance(bic_scores[k], float)
        
        # Verify that best BIC score is defined
        best_bic = self.learner.get_best_bic_score()
        self.assertIsNotNone(best_bic)
        self.assertIsInstance(best_bic, float)

    def test_bic_minimization(self):
        """
        Test that the optimal k indeed minimizes the BIC score.
        """
        self.learner.learn_ktbn(max_k=5)
        
        bic_scores = self.learner.get_bic_scores()
        best_k = self.learner.get_k()
        best_bic = self.learner.get_best_bic_score()
        
        # Verify that the best k has the minimum BIC score
        min_bic = min(bic_scores.values())
        self.assertEqual(best_bic, min_bic)
        self.assertEqual(bic_scores[best_k], min_bic)
        

    def test_log_likelihood_compatibility(self):
        """
        Test that log-likelihood methods still work for backward compatibility.
        """
        self.learner.learn_ktbn(max_k=4)
        
        # Get log-likelihoods
        log_likelihoods = self.learner.get_log_likelihoods()
        
        # Verify that log-likelihoods are calculated for each k
        self.assertEqual(len(log_likelihoods), 3)  # k from 2 to 4
        for k in range(2, 5):
            self.assertIn(k, log_likelihoods)
            self.assertIsInstance(log_likelihoods[k], float)

    def test_bic_formula_validation(self):
        """
        Test that BIC formula is correctly applied.
        """
        self.learner.learn_ktbn(max_k=3)
        
        bic_scores = self.learner.get_bic_scores()
        log_likelihoods = self.learner.get_log_likelihoods()
        
        # Calculate expected n_data_points
        expected_n = len(self.trajectories) * len(self.trajectories[0])
        
        # For one value of k, manually verify BIC calculation
        k_test = 2
        if k_test in bic_scores:
            # Get the learned network for this k
            learner = Learner.Learner(self.trajectories, self.discreteTypeProcessor, delimiter='_', k=k_test)
            bn = learner.learn_ktbn().to_bn()
            
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
        
        # Learn the KTBN with optimal k (testing up to k=8)
        ktbn = self.learner.learn_ktbn(max_k=8)
        
        # Display the optimal k found
        print(f"\n=== BIC-based Learning Results ===")
        print(f"Optimal k found (BIC): {self.learner.get_k()}")
        print(f"Best BIC score: {self.learner.get_best_bic_score():.2f}")
        
        # Display BIC scores and log-likelihoods for each k
        bic_scores = self.learner.get_bic_scores()
        log_likelihoods = self.learner.get_log_likelihoods()
        
        print("\nComparison BIC vs Log-likelihood by k:")
        for k in sorted(bic_scores.keys()):
            print(f"  k={k}: BIC={bic_scores[k]:.2f}, log-likelihood={log_likelihoods[k]:.2f}")
        
        # Find what would be optimal with log-likelihood only
        best_ll_k = max(log_likelihoods.keys(), key=lambda k: log_likelihoods[k])
        print(f"\nBest k by log-likelihood only: {best_ll_k}")
        print(f"Best k by BIC: {self.learner.get_k()}")
        
        if best_ll_k != self.learner.get_k():
            print("BIC and log-likelihood give different optimal k values!")
        else:
            print("BIC and log-likelihood agree on optimal k.")
        
        # Verify that optimal k is defined
        self.assertIsNotNone(self.learner.get_k())
        
        # Verify that BIC scores are consistent
        best_bic = self.learner.get_best_bic_score()
        for k, bic in bic_scores.items():
            # The optimal k must have the lowest BIC score
            self.assertGreaterEqual(bic, best_bic)


        # Ensure that the KTBN has k=best_k
        self.assertEqual(self.learner.get_k(), ktbn.get_k())
        self.assertEqual(self.learner.get_best_bic_score(), bic_scores[self.learner.get_k()])


