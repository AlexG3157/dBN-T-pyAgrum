import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from pyAgrum.lib.discretizer import Discretizer

from KTBN import KTBN
from Learner import Learner

class KLearner:
    """
    A class for learning a K-Time Slice Bayesian Network (KTBN) without knowing the k in advance.
    It finds the optimal k by maximizing the log-likelihood of the trajectories given the model.
    """
    
    def __init__(self, trajectories: List[pd.DataFrame], discretizer: Discretizer, delimiter: str = '#'):
        """
        Initialize the KLearner.
        
        Args:
            trajectories (List[pd.DataFrame]): List of trajectories.
            discretizer (Discretizer): Discretizer for template generation.
            delimiter (str, optional): Delimiter for variable names. Defaults to '#'.
        """
        self.trajectories = trajectories
        self.discretizer = discretizer
        self.delimiter = delimiter
        self._best_k = None
        self._best_ktbn = None
        self._log_likelihoods = {}
    
    def learn(self, max_k: int = 10) -> KTBN:
        """
        Learn a KTBN with the optimal k.
        
        Args:
            max_k (int, optional): Maximum k to test. Defaults to 10.
            
        Returns:
            KTBN: KTBN with the optimal k.
        """
        best_log_likelihood = float('-inf')
        best_k = None
        best_ktbn = None
        
        # Create a temporary KTBN only for preparing trajectories
        # No need to add variables, just to use prepare_for_learner
        temp_ktbn = KTBN(k=1, delimiter=self.delimiter)
        
        # Prepare trajectories once for all k values
        prepared_dfs = temp_ktbn.prepare_for_learner(self.trajectories)
        
        for k in range(1, max_k + 1):
            # Pass prepared data directly to Learner
            # Learner will detect temporal vs. atemporal variables automatically
            learner = Learner(prepared_dfs, self.discretizer, delimiter=self.delimiter, k=k)
            bn = learner.learn_ktbn()
            ktbn = KTBN.from_bn(bn, self.delimiter)
            
            # Calculate log-likelihood on original trajectories
            log_likelihood = ktbn.log_likelihood(self.trajectories)
            self._log_likelihoods[k] = log_likelihood
            
            # Keep the k that maximize the log likelihood
            if log_likelihood > best_log_likelihood:
                best_log_likelihood = log_likelihood
                best_k = k
                best_ktbn = ktbn
        
        self._best_k = best_k
        self._best_ktbn = best_ktbn
        
        return best_ktbn
    
    def get_best_k(self) -> Optional[int]:
        """
        Get the best k found.
        
        Returns:
            Optional[int]: Best k or None if learn() has not been called.
        """
        return self._best_k
    
    def get_log_likelihoods(self) -> Dict[int, float]:
        """
        Get the log-likelihoods for each k tested.
        
        Returns:
            Dict[int, float]: Dictionary mapping k to log-likelihood.
        """
        return self._log_likelihoods
