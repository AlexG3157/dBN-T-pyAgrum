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
        self._constant_variables = []
        self._filtered_trajectories = []
        
        # Detect and filter constant variables during initialization
        self._detect_and_filter_constant_variables()
    
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
        
        
        for k in range(2, max_k + 1):
            # Pass filtered trajectories to Learner (without constant variables)
            # Learner will detect temporal vs. atemporal variables automatically
            learner = Learner(self._filtered_trajectories, self.discretizer, delimiter=self.delimiter, k=k)
            bn = learner.learn_ktbn()
            ktbn = KTBN.from_bn(bn, self.delimiter)
            
            # Calculate log-likelihood on filtered trajectories (without constant variables)
            log_likelihood = ktbn.log_likelihood(self._filtered_trajectories)
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
    
    def _detect_and_filter_constant_variables(self) -> None:
        """
        Detect constant variables across all trajectories and create filtered trajectories.
        """
        if not self.trajectories:
            return
        
        # Detect constant variables
        self._constant_variables = self._detect_constant_variables()
        
        if self._constant_variables:
            # Create filtered trajectories without constant variables
            self._filtered_trajectories = self._filter_constant_variables()
        else:
            # Use original trajectories if no constant variables
            self._filtered_trajectories = self.trajectories.copy()
    
    def _detect_constant_variables(self) -> List[str]:
        """
        Detect which variables are constant across all trajectories.
        
        Returns:
            List[str]: List of variable names that are constant.
        """
        if not self.trajectories:
            return []
        
        constant_vars = []
        
        # Check each column
        for col in self.trajectories[0].columns:
            all_values = set()
            
            # Collect all unique values across all trajectories
            for traj in self.trajectories:
                all_values.update(traj[col].unique())
            
            # If only one unique value, variable is constant
            if len(all_values) == 1:
                constant_vars.append(col)
        
        return constant_vars
    
    def _filter_constant_variables(self) -> List[pd.DataFrame]:
        """
        Create new trajectories without constant variables.
        
        Returns:
            List[pd.DataFrame]: Filtered trajectories.
        """
        filtered_trajectories = []
        
        for traj in self.trajectories:
            # Remove constant variable columns
            filtered_traj = traj.drop(columns=self._constant_variables)
            filtered_trajectories.append(filtered_traj)
        
        return filtered_trajectories
    
    def get_constant_variables(self) -> List[str]:
        """
        Get the list of constant variables that were excluded.
        
        Returns:
            List[str]: List of constant variable names.
        """
        return self._constant_variables.copy()
