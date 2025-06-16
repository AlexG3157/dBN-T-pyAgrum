import pandas as pd
import numpy as np

import pyagrum as gum
from pyagrum.lib.discreteTypeProcessor import DiscreteTypeProcessor

from typing import List, Optional, Dict

from KTBN import KTBN
from Learner import Learner


class KLearner:
    """
    A class for learning a K-Time Slice Bayesian Network (KTBN) without knowing the k in advance.
    It finds the optimal k by minimizing the BIC score of the trajectories given the model.
    """
    
    def __init__(self, trajectories: List[pd.DataFrame], discreteTypeProcessor: DiscreteTypeProcessor, delimiter: str = '#'):
        """
        Initialize the KLearner.
        
        Args:
            trajectories (List[pd.DataFrame]): List of trajectories.
            discreteTypeProcessor (DiscreteTypeProcessor): DiscreteTypeProcessor for template generation.
            delimiter (str, optional): Delimiter for variable names. Defaults to '#'.
        """
        self.trajectories = trajectories
        self._discreteTypeProcessor = discreteTypeProcessor
        self.delimiter = delimiter
        self._best_k = None
        self._best_ktbn = None
        self._best_bic_score = None
        self._bic_scores = {}
        self._log_likelihoods = {}
        self._constant_variables = []
        self._filtered_trajectories = []
        
        # Detect and filter constant variables during initialization
        self._detect_and_filter_constant_variables()
    
    def learn(self, max_k: int = 10) -> KTBN:
        """
        Learn a KTBN with the optimal k by minimizing the BIC score.
        
        Args:
            max_k (int, optional): Maximum k to test. Defaults to 10.
            
        Returns:
            KTBN: KTBN with the optimal k.
        """
        best_bic_score = float('inf')  # Minimiser le BIC
        best_k = None
        best_ktbn = None
                
        for k in range(2, max_k + 1):
            learner = Learner(self._filtered_trajectories, self._discreteTypeProcessor, delimiter=self.delimiter, k=k)
            ktbn = learner.learn_ktbn()
            
            # Calculer le score BIC
            bic_score = self._calculate_bic_score(ktbn, self._filtered_trajectories)
            self._bic_scores[k] = bic_score
            
            # Garder également la log-vraisemblance pour compatibilité
            log_likelihood = ktbn.log_likelihood(self._filtered_trajectories)
            self._log_likelihoods[k] = log_likelihood
            
            # Garder le k qui minimise le BIC
            if bic_score < best_bic_score:
                best_bic_score = bic_score
                best_k = k
                best_ktbn = ktbn
        
        self._best_k = best_k
        self._best_ktbn = best_ktbn
        self._best_bic_score = best_bic_score
        
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
    
    def get_bic_scores(self) -> Dict[int, float]:
        """
        Get the BIC scores for each k tested.
        
        Returns:
            Dict[int, float]: Dictionary mapping k to BIC score.
        """
        return self._bic_scores
    
    def get_best_bic_score(self) -> Optional[float]:
        """
        Get the best BIC score found.
        
        Returns:
            Optional[float]: Best BIC score or None if learn() has not been called.
        """
        return self._best_bic_score
    
    @staticmethod
    def _count_data_points(trajectories: List[pd.DataFrame]) -> int:
        """
        Calculate the total number of data points across all trajectories.
        
        Args:
            trajectories (List[pd.DataFrame]): List of trajectories
            
        Returns:
            int: n_trajectories × trajectory_len
        """
        total_points = 0
        for trajectory in trajectories:
            total_points += len(trajectory)
        return total_points
    
    @staticmethod
    def _calculate_bic_score(ktbn: KTBN, trajectories: List[pd.DataFrame]) -> float:
        """
        Calculate the BIC score for a given KTBN.
        
        BIC = k * ln(n) - 2 * ln(L̂)
        where k = number of parameters, n = number of data points, L̂ = likelihood
        
        Args:
            ktbn (KTBN): The KTBN model
            trajectories (List[pd.DataFrame]): The trajectories
            
        Returns:
            float: BIC score (lower is better)
        """
        bn = ktbn.to_bn()
        
        # Log-vraisemblance
        log_likelihood = ktbn.log_likelihood(trajectories)
        
        # Nombre de paramètres (dimension du réseau)
        k_params = bn.dim()
        
        # Nombre total de points de données
        n_data_points = KLearner._count_data_points(trajectories)
        
        # Formule BIC
        bic_score = k_params * np.log(n_data_points) - 2 * log_likelihood
        
        return bic_score
    
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
    