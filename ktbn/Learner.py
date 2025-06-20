import numpy as np
import pandas as pd
import pyagrum as gum
from pyagrum.lib.discreteTypeProcessor import DiscreteTypeProcessor
from typing import Dict, List, Optional, Set
from typing import Tuple
from KTBN import KTBN

class Learner():
    """
    A class for learning a K-Time Slice Bayesian Network (KTBN) from trajectory data.
    """
    def __init__(self, dfs : List[pd.DataFrame], discreteTypeProcessor : DiscreteTypeProcessor,delimiter : str = '$', k : int = None ):
        """
        Initializes the learner with trajectory data and parameters.

        Args:
            dfs (List[pd.DataFrame]): A list of trajectory datasets for learning the KTBN.
            discreteTypeProcessor (DiscreteTypeProcessor): A DiscreteTypeProcessor used for template generation.
            delimiter (str, optional): The separator between variable names and time slices. Defaults to '_'.
            k (int, optional): The k for the ktbn. Defaults to None for automatic learning of K.

        Raises:
            ValueError: If a constant variable exists in the dfs and k is not given.
        """

        self._delimiter = delimiter
        self._discreteTypeProcessor = discreteTypeProcessor

        self._dfs = dfs
        self._data_points = self._count_data_points()

        self._atemporal_vars = [col for col in dfs[0].columns if _is_atemporal(dfs, col)]
        self._temporal_vars = [col for col in dfs[0].columns if col not in self._atemporal_vars]

        self._best_bic_score = None
        self._bic_scores = None
        self._log_likelihoods = None
        self._learned_ktbn = None
        self._lags_learner = None
        self._first_learner = None

        if k == None:
            self._k = None
            self._detect_constant_variables()
        
        else:
            self._init_learners(k)

    def _init_learners(self, k : int):
        """
        Generates sequences and initializes learners according to k.

        Args:
            k (int): The parameter K of the KTBN.
        """

        self._k = k

        self._lags, self._first_rows = _create_sequences(self._dfs, k, self._delimiter, self._temporal_vars, self._atemporal_vars)

        self._first_template, self._lags_template = _create_templates(self._discreteTypeProcessor,self._dfs, self._delimiter,k, self._temporal_vars, self._atemporal_vars)
        
        
        lags_learner = gum.BNLearner(self._lags, self._lags_template)
        first_learner = gum.BNLearner(self._first_rows, self._first_template)

        if self._lags_learner != None:
            lags_learner.copyState(self._lags_learner)
            first_learner.copyState(self._first_learner)

        self._lags_learner= lags_learner
        self._first_learner = first_learner

    def learn_ktbn(self, max_k = 10) -> 'KTBN':
        """
        Learns a KTBN from a list of trajectories. If k=-1, it also finds the optimal K.

        Args:
            max_k (int, optional): The maximum K to test if k is not given. Defaults to 10.

        Returns:
            KTBN: The learned KTBN
        """
        if self._k != None:
            return self._learn()
        
        best_bic_score = float('inf')  # Minimiser le BIC
        best_k = None
        best_ktbn = None
        self._bic_scores = dict()
        self._log_likelihoods = dict()
                
        for k in range(2, max_k + 1):

            self._init_learners(k)
            ktbn = self._learn()
            
            # Calculer le score BIC
            bic_score = self._calculate_bic_score(ktbn)
            self._bic_scores[k] = bic_score
            
            # Garder également la log-vraisemblance pour compatibilité
            log_likelihood = ktbn.log_likelihood(self._dfs)
            self._log_likelihoods[k] = log_likelihood
            
            # Garder le k qui minimise le BIC
            if bic_score < best_bic_score:
                best_bic_score = bic_score
                best_k = k
                best_ktbn = ktbn
        
        self._k= best_k
        self._learned_ktbn= best_ktbn
        self._best_bic_score = best_bic_score

        return best_ktbn

    def _learn(self) -> 'KTBN':
        """
        Learns a KTBN from a list of trajectories, given k.

        Returns:
            KTBN: The learned KTBN
        """

        # Atemporal variables can't have parents.
        for var in self._atemporal_vars:
            
            self._first_learner.addNoParentNode(var)
            self._lags_learner.addNoParentNode(var)

        # Learn transition
        slices = [[KTBN.encode_name_static(col,i,self._delimiter) for col in self._temporal_vars] for i in range(self._k)]
        self._lags_learner.setSliceOrder(slices)

        lags_bn = self._lags_learner.learnBN()


        #Learn first time slices
        slices = [[KTBN.encode_name_static(col,i,self._delimiter) for col in self._temporal_vars] for i in range(self._k-1)]
        self._first_learner.setSliceOrder(slices)

        first_bn = self._first_learner.learnBN()
        

        bn = gum.BayesNet(first_bn)
        
        # Add the variables of the last time slice
        for col in self._temporal_vars:
            
            bn.add(lags_bn.variable(KTBN.encode_name_static(col,self._k-1,self._delimiter)))
            

        #Add arcs
        for col in self._temporal_vars:

            name = KTBN.encode_name_static(col,self._k-1,self._delimiter)
            parents = lags_bn.parents(name)

            for parent in parents:

                p_name = lags_bn.variable(parent).name()
                bn.addArc(p_name,name)

        # Add atemporal cpts
        for col in self._atemporal_vars:

            bn.cpt(col).fillWith(lags_bn.cpt(col), bn.cpt(col).names)

        # Add temporal cpts
        for col in self._temporal_vars:
            name = KTBN.encode_name_static(col,self._k-1,self._delimiter)
            bn.cpt(name).fillWith(lags_bn.cpt(name), bn.cpt(name).names)

        self._learned_ktbn = KTBN.from_bn(bn, delimiter=self._delimiter)
        return self._learned_ktbn
    
    def get_learned_ktbn(self) -> 'KTBN':
        """
        Returns:
            KTBN: The learned KTBN, if learn_ktbn() has been called.
        """
        return self._learned_ktbn
        

    def get_delimiter(self) -> str:
        """
        Returns the delimiter used to separate variable names from time slices.

        Returns:
            str: The delimiter.
        """

        return self._delimiter
    
    def get_log_likelihoods(self) -> Optional[Dict[int, float]]:
        """
        Get the log-likelihoods for each k tested.
        
        Returns:
            Dict[int, float]: Dictionary mapping k to log-likelihood. None if k was given.
        """
        return self._log_likelihoods
    
    def get_bic_scores(self) -> Optional[Dict[int, float]]:
        """
        Get the BIC scores for each k tested.
        
        Returns:
            Dict[int, float]: Dictionary mapping k to BIC score. None if k was given.
        """
        return self._bic_scores
    
    def get_best_bic_score(self) -> Optional[float]:
        """
        Get the best BIC score found.
        
        Returns:
            Optional[float]: Best BIC score or None if learn_KTBN() has not been called or k was given.
        """
        return self._best_bic_score

    def get_k(self) -> int:
        """
        Returns:
            int: The parameter k of the KTBN, -1 if it has not been learned.
        """
        return self._k

    def addMandatoryArc(self, tail : Tuple[str, int] | str, head : Tuple[str, int]):
        """
        Adds a mandatory arc from `tail` to `head` in the KTBN structure.

        Ensures that the arc respects time slice constraints (i.e., no future-to-past dependencies)
        and updates the necessary learners accordingly.

        Args:
            tail (Tuple[str, int] | str): The parent variable, either static (str) or time-dependent (Tuple[str, int]).
            head (Tuple[str, int]): The child variable, always time-dependent.

        Raises:
            ValueError: If the arc violates time slice constraints (i.e., points from future to past).
            pyagrum.InvalidDetectedCycle: If adding the arc creates a directed cycle in the graph.
        """
        ValueError
        _verify_timeslice(tail[1] if type(tail) == tuple else -1, head[1])
        
        if self._k == -1:
            raise RuntimeError("Cannot add mandatory arcs if k is not set.")

        tail_name, head_name = self._encode_head_tail(tail,head)
        self._lags_learner.addMandatoryArc(tail_name, head_name)
        
        if(head[1] < self._k - 1):
            self._first_learner.addMandatoryArc(tail_name, head_name)
    
    def eraseMandatoryArc(self, tail : Tuple[str, int] | str, head : Tuple[str,int]):
        """
        Removes a mandatory arc from `tail` to `head` in the KTBN structure.

        Ensures that the arc respects time slice constraints before attempting removal 
        and updates the necessary learners accordingly.

        Args:
            tail (Tuple[str, int] | str): The parent variable, either static (str) or time-dependent (Tuple[str, int]).
            head (Tuple[str, int]): The child variable, always time-dependent.

        Raises:
            ValueError: If the arc violates time slice constraints (i.e., points from future to past).
        """

        _verify_timeslice(tail[1] if type(tail) == tuple else -1, head[1])
        
        if self._k == -1:
            raise RuntimeError("Cannot erase mandatory arcs if k is not set.")

        tail_name, head_name = self._encode_head_tail(tail,head)
        self._lags_learner.eraseMandatoryArc(tail_name, head_name)
        
        if(head[1] < self._k - 1):
            self._first_learner.eraseMandatoryArc(tail_name, head_name)

    def addForbiddenArc(self, tail : Tuple[str, int] | str, head : Tuple[str, int]):
        """
        Marks an arc from `tail` to `head` as forbidden in the KTBN structure.

        Ensures the arc adheres to time slice constraints before registering it as forbidden, and
        updates the necessary learners accordingly.

        Args:
            tail (Tuple[str, int] | str): The parent variable, either static (str) or time-dependent (Tuple[str, int]).
            head (Tuple[str, int]): The child variable, always time-dependent.

        Raises:
            ValueError: If the arc violates time slice constraints (i.e., points from future to past).
        """
        _verify_timeslice(tail[1] if type(tail) == tuple else -1, head[1])
        
        if self._k == -1:
            raise RuntimeError("Cannot add forbidden arcs if k is not set.")

        tail_name, head_name = self._encode_head_tail(tail,head)
        self._lags_learner.addForbiddenArc(tail_name, head_name)
        
        if(head[1] < self._k - 1):
            self._first_learner.addForbiddenArc(tail_name, head_name)

    def eraseForbiddenArc(self, tail : Tuple[str, int] | str, head : Tuple[str, int]):
        """
        Removes a previously forbidden arc from `tail` to `head` in the KTBN structure.

        Verifies time slice constraints before attempting the removal. This enables the arc to be 
        considered again during the learning process.

        Args:
            tail (Tuple[str, int] | str): The parent variable, either static (str) or time-dependent (Tuple[str, int]).
            head (Tuple[str, int]): The child variable, always time-dependent.

        Raises:
            ValueError: If the arc violates time slice constraints (i.e., points from future to past).
        """
        _verify_timeslice(tail[1] if type(tail) == tuple else -1, head[1])
        
        if self._k == -1:
            raise RuntimeError("Cannot erase forbidden arcs if k is not set.")

        tail_name, head_name = self._encode_head_tail(tail,head)
        self._lags_learner.eraseForbiddenArc(tail_name, head_name)
        
        if(head[1] < self._k - 1):
            self._first_learner.eraseForbiddenArc(tail_name, head_name)
    
    def useSmoothingPrior(self, weight : float = 1):
        """
        Use the prior smoothing.

        Args:
            weight(float) : pass in argument a weight if you wish to assign a weight to the smoothing, 
                otherwise the current weight of the learner will be used.

        """
        self._lags_learner.useSmoothingPrior(weight)
        self._first_learner.useSmoothingPrior(weight)
    
    def addNoChildrenNode(self, node : Tuple[str,int]):
        """
        Adds a constraint preventing the given node from having any children.

        Args:
            node (Tuple[str,int]): The node to constrain, as (name, time slice).
        """
        if self._k == -1:
            raise RuntimeError("Cannot add no children node if k is not set.")

        name = KTBN.encode_name_static(node[0], node[1], self._delimiter)
        self._lags_learner.addNoChildrenNode(name)

        if node[1] < self._k - 1:
            self._first_learner.addNoChildrenNode(name)
        
    def addNoParentNode(self, node : Tuple[str,int]):
        """
        Adds a constraint preventing the given node from having any parents. 

        Args:
            node (Tuple[str,int]): The node to constraint, as (name, time slice).
        """
        if self._k == -1:
            raise RuntimeError("Cannot add no parent node if k is not set.")

        name = KTBN.encode_name_static(node[0], node[1], self._delimiter)
        self._lags_learner.addNoParentNode(name)

        if node[1] < self._k - 1:
            self._first_learner.addNoParentNode(name)

    def addPossibleEdge(self, tail : Tuple[str,int], head : Tuple[str,int]):
        """
        Adds a possible edge from `tail` to `head` in the KTBN structure.

        Args:
            tail (str | int): The parent variable, either static (str) or time-dependent (int).
            head (str | int): The child variable, either static (str) or time-dependent (int).
        """
        if self._k == -1:
            raise RuntimeError("Cannot add possible edges if k is not set.")

        tail_name, head_name = self._encode_head_tail(tail, head)
        self._lags_learner.addPossibleEdge(tail_name, head_name)

        if head[1] < self.k - 1:
            self._first_learner.addPossibleEdge(tail_name, head_name)

    def eraseNoChildrenNode(self, node : Tuple[str,int]):
        """
        Removes the constraint preventing the given node from having any children.

        Args:
            node (Tuple[str, int]): The node whose constraint should be removed, as (name, time slice). 
        """
        
        if self._k == -1:
            raise RuntimeError("Cannot erase no children node if k is not set.")
            
        name = KTBN.encode_name_static(node[0], node[1], self._delimiter)
        self._lags_learner.eraseNoChildrenNode(name)
        if node[1] < self._k -1:
            self._first_learner.eraseNoChildrenNode(name)
    
    def eraseNoParentNode(self, node : Tuple[str,int]):
        """
        Removes the constraint preventing the given node from having any parents.

        Args:
            node (Tuple[str,int]): The node whose constraint should be removed, as (name, time slice). 
        """
        if self._k == -1:
            raise RuntimeError("Cannot erase no parent node if k is not set.")
        
        name = KTBN.encode_name_static(node[0], node[1], self._delimiter)
        self._lags_learner.eraseNoParentNode(name)
        if node[1] < self._k - 1:
            self._first_learner.eraseNoParentNode(name)

    def erasePossibleEdge(self, tail : Tuple[str, int], head : Tuple[str,int]):
        """
        Removes the possible edge between tail and head. 

        Args:
            tail (Tuple[str, int] | str): A node (name, time slice)
            head (Tuple[str,int]): A node (name, time slice)
        """
        if self._k == -1:
            raise RuntimeError("Cannot erase possible edge if k is not set.")

        tail_name, head_name = self._encode_head_tail(tail, head)
        self._lags_learner.erasePossibleEdge(tail_name, head_name)

        if head[1] < self.k - 1:
            self._first_learner.erasePossibleEdge(tail_name, head_name)

    def isScoreBased(self) ->bool:
        """
        Return wether the current learning method is score-based or not.
        """
        return self._first_learner.isScoreBased()
    
    def isConstraintBased(self) ->bool:
        """
        Return wether the current learning method is score-based or not.
        """
        return self._first_learner.isConstraintBased()
    
    def names(self)->List[str]:
        """
        Returns:
            List[str]: the names of the variables in the database.
        """
        return self._atemporal_vars+self._temporal_vars

    def nbCols(self)->int:
        """
        Returns:
            int: The number of columns in the database.
        """
        return len(self._atemporal_vars)+len(self._temporal_vars)
    
    def setMaxIndegree(self, max_indegree : int):
        """
        Sets the limit of the number of parents.
        Args:
            max_indegree (int): The limit number of parents. 
        """

        self._first_learner.setMaxIndegree(max_indegree)
        self._lags_learner.setMaxIndegree(max_indegree)

    def setPossibleEdges(self, edges : Set[Tuple[Tuple[str, int], Tuple[str, int]]]):
        """
        Sets the fixed set of possible edges.

        Each edge is a tuple of two (variable, time slice) pairs: (tail, head).

        Args:
            edges (Set[Tuple[Tuple[str, int], Tuple[str, int]]]): 
                A set of edges represented as ((tail_var, tail_time), (head_var, head_time)) tuples.
        """

        if self._k == -1:
            raise RuntimeError("Cannot set possible edges if k is not set.")

        first_id_set = set()
        lags_id_set = set()

        for (tail, head) in edges:

            tail_name, head_name = self._encode_head_tail(tail, head)

            if tail[1] < self._k - 1 and head[1] < self._k - 1:
                
                tail_id = self._first_template.idFromName(tail_name)
                head_id = self._first_template.idFromName(head_name)

                first_id_set.add((tail_id, head_id))
            

            tail_id = self._lags_template.idFromName(tail_name)
            head_id = self._lags_template.idFromName(head_name)

            lags_id_set.add((tail_id, head_id))

        self._first_learner.setPossibleEdges(first_id_set)
        self._lags_learner.setPossibleEdges(lags_id_set)
   
    def setPossibleSkeleton(self, skeleton : gum.UndiGraph):
        """
        Sets the fixed skeleton, given as an undirected graph.

        Args:
            skeleton (gum.UndiGraph): The skeleton as a `gum.UndiGraph`
        """
        if self._k == -1:
            raise RuntimeError("Cannot set possible skeleton if k is not set.")

        edges = skeleton.edges()
        first = {(t,h) for (t,h) in edges if self._first_template.exists(t) and self._first_template.exists(h)}
        
        self._first_learner.setPossibleEdges(first)
        self._lags_learner.setPossibleEdges(edges)

    def useBDeuPrior(self, weight : float = 1):
        """
        The BDeu prior adds weight to all the cells of the counting tables. 
        In other words, it adds weight rows in the database with equally 
        probable values.

        Args:
            weight (float, optional): The prior weight. Defaults to 1.
        """
        self._first_learner.useBDeuPrior(weight)
        self._lags_learner.useBDeuPrior(weight)

    def useDirichletPrior(self, source : str | gum.BayesNet, weight : float):
        """
        Use the Dirichlet prior.

        Args:
            source (str | gum.BayesNet): the Dirichlet related source (filename of a database or a Bayesian network).
            weight (float): the weight of the prior (the 'size' of the corresponding 'virtual database')
        """
        self._first_learner.useDirichletPrior(source, weight)
        self._lags_learner.useDirichletPrior(source, weight)

    def useGreedyHillClimbing(self):
        """
        Use greedy hill climbing algorithm.
        """
        self._first_learner.useGreedyHillClimbing()
        self._lags_learner.useGreedyHillClimbing()

    def useLocalSearchWithTabuList(self, tabu_size : int, nb_decrease : int):
        """
        Use a local search with tabu list.

        Args:
            tabu_size (int): The size of the tabu list.
            nb_decrease (int): The max allowed number of consecutive changes decreasing the score.
        """
        self._lags_learner.useLocalSearchWithTabuList()
        self._first_learner.useLocalSearchWithTabuList()

    def useMDLCorrection(self):
        """
        Use MDL correction for MIIC.
        """
        self._first_learner.useMDLCorrection()
        self._lags_learner.useMDLCorrection()
    
    def useMIIC(self):
        """
        Use MIIC.
        """
        self._lags_learner.useMIIC()
        self._first_learner.useMIIC()

    def useNMLCorrection(self):
        """
        Use NMLCorrection for MIIC.
        """
        self._lags_learner.useNMLCorrection()
        self._first_learner.useNMLCorrection()
    
    def useNoCorrection(self):
        """
        Use NoCorr for MIIC.
        """
        self._lags_learner.useNoCorrection()
        self._first_learner.useNoCorrection()
    
    def useNoPrior(self):
        """
        Use no prior.
        """
        self._lags_learner.useNoPrior()
        self._first_learner.useNoPrior()
    
    def useScoreAIC(self):
        """
        Use an AIC score.
        """
        self._lags_learner.useScoreAIC()
        self._first_learner.useScoreAIC()
    
    def useScoreBD(self):
        """
        Use a BD score.
        """
        self._lags_learner.useScoreBD()
        self._first_learner.useScoreBD()

    def useScoreBDeu(self):
        """
        Use a BDeu score.
        """
        self._lags_learner.useScoreBDeu()
        self._first_learner.useScoreBDeu()

    def useScoreBIC(self):
        """
        Use a BIC score.
        """
        self._lags_learner.useScoreBIC()
        self._first_learner.useScoreBIC()

    def useScoreK2(self):
        """
        Use a K2 score.
        """
        self._lags_learner.useScoreK2()    
        self._first_learner.useScoreK2()    
    
    def useScoreLog2Likelihood(self):
        """
        Use a log2 likelihood score. 
        """
        self._lags_learner.useScoreLog2Likelihood()
        self._first_learner.useScoreLog2Likelihood()
    
    def _encode_head_tail(self, tail : Tuple[str,int]|str, head : Tuple[str,int]) -> Tuple[str,str]:
        """
        Encodes a (variable, time slice) tuple or atemporal variable into its corresponding 
        Bayes Net name. Tail can also be an atemporal variable, in which case it can either 
        be a string, or a tuple (string, -1). 

        Args:
            tail (Tuple[str,int] | str): Tail variable (temporal or atemporal). The tail.
            head (Tuple[str,int]): Head variable (must be temporal).The head. 

        Returns:
            Tuple[str,str]: Encoded names as (tail_name, head_name).
        """

        tail_name = KTBN.encode_name_static(tail[0],tail[1],self._delimiter) if type(tail) == tuple else tail
        head_name = KTBN.encode_name_static(head[0],head[1],self._delimiter) 

        return tail_name, head_name
    
    def _detect_constant_variables(self):
        """
        Detect if there are variables that are constant across all trajectories.

        Raises:
            ValueError: If a constant variable exists in the dfs.

        """
        
        for col in self._dfs[0].columns:
            all_values = set()
            
            # Collect all unique values across all trajectories
            for df in self._dfs:
                all_values.update(df[col].unique())
            
            # If only one unique value, variable is constant
            if len(all_values) == 1:
                raise ValueError("Variables shouldn't be constant across all trajectories.")
    
    def _calculate_bic_score(self, ktbn: KTBN) -> float:
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
        log_likelihood = ktbn.log_likelihood(self._dfs)
        
        # Nombre de paramètres (dimension du réseau)
        k_params = bn.dim()
        
        # Nombre total de points de données
        n_data_points = self._data_points
        
        # Formule BIC
        bic_score = k_params * np.log(n_data_points) - 2 * log_likelihood
        
        return bic_score
    
    def _count_data_points(self) -> int:
        """
        Calculate the total number of data points across all trajectories.
        
            
        Returns:
            int: n_trajectories × trajectory_len
        """

        total_points = 0
        for trajectory in self._dfs:
            total_points += len(trajectory)
        return total_points


def _is_atemporal(dfs: List[pd.Series], column: str | int) -> bool:
    """
    Checks whether a variable is atemporal, meaning it remains constant 
    across all given DataFrames.

    Args:
        dfs (List[pd.Series]): A list of pandas Series, each containing the column to check.
        column (str | int): The column name or index.

    Returns:
        bool: True if the variable is constant across all DataFrames, otherwise False.
    """
    return all(df[column].nunique() == 1 for df in dfs)

def _create_sequences(dfs: List[pd.DataFrame], k: int, delimiter : str, temporal_variables : List, atemporal_variables : List) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates lagged feature sequences from a list of independent time series.

    Args:
        dfs (List[pd.DataFrame]): 
            A list of pandas DataFrames representing independent time series. 
            Each DataFrame must have the same columns.
        
        k (int): 
            The number of time steps to generate.

    Raises:
        ValueError: 
            If the input DataFrames do not have the same columns.

    Returns:
        Tuple[pd.DataFrame,pd.DataFrame]:  
        - The first DataFrame contains the concatenated sequences with lagged features.
        - The second DataFrame contains the first row of each time series with its corresponding lags.
    
    """
    columns = dfs[0].columns

    if not all(df.columns.equals(columns) for df in dfs):
        raise ValueError("All input DataFrames in the list must have the same columns.")



    # Variable types
    types = dfs[0].dtypes
    unrolled_types = {
        KTBN.encode_name_static(col, i, delimiter): types[col]
        for col in temporal_variables
        for i in range(k)
    }

    unrolled_types.update(types[atemporal_variables])

    # Temporal sequences 

    temporal_dfs = [df[temporal_variables] for df in dfs]

    temp_sequences = []
    for df in temporal_dfs:
        shifted_versions = []
        for i in range(k):
            shifted = df.shift(-i).copy()
            shifted.columns = [KTBN.encode_name_static(col, i, delimiter) for col in df.columns]
            shifted_versions.append(shifted.reset_index(drop=True))
        temp_sequences.append(pd.concat(shifted_versions, axis=1))

    # Atemporal and temporal sequences
    sequences = [pd.concat([df[atemporal_variables], temp], axis = 1) for (df, temp) in zip(dfs, temp_sequences)]

    #Final DFs
    lags = pd.concat(sequences, axis=0,ignore_index=True).dropna().astype(unrolled_types).reset_index(drop=True)    
    first_rows = pd.concat([sequence.iloc[[0]] for sequence in sequences], axis=0).dropna().astype(unrolled_types).reset_index(drop=True)

    return lags, first_rows

def _create_templates(
    discreteTypeProcessor: DiscreteTypeProcessor,
    dfs: List[pd.DataFrame],
    delimiter: str,
    k: int,
    temporal_variables: List[str],
    atemporal_variables: List[str]
) -> Tuple[gum.BayesNet, gum.BayesNet]:
    """
    Creates the structure templates for learning a KTBN from data.

    This function builds two Bayesian network templates:
    - `first_template`: Represents the first `k-1` time slices.
    - `lags_template`: Represents the full network up to time `k`.

    Args:
        discreteTypeProcessor (DiscreteTypeProcessor): The DiscreteTypeProcessor used to create the templates.
        dfs (List[pd.DataFrame]): The dataset containing trajectories.
        delimiter (str): Delimiter separating variable names from time indices.
        k (int): The k to use.
        temporal_variables (Set[str]): The set of time-dependent variables.
        atemporal_variables (Set[str]): The set of static variables.

    Returns:
        Tuple[gum.BayesNet, gum.BayesNet]: 
            - `first_template`: The template for the first `k-1` time slices.
            - `lags_template`: The full KTBN structure with all `k` time slices.
    """

    db = pd.concat(dfs, axis = 0).reset_index(drop=True)
    static_template = discreteTypeProcessor.discretizedTemplate(db)
    
    lags_template = gum.BayesNet()
    first_template = gum.BayesNet()

    for var in atemporal_variables:

        lags_template.add(static_template.variable(var))
        first_template.add(static_template.variable(var))

    for var in temporal_variables:

        static_var = static_template.variable(var)

        for i in range(k):

            static_var.setName(var+delimiter+str(i))
            lags_template.add(static_var)

            if i < k-1:
                first_template.add(static_var)

    return first_template, lags_template

def _verify_timeslice(tail: int, head: int) -> None:
    """
    Ensures that an arc does not violate temporal constraints. 

    Args:
        tail (int): The time slice of the parent node.
        head (int): The time slice of the child node.

    Raises:
        ValueError: If an arc from a future time slice to a past 
        time slice is detected.
    """
    if head < tail:
        raise ValueError(
            f"Invalid arc detected: Cannot create an edge from time slice {tail} "
            f"to time slice {head} in a KTBN."
        )

