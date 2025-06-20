import copy
import random
import pyagrum as gum
import pandas as pd
import os
import numpy as np
from multiprocessing import Pool
from typing import Tuple, List, Union

def _sample_worker(ktbn: 'KTBN', trajectory_len: int) -> pd.DataFrame:
    """
    Worker function for parallel trajectory sampling.
    
    Args:
        ktbn (KTBN): The KTBN instance
        trajectory_len (int): Length of trajectory to sample
        
    Returns:
        pd.DataFrame: A single sampled trajectory
    """
    gum.initRandom()
    return ktbn._sample(trajectory_len)

class KTBN:
    """
    Class that represents a KTBN
    
    Attributes:
        _k (int): number of time_slices
        _delimiter (str): delimiter to separate variable names from the index
        _temporal_variables (set): set of temporal variables
        _atemporal_variables (set): set of atemporal variables
        _bn (gum.BayesNet): underlying Bayesian network
    """
    
    def __init__(self, k: int, delimiter: str):
        """
        Initialize a KTBN
        
        Args:
            k (int): number of time-slices.
            delimiter (str):  delimiter 
        """
        self._k = k
        self._delimiter = delimiter
        self._temporal_variables = set()
        self._atemporal_variables = set()
        self._bn = gum.BayesNet()
    
    def set_k(self, k: int) -> None:
        """
        Set the number of time-slices
        
        Args:
            k (int): number of time-slices
        """
        self._k = k
    
    def get_k(self) -> int:
        """
        Returns the number of time-slices
        
        Returns:
            int: number of time-slices
        """
        return self._k
    
    def addVariable(self, variable: Union[str, gum.DiscreteVariable], temporal: bool) -> None:
        """
        Adds a variable to the KTBN.
        
        If the variable is temporal, the function adds k versions of the variable to the KTBN. 
        If the variable is atemporal, it adds only a single version of the variable.
        
        Args:
            variable (Union[str, gum.DiscreteVariable]): the variable to add
            temporal (bool): if the variable is temporal or not.
        """
        if isinstance(variable, str):
            var = gum.fastVariable(variable)
        else:
            var = variable
        
        if temporal:
            self._temporal_variables.add(var.name())
            for i in range(self._k):
                var_clone = var.clone()
                var_name = self.encode_name(var.name(), i)
                
                # Add the variable to the Bayesian network
                var_id = self._bn.add(var_clone)
                
                if var_clone.name() != var_name:
                    self._bn.changeVariableName(var_id, var_name)
        else:
            self._atemporal_variables.add(var.name())
            self._bn.add(var)
    
    def _validate_variable(self, var_tuple: Tuple[str, int]) -> str:
        """
        Validates a variable
        
        Args:
            var_tuple (Tuple[str, int]): A tuple containing the variable name and time slice index
        
        Returns:
            str: the name of the variable encoded
            
        Raises:
            ValueError: If the variable doesn't exist, is used incorrectly, or has an invalid time slice
        """
        var_name, time_slice = var_tuple
        
        # Check if the variable exists
        if var_name not in self._temporal_variables and var_name not in self._atemporal_variables:
            raise ValueError(f"The variable {var_name} does not exist")
        
        # Check if the variable is indeed temporal or atemporal
        if time_slice == -1:
            if var_name in self._temporal_variables:
                raise ValueError(f"The variable {var_name} is temporal but used as atemporal")
            return var_name
        else:
            if var_name in self._atemporal_variables:
                raise ValueError(f"The variable {var_name} is atemporal but used as temporal")
            if time_slice < 0 or time_slice >= self._k:
                raise ValueError(f"Invalid temporal index {time_slice} for variable {var_name}")
            return self.encode_name(var_name, time_slice)
    
    def addArc(self, tail: Tuple[str, int], head: Tuple[str, int]) -> None:
        """
        Adds an arc between two variables.
        
        Args:
            tail (Tuple[str, int]): the name of the variable and the time slice to which it belongs
            head (Tuple[str, int]): same
            If the value is -1, the variable is atemporal        
        
        Raises:
            ValueError: if the time-slice index is not valid.
        """
        tail_name = self._validate_variable(tail)
        head_name = self._validate_variable(head)
        
        if tail[1] == -1 and head[1] == -1:
            raise ValueError("Cannot add this arc. Both variables are atemporal")
        
        self._bn.addArc(tail_name, head_name)
    
    @staticmethod
    def encode_name_static(variable : str, time_slice : int, delimiter : str):
        """
        Encodes a variable name and a temporal index into a complete name.
        
        Args:
            variable (str): The name of the variable
            time_slice (int): The temporal index (-1 for atemporal variables)
        
        Returns:
            str: The complete encoded name
        """
        if time_slice == -1:
            return variable
        else:
            return f"{variable}{delimiter}{time_slice}"
        
    @staticmethod 
    def decode_name_static(name: str, delimiter : str) -> Tuple[str, int]:
        """
        Decodes a complete name into variable name and temporal index.
        
        Args:
            name (str): The complete name to decode
        
        Returns:
            Tuple[str, int]: The name of the variable and the temporal index (-1 for atemporal variables)
        """
        parts = name.split(delimiter)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0], int(parts[1])
        return name, -1
    
    def encode_name(self, variable: str, time_slice: int) -> str:
        """
        Encodes a variable name and a temporal index into a complete name.
        
        Args:
            variable (str): The name of the variable
            time_slice (int): The temporal index (-1 for atemporal variables)
        
        Returns:
            str: The complete encoded name
        """
        return KTBN.encode_name_static(variable,time_slice, self._delimiter)
    
    def decode_name(self, name: str) -> Tuple[str, int]:
        """
        Decodes a complete name into variable name and temporal index.
        
        Args:
            name (str): The complete name to decode
        
        Returns:
            Tuple[str, int]: The name of the variable and the temporal index (-1 for atemporal variables)
        """
        return KTBN.decode_name_static(name,self._delimiter)
    
    def unroll(self, n: int)->gum.BayesNet:
        """
        Unrolls a KTBN for `n` time slices        
        Args:
            n (int): number of time slices to unroll
            
        Returns: 
            the KTBN unrolled
        """
        unrolled_bn = gum.BayesNet()
        
        # 1. VARIABLES
        
        # Add atemporal variables
        for var_name in self._atemporal_variables:
            var_id = self._bn.idFromName(var_name)
            template_var = self._bn.variable(var_id)
            
            new_var = template_var.clone()
            new_var.setName(var_name)
            
            unrolled_bn.add(new_var)
        
        # Add temporal variables
        for var_name in self._temporal_variables:
            template_name = self.encode_name(var_name, self._k - 1)  
            template_id = self._bn.idFromName(template_name)
            template_var = self._bn.variable(template_id)
            
            for t in range(self._k + n):  
                new_var = template_var.clone()
                new_name = self.encode_name(var_name, t)
                new_var.setName(new_name)
                
                unrolled_bn.add(new_var)
                
        
        # 2. ARCS
        intra_slice_arcs = []  
        inter_slice_arcs = []  
        atemporal_arcs = []    
        
        for arc in self._bn.arcs():
            tail_id, head_id = arc
            tail_name = self._bn.variable(tail_id).name()
            head_name = self._bn.variable(head_id).name()
            
            tail_var, tail_time = self.decode_name(tail_name)
            head_var, head_time = self.decode_name(head_name)
            
            # Case 1: Arc from an atemporal variable
            if tail_time == -1 and head_time != -1:
                atemporal_arcs.append((tail_var, head_var, head_time))
            
            # Case 2: Arc between two consecutive time slices 
            elif tail_time != -1 and head_time != -1 and head_time == tail_time + 1:
                inter_slice_arcs.append((tail_var, head_var))
            
            # Case 3: Arc between two variables in the same time slice
            elif tail_time != -1 and head_time != -1 and head_time == tail_time:
                intra_slice_arcs.append((tail_var, head_var, tail_time))
        
        # Add arcs between two variables in the same time slice
        for tail_var, head_var, time_slice in intra_slice_arcs:
            # Add originals arcs (for time slices 0 to k-1)
            for t in range(self._k):
                tail_name = self.encode_name(tail_var, t)
                head_name = self.encode_name(head_var, t)
                if self._bn.existsArc(tail_name, head_name):
                    unrolled_bn.addArc(tail_name, head_name)
            
            # Check if there is an arc in the last slice
            last_slice_tail_name = self.encode_name(tail_var, self._k - 1)
            last_slice_head_name = self.encode_name(head_var, self._k - 1)
            if self._bn.existsArc(last_slice_tail_name, last_slice_head_name):
                # If yes, add similar arcs for the new slices
                for t in range(self._k, self._k + n):
                    new_tail_name = self.encode_name(tail_var, t)
                    new_head_name = self.encode_name(head_var, t)
                    unrolled_bn.addArc(new_tail_name, new_head_name)

        
        # Add arcs between two consecutive time slices
        for tail_var, head_var in inter_slice_arcs:
            for t in range(self._k + n - 1):  # Modified here: k+n-1 instead of k+n
                tail_name = self.encode_name(tail_var, t)
                head_name = self.encode_name(head_var, t + 1)
                unrolled_bn.addArc(tail_name, head_name)
        
        # Add arcs from atemporal variables
        for tail_var, head_var, head_time in atemporal_arcs:
            # Add originals arcs
            for t in range(self._k):
                head_name = self.encode_name(head_var, t)
                if self._bn.existsArc(tail_var, head_name):
                    unrolled_bn.addArc(tail_var, head_name)
            
            # 2. Check if there is an arc between the atemporal variable and the temporal variable in the last slice
            last_slice_head_name = self.encode_name(head_var, self._k - 1)
            if self._bn.existsArc(tail_var, last_slice_head_name):
                # If yes, add similar arcs for the new slices
                for t in range(self._k, self._k + n):
                    new_head_name = self.encode_name(head_var, t)
                    unrolled_bn.addArc(tail_var, new_head_name)
                
        # 3. CPTs
        for node_id in self._bn.nodes():
            name = self._bn.variable(node_id).name()
            static_name, t_slice = self.decode_name(name)

            if t_slice < self._k:
                unrolled_bn.cpt(name).fillWith(self._bn.cpt(name), unrolled_bn.cpt(name).names)
            else:
                for t in range(self._k, self._k + n):
                    new_name = self.encode_name(static_name, t)
                    dict_names = {
                        p_name: p_name if (p_t_slice := self.decode_name(p_name)[1]) == -1 
                        else self.encode_name(self.decode_name(p_name)[0], p_t_slice - t + self._k)
                        for p_name in unrolled_bn.cpt(new_name).names
                    }
                    unrolled_bn.cpt(new_name).fillWith(self._bn.cpt(name), dict_names)
        
        return unrolled_bn
        
    def cpt(self, variable: str, time_slice: int) -> gum.Potential:
        """
        Returns the cpt of a variable
        
        Args:
            variable (str): the name of the variable
            time_slice (int): the time-slice
        
        Returns:
            gum.Potential: cpt of the variable
        """
        var_name = self._validate_variable((variable, time_slice))
        return self._bn.cpt(var_name)
    
    def to_bn(self) -> gum.BayesNet:
        """
        Returns a deep copy of the underlying BN
        
        Returns:
            gum.BayesNet: deep copy of the BN
        """
        return copy.deepcopy(self._bn)
    
    def save(self, filename: str) -> None:
        """
        Saves the KTBN in BIFXML format.
        
        Args:
            filename (str): the filename
        """
        if not filename.endswith('.bifxml'):
            filename += '.bifxml'
        
        self._bn.saveBIFXML(filename) 
    
    @classmethod
    def load(cls, filename: str, delimiter = '$') -> 'KTBN':
        """
        Loads a KTBN from a BIFXML file.
        
        Args:
            filename (str): the filename (with or without .bifxml extension)
        
        Returns:
            KTBN: the loaded KTBN
        """
        if not filename.endswith('.bifxml'):
            filename += '.bifxml'
        
        # loadBN works with various formats including BIFXML
        bn = gum.loadBN(filename)
        
        temporal_variables = set()
        atemporal_variables = set()
        max_time_slice = 0
        
        for node_id in bn.nodes():
            name = bn.variable(node_id).name()
            parts = name.split(delimiter)
            
            if len(parts) == 2 and parts[1].isdigit():
                base_name = parts[0]
                time_slice = int(parts[1])
                temporal_variables.add(base_name)
                max_time_slice = max(max_time_slice, time_slice)
            else:
                atemporal_variables.add(name)
        
        k = max_time_slice + 1  # k is the number of slices, not the maximum index
        
        ktbn = cls(k=k, delimiter=delimiter)
        ktbn._temporal_variables = temporal_variables
        ktbn._atemporal_variables = atemporal_variables
        ktbn._bn = bn
        
        return ktbn
    
    @classmethod
    def from_bn(cls, bn: gum.BayesNet, delimiter: str = '$') -> 'KTBN':
        """
        Creates a KTBN from an existing Bayesian network.
        
        Args:
            bn (gum.BayesNet): The Bayesian network that needs to be converted to a KTBN
            delimiter (str, optional): delimiter
        
        Returns:
            KTBN: A new instance of KTBN
        """
        temporal_variables = set()
        atemporal_variables = set()
        max_time_slice = 0
        
        temp_ktbn = cls(k=0, delimiter=delimiter)
        
        for node_id in bn.nodes():
            name = bn.variable(node_id).name()
            base_name, time_slice = temp_ktbn.decode_name(name)
            
            if time_slice == -1:
                atemporal_variables.add(base_name)
            else:
                temporal_variables.add(base_name)
                max_time_slice = max(max_time_slice, time_slice)
        
        k = max_time_slice + 1  # k is the number of slices, not the maximum index
        
        # Creation of a new KTBN
        ktbn = cls(k=k, delimiter=delimiter)
        ktbn._temporal_variables = temporal_variables
        ktbn._atemporal_variables = atemporal_variables
        ktbn._bn = copy.deepcopy(bn)
        
        return ktbn

    def sample(self, n_trajectories: int, trajectory_len: int, processes: int = None) -> List[pd.DataFrame]:
        """
        Generate a list of trajectories sampled from the ktbn in parallel 
        using Python's multiprocessing module.

        Args:
            n_trajectories (int): The number of trajectories to sample.
            trajectory_len (int): The length of each trajectory.
            processes (int, optional): Number of worker processes to use. Defaults to the number of CPU cores.

        Returns:
            List[pd.DataFrame]: A list of data frames, each containing a sample trajectory.

        ValueError
            If `trajectory_len` is less than the order K of the KTBN.
        """ 
        if trajectory_len < self._k:
            raise ValueError("Trajectory length can't be smaller than k.")
        
        with Pool(processes=processes) as pool:
            
            results = [pool.apply_async(_sample_worker, args=(self, trajectory_len)) for _ in range(n_trajectories)]
            output = [res.get() for res in results]
        
        return output
              
    def _sample(self, trajectory_len: int) -> pd.DataFrame:
        """
        Generates a trajectory sampled from the ktbn.

        Args:
            trajectory_len (int): The length of the trajectory.

        Returns:
            pd.DataFrame: A data frame containing a sample trajectory.

        ValueError
            If `trajectory_len` is less than the order K of the KTBN.
        """

        if trajectory_len < self._k:
            raise ValueError("Trajectory length can't be smaller than k.")

        vars = list(self._temporal_variables.union(self._atemporal_variables))
        
        dtypes = {var : str if self._bn.variable(self.encode_name(var,0)).varType() == gum.VarType_LABELIZED else int for var in self._temporal_variables}
        dtypes.update({var : str if self._bn.variable(var).varType() == gum.VarType_LABELIZED else int for var in self._atemporal_variables})

        df = pd.DataFrame(np.zeros((trajectory_len, len(vars))), columns = vars).astype(dtypes)
        order = self._bn.topologicalOrder()
        last_order = [var for var in order if self.decode_name(self._bn.variable(var).name())[1] == self._k-1]
        
        trajectory = gum.Instantiation()

        # First time slices.
        for node in order:
            
            var = self._bn.variable(node)
            val = var.label(self._bn.cpt(node).extract(trajectory).draw())

            trajectory.add(var)
            trajectory[var.name()] = val

            name, index = self.decode_name(var.name())

            if index == -1:
                df[name] = dtypes[name](val)
            else:
                df.loc[index, name] = dtypes[name](val)
        
        # Transition
        for i in range(1,trajectory_len - self._k + 1):

            I = gum.Instantiation()

            for node in last_order:

                var = self._bn.variable(node)
                # Add parents to instantiation 
                for parent in self._bn.parents(node):
                    
                    parent_var = self._bn.variable(parent)
                    if I.contains(parent_var):
                        continue

                    I.add(parent_var)
                    name, index = self.decode_name(parent_var.name())
                    
                    if index == -1:
                        I[name] = trajectory[name]
                    else :
                        I[parent_var.name()] = trajectory[self.encode_name(name, index+i)]

                # Sample node
                val = var.label(self._bn.cpt(node).extract(I).draw())
                I.add(var)
                I[var.name()] = val
                
                new_var = var.clone()
                name, index = self.decode_name(new_var.name())
                new_var.setName(self.encode_name(name, index+i))

                trajectory.add(new_var)
                trajectory[new_var.name()] = val

                df.loc[index+i,name] = dtypes[name](val)

        return df
        
    @staticmethod
    def random(k : int, n_vars : int, n_mods : int, n_arcs : int, delimiter = '$') -> 'KTBN':
        """
        Generates a random KTBN with hyperparameter k, n_vars number of variables
        with each n_mods number of modalities.
        
        Args :
            k (int): The k of the KTBN
            n_vars (int): The number of variables per time slice.
            n_mods (int): The number of modalities for each variable.
            n_arcs (int): The total number of arcs.
            delimiter (str): The delimiter to use for encoding variable names.
        
        Returns:
            KTBN: A random KTBN with the specified parameters.
        """
        # Check if the number of arcs is valid    
        if n_arcs > (k/2) * ( (k-1) * n_vars**2  + n_vars*(n_vars-1)):

            raise ValueError("Too many arcs requested.")
        
        ktbn = KTBN(k,delimiter)

        # Add variables
        for i in range(n_vars):
            var = gum.RangeVariable(f"X{i}", f"variable{i}",1, n_mods)
            ktbn.addVariable(var, True)

        arcs = set()
        while len(arcs) < n_arcs:

            # Randomly select two variables to create an arc between
            head = random.randint(0, n_vars - 1), random.randint(0, k - 1)
            tail = random.randint(0, n_vars - 1), random.randint(0, k - 1)

            if head == tail:
                continue

            # Ensure that the arc is directed from the past to the future time slice.
            if head[1] < tail[1]:

                bubble = head
                head = tail
                tail = bubble 
            # Ensure that the arc is not already present
            if (head, tail) in arcs:
                continue

            try:
                ktbn.addArc( (f"X{tail[0]}", tail[1]), (f"X{head[0]}", head[1]) )
                arcs.add((head, tail))
            except gum.InvalidDirectedCycle:
                continue
            
        #Add random cpts
        for v in ktbn._temporal_variables:
            for i in range(k):
                
                cpt = np.random.random(ktbn.cpt(v,i).shape)

                # Normalize depending on the shape
                if len(ktbn.cpt(v,i).shape) ==1:
                    cpt/=cpt.sum()
                else:
                    cpt/=cpt.sum(axis=1,keepdims=True)
                ktbn.cpt(v,i).fillWith(cpt.ravel().tolist())

        return ktbn
                
    def id_from_name(self, name : str, time_slice : int = -1):
        """
        Returns the variable's id given its name and its time slice.

        Args:
            name (str): The name of the variable
            time_slice (int, optional): The time slice of the variable, -1 for atemporal. Defaults to -1.
        """
        return self._bn.idFromName(self.encode_name(name, time_slice))
    
    def names(self) -> set[str]:
        """
        Returns the set of variable names in the KTBN, excluding time slice annotations.

        Returns:
            set[str]: Set of unique variable names without time slices.
        """

        return self._atemporal_variables.union(self._temporal_variables) 

    def _get_value_from_trajectory(self, trajectory, var_base, time_slice):
        """
        Retrieves the value of a variable from a trajectory.
        
        Args:
            trajectory (pd.DataFrame): The trajectory
            var_base (str): The base name of the variable
            time_slice (int): The time slice (-1 for atemporal variables)
            
        Returns:
            The value of the variable at the given time slice
        """
        if time_slice == -1:
            return trajectory.loc[0, var_base]
        else:
            return trajectory.loc[time_slice, var_base]
    
    def _get_var_index(self, variable: gum.DiscreteVariable, value) -> int:
        """
        Converts a value to its corresponding index in a variable.
        
        Args:
            variable (gum.DiscreteVariable): The variable
            value: The observed value
            
        Returns:
            int: The index corresponding to the value
        """
        return variable.labels().index(str(value))
    
    
    def log_likelihood(self, trajectories: List[pd.DataFrame]) -> float:
        """
        Compute the log-likelihood of a list of trajectories given a KTBN
        
        Args:
            trajectories (List[pd.DataFrame]): List of trajectories from sample()
            
        Returns:
            float: Log-likelihood value
        """
        if not trajectories:
            return 0.0
        
        total_log_likelihood = 0.0
        
        for trajectory in trajectories:
            trajectory_log_likelihood = 0.0
            trajectory_length = len(trajectory)
            
            # Phase 1: Calculate likelihood for the first k time slices (template nodes)
            for node_id in self._bn.nodes():
                var = self._bn.variable(node_id)
                var_name = var.name()
                var_base, var_time = self.decode_name(var_name)
                
                # If the variable belongs to a time slice that doesn't exist in the trajectory
                if var_time != -1 and var_time >= trajectory_length:
                    continue
                
                # Creation of an instanciation for the parents
                inst = gum.Instantiation()
                
                # Add parents to the instantiation
                for parent_id in self._bn.parents(node_id):
                    parent_var = self._bn.variable(parent_id)
                    parent_name = parent_var.name()
                    parent_base, parent_time = self.decode_name(parent_name)
                    
                    if parent_time != -1 and parent_time >= trajectory_length:
                        continue
                    
                    inst.add(parent_var)
                    
                    # Get the value of the parent in the trajectory
                    parent_value = self._get_value_from_trajectory(trajectory, parent_base, parent_time)
                    
                    # Convert the value into index 
                    inst[parent_name] = self._get_var_index(parent_var, parent_value)
                
                # Add the variable to the instantiation
                inst.add(var)
                
                # Get the value of the variable in the trajectory
                var_value = self._get_value_from_trajectory(trajectory, var_base, var_time)
                
                # Convert the value into index
                inst[var_name] = 0
                inst[var_name] = self._get_var_index(var, var_value)
                
                # Compute the probability of the variable knowing its parents 
                cpt = self._bn.cpt(node_id)
                prob = cpt[inst]
                
                # A probability equals to 0 is replaced by 1e-6
                prob = max(prob, 1e-6)  
                trajectory_log_likelihood += np.log(prob)
            
            # Phase 2: Calculate likelihood for the remaining time slices (t >= k)
            if trajectory_length > self._k:
                # Get nodes from the last time slice (k-1) to use for transition
                last_slice_nodes = []
                for node_id in self._bn.nodes():
                    var = self._bn.variable(node_id)
                    var_name = var.name()
                    var_base, var_time = self.decode_name(var_name)
                    
                    if var_time == self._k - 1:  # Last time slice
                        last_slice_nodes.append(node_id)
                
                # For each time step from k to trajectory_length-1
                for t in range(self._k, trajectory_length):
                    shift = t - (self._k - 1)  # How much to shift parent indices
                    
                    # For each node in the last time slice
                    for node_id in last_slice_nodes:
                        var = self._bn.variable(node_id)
                        var_name = var.name()
                        var_base, var_time = self.decode_name(var_name)
                        
                        # Create instantiation for shifted parents
                        inst = gum.Instantiation()
                        
                        # Add parents with shifted time indices
                        for parent_id in self._bn.parents(node_id):
                            parent_var = self._bn.variable(parent_id)
                            parent_name = parent_var.name()
                            parent_base, parent_time = self.decode_name(parent_name)
                            
                            # Calculate the shifted time for this parent
                            if parent_time == -1:  # Atemporal variable
                                shifted_parent_time = -1
                            else:
                                shifted_parent_time = parent_time + shift
                            
                            # Skip if shifted time is out of bounds
                            if shifted_parent_time != -1 and shifted_parent_time >= trajectory_length:
                                continue
                            
                            inst.add(parent_var)
                            
                            # Get the value of the parent in the trajectory at shifted time
                            parent_value = self._get_value_from_trajectory(trajectory, parent_base, shifted_parent_time)
                            
                            # Convert the value into index
                            inst[parent_name] = self._get_var_index(parent_var, parent_value)
                        
                        # Add the current variable to the instantiation
                        inst.add(var)
                        
                        # Get the value of the variable at time t
                        var_value = self._get_value_from_trajectory(trajectory, var_base, t)
                        
                        # Convert the value into index
                        inst[var_name] = 0
                        inst[var_name] = self._get_var_index(var, var_value)
                        
                        # Compute the probability using the same CPT as the template
                        cpt = self._bn.cpt(node_id)
                        prob = cpt[inst]
                        
                        # A probability equals to 0 is replaced by 1e-6
                        prob = max(prob, 1e-6)  
                        trajectory_log_likelihood += np.log(prob)
            
            total_log_likelihood += trajectory_log_likelihood
        
        return total_log_likelihood
