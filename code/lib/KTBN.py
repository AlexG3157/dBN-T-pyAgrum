
import pyAgrum as gum
import numpy as np
import re
from typing import Tuple, List, Union, Dict, Any, Optional

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
            for i in range(self._k + 1):
                var_clone = var.clone()
                var_name = self.encode_name(var.name(), i)
                
                # Add the variable to the Bayesian network
                var_id = self._bn.add(var_clone)
                
                # Change its name if necessary
                if var_clone.name() != var_name:
                    self._bn.changeVariableName(var_id, var_name)
        else:
            self._atemporal_variables.add(var.name())
            self._bn.add(var)
    
    def _validate_variable(self, var_tuple: Tuple[str, int]) -> str:
        """
        Validates a variable tuple and returns the full variable name.
        
        Args:
            var_tuple (Tuple[str, int]): A tuple containing the variable name and time slice index
                                     (-1 for atemporal variables)
        
        Returns:
            str: The full variable name (encoded with delimiter if temporal)
            
        Raises:
            ValueError: If the variable doesn't exist, is used incorrectly, or has an invalid time slice
        """
        var_name, time_slice = var_tuple
        
        # Check if the variable exists
        if var_name not in self._temporal_variables and var_name not in self._atemporal_variables:
            raise ValueError(f"The variable {var_name} does not exist")
        
        # Validate temporal/atemporal usage and return the full name
        if time_slice == -1:
            if var_name in self._temporal_variables:
                raise ValueError(f"The variable {var_name} is temporal but used as atemporal")
            return var_name
        else:
            if var_name in self._atemporal_variables:
                raise ValueError(f"The variable {var_name} is atemporal but used as temporal")
            if time_slice < 0 or time_slice > self._k:
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
    
    def encode_name(self, variable: str, time_slice: int) -> str:
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
            return f"{variable}{self._delimiter}{time_slice}"
    
    def decode_name(self, name: str) -> Tuple[str, int]:
        """
        Decodes a complete name into variable name and temporal index.
        
        Args:
            name (str): The complete name to decode
        
        Returns:
            Tuple[str, int]: The name of the variable and the temporal index (-1 for atemporal variables)
        """
        parts = name.split(self._delimiter)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0], int(parts[1])
        return name, -1
    
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
        atemporal_var_ids = {}
        for var_name in self._atemporal_variables:
            var_id = self._bn.idFromName(var_name)
            template_var = self._bn.variable(var_id)
            
            # Creation of a new variable with the same labels and the same name
            new_var = gum.LabelizedVariable(var_name, var_name, template_var.domainSize())
            for i in range(template_var.domainSize()):
                new_var.changeLabel(i, template_var.label(i))
            
            # Add the new variable to the new BN
            new_id = unrolled_bn.add(new_var)
            atemporal_var_ids[var_name] = new_id
        
        # Add temporal variables (0 to k+n)
        temporal_var_ids = {} 
        for var_name in self._temporal_variables:
            # Utiliser la variable du time-slice k comme modèle pour toutes les variables
            template_name = self.encode_name(var_name, self._k)
            template_id = self._bn.idFromName(template_name)
            template_var = self._bn.variable(template_id)
            
            # Initialiser le dictionnaire pour cette variable
            temporal_var_ids[var_name] = {}
            
            for t in range(self._k + n + 1):
                # Créer une nouvelle variable avec les mêmes caractéristiques
                new_name = self.encode_name(var_name, t)
                new_var = gum.LabelizedVariable(new_name, new_name, template_var.domainSize())
                for i in range(template_var.domainSize()):
                    new_var.changeLabel(i, template_var.label(i))
                
                # Ajouter la variable au BN
                new_id = unrolled_bn.add(new_var)
                
                # Stocker l'ID pour une utilisation ultérieure
                temporal_var_ids[var_name][t] = new_id
        
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
                atemporal_arcs.append((tail_var, head_var))
            
            # Case 2: Arc between two consecutive time slices 
            elif tail_time != -1 and head_time != -1 and head_time == tail_time + 1:
                inter_slice_arcs.append((tail_var, head_var))
            
            # Case 3: Arc between two variables in the same time slice
            elif tail_time != -1 and head_time != -1 and head_time == tail_time:
                intra_slice_arcs.append((tail_var, head_var, tail_time))
        
        # Add arcs between two variables in the same time slice
        for tail_var, head_var, time_slice in intra_slice_arcs:
            for t in range(self._k + n + 1):
                tail_name = self.encode_name(tail_var, t)
                head_name = self.encode_name(head_var, t)
                try:
                    unrolled_bn.addArc(tail_name, head_name)
                except:
                    # It means that the arc already exists
                    pass
        
        # Add arcs between two consecutive time slices
        for tail_var, head_var in inter_slice_arcs:
            for t in range(self._k + n):
                tail_name = self.encode_name(tail_var, t)
                head_name = self.encode_name(head_var, t + 1)
                try:
                    unrolled_bn.addArc(tail_name, head_name)
                except:
                    # It means that the arc already exists
                    pass
        
        # Add arcs from atemporal variables
        for tail_var, head_var in atemporal_arcs:
            for t in range(self._k + n + 1):  
                head_name = self.encode_name(head_var, t)
                try:
                    unrolled_bn.addArc(tail_var, head_name)
                except:
                    # It means that the arc already exists
                    pass
        
        # 3. CPTs
        
        for node_id in unrolled_bn.nodes():
            cpt = unrolled_bn.cpt(node_id)
            cpt.fillWith(1.0)
            cpt.normalize()
        
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
        Returns a deep copy of the underlying Bayesian network.
        
        Returns:
            gum.BayesNet: A deep copy of the Bayesian network
        """
        import copy
        return copy.deepcopy(self._bn)

    
    def save(self, filename: str) -> None:
        """
        Saves the KTBN in BIFXML format.
        
        Args:
            filename (str): the filename (with or without .bifxml extension)
        """
        if not filename.endswith('.bifxml'):
            filename += '.bifxml'
        
        self._bn.saveBIFXML(filename) 
    
    @classmethod
    def load(cls, filename: str) -> 'KTBN':
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
        
        delimiter = '#'
        
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
        
        k = max_time_slice
        
        ktbn = cls(k=k, delimiter=delimiter)
        ktbn._temporal_variables = temporal_variables
        ktbn._atemporal_variables = atemporal_variables
        ktbn._bn = bn
        
        return ktbn
    
    @classmethod
    def from_bn(cls, bn: gum.BayesNet, delimiter: str = '#') -> 'KTBN':
        """
        Creates a KTBN from an existing Bayesian network.
        
        Args:
            bn (gum.BayesNet): The Bayesian network to convert to KTBN
            delimiter (str, optional): The delimiter to use to separate variable names 
                                      from temporal indices. Default '#'.
        
        Returns:
            KTBN: A new instance of KTBN
        """
        # Analysis of variable names to identify temporal and atemporal variables
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
        
        k = max_time_slice
        
        # Creation of a new KTBN instance
        ktbn = cls(k=k, delimiter=delimiter)
        ktbn._temporal_variables = temporal_variables
        ktbn._atemporal_variables = atemporal_variables
        
        # Using a deep copy of the Bayesian network
        import copy
        ktbn._bn = copy.deepcopy(bn)
        
        return ktbn
