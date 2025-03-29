import pandas as pd
import pyAgrum as gum
from pyAgrum.lib.discretizer import Discretizer
from typing import List, Set, Union
from typing import Tuple

class Learner():
    """
    A class for learning a K-Time Slice Bayesian Network (KTBN) from trajectory data.
    """
    def __init__(self, dfs : List[pd.DataFrame], discretizer : Discretizer,delimiter : str = '_', k : int = -1 ):
        """
        Initializes the learner with trajectory data and parameters.

        Args:
            dfs (List[pd.DataFrame]): A list of trajectory datasets for learning the KTBN.
            discretizer (Discretizer): A discretizer used for template generation.
            delimiter (str, optional): The separator between variable names and time slices. Defaults to '_'.
            k (int, optional): The k for the ktbn. Defaults to -1.

        Raises:
            NotImplementedError: If `k=-1`, as automatic k learning is not yet supported.
        """
        if k == -1:

            raise NotImplementedError("Learning the hyperparameter k is not yet supported!")
        
        self.k = k
        self.delimiter = delimiter

        self.atemporal_vars = [col for col in dfs[0].columns if is_atemporal(dfs, col)]
        self.temporal_vars = [col for col in dfs[0].columns if col not in self.atemporal_vars]
        
        self.lags, self.first_rows = create_sequences(dfs, k, delimiter, self.temporal_vars, self.atemporal_vars)

        first_template, lags_template = create_templates(discretizer,dfs, delimiter,k, self.temporal_vars, self.atemporal_vars)
        
        self.lags_learner = gum.BNLearner(self.lags, lags_template)
        self.first_learner = gum.BNLearner(self.first_rows, first_template)

    
    def learn_ktbn(self) -> gum.BayesNet:
        """
        Learns a KTBN from a list of trajectories.

        Returns:
            gum.BayesNet: The learned ktBN
        """

        # Atemporal variables can't have parents.
        for var in self.atemporal_vars:
            
            self.first_learner.addNoParentNode(var)
            self.lags_learner.addNoParentNode(var)

        # Learn transition
        slices = [[f'{col}{self.delimiter}{i}' for col in self.temporal_vars] for i in range(self.k)]
        self.lags_learner.setSliceOrder(slices)

        lags_bn = self.lags_learner.learnBN()


        #Learn first time slices
        slices = [[f'{col}{self.delimiter}{i}' for col in self.temporal_vars] for i in range(self.k-1)]
        self.first_learner.setSliceOrder(slices)

        first_bn = self.first_learner.learnBN()
        

        ktbn = gum.BayesNet(first_bn)
        
        # Add the variables of the last time slice
        for col in self.temporal_vars:
            
            ktbn.add(lags_bn.variable(f'{col}{self.delimiter}{self.k-1}'))
            

        #Add arcs
        for col in self.temporal_vars:

            name = f'{col}{self.delimiter}{self.k-1}'
            parents = lags_bn.parents(name)

            for parent in parents:

                p_name = lags_bn.variable(parent).name()
                ktbn.addArc(p_name,name)

        # Add atemporal cpts
        for col in self.atemporal_vars:

            ktbn.cpt(col).fillWith(lags_bn.cpt(col), ktbn.cpt(col).names)

        # Add temporal cpts
        for col in self.temporal_vars:
            name = f'{col}{self.delimiter}{self.k-1}'
            ktbn.cpt(name).fillWith(lags_bn.cpt(name), ktbn.cpt(name).names)

        return ktbn


    def get_delimiter(self) -> str:
        """
        Returns the delimiter used to separate variable names from time slices.

        Returns:
            str: The delimiter.
        """

        return self.delimiter
    
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
            pyAgrum.InvalidDetectedCycle: If adding the arc creates a directed cycle in the graph.
        """

        verify_timeslice(tail[1] if type(tail) == tuple else -1, head[1])
        
        tail_name = f'{tail[0]}{self.delimiter}{tail[1]}' if type(tail) == tuple else tail
        head_name = f'{head[0]}{self.delimiter}{head[1]}' 
        self.lags_learner.addMandatoryArc(tail_name, head_name)
        
        if(head[1] < self.k-1):
            self.first_learner.addMandatoryArc(tail_name, head_name)
    
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

        verify_timeslice(tail[1] if type(tail) == tuple else -1, head[1])
        
        tail_name = f'{tail[0]}{self.delimiter}{tail[1]}' if type(tail) == tuple else tail
        head_name = f'{head[0]}{self.delimiter}{head[1]}' 
        self.lags_learner.eraseMandatoryArc(tail_name, head_name)
        
        if(head[1] < self.k-1):
            self.first_learner.eraseMandatoryArc(tail_name, head_name)

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
        verify_timeslice(tail[1] if type(tail) == tuple else -1, head[1])
        
        tail_name = f'{tail[0]}{self.delimiter}{tail[1]}' if type(tail) == tuple else tail
        head_name = f'{head[0]}{self.delimiter}{head[1]}' 
        self.lags_learner.addForbiddenArc(tail_name, head_name)
        
        if(head[1] < self.k-1):
            self.first_learner.addForbiddenArc(tail_name, head_name)

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
        verify_timeslice(tail[1] if type(tail) == tuple else -1, head[1])
        
        tail_name = f'{tail[0]}{self.delimiter}{tail[1]}' if type(tail) == tuple else tail
        head_name = f'{head[0]}{self.delimiter}{head[1]}' 
        self.lags_learner.eraseForbiddenArc(tail_name, head_name)
        
        if(head[1] < self.k-1):
            self.first_learner.eraseForbiddenArc(tail_name, head_name)
    
    def useSmoothingPrior(self, weight : float = 1):
        """
        Use the prior smoothing.

        Args:
            weight(float) : pass in argument a weight if you wish to assign a weight to the smoothing, 
                otherwise the current weight of the learner will be used.

        """
        self.lags_learner.useSmoothingPrior(weight)
        self.first_learner.useSmoothingPrior(weight)
        
  

def is_atemporal(dfs: List[pd.Series], column: Union[str, int]) -> bool:
    """
    Checks whether a variable is atemporal, meaning it remains constant 
    across all given DataFrames.

    Args:
        dfs (List[pd.Series]): A list of pandas Series, each containing the column to check.
        column (Union[str, int]): The column name or index.

    Returns:
        bool: True if the variable is constant across all DataFrames, otherwise False.
    """
    return all(df[column].nunique() == 1 for df in dfs)

def create_sequences(dfs: List[pd.DataFrame], k: int, delimiter : str, temporal_variables : List, atemporal_variables : List) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
    unrolled_types = {f'{col}{delimiter}{i}': types[col] for col in temporal_variables for i in range(k)}
    unrolled_types.update(types[atemporal_variables])

    # Temporal sequences 

    temporal_dfs = [df[temporal_variables] for df in dfs]
    temp_sequences = [
        pd.concat([df.shift(-i).add_suffix(f"{delimiter}{i}").reset_index(drop=True) for i in range(k)], axis=1) 
        for df in temporal_dfs
    ]

    # Atemporal and temporal sequences
    sequences = [pd.concat([df[atemporal_variables], temp], axis = 1) for (df, temp) in zip(dfs, temp_sequences)]

    #Final DFs
    lags = pd.concat(sequences, axis=0,ignore_index=True).dropna().astype(unrolled_types).reset_index(drop=True)    
    first_rows = pd.concat([sequence.iloc[[0]] for sequence in sequences], axis=0).dropna().astype(unrolled_types).reset_index(drop=True)

    return lags, first_rows

def create_templates(
    discretizer: Discretizer,
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
        discretizer (Discretizer): The discretizer used to create the templates.
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
    static_template = discretizer.discretizedTemplate(db)
    
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

def verify_timeslice(tail: int, head: int) -> None:
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
