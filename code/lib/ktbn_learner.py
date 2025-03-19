import pandas as pd
import pyAgrum as gum
import ktbn
from typing import List
from typing import Tuple

def create_sequences(dfs: List[pd.DataFrame], k: int, delimiter : str) -> Tuple[pd.DataFrame, pd.DataFrame]:
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

    types = dfs[0].dtypes
    unrolled_types = {f'{col}{delimiter}{i}': types[col] for col in columns for i in range(k)}

    sequences = [
        pd.concat([df.shift(-i).add_suffix(f"{delimiter}{i}").reset_index(drop=True) for i in range(k)], axis=1) 
        for df in dfs
    ]

    lags = pd.concat(sequences, axis=0,ignore_index=True).dropna().astype(unrolled_types).reset_index(drop=True)
    
    first_rows = pd.concat([sequence.iloc[[0]] for sequence in sequences], axis=0).astype(unrolled_types).reset_index(drop=True)

    return lags, first_rows


def learn_ktbn(dfs: List[pd.DataFrame], k : int, delimiter : str) -> gum.BayesNet:
    """
    Learns a KTBN from a list of trajectories.

    Args:
        dfs (List[pd.DataFrame]): List of DataFrames of the trajectories.
        k (int): The k to use.
        delimiter (str): The separator between variable names and time indices.

    Returns:
        gum.BayesNet: The learned ktBN
    """

    columns = dfs[0].columns

    lags, first_rows = create_sequences(dfs, k, delimiter)

    # Learn transition
    lags_learner = gum.BNLearner(lags)

    slices = [[f'{col}{delimiter}{i}' for col in columns] for i in range(k)]
    lags_learner.setSliceOrder(slices)

    lags_bn = lags_learner.learnBN()


    #Learn first time slices
    template = gum.BayesNet(lags_bn)
    for col in columns:

        template.erase(f'{col}{delimiter}{k-1}')

    first_learner = gum.BNLearner(first_rows.iloc[:,:len(columns)*(k-1)], template)

    slices = [[f'{col}{delimiter}{i}' for col in columns] for i in range(k-1)]
    first_learner.setSliceOrder(slices)

    first_bn = first_learner.learnBN()
    

    ktbn = gum.BayesNet(first_bn)

    # Add the variables of the last time slice
    for col in columns:
        ktbn.add(lags_bn.variable(f'{col}{delimiter}{k-1}'))
        

    #Add arcs
    for col in columns:

        name = f'{col}{delimiter}{k-1}'
        parents = lags_bn.parents(name)

        for parent in parents:
            p_name = lags_bn.variable(parent).name()
            ktbn.addArc(p_name,name)

    # Add cpts
    for col in columns:
        name = f'{col}{delimiter}{k-1}'
        ktbn.cpt(name).fillWith(lags_bn.cpt(name), ktbn.cpt(name).names)

    return ktbn




     
