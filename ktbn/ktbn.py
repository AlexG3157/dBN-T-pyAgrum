import re
import pyAgrum as gum
import numpy as np
from typing import Tuple

def split_name(name : str, delimiter : str) -> Tuple[str,int]:
    """
    Splits a dBN variable name into a static prefix and a time slice index.

    In a dBN, time-dependent variables follow the convention "X" + str(t), 
    where `X` is the base name and `t` is the time slice. Variables that 
    do not follow this pattern are considered static.

    Args:
        name (str): The variable name, which may include a numeric time suffix.

    Returns:
        Tuple[str,int]: The base name of the variable and the time slice. If the 
        variable is static, returns `-1` as the time slice.
    """

    # TODO mettre delimiteur (defaut #) 
    match = re.match(r"(.*?)(\d+)$", name)  
    if match:
        return match.group(1), int(match.group(2))  
    else:
        return name, -1
    
def get_time_slice(name: str) -> int:
    """
    Extracts the time slice index from a dynamic Bayesian network (dBN) variable name.

    In dBNs, time-dependent variables follow the convention "X" + str(t), 
    where `X` is the base name and `t` is the time slice. If the variable 
    does not follow this pattern, it is considered static.

    Args:
        name (str): The variable name, potentially including a time slice suffix.

    Returns:
        int: The extracted time slice index, or -1 if the variable is static.
    """
    return split_name(name)[1]  

def get_k(dbn: gum.BayesNet) -> int:
    """
    Determines the order `k` of a K-Time-Slice Bayesian Network (K-TBN).

    A K-TBN follows the convention where the first `k` time slices are named 
    as `base_name + time_slice`, representing dependencies up to `k` past slices.

    Args:
        dbn (gum.BayesNet): The K-TBN whose order `k` is to be determined.

    Returns:
        int: The maximum time slice index (`k`) inferred from the variable names.
    """
    return np.max([split_name(name)[1] for name in dbn.names()])

def unrollKTBN(dbn : gum.BayesNet, n : int) -> gum.BayesNet:
    """
    Unrolls a KTBN for `n` time slices

    Args:
        dbn (gum.BayesNet): The KTBN to unroll
        n (int): The number of time slices to unroll

    Returns:
        gum.BayesNet: The unrolled KTBN.
    """
    # TODO Entree liste variables, K, et bn (fonctions deviennent methodes)
    k = get_k(dbn)

    bn = gum.BayesNet()

    # Variable creation
    for node_id in dbn.nodes():
        
        name = dbn.variable(node_id).name()
        static_name, t_slice = split_name(name)

        if t_slice < k:
            bn.add(dbn.variable(node_id))
        else:
            for t in range(k+n, k-1,-1):
                new_id = bn.add(dbn.variable(node_id),)
                if t > k:
                    bn.changeVariableName(new_id,static_name+str(t))
    
    #Arcs  
    for node_id in dbn.nodes():

        name = dbn.variable(node_id).name()
        static_name, t_slice = split_name(name)

        parents = list(dbn.cpt(node_id).names)
        parents.remove(name)


        for parent_name in parents:

            static_p_name, p_t_slice = split_name(parent_name)

            if t_slice < k:

                if p_t_slice < k:

                    bn.addArc(parent_name, name)
                else:
                    # TODO meme chose pour t < k
                    raise TypeError(f"Found arc from time slice k to time slice {t_slice}!")
            else:

                for t in range(k, k+n+1):
                    if p_t_slice == -1:
                        bn.addArc(parent_name, static_name+str(t))
                    else:
                        bn.addArc(static_p_name+str(t-k+p_t_slice),static_name+str(t))

    #CPT
    for node_id in dbn.nodes():
        name = dbn.variable(node_id).name()
        static_name, t_slice = split_name(name)
        if t_slice < k:
            bn.cpt(name).fillWith(dbn.cpt(name), bn.cpt(name).names)
        else:

            for t in range(k, k+n+1):

                dict_names = {
                p_name: p_name if (p_t_slice := split_name(p_name)[1]) == -1 
                    else split_name(p_name)[0] + str(p_t_slice - t + k)
                    for p_name in bn.cpt(static_name + str(t)).names
                }

                bn.cpt(static_name+str(t)).fillWith(dbn.cpt(name), dict_names)
    return bn

