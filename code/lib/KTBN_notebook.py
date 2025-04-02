import pyAgrum as gum
import pyAgrum.lib.notebook as gnb
from KTBN import KTBN

# Functions for displaying KTBN in notebooks
def show_ktbn(ktbn, **kwargs):
    """
    Display a KTBN in a notebook.
    
    Args:
        ktbn (KTBN): The KTBN to display
        **kwargs: Additional arguments for gnb.showBN
    """
    bn = ktbn.to_bn()
    return gnb.showBN(bn, **kwargs)

def show_unroll(ktbn, n, **kwargs):
    """
    Display an unrolled KTBN in a notebook.
    
    Args:
        ktbn (KTBN): The KTBN to unroll and display
        n (int): Number of additional time slices to unroll
        **kwargs: Additional arguments for gnb.showBN
    """
    unrolled_bn = ktbn.unroll(n)
    return gnb.showBN(unrolled_bn, **kwargs)
