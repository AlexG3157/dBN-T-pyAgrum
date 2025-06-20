import pyagrum as gum
import pyagrum.lib.notebook as gnb
from KTBN import KTBN
import pydot as dot

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
    
def show_time_slice(ktbn, **kwargs):
    """
    Display all time slices of a KTBN in the notebook with variables grouped by time slice.
    Similar to pyagrum's showTimeSlices function for dynamic Bayesian networks.
    
    Args:
        ktbn (KTBN): The KTBN to display
        **kwargs: Additional arguments for gnb.showGraph
    
    Returns:
        The graphical representation of all time slices
    """
    # Get the original BN from the KTBN
    bn = ktbn.to_bn()
    
    # Create a dictionary to organize nodes by time slice
    time_slices = {}
    atemporal_vars = []
    
    # Organize variables by time slice
    for node_id in bn.nodes():
        var_name = bn.variable(node_id).name()
        var_base, var_time = ktbn.decode_name(var_name)
        
        if var_time == -1:
            atemporal_vars.append(var_name)
        else:
            if var_time not in time_slices:
                time_slices[var_time] = []
            time_slices[var_time].append(var_name)
    
    # Create a dot graph
    g = dot.Dot(graph_type='digraph')
    g.set_rankdir("LR")  # Left to right layout
    g.set_splines("ortho")  # Orthogonal edges
    g.set_node_defaults(color="#000000", fillcolor="white", style="filled")
    
    # Add clusters for time slices
    for ts in sorted(time_slices.keys()):
        cluster = dot.Cluster(str(ts), label=f"Time slice {ts}", bgcolor="#DDDDDD", rankdir="same")
        g.add_subgraph(cluster)
        
        # Add nodes for this time slice
        for var_name in sorted(time_slices[ts]):
            var_base, _ = ktbn.decode_name(var_name)
            cluster.add_node(dot.Node(f'"{var_name}"', label=f'"{var_base}"'))
    
    # Add atemporal variables to the main graph
    if atemporal_vars:
        cluster = dot.Cluster("atemporal", label="Atemporal variables", bgcolor="#EEFFEE", rankdir="same")
        g.add_subgraph(cluster)
        for var_name in sorted(atemporal_vars):
            cluster.add_node(dot.Node(f'"{var_name}"', label=f'"{var_name}"'))
    
    # Add arcs between variables
    g.set_edge_defaults(color="blue", constraint="False")
    for tail_id, head_id in bn.arcs():
        tail_name = bn.variable(tail_id).name()
        head_name = bn.variable(head_id).name()
        g.add_edge(dot.Edge(f'"{tail_name}"', f'"{head_name}"'))
    
    # Add invisible edges to maintain time slice order
    g.set_edge_defaults(style="invis", constraint="True")
    
    # For each variable that appears in multiple time slices,
    # add invisible edges to maintain ordering
    all_vars = set()
    for ts_vars in time_slices.values():
        for var_name in ts_vars:
            base_name, _ = ktbn.decode_name(var_name)
            all_vars.add(base_name)
    
    for base_name in all_vars:
        prev_node = None
        for ts in sorted(time_slices.keys()):
            current_node = None
            for var_name in time_slices[ts]:
                var_base, _ = ktbn.decode_name(var_name)
                if var_base == base_name:
                    current_node = var_name
                    break
            
            if current_node and prev_node:
                g.add_edge(dot.Edge(f'"{prev_node}"', f'"{current_node}"'))
            
            if current_node:
                prev_node = current_node
    
    # Display the graph using gnb's showGraph function
    return gnb.showGraph(g, **kwargs)
