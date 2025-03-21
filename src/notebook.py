from pyAgrum.lib.notebook import showGraph
import pydot as dot 
from pyAgrum.lib.dynamicBN import getTimeSlicesRange, noTimeCluster
import pyAgrum as gum

def showKTBN(dbn):
    """
    Display a dynamic Bayesian network using the custom DOT graph format.

    Parameters
    ----------
    dbn : pyAgrum.DynamicBayesNet
        The dynamic Bayesian network to display.
     """
    return dbn.kTBN

def _TimeSlicesToDot(dbn):
    """
    Try to correctly represent dBN and kTBN in dot format.
    
    This version uses a composite node ID (letter + time slice) so that invisible
    edges can order nodes correctly, but displays only the letter.
    
    Parameters
    ----------
    dbn: pyAgrum.BayesNet
      a 2TBN or an unrolled BN with a kTBN attribute.
    """
    timeslices = getTimeSlicesRange(dbn.kTBN)
    kts = sorted(timeslices.keys(), key=lambda x: -1 if x == noTimeCluster 
                   else 1e8 if x == 't' else int(x))
    
    g = dot.Dot(graph_type='digraph')
    g.set_rankdir("LR")
    g.set_splines("ortho")
    g.set_node_defaults(color="#000000", fillcolor="white", style="filled")
    
    for k in kts:
        if k != noTimeCluster:
            cluster = dot.Cluster(k, label=f"Time slice {k}", bgcolor="#DDDDDD", rankdir="same")
            g.add_subgraph(cluster)
        else:
            cluster = g  
        
        # Add nodes for this time slice.
        # Here, each node is given a composite id: letter + time slice.
        # However, the displayed label is only the letter.
        for (n, label) in sorted(timeslices[k]):
            # n is the variable name (assumed to be of the form "letter_time")
            # _codeToUserName returns a tuple (letter, time) – we take the letter.
            letter = str(dbn._codeToUserName(n)[0])
            composite_id = f"{letter}{k}"  # e.g., "A0" if letter is "A" and k is "0"
            cluster.add_node(dot.Node(composite_id, label=letter))
    
    g.set_edge_defaults(color="blue", constraint="False")
    for tail, head in dbn.kTBN.arcs():
        # Retrieve the full variable name from the kTBN, then split into (letter, time)
        tail_tuple = dbn._codeToUserName(dbn.kTBN.variable(tail).name())
        head_tuple = dbn._codeToUserName(dbn.kTBN.variable(head).name())
        tail_id = f"{tail_tuple[0]}{tail_tuple[1]}"
        head_id = f"{head_tuple[0]}{head_tuple[1]}"
        g.add_edge(dot.Edge(tail_id, head_id))
    
    g.set_edge_defaults(style="invis", constraint="True")
    for x in timeslices["0"]:
        # x[0] is the variable name; extract its letter.
        letter = str(dbn._codeToUserName(x[0])[0])
        prec = None
        for k in kts:
            if k == noTimeCluster:
                continue
            if prec is not None:
                g.add_edge(dot.Edge(f"{letter}{prec}", f"{letter}{k}"))
            prec = k
    
    return g

def showTimeSlices(dbn):
    """
    Display a dynamic Bayesian network using the custom DOT graph format.

    Parameters
    ----------
    dbn : pyAgrum.DynamicBayesNet
        The dynamic Bayesian network to display.
    """
    return showGraph(_TimeSlicesToDot(dbn))