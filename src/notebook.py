from pyAgrum.lib.notebook import showGraph
import pyAgrum as gum
import pydot as dot 
from pyAgrum.lib.dynamicBN import getTimeSlicesRange, noTimeCluster
import copy

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


def unrollKTBN(dbn, nbr):
    """
    Unrolls a dynamic Bayesian network into a static Bayesian network.

    Parameters
    ----------
    dbn : pyAgrum.DynamicBayesNet
        The dynamic Bayesian network to unroll.

    nbr : int
        The total number of time slices to unroll.

    Returns
    -------
    pyAgrum.BayesNet
        The unrolled Bayesian network.
    """  
    # Check that the number of time slices to unroll is greater than the current number of time slices.
    k = dbn.k
    if nbr < k:
        raise ValueError("The number of time slices to unroll must be greater than or equal to the current number of time slices.")
    
    # Create a deep copy of the current time-slice network
    bn = copy.deepcopy(dbn.kTBN)
    
    # Add variables for the new time slices.
    for var_name in dbn.variables:
        for time_slice in range(k, nbr):
            bn.add(dbn._userToCodeName(var_name, time_slice))
    
    # -----------------------------------------------------------------------
    # Collect information on arcs from the original network.
    # For every arc in dbn.kTBN where the head is in the last original time slice (k-1):
    #   - Record the head variable base name (for later CPT updates)
    #   - Record the tail information as a tuple: (tail_base, tail_diff)
    #
    # Here, tail_diff = (head_time - tail_time). For example, if we have an arc from a#1 to b#2,
    # then tail_diff is 2 - 1 = 1.
    # -----------------------------------------------------------------------
    head_vars = set()
    tails = {}  # mapping: head_base -> set of tuples (tail_base, tail_diff)
    
    for arc in dbn.kTBN.arcs():
        tail_code, head_code = arc
        
        # Extract the base name and time slice of the head variable.
        head_user = dbn._codeToUserName(dbn.kTBN.variable(head_code).name())
        head_base, head_ts = head_user[0], int(head_user[1])
        
        # Check if the head variable is in the last original time slice.
        if head_ts == k - 1:

            # used later for CPT update
            head_vars.add(head_base)
            if head_base not in tails:
                tails[head_base] = set()

            # Extract the base name and time slice of the tail variable.
            tail_user = dbn._codeToUserName(dbn.kTBN.variable(tail_code).name())
            tail_base, tail_ts = tail_user[0], int(tail_user[1])

            # Calculate the difference in time slices between the head and tail variables. Use later for CPT update.
            tail_diff = head_ts - tail_ts  # difference in time slices between head and tail
            tails[head_base].add((tail_base, tail_diff)) 
            
            # Extend the arc into the new time slices.
            for i in range(k + 1, nbr + 1):
                new_tail = dbn._userToCodeName(tail_base, tail_ts + i - k)
                new_head = dbn._userToCodeName(head_base, head_ts + i - k)
                bn.addArc(new_tail, new_head)
    
    # -----------------------------------------------------------------------
    # Update the CPTs for the new head variable instances.
    # For each head variable base that we recorded:
    #   For each new time slice t (from k to nbr-1), copy its CPT from the previous time slice
    #   and update the tail variables in the CPT.
    #
    # The mapping for fillWith is built as follows:
    #   - Map the new head variable (e.g. "B#3") to its previous version (e.g. "B#2").
    #   - For each tail feeding into it (e.g. from "A"), if tail_diff = d then for the new head at time t,
    #     we map the new tail (which is "A#t" if d==1, or "A#(t - d + 1)" in general) to its previous instance.
    #     Here we assume that the correct relation is:
    #         new tail at time t: dbn._userToCodeName(tail_base, t - d)
    #         previous tail:    dbn._userToCodeName(tail_base, t - d - 1)
    # -----------------------------------------------------------------------
    for head_base in head_vars:
        for t in range(k, nbr):
            new_head = dbn._userToCodeName(head_base, t)
            prev_head = dbn._userToCodeName(head_base, t - 1)
            
            # Build the mapping for the CPT update.
            mapping = { new_head: prev_head }
            for (tail_base, tail_diff) in tails[head_base]:
                # For the new head at time t, the tail variable appears at time: t - tail_diff.
                new_tail = dbn._userToCodeName(tail_base, t - tail_diff)
                prev_tail = dbn._userToCodeName(tail_base, t - tail_diff - 1)
                mapping[new_tail] = prev_tail
        
            bn.cpt(new_head).fillWith(bn.cpt(prev_head), mapping)
    
    return bn

def showCPT(dbn, var):
    """
    Display the conditional probability table (CPT) of a variable in a dynamic Bayesian network.

    Parameters
    ----------
    dbn : pyAgrum.DynamicBayesNet
        The dynamic Bayesian network containing the variable.

    var : str
        The name of the variable whose CPT to display.
    """
    # Get pyAgrum.Potential CPT table of the variable
    cpt = dbn.cpt(var)._potential
    
    # Create a new Potential with user-friendly variable names
    userCPT = gum.Potential()
    
    # Get the list of variables in the CPT
    internalVariables = cpt.variablesSequence()

    # Create a mapping from user-friendly variable names to internal variable names
    mapping = {}

    for v in internalVariables:
        # Create a new variable with a user-friendly name and put it into the new CPT
        userVar = v.clone()
        userVar.setName(dbn._nameToString(v.name()))

        userCPT.add(userVar)
        mapping[userVar.name()] = v.name()

    # bn.cpt(new_head).fillWith(bn.cpt(prev_head), mapping)
    userCPT.fillWith(cpt, mapping)
    return userCPT

    # Create a new Potential with user-friendly variable names
    
