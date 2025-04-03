from pyAgrum.lib.notebook import showGraph
import pyAgrum as gum
import pydot as dot 
from pyAgrum.lib.dynamicBN import getTimeSlicesRange, noTimeCluster, Rectangle
import numpy as np
import pyAgrum.lib.notebook as gnb 
from matplotlib import pyplot as plt

def _kTBNToDot(dbn):
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

def showKTBN(dbn):
    """
    Display a dynamic Bayesian network using the custom DOT graph format.

    Parameters
    ----------
    dbn : pyAgrum.DynamicBayesNet
        The dynamic Bayesian network to display.
    """
    return showGraph(_kTBNToDot(dbn))

def _TimeSlicesToDot(bn):
  """
  Try to correctly represent BN in dot format (user friendly)

  Parameters
  ----------
  dbn: pyAgrum.BayesNet
        an unrolled BN
  """
  timeslices = getTimeSlicesRange(bn)
  kts = sorted(timeslices.keys(), key=lambda x: -1 if x == noTimeCluster else 1e8 if x == 't' else int(x))

  # dynamic member makes pylink unhappy
  # pylint: disable=no-member
  g = dot.Dot(graph_type='digraph')
  g.set_rankdir("LR")
  g.set_splines("ortho")
  g.set_node_defaults(color="#000000", fillcolor="white", style="filled")

  for k in kts:
    if k != noTimeCluster:
      cluster = dot.Cluster(k, label=f"Time slice {k}", bgcolor="#DDDDDD", rankdir="same")
      g.add_subgraph(cluster)
    else:
      cluster = g  # small trick to add in graph variable in no timeslice
    for (n, label) in sorted(timeslices[k]):
      letter, t = n.split("#")
      composite_id = f"{letter}{t}" 
      cluster.add_node(dot.Node(composite_id, label=letter))

  g.set_edge_defaults(color="blue", constraint="False")
  for tail, head in bn.arcs():
    tail_tuple = bn.variable(tail).name().split("#")
    head_tuple = bn.variable(head).name().split("#")
    tail_id = f"{tail_tuple[0]}{tail_tuple[1]}"
    head_id = f"{head_tuple[0]}{head_tuple[1]}"
    g.add_edge(dot.Edge(tail_id, head_id))

  g.set_edge_defaults(style="invis", constraint="True")
  for x in timeslices["0"]:
    name = x[1].split("#")[0]
    prec = None
    for k in kts:
      if k == noTimeCluster:
        continue
      if prec is not None:
        g.add_edge(dot.Edge(f'"{name}{prec}"', f'"{name}{k}"'))
      prec = k

  return g


def showUnrolled(dbn, size=None):
  """
  Try to show correctly an unrolled BN (user friendly)

  Parameters
  ----------
  dbn: pyAgrum.BayesNet
        an unrolled BN
  size: int or str
    size of the fig
  """

  # jupyter notebooks is optional
  # pylint: disable=import-outside-toplevel
  if size is None:
    size = gum.config["dynamicBN", "default_graph_size"]

  showGraph(_TimeSlicesToDot(dbn), size)

def unrollKTBN(dbn, nbr):
    """
    Unrolls a dynamic Bayesian network (DBN) into a static Bayesian network.
    This version leverages the parent information from the initial (template) network.
    
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
    # The original number of time slices in the template
    k = dbn.k
    if nbr < k:
        raise ValueError("The number of time slices to unroll must be greater than or equal to the current number of time slices.")
    
    # Start by copying the initial (template) network.
    bn = gum.BayesNet(dbn.kTBN)
    
    # For each new time slice beyond the template.
    for t in range(k, nbr):
        for var in dbn.variables:
            # Create the new instance for this variable.
            new_var = dbn._userToCodeName(var, t)
            # Retrieve a template instance to copy its domain.
            template_var = bn.variable(dbn._userToCodeName(var, 0))
            new_label_var = template_var.clone()
            new_label_var.setName(new_var)
            new_label_var.setDescription(f"{template_var.description()} (t={t})")
            bn.add(new_label_var) 

            
            # Use the template instance (in the last time slice of the template) to retrieve parent info.
            template_node = dbn._userToCodeName(var, k - 1)         # Template node
            mapping = { new_var: dbn._userToCodeName(var, t - 1) }  # Mapping for CPT update
            for parent in bn.parents(template_node):
                # Decode the parent's base name and its time slice from the template.
                parent_user = dbn._codeToUserName(bn.variable(parent).name())
                parent_base, parent_ts = parent_user[0], int(parent_user[1])
                
                # Calculate the time distance between the template node and its parent.
                d = (k - 1) - parent_ts
                # The corresponding parent for the new time slice is at time: t - d.
                parent_new = dbn._userToCodeName(parent_base, t - d)
                bn.addArc(parent_new, new_var)

                # Build the mapping for the CPT update:
                # - Map the new variable to its previous instance.
                parent_user = dbn._codeToUserName(bn.variable(parent).name())
                parent_base, parent_ts = parent_user[0], int(parent_user[1])
                d = (k - 1) - parent_ts
                new_parent = dbn._userToCodeName(parent_base, t - d)
                prev_parent = dbn._userToCodeName(parent_base, t - d - 1)
                mapping[new_parent] = prev_parent
            
            # Copy the CPT from the previous time slice using the mapping.
            bn.cpt(new_var).fillWith(bn.cpt(dbn._userToCodeName(var, t - 1)), mapping)
    
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

flow = gnb.FlowLayout()

def getPosterior(bn, evs, target):
    """
    shortcut for proba2histo(gum.getPosterior(bn,evs,target))

    Parameters
    ----------
    bn: "pyAgrum.BayesNet"
        the BayesNet
    evs: Dict[(str, int):int|str|List[float]]
        map of evidence
    target: (str, int) user friendly format
        name of target variable

    Returns
    ------
        the matplotlib graph
    """
        # we want to transform for target (str, int) to (strint)
    raw_target = target[0] + str(target[1])
    raw_evs = {}
    for k, v in evs.items():
        raw_evs[k[0] + str(k[1])] = v

    return gnb.getPosterior(bn, evs=raw_evs, target=raw_target)

def plotFollow(lovars, kTBN, T, evs):
  """
  plots modifications of variables in a kTBN knowing the size of the time window (T) and the evidence on the sequence.

  :param lovars: list of variables to follow
  :param twoTdbn: the kTBN
  :param T: the time range
  :param evs: observations
  """
  # variables input as atemporal so no need to change
  raw_evs = {}
  for key, value in evs.items():
    raw_evs[key[0] +"#"+ str(key[1])] = value

  plotFollowUnrolled(lovars, unrollKTBN(kTBN, T), T, raw_evs)

def plotFollowUnrolled(lovars, dbn, T, evs, vars_title=None):
  """
  plot the dynamic evolution of a list of vars with a dBN

  :param lovars: list of variables to follow
  :param dbn: the unrolled dbn
  :param T: the time range
  :param evs: observations
  :param vars_title: string for default or a dictionary with the variable name as key and the respective title as value.
  """
  ie = gum.LazyPropagation(dbn)
  ie.setEvidence(evs)
  ie.makeInference()

  x = np.arange(T)

  for var in lovars:
    v0 = dbn.variableFromName(var + "#0")
    lpots = []
    for i in range(v0.domainSize()):
      serie = []
      for t in range(T):
        serie.append(ie.posterior(dbn.idFromName(var + "#" + str(t)))[i])
      lpots.append(serie)

    _, ax = plt.subplots()
    plt.xlim(left=0, right=T - 1)
    plt.ylim(top=1, bottom=0)
    ax.xaxis.grid()

    # Setting a customized title
    if vars_title is None:
      plt.title(f"Following variable {var}", fontsize=20)
    elif len(vars_title) != 0:
      plt.title(vars_title[var], fontsize=20)
    else:
      raise TypeError("Incorrect format of the plots title dictionary")

    plt.xlabel('time')

    stack = ax.stackplot(x, lpots)

    proxy_rects = [Rectangle((0, 0), 1, 1, fc=pc.get_facecolor()[0])
                   for pc in stack]
    labels = [v0.label(i) for i in range(v0.domainSize())]
    plt.legend(proxy_rects, labels, loc='center left',
               bbox_to_anchor=(1, 0.5), ncol=1, fancybox=True, shadow=True
               )


