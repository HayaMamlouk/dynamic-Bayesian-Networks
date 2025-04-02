import pyAgrum as gum
import logging

logging.basicConfig(
    filename='mylog.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class dCommon:
    """
    A common class for Dynamic Bayesian Networks (DBNs) and Dynamic Tensors.
    """

    def __init__(self, separator="#"):
        """
        Initializes the dCommon object.
        """
        self.separator = separator

    def _userToCodeName(self, name, time_slice):
        r"""
        Converts a user-friendly variable name to a code-friendly name.

        
        Parameters
        ----------
            name (str): The user-friendly variable name.
            time_slice (int): The time slice to which the variable belongs.
       
        Returns
        -------
            str: The code-friendly variable name.
        """
        return f"{name}{self.separator}{time_slice}"
    
    def _codeToUserName(self, name):
        """
        Converts a code-friendly variable name to a user-friendly name. The name should be in the format: {string + separator + time_slice}.

        Parameters
        ----------
            name (str): The code-friendly variable name.

        Returns
        -------
            tuple: A tuple containing the user-friendly variable name and the time slice
        """
        split = name.split(self.separator)
        return split[0], split[1]
    
    def _nameToString(self, var_name):
        r"""
        Converts a variable name to a string representation. Name should be in the format: {string + separator + time_slice}.

        Parameters
        ----------
            var_name (str): The variable name.

        Returns
        -------
            str: A string representation of the variable name.
        """
        name, t = self._codeToUserName(var_name)
        return f"('{name}', {t})"

    def _arcToString(self, tail, head):
        r"""
        Converts a pyAgrum type arc to a string representation.
        
        Parameters
        ----------
        tail : str
            The name of the tail variable.

        head : str
            The name of the head variable.

        Returns
        -------
        str
            A string representation of the arc.
        """

        return f"{self._nameToString(self._internalNameFromId(tail))} -> {self._nameToString(self._internalNameFromId(head))}"
    
    def _internalNameFromId(self, id):
        r"""
        Returns the internal name of a variable given its id.

        Parameters
        ----------
        id : int
            The id of the variable.
        
        Returns
        -------
        str
            The internal name of the variable (shows separator).
        """
        return self.kTBN.variable(id).name()
    
    def nameFromId(self, id):
        r"""
        Returns the name of a variable given its id.

        Parameters
        ----------
        id : int
            The id of the variable.
        
        Returns
        -------
        str
            The name of the variable.
        """
        return self._codeToUserName(self._internalNameFromId(id))

    def idFromName(self, var):
        r"""

        Returns a variable's id given its name in the graph.

        Parameters
        ----------
        name : tuple
        	The variable's name from which the id is returned.

        Notes
        -----
          A convenient shortcut for `g.variableFromName(name)` is `g[name]`.

        Returns
        -------
        int :
        	The variable's node id.

        Raises
        ------
        pyAgrum.NotFound
        	If name does not match a variable in the graph

        """
        n, t = var
        name = self._userToCodeName(n, t)
        return self.kTBN.idFromName(name)
    

class DynamicBayesNet(dCommon):
    """
    A class to represent and manipulate Dynamic Bayesian Networks (kTBNs).
    This class extends the functionality of a standard Bayesian Network by incorporating
    a temporal dimension, allowing the modeling of time-evolving systems.
    """

    def __init__(self, k, separator="#"):
        """
        Initializes the kTBN object.
        """
        self.kTBN = gum.BayesNet()  # The underlying Bayesian Network (pyAgrum.BayesNet)
        self.variables = set()       # List of variables in the kTBN (atemporal)
        self.k = k                # Time horizon (number of time slices)
        super().__init__(separator)
    
    def add(self, v):
        r"""
        Adds a variable to the kTBN across all time slices.

        Parameters
        ----------
            v (pyAgrum.Variable): The variable to be added. This variable is created using one of pyAgrum's variable creation methods.
        """

        # Check if the variable is already in the kTBN
        # if name in self.variables:
        #     raise ValueError(f"Variable '{v.name()}' already exists in the kTBN.")

        
        # Extract the variable's name and description
        name = v.name()
        description = v.description()

        # Check if the variable name contains the separator
        if self.separator in name:
            raise ValueError(f"Variable name '{name}' contains the separator '{self.separator}'.")

        # Add the variable to all time slices
        for t in range(self.k):
            var_name = super()._userToCodeName(name, t)  # Format: {variable_name}#{time_slice} it checks if separator is in name
            var = v.clone()
            var.setName(var_name)
            var.setDescription(f"{description} (t={t})")

            self.kTBN.add(var)  # Add to the base network

        # Add the variable to the set of variables
        self.variables.add(name)

        logger.info(f"Added variable '{v.name()}' across {self.k} time slices.")

    def addFast(self, var_description):
        r"""
        Adds a variable to the kTBN across all time slices. This method uses fast syntax to add a variable in the dbn.

        Parameters
        ----------
            var_description (str): String following fast syntax description.
            domaine (int): The domain size of the variable (default: 2).
        """

        # Create the variable
        var = gum.fastVariable(var_description)

        # Add the variable to the kTBN
        self.add(var)

    def addArc(self, tail, head):
        r"""
        Adds a directed arc between variables v1 and v2 across time slices.

        Parameters
        ----------
            tail (tuple): The variable the arc starts from, represented as (n, t)
            head (tuple): The variable the arc ends at, represented as (n, t) where:
                - n is the variable name (string).
                - t is the time slice (int).
        """
        # Extract the variable names and time slices
        _, t1 = tail
        _, t2 = head

        # Check for backward arcs (t1 > t2)
        if t1 > t2:
            raise ValueError(f"Backward arc not allowed: {t1} -> {t2}.")

        # Check if the arc spans more than k time slices
        if t1 >= self.k or t2 >= self.k:
            raise ValueError(f"Arc spans more than {self.k} time slices.")
         
        # Get the variable ids
        i1 = self.idFromName(tail)
        i2 = self.idFromName(head)

        # Add the arc to the base network
        self.kTBN.addArc(i1, i2)
        
        logger.info(f"Added arc {self._arcToString(i1, i2)} to the kTBN.")

    def arcs(self):
        r"""

        Returns
        -------
        list
        	The list of arcs in the IBayesNet

        """

        arcs = self.kTBN.arcs()
        l_arcs = []

        for arc in arcs:
            i1, i2 = arc
            l_arcs.append(self._arcToString(i1, i2))
        
        return l_arcs

    def eraseArc(self, tail, head):
        r"""

        Removes an arc in the BN, and update head's CTP.

        If (tail, head) doesn't exist, the nothing happens.

        Parameters
        ----------
        arc : pyAgrum.Arc when calling eraseArc(arc)
        	The arc to be removed.
        head : Union[int,str]
        	a variable's id (int) or name for the head when calling eraseArc(head,tail)
        tail : Union[int,str]
        	a variable's id (int) or name for the tail when calling eraseArc(head,tail)

        """
        i1 = self.idFromName(tail)
        i2 = self.idFromName(head)

        # Remove the arc from the base network
        self.kTBN.eraseArc(i1, i2)

        logger.info(f"Deleted arc {self._arcToString(i1, i2)} from the kTBN.")       

    def erase(self, var):
        r"""
        Deletes a variable and its associated arcs from all time slices.

        Parameters
        ----------
            var (String): The name of the variable to be deleted.
        """

        # Delete the variable from all time slices
        for t in range(self.k):
            var_name = super()._userToCodeName(var, t)  # Format: {variable_name}#{time_slice}
            self.kTBN.erase(var_name)  # Delete the variable from the base network

        self.variables.remove(var)  # Remove the variable from the set of variables

        logger.info(f"Deleted variable '{var}' and its associated arcs from the kTBN.")

    def cpt(self, var):
        r"""
        Returns the Conditional Probability Table (CPT) of a variable.

        Parameters
        ----------
            var (tuple): The name and time slice of the variable.

        Returns
        -------
            pyAgrum.Potential: The CPT of the variable.
        """
        name, ts = var
        raw_potential = self.kTBN.cpt(self._userToCodeName(name, ts))
        return dPotential(raw_potential, self.separator)
    
    def generateCPTs(self) :
        r"""
        Randomly generates CPTs for a given structure.
        """
        return self.kTBN.generateCPTs()

class dPotential(dCommon):
    """
    A wrapper for a pyAgrum.Potential object that exposes a user-friendly
    interface using tuple-based variable identifiers (e.g., ('A', 1))
    instead of the internal string format (e.g., 'A#1').
    
    Inherits from dCommon to directly access name conversion functions.
    """
    def __init__(self, potential, separator="#"):
        """
        Parameters
        ----------
        potential : pyAgrum.Potential
            The underlying potential (CPT) from pyAgrum.
        separator : str, optional
            The separator used for variable names, by default "#".
        """
        super().__init__(separator)
        self._potential = potential

    def __getitem__(self, key):
        """
        Allows user-friendly indexing of the potential.
        For example: dBN.cpt(('A', 1))[{('B', 0): 1}]
        """
        if isinstance(key, dict):
            internal_key = {}
            for var, state in key.items():
                if isinstance(var, tuple):
                    # Convert tuple ('B', 0) to "B#0"
                    internal_key[self._userToCodeName(var[0], var[1])] = state
                else:
                    raise ValueError(f"Invalid key format: {var}")
            key = internal_key
        return self._potential[key]

    def __setitem__(self, key, value):
        """
        Allows setting values in the potential using user-friendly keys.
        For example: dBN.cpt(('A', 1))[{('B', 0): 1}] = some_value
        """
        if isinstance(key, dict):
            internal_key = {}
            for var, state in key.items():
                if isinstance(var, tuple):
                    internal_key[self._userToCodeName(var[0], var[1])] = state
                else:
                    raise ValueError(f"Invalid key format: {var}")
            key = internal_key
        self._potential[key] = value

    def fillWith(self, value):
        """
        Fills the potential with a constant or with a list of values.
        
        Parameters
        ----------
        value : int, float, or list
            - If a constant, fills the entire potential with that constant.
            - If a list, it is assumed to be a flattened version of the potential.
              The length of the list must match the number of cells in the potential.
        """
        self._potential.fillWith(value)


    def __str__(self):
        """
        Returns a user-friendly string representation of the potential.
        """
        # Get pyAgrum.Potential CPT table of the variable
        cpt = self._potential
        
        # Create a new Potential with user-friendly variable names
        userCPT = gum.Potential()
        
        # Get the list of variables in the CPT
        internalVariables = cpt.variablesSequence()

        # Create a mapping from user-friendly variable names to internal variable names
        mapping = {}

        for v in internalVariables:
            # Create a new variable with a user-friendly name and put it into the new CPT
            userVar = v.clone()
            userVar.setName(self._nameToString(v.name()))

            userCPT.add(userVar)
            mapping[userVar.name()] = v.name()

        # bn.cpt(new_head).fillWith(bn.cpt(prev_head), mapping)
        userCPT.fillWith(cpt, mapping)
        return userCPT.__str__()




    