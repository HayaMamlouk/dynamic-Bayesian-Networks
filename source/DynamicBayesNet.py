import pyAgrum as gum

class DynamicBayesNet:
    """
    A class to represent and manipulate Dynamic Bayesian Networks (DBNs).
    This class extends the functionality of a standard Bayesian Network by incorporating
    a temporal dimension, allowing the modeling of time-evolving systems.
    """

    def __init__(self, k, separator="#"):
        """
        Initializes the dBN object.
        """
        self.dBN = gum.BayesNet()  # The underlying Bayesian Network (pyAgrum.BayesNet)
        self.variables = set()       # List of variables in the DBN
        self.k = k                # Time horizon (number of time slices)
        self.separator = separator  # Separator used to distinguish variables across time slices

    def _userToCodeName(self, name, time_slice):
        """
        Converts a user-friendly variable name to a code-friendly name.

        --------------------------------------------
        Parameters:
            name (str): The user-friendly variable name.
            time_slice (int): The time slice to which the variable belongs.
        --------------------------------------------
        Returns:
            str: The code-friendly variable name.
        """
        if self.separator in name:
            raise ValueError(f"Variable name '{name}' contains the separator '{self.separator}'.")
        
        return f"{name}{self.separator}{time_slice}"
    
    def _codeToUserName(self, name):
        """
        Converts a code-friendly variable name to a user-friendly name. The name should be in the format: {string + separator + time_slice}.

        --------------------------------------------
        Parameters:
            name (str): The code-friendly variable name.

        --------------------------------------------
        Returns:
            tuple: A tuple containing the user-friendly variable name and the time slice
        """
        split = name.split(self.separator)
        return split[0], split[1]
    
    def _nameToString(self, var_name):
        """
        Converts a variable name to a string representation. Name should be in the format: {string + separator + time_slice}.

        --------------------------------------------
        Parameters:
            var_name (str): The variable name.

        --------------------------------------------
        Returns:
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

        return f"({self._nameToString(self._nameFromId(tail))} -> {self._nameToString(self._nameFromId(head))})"

    def _nameFromId(self, id):
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
        return self.dBN.variable(id).name()

    def add(self, v):
        """
        Adds a variable to the dBN across all time slices.

        Parameters:
            v (pyAgrum.Variable): The variable to be added. This variable is created using one of pyAgrum's variable creation methods.
        """

        # Check if the variable is already in the DBN
        # if name in self.variables:
        #     raise ValueError(f"Variable '{v.name()}' already exists in the dBN.")
        

        # Extract the variable's name and description
        name = v.name()
        description = v.description()

        # Convert name to be added to the dBN
        # name = self._userToCodeName(name, len(self.variables))

        # Add the variable to all time slices
        for t in range(self.k):
            var_name = self._userToCodeName(name, t)  # Format: {variable_name}#{time_slice}
            var = v.clone()
            var.setName(var_name)
            var.setDescription(f"{v.description()} (t={t})")

            self.dBN.add(var)  # Add to the base network

        # Add the variable to the set of variables
        self.variables.add(name)

        print(f"Added variable '{v}' across {self.k} time slices.")

    def idFromName(self, name):
        r"""

        Returns a variable's id given its name in the graph.

        Parameters
        ----------
        name : str
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

        return self.dBN.idFromName(name)

    def addArc(self, v1, v2):
        r"""
        Adds a directed arc between variables v1 and v2 across time slices.

        Parameters
        ----------
            v1 (tuple): The variable the arc starts from, represented as (n, t)
            v2 (tuple): The variable the arc ends at, represented as (n, t) where:
                - n is the variable name (string).
                - t is the time slice (int).
        """
        # Extract the variable names and time slices
        n1, t1 = v1
        n2, t2 = v2

        # Check for backward arcs (t1 > t2)
        if t1 > t2:
            raise ValueError(f"Backward arc not allowed: {t1} -> {t2}.")

        # Check if the arc spans more than k time slices
        if t1 >= self.k or t2 >= self.k:
            raise ValueError(f"Arc spans more than {self.k} time slices.")
        
        # Convert the variable names to the format used in the DBN
        var_name1 = self._userToCodeName(n1, t1)
        var_name2 = self._userToCodeName(n2, t2)
        
        i1 = self.idFromName(var_name1)
        i2 = self.idFromName(var_name2)

        # Add the arc to the base network
        self.dBN.addArc(i1, i2)
        
        print(f"Added arc ({n1, t1}) -> ({n2, t2}) to the DBN.")

    def arcs(self):
        r"""

        Returns
        -------
        list
        	The list of arcs in the IBayesNet

        """

        arcs = self.dBN.arcs()
        l_arcs = []

        for arc in arcs:
            i1, i2 = arc
            n1 = self._codeToUserName(self._nameFromId(i1))
            n2 = self._codeToUserName(self._nameFromId(i2))
            l_arcs.append(f"({n1}) -> ({n2})")
        
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
        # Extract the variable names and time slices
        n1, t1 = tail
        n2, t2 = head

        # Construct the variable names for the given time slices
        var1 = self._userToCodeName(n1, t1)
        var2 = self._userToCodeName(n2, t2)

        i1 = self.idFromName(var1)
        i2 = self.idFromName(var2)

        # Remove the arc from the base network
        self.dBN.eraseArc(i1, i2)

        print(f"Deleted arc {self._arcToString(i1, i2)} from the DBN.")       

    def erase(self, var):
        r"""
        Deletes a variable and its associated arcs from all time slices.

        Parameters
        ----------
            var (String): The name of the variable to be deleted.
        """

        # Delete the variable from all time slices
        for t in range(self.k):
            var_name = self._userToCodeName(var, t)  # Format: {variable_name}#{time_slice}
            self.dBN.erase(var_name)  # Delete the variable from the base network

        self.variables.remove(var)  # Remove the variable from the set of variables

        print(f"Deleted variable '{var}' and its associated arcs from the DBN.")

    def showDBN(self):
        """
        Displays the DBN.
        """
        return self.dBN