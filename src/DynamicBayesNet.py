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

        return f"{self._nameToString(self.nameFromId(tail))} -> {self._nameToString(self.nameFromId(head))}"
    
    def nameFromId(self, id):
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

    def idFromName(self, var):
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

    

   






