import pyAgrum as gum

class dBN:
    """
    A class to represent and manipulate Dynamic Bayesian Networks (DBNs).
    This class extends the functionality of a standard Bayesian Network by incorporating
    a temporal dimension, allowing the modeling of time-evolving systems.
    """

    def __init__(self):
        """
        Initializes the dBN object.
        """
        self.base_network = None  # The underlying Bayesian Network (pyAgrum.BayesNet)
        self.variables = {}       # List of variables in the DBN
        self.arcs = []            # List of arcs (dependencies) in the DBN
        self.k = 0                # Time horizon (number of time slices)
        self.num_vars = 0         # Number of variables added to the DBN

    def createDBN(self, k: int):
        """
        Creates the base Dynamic Bayesian Network and sets the time horizon `k`.

        Parameters:
        k (int): The number of time slices in the DBN.
        """
        # Set the time horizon
        self.k = k

        # Initialize the base Bayesian Network
        self.base_network = gum.BayesNet()

        # Clear any existing variables and arcs
        self.variables = {}
        self.arcs = []

        print(f"Created a dBN with {k} time slices.")



    def addVar(self, v):
        """
        Adds a variable to the dBN across all time slices.

        Parameters:
            v (pyAgrum.Variable): The variable to be added. This variable is created using one of pyAgrum's variable creation methods.
        """
        # Check if the variable is already in the DBN
        if v.name() in [var for var in self.variables]:
            raise ValueError(f"Variable '{v.name()}' already exists in the dBN.")

        # Add the variable to all time slices
        for t in range(self.k):
            var_name = f"{v.name()}#{t}"  # Format: {variable_name}#{time_slice}
            var = v.clone()
            var.setName(var_name)
            var.setDescription(f"{v.description()} (t={t})")

            n = self.base_network.add(var)  # Add to the base network
            self.variables[var_name] = n  # Add to the list of variables

        print(f"Added variable '{v.name()}' across {self.k} time slices.")


    def getVariables(self):
        """
        Returns the list of variables in the DBN.
        """
        return list(self.variables.keys()
)

    def addArc(self, a):
        """
        Adds a directed arc between variables across time slices.

        Parameters:
            a (tuple): The arc to be added, represented as ((v1, t1), (v2, t2)), where:
                - v1 and v2 are variables.
                - t1 and t2 are time slices.
        """
        (v1, t1), (v2, t2) = a


        arc = ((v1.name(), t1), (v2.name(), t2))

        # Check if the arc already exists
        if arc in self.arcs:
            raise ValueError(f"Arc {arc} already exists in the DBN.")

        # Check for backward arcs (t1 > t2)
        if t1 > t2:
            raise ValueError(f"Backward arc not allowed: {arc}.")

        # Check if the arc spans more than k time slices
        if abs(t2 - t1) > self.k:
            raise ValueError(f"Arc {arc} spans more than {self.k} time slices.")

        # Check if the variables exist in the DBN

        var1 = f"{v1.name()}#{t1}"
        var2 = f"{v2.name()}#{t2}"
        if var1 not in [var for var in self.variables] or var2 not in [var for var in self.variables]:
            raise ValueError(f"One or both variables in arc {arc} do not exist in the DBN.")

        # Add the arc to the base network
        self.base_network.addArc(self.variables[var1], self.variables[var2])
        self.arcs.append(arc)  # Add to the list of arcs
        
        print(f"Added arc {arc} to the DBN.")

    def getArcs(self):
        """
        Returns the list of arcs in the DBN.
        """
        arcss = []
        for arc in self.arcs:
            (v1, t1), (v2, t2) = arc
            arcss.append(f"({v1}#{t1}, {v2}#{t2})")
        return arcss


    def deleteArc(self, a):
        """
        Deletes a specified arc from the DBN.

        Parameters:
            a (tuple): The arc to be deleted, represented as ((v1, t1), (v2, t2)), where:
                - v1 and v2 are variables.
                - t1 and t2 are time slices.
        """
        (v1, t1), (v2, t2) = a

        # Construct the arc in the format stored in self.arcs
        arc = ((v1.name(), t1), (v2.name(), t2))

        # Check if the arc exists in the DBN
        if arc not in self.arcs:
            raise ValueError(f"Arc {arc} does not exist in the DBN.")

        # Construct the variable names for the given time slices
        var1 = f"{v1.name()}#{t1}"
        var2 = f"{v2.name()}#{t2}"

        # Remove the arc from the base network
        self.base_network.eraseArc(self.variables[var1], self.variables[var2])

        # Remove the arc from the list of arcs
        self.arcs.remove(arc)

        print(f"Deleted arc {arc} from the DBN.")       



    def deleteVar(self, v):
        """
        Deletes a variable and its associated arcs from all time slices.

        Parameters:
            v (pyAgrum.Variable): The variable to be deleted.
        """
        # Check if the variable exists in the DBN
        if not any(var.startswith(v.name() + "#") for var in self.variables):
            raise ValueError(f"Variable '{v.name()}' does not exist in the DBN.")

        # Delete the variable from all time slices
        for t in range(self.k):
            var_name = f"{v.name()}#{t}"
            if var_name in self.variables:
                self.base_network.erase(self.variables[var_name])  # Remove from the base network
                del self.variables[var_name]  # Remove from the list of variables

        # Collect arcs associated with the variable to be removed
        arcs_to_remove = []
        for arc in self.arcs:
            (v1, t1), (v2, t2) = arc
            if v1 == v.name() or v2 == v.name():
                arcs_to_remove.append(arc)

        # Remove the collected arcs from the list of arcs
        for arc in arcs_to_remove:
            self.arcs.remove(arc)

        print(f"Deleted variable '{v.name()}' and its associated arcs from the DBN.")