<center>
<div style="font-size: 36px; font-weight: bold; color: blue;">
    Extending Dynamic Bayesian Networks in pyAgrum for Multi-Temporal Modeling
</div>

<div style="font-size: 24px">
    Implementation, Inference, and Structure Learning for Temporal Modeling

</div>

<h3> Haya Mamlouk - Doruk Ozgenc <br>
Pierre-Henri Wuillemin <br>
9 May 2025 </h3>
</center>
<br>

---

<br>
A Bayesian Network (BN) is a probabilistic graphical model that represents a set of variables and their conditional dependencies through a directed acyclic graph (DAG). Each node in the network corresponds to a random variable, and edges represent probabilistic dependencies between them. BNs are widely used for reasoning under uncertainty, as they allow for the computation of marginal distributions and conditional probabilities, making them ideal for tasks involving decision-making, diagnosis, and prediction.

When extended to dynamic systems, Dynamic Bayesian Networks (dBNs) model the evolution of variables over time. In a dBN, the network's structure and parameters are time-dependent, capturing how the relationships between variables change over successive time steps. dBNs are valuable for modeling sequential processes such as speech recognition, digital forensics, and bioinformatics, where time plays a critical role in understanding and predicting outcomes.

However, the current implementation of dBNs in the pyAgrum library has limitations. The library primarily supports one-step temporal transitions, which restricts its applicability to more complex dynamic systems where multiple time steps need to be modeled simultaneously. This limitation hinders its ability to capture more intricate temporal dependencies that are often present in real-world applications.

To address this challenge, our project aims to extend pyAgrum to support multi-step time windows (e.g., 2, 3, or more steps) for dBN modeling. This extension will enable more accurate representation and inference of dynamic systems, providing a more flexible and powerful tool for modeling and reasoning in temporal domains. Additionally, we will focus on enhancing the visualization and inference as well as structure learning capabilities to ensure that the extended functionality is both accessible and effective for practical use cases.

<div>

## **1. Bayesian Networks (BNs)**

### *Bayesian Networks*

A **Bayesian Network (BN)** is a probabilistic graphical model that represents a set of random variables and their conditional dependencies using a **directed acyclic graph (DAG)**. It provides a compact representation of the **joint probability distribution (JPD)** of a system, allowing for efficient reasoning under uncertainty.

### Factorization of the Joint Probability

Given a set of variables $(X = \{X_1, X_2, \dots, X_n\})$ with a DAG structure, the joint probability distribution can be factorized as:

$$
P(X_1, X_2, \dots, X_n) = \prod_{i=1}^{n} P(X_i | Pa(X_i))
$$

where $Pa(X_i)$ denotes the set of parent nodes of $X_i$ in the DAG. This factorization follows from the assumption that each variable is **conditionally independent** of its non-descendants, given its parents, making Bayesian networks highly efficient for probabilistic inference.

--- 
### **Local Markov Property**

A Bayesian network $X$ with respect to $G$ satisfies the **local Markov property**, meaning that each variable is conditionally independent of its **non-descendants** given its parent variables:

$$X_v \perp\!\!\!\perp X_{V \setminus de(v)} | X_{Pa(v)}, \quad \forall v \in V$$

where $de(v)$ is the set of **descendants** of $v$, and $V \setminus de(v)$ represents the set of **non-descendants** of $v$. This ensures that probabilistic dependencies are explicitly encoded in the graph structure.

Using the above definition, the conditional independence can be expressed as:

$$P(X_v = x_v | X_i = x_i \text{ for each } X_i \text{ that is not a descendant of } X_v) = P(X_v = x_v | X_j = x_j \text{ for each } X_j \text{ that is a parent of } X_v)$$

Since the graph is acyclic, the set of parent nodes is always a subset of the non-descendant nodes.

<br>

## 2. Dynamic Bayesian Networks (dBNs)

A **Dynamic Bayesian Network (DBN)** is a probabilistic graphical model that extends Bayesian Networks (BNs) to model time-evolving systems. Unlike a standard BN, which represents a static snapshot of variables and their dependencies, a DBN captures how these variables and their relationships evolve over time. Below is a detailed explanation of DBNs, considering **t-time-steps** and their general structure.

---

### **1. Core Components of a DBN**

A DBN is defined by the following components:

1. **Random Variables Across Time**:

   - At each time step $t$, there is a set of random variables $\mathbf{X}\_t = \{X_t^1, X_t^2, \dots, X_t^n\}$,  where $n$ is the number of variables in the system.
   - These variables represent the state of the system at time $t$.

2. **Intra-Slice Dependencies**:

   - Within a single time slice $t$, the dependencies between variables are represented by a **Directed Acyclic Graph (DAG)**. This is similar to a standard BN, where edges represent conditional dependencies.

3. **Inter-Slice Dependencies**:

   - Between time slices, dependencies are modeled to capture how the state at time $t$ depends on the state at previous time steps (e.g., $t-1$, $t-2$, etc.).
   - These dependencies are typically represented by edges connecting variables across time slices.

4. **Conditional Probability Distributions (CPDs)**:
   - Each variable $X_t^i$ is associated with a CPD that defines its probability distribution given its parents in the DAG (both within the same time slice and from previous time slices).

---

### **2. Temporal Structure**

A DBN is often represented as a **two-time-slice Bayesian Network (2-TBN)**, which captures the transition model between two consecutive time steps. The 2-TBN assumes the **Markov property**, meaning the state at time $t$ depends only on the state at $t-1$.

### **First-Order Markov Assumption**

The first-order Markov assumption simplifies the model by stating:

$$
P(\mathbf{X}_t \mid \mathbf{X}_{t-1}, \mathbf{X}_{t-2}, \dots, \mathbf{X}_0) = P(\mathbf{X}_t \mid \mathbf{X}_{t-1}).
$$

This means the future state depends only on the current state, not on the entire history.

### **Transition Model**

The transition model describes how the system evolves over time. It is defined as:

$$
P(\mathbf{X}_t \mid \mathbf{X}_{t-1}) = \prod_{i=1}^n P(X_t^i \mid \text{Pa}(X_t^i)),
$$

where $\text{Pa}(X_t^i)$ represents the parents of $X_t^i$ in the DBN (which may include variables from the same time slice $t$ or the previous time slice $t-1$).

In the case of k-Order dBN $Pa(X_t^i)$ represents the parents of node $X_t^i$ in the dBN.

---

### **3. Joint Probability Distribution**

The joint probability distribution (JPD) over all time steps $t = 0$ to $t = T$ is factorized as:

$$
P(\mathbf{X}_0, \mathbf{X}_1, \dots, \mathbf{X}_T) = P(\mathbf{X}_0) \prod_{t=1}^T P(\mathbf{X}_t \mid \mathbf{X}_{t-1}).
$$

Here:

- $P(\mathbf{X}\_0)$ is the **initial state distribution**, representing the probabilities at time $t = 0$.
- $P(\mathbf{X}_t \mid \mathbf{X}_{t-1})$ is the **transition model**, describing how the system evolves from $t-1$  to $t$.

For a first-order dBN, this simplifies to:

$$
P(\mathbf{X}_0, \mathbf{X}_1, \dots, \mathbf{X}_T) = P(\mathbf{X}_0) \prod_{t=1}^T \prod_{i=1}^n P(X_t^i \mid \text{Pa}(X_t^i)).
$$

---

### **4. Multi-Step Transitions**

While the first-order Markov assumption is common, dBNs can also model **higher-order dependencies** by extending the transition model to depend on multiple previous time steps. For example:

$$
P(\mathbf{X}_t \mid \mathbf{X}_{t-k}, \dots, \mathbf{X}_{t-1}),
$$

where $k$ is the temporal window length. This is useful for capturing long-term dependencies in systems where the state at time $t$ depends on states further back in time.

---

### **5. Applications of DBNs**

DBNs are widely used in domains where systems evolve over time, such as:

- **Speech Recognition**: Modeling temporal dependencies in audio signals.
- **Robotics**: Tracking and predicting robot states.
- **Bioinformatics**: Modeling gene regulatory networks over time.
- **Finance**: Predicting stock prices or market trends.

<br>
<br>

# Cahier des Charges: Extension of pyAgrum for Multi-Step Dynamic Bayesian Networks

# 1. Objectif

The goal of this project is to extend the pyAgrum library to support multi-step time windows (e.g., 2, 3, or 4 steps) for Dynamic Bayesian Networks (dBNs). This extension will involve:

- Implementing new functions for multi-temporal dependencies.
- Enhancing visualization and inference capabilities.
- Providing unit testing and documentation.

# 2. Key Deadlines

- **Submission Deadline:** May 9, 2025
- **Presentation Date:** May 15, 2025

# 3. Project Breakdown

## Part 1: Modelization (Multi-Temporal Dependencies)

### **Class Attributes**

The `dBN` class has the following attributes:

1. **`base_network`**:

   - The underlying Bayesian Network, created using `pyAgrum.BayesNet()`.
   - Represents the structure and dependencies of the dBN.

2. **`variables`**:

   - A list of variables in the dBN.
   - Each variable is replicated across `k` time slices.

3. **`arcs`**:

   - A list of directed edges (arcs) between variables across time slices.
   - Each arc is represented as a tuple of tuples, e.g., `((v1, 3), (v1, 4))`.

4. **`k`**:
   - The time horizon, representing the number of time slices in the DBN.
   - Defines the temporal dimension of the network.

---

### **Class Methods**

### **1. `createDBN(int k)`**

Creates the base Dynamic Bayesian Network and sets the time horizon `k`.

- **Parameters**:
  - `k` (int): The number of time slices in the DBN.
- **Returns**:
  - Initializes the `base_network` as a `pyAgrum.BayesNet()` and sets the `k` attribute.
- **Example**:

  ```python
  dbn = dBN()
  dbn.createDBN(5)  # Creates a DBN with 5 time slices
  ```

### **2. AddVar methods**
  ### **2.1 `addVar(Variable a)`**

  Adds a variable to the DBN across all time slices.

  - **Parameters**:
    - `a` (Variable): The variable to be added. This variable is created using one of `pyAgrum`'s variable creation methods.
  - **Behavior**:
    - Replicates the variable ``across all`k` time slices.
    - Names the variables as `{variable_name}#{time_slice}` (e.g., `A#2` for variable `A` at time slice 2).
    - Uses `pyAgrum`'s `add` method to add the variables to the `base_network`.
  - **Example**:

    ```python
    a = gum.LabelizedVariable("A", "Variable A", 2)  # Binary variable
    dbn.addVar(a)  # Adds A#0, A#1, ..., A#k to the DBN
    ```

    ### **2.2 `AddFastVar(str fast_description, int default_domain_size=2)`**  

    Adds a variable to the DBN efficiently using `pyAgrum.fastVariable()`, a method for rapid variable creation with minimal overhead.

    ### **Parameters**:
    - `fast_description` (str): A string following `pyAgrum`'s fast syntax for variable creation.
    - `default_domain_size` (int, optional): The number of modalities (default is `2`). If `fast_description` does not specify the number of modalities, this value is used.

    ### **Behavior**:
    - Uses `pyAgrum.fastVariable()` to create and add the variable efficiently. 
    - Replicates the variable across all `k` time slices.
    - Names the variables as `{variable_name}#{time_slice}` (e.g., `B#3` for variable `B` at time slice 3).
    - Ensures faster variable creation and addition compared to `addVar()`.

    ### **Example**:

    ```python
    dbn.AddFastVar("B[0,1]")  # Adds B#0, B#1, ..., B#k to the DBN efficiently
    ```

### **3. `addArc(Arc a)`**

  Adds a directed arc between variables across time slices.

- **Parameters**:
  - `a` (tuple): The arc to be added, represented as `((a1, t1), (a2, t2))`, where:
    - `a1` and `a2` are variables.
    - `t1` and `t2` are time slices.
- **Checks**:
  - Ensures the arc does not already exist.
  - Ensures `t1 <= t2` (no backward arcs).
  - Ensures `|t2 - t1| <= k` (arcs cannot span more than `k` time slices).
  - All other checks are handled by pyAgrum's own functions.
- **Behavior**:
  - Uses `pyAgrum`'s `addArc` method to add the arc to the `base_network` if all checks pass.
- **Example**:

  ```python
  dbn.addArc(((a, 3), (a, 4)))  # Adds an arc from A#3 to A#4
  ```

### **4. `deleteVar(Variable a)`**

  Deletes a variable and its associated arcs from all time slices.

- **Parameters**:
  - `a` (Variable): The variable to be deleted.
- **Behavior**:
  - Removes all instances of the variable across all time slices.
  - Uses `pyAgrum`'s `erase` method to remove the variable and its associated arcs.
- **Example**:
  ```python
  dbn.deleteVar(a)  # Deletes A#0, A#1, ..., A#k and associated arcs
  ```

### **5. `deleteArc(Arc a)`**

Deletes a specified arc from the DBN.

- **Parameters**:
  - `a` (tuple): The arc to be deleted, represented as `((a1, t1), (a2, t2))`.
- **Behavior**:
  - Uses `pyAgrum`'s `eraseArc` method to remove the arc from the `base_network`.
- **Example**:
  ```python
  dbn.deleteArc(((a, 3), (a, 4)))  # Deletes the arc from A#3 to A#4
  ```

### **Usage Example**

```python
import pyAgrum as gum

# Initialize the DBN
dbn = dBN()
dbn.createDBN(5)  # Create a DBN with 5 time slices

# Add variables
a = gum.LabelizedVariable("A", "Variable A", 2)  # Binary variable
b = gum.LabelizedVariable("B", "Variable B", 3)  # Ternary variable
dbn.addVar(v1)
dbn.addVar(v2)

# Add arcs
dbn.addArc(((a, 0), (a, 1)))  # A#0 → A#1
dbn.addArc(((a, 1), (b, 2)))  # A#1 → B#2

# Delete a variable
dbn.deleteVar(a)  # Removes A#0, A#1, ..., A#5 and associated arcs

# Delete an arc
dbn.deleteArc(((b, 2), (b, 3)))  # Removes B#2 → B#3
```

## Part 2: Inference (Non-Optimized)

**Goal:** Implement inference without optimization.

- Takes a dBN, unrolls it, and performs classical inference techniques.

## Part 3: Non-Stationary dBNs (If Time Permits)

- Extends dBNs to support time-varying structures by associating each dBN in a family with a specific usage period. This means that instead of having a fixed structure, the model can adapt its dependencies and parameters based on different phases of usage. As time progresses, different dBNs from the family are applied depending on the defined utilization period, allowing the network to better capture changing dynamics in the system.

## Part 4: Structure Learning (If Time Permits)

**Goal:** Implement structure learning techniques for dynamic Bayesian networks.

## Part 5: Unit Testing (Obligatory)

**Goal:** Ensure functionality correctness through testing and documentation.

- Develop unit tests for all implemented functions.

</br>
</br>

# 4. Project Timeline (12 Weeks)

| Phase              | Duration |
| ------------------ | -------- |
| Cahier des Charges | 2 weeks  |
| Modelization       | 4 weeks  |
| Inference          | 2 weeks  |
| Non stationnary    | 2 weeks  |
| Structure Learning | 2 weeks  |

Unit testing will be done in parallell.
