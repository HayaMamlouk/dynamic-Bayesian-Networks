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

To address this challenge, our project aims to extend pyAgrum to support multi-step time windows (e.g., 2, 3, or 4 steps) for dBN modeling. This extension will enable more accurate representation and inference of dynamic systems, providing a more flexible and powerful tool for modeling and reasoning in temporal domains. Additionally, we will focus on enhancing the visualization and inference as well as structure learning capabilities to ensure that the extended functionality is both accessible and effective for practical use cases.

<div>
## 1. Bayesian Networks (BNs)
A **Bayesian Network (BN)** is a probabilistic graphical model that represents a set of random variables and their conditional dependencies using a **directed acyclic graph (DAG)**. It provides a compact representation of the **joint probability distribution (JPD)** of a system.

### **Factorization of the Joint Probability**

Given a set of variables $X = \{X_1, X_2, \dots, X_n\}$ with a DAG structure, the joint probability distribution can be factorized as:
$$P(X_1, X_2, \dots, X_n) = \prod_{i=1}^{n} P(X_i \mid Pa(X_i))$$
where $Pa(X_i)$ denotes the parent nodes of $X_i$ in the DAG.

### **Inference in Bayesian Networks**

Inference in a BN involves computing probabilities of unknown variables given observed evidence. Common methods include:

- **Variable Elimination**
- **Belief Propagation (Message Passing Algorithm)**
- **Sampling Methods (Monte Carlo, Gibbs Sampling)**
</div>

<br>

## 2. Dynamic Bayesian Networks (dBNs)

A **Dynamic Bayesian Network (DBN)** is a probabilistic graphical model that extends Bayesian Networks (BNs) to model time-evolving systems. Unlike a standard BN, which represents a static snapshot of variables and their dependencies, a DBN captures how these variables and their relationships evolve over time. Below is a detailed explanation of DBNs, considering **t-time-steps** and their general structure.

---

## **1. Core Components of a DBN**

A DBN is defined by the following components:

1. **Random Variables Across Time**:

   - At each time step $t$, there is a set of random variables $$ \mathbf{X}\_t = \{X_t^1, X_t^2, \dots, X_t^n\}, $$ where $n$ is the number of variables in the system.
   - These variables represent the state of the system at time $t$.

2. **Intra-Slice Dependencies**:

   - Within a single time slice $t$, the dependencies between variables are represented by a **Directed Acyclic Graph (DAG)**. This is similar to a standard BN, where edges represent conditional dependencies.

3. **Inter-Slice Dependencies**:

   - Between time slices, dependencies are modeled to capture how the state at time $t$ depends on the state at previous time steps (e.g., $t-1$, $t-2$, etc.).
   - These dependencies are typically represented by edges connecting variables across time slices.

4. **Conditional Probability Distributions (CPDs)**:
   - Each variable $X_t^i$ is associated with a CPD that defines its probability distribution given its parents in the DAG (both within the same time slice and from previous time slices).

---

## **2. Temporal Structure**

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

where $ \text{Pa}(X_t^i) $ represents the parents of $X_t^i$ in the DBN (which may include variables from the same time slice $ t $ or the previous time slice $ t-1 $).

---

## **3. Joint Probability Distribution**

The joint probability distribution (JPD) over all time steps $ t = 0 $ to $ t = T $ is factorized as:

$$
P(\mathbf{X}_0, \mathbf{X}_1, \dots, \mathbf{X}_T) = P(\mathbf{X}_0) \prod_{t=1}^T P(\mathbf{X}_t \mid \mathbf{X}_{t-1}).
$$

Here:

- $ P(\mathbf{X}\_0) $ is the **initial state distribution**, representing the probabilities at time $ t = 0 $.
- $ P(\mathbf{X}_t \mid \mathbf{X}_{t-1}) $ is the **transition model**, describing how the system evolves from $ t-1 $ to $ t $.

For a first-order DBN, this simplifies to:

$$
P(\mathbf{X}_0, \mathbf{X}_1, \dots, \mathbf{X}_T) = P(\mathbf{X}_0) \prod_{t=1}^T \prod_{i=1}^n P(X_t^i \mid \text{Pa}(X_t^i)).
$$

---

## **4. Multi-Step Transitions**

While the first-order Markov assumption is common, DBNs can also model **higher-order dependencies** by extending the transition model to depend on multiple previous time steps. For example:

$$
P(\mathbf{X}_t \mid \mathbf{X}_{t-k}, \dots, \mathbf{X}_{t-1}),
$$

where $ k $ is the temporal window length. This is useful for capturing long-term dependencies in systems where the state at time $ t $ depends on states further back in time.

---

## **5. Inference in DBNs**

Inference in DBNs involves computing posterior distributions over hidden variables given observed evidence over time. Common inference tasks include:

- **Filtering**: Compute $ P(\mathbf{X}\_t \mid \mathbf{Y}\_1, \dots, \mathbf{Y}\_t) $, where $ \mathbf{Y}\_t $ are observations up to time $ t $.
- **Smoothing**: Compute $ P(\mathbf{X}\_k \mid \mathbf{Y}\_1, \dots, \mathbf{Y}\_t) $ for $ k < t $.
- **Prediction**: Compute $ P(\mathbf{X}\_{t+\Delta} \mid \mathbf{Y}\_1, \dots, \mathbf{Y}\_t) $ for future time steps $ \Delta $.

Inference algorithms for DBNs include:

- **Forward-Backward Algorithm**: For exact inference in linear chains.
- **Kalman Filters**: For continuous-state DBNs (e.g., in linear dynamical systems).
- **Particle Filters**: For approximate inference in non-linear or non-Gaussian systems.

---

## **6. Learning in DBNs**

Learning a DBN involves two main tasks:

1. **Parameter Learning**:

   - Estimate the CPDs from data using methods like:
     - **Maximum Likelihood Estimation (MLE)**: Directly estimate parameters from observed data.
     - **Expectation-Maximization (EM)**: For cases with missing or hidden data.

2. **Structure Learning**:
   - Discover the underlying graph structure (intra-slice and inter-slice dependencies) using:
     - **Score-Based Methods**: Optimize a scoring function (e.g., BIC, AIC) over possible structures.
     - **Constraint-Based Methods**: Use conditional independence tests to infer dependencies.

---

## **7. Applications of DBNs**

DBNs are widely used in domains where systems evolve over time, such as:

- **Speech Recognition**: Modeling temporal dependencies in audio signals.
- **Robotics**: Tracking and predicting robot states.
- **Bioinformatics**: Modeling gene regulatory networks over time.
- **Finance**: Predicting stock prices or market trends.

<br>
<br>

# Cahier des Charges: Extension of pyAgrum for Multi-Step Dynamic Bayesian Networks

## 1. Objectif

The goal of this project is to extend the pyAgrum library to support multi-step time windows (e.g., 2, 3, or 4 steps) for Dynamic Bayesian Networks (dBNs). This extension will involve:

- Implementing new functions for multi-temporal dependencies.
- Enhancing visualization and inference capabilities.
- Providing unit testing and documentation.

## 2. Key Deadlines

- **Submission Deadline:** May 9, 2025
- **Presentation Date:** May 15, 2025

## 3. Project Breakdown

### Part 1: Modelization (Multi-Temporal Dependencies)

**Goal:** Implement support for multi-step time windows in dBNs.

#### 1 Initialization

- The user initializes a dBN using:
  ```python
  init_dBN(variables, arcs, p)
  ```
- **Variables:** A list of variables, where each element is either:

  - A tuple `(name, domain_size)`, e.g., `("a", 2)` for a binary variable `a`.
  - A variable `a` (defaulting to binary domain `2`)
  - Syntax options :
    - A single character. **(Decision needed)** Should uppercase and lowercase letters represent different variables?
    - Variable names must be strings without numbers to avoid confusion with time slices, e.g., `"Flu", "Cough", "Fever"`

- **Arcs:** A list of directed edges in the form:

  ```python
  ((variable_name, time_slice), (variable_name, time_slice))
  ```

  Example: `(("a", 0), ("a", 1))` represents `a_0 → a_1`.

- **Time Dimension (p):** Defines the number of time slices to generate.
- The initialization automatically expands variables across time slices (e.g., `init_dBN([a, b], [((a,0), (a,1)), ((a,0), (b,0))], 2)` creates `a_0, a_1, b_0, b_1`).

#### 2 Verification

- **Variable Checking (`check_variable`)**
  - Ensures variable naming rules are followed.
- **Arc Checking (`check_arc`)**
  - ((variable_name, i), (variable_name, j)
  - Prevents **backward arcs** (`i > j` error) and **reflexive arcs** (`i = j` error for the same variable).
- **Time-Slice Validation (`is_kTBN(dBN, k)`)**
  - Ensures that for each variable, there exist `a_0, a_1, ..., a_k`.
  - Calls `check_variable`
  - Calls `check_arc`

#### 3 Construction

- **Building the Network:**
  - Create a Bayesian Network (`gum.BayesNet()`).
  - Iterate over variables and time slices to create nodes. Format accepted by gum.LabelizedVariable().
  - Iterate over arcs and call `addArc()` ensuring arcs do not exceed `p`. We have to translate into a format accepted by addArc().

#### 4 Functionalities

- **Unrolling (`unroll_dBN`)**

  - When unrolling a DBN, the structure of incoming arcs from the last existing time slices should be extended to the newly generated time slices. Specifically, if the original DBN has $ p $ time slices and there are dependencies at time slices $ t - 3 $ and $ t - 1 $, these same dependencies should be replicated when new time slices are added. For example, if a DBN initially has three time slices (0, 1, 2), and there are incoming arcs at time slice $ 3 - 3 $ (i.e., 0) and $ 3 - 1 $ (i.e., 2), then when extending the DBN beyond time slice 3, the same pattern should apply. That means at time slice 4, the arcs should follow the same structure as they did in the original DBN, ensuring a consistent extension of dependencies over time.

- **Adding Variables (`addVar_dBN`)**
  - `addVar_dBN("c")`: Adds a new variable.
  - `addVar_dBN(("a", i))`: Adds an existing variable to a new time slice i.
- **Adding Arcs (`addArc_dBN`)**
  - Example: `addArc_dBN(("x", i), ("y", j))` while ensuring `i, j` are within bounds and `x, y` are existing variables.
- **Removing Variables (`removeVar_dBN`)**
  - `removeVar_dBN("a")`: Removes a variable entirely.
  - `addVar_dBN(("a", i))`: Removes a variable from a specific time slice i.
  - This function ensures that the depending arcs are removed as well.
- **Removing Arcs (`removeArc_dBN`)**
  - Removes a specific arc.
- **CPT Modification (`changeCPT_dBN`)**
  - **(To be defined)** How to modify conditional probability tables.

### Part 2: Inference (Non-Optimized)

**Goal:** Implement inference without optimization.

- Takes a dBN, unrolls it, and performs classical inference techniques.

### Part 3: Non-Stationary dBNs (If Time Permits)

- Extends dBNs to support time-varying structures by associating each dBN in a family with a specific usage period. This means that instead of having a fixed structure, the model can adapt its dependencies and parameters based on different phases of usage. As time progresses, different dBNs from the family are applied depending on the defined utilization period, allowing the network to better capture changing dynamics in the system.

### Part 4: Structure Learning (If Time Permits)

**Goal:** Implement structure learning techniques for dynamic Bayesian networks.

### Part 5: Unit Testing (Obligatory)

**Goal:** Ensure functionality correctness through testing and documentation.

- Develop unit tests for all implemented functions.

## 4. Project Timeline (12 Weeks)

| Phase              | Duration |
| ------------------ | -------- |
| Cahier des Charges | 2 weeks  |
| Modelization       | 4 weeks  |
| Inference          | 2 weeks  |
| Non stationnary    | 2 weeks  |
| Structure Learning | 2 weeks  |

Unit testing will be done in parallell.
