# 📊 Multi-Temporal Dynamic Bayesian Networks (kTBN)

This project provides a flexible and modular Python framework for modeling, manipulating, and inferring **Dynamic Bayesian Networks (DBNs)** with a fixed time horizon (`kTBN`). Built on top of [pyAgrum](https://agrum.gitlab.io/), it extends the classic Bayesian Network interface to support temporal reasoning, with a clean and user-friendly API.

---

## 📦 Project Structure

```
.
├── src/
│   ├── DynamicBayesNet.py    # Main logic for building kTBNs
│   ├── notebook.py           # Display utilities and inference helpers
    ├── tutorial.ipynb           # Guided notebook to explore features
│
├── tests/
│   ├── TestDynamicBayesNet.py  # Unit tests for core functionality
    ├── experiments.ipynb        # Real-world use case (weather scenario)
│
└── README.md
```

---

## ✅ Features

* ✅ Simple API to define and link variables across multiple time slices.
* ✅ Flexible separator system (`#`, `|`, etc.) for variable naming.
* ✅ Easy unrolling of kTBNs into standard Bayesian Networks for inference.
* ✅ Custom display modes (`compact`, `reverse`, `classic`) for readable CPTs.
* ✅ Built-in plotting and posterior visualization (`plotFollow`, `getPosterior`).
* ✅ Full test suite with edge cases and structure preservation validation.

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install pyAgrum pydot matplotlib
```

> ⚠️ You must also ensure Graphviz is installed and available in your system path (for graph rendering).

### 2. Clone the repository

```bash
git clone https://github.com/HayaMamlouk/dynamic-bayesnet.git
cd dynamic-bayesnet
```

### 3. Try the tutorial

Open `tutorial.ipynb` to explore all core functionalities in a self-contained and beginner-friendly notebook.

---

## 🧪 Testing

All core components are rigorously tested using `unittest`.

To run the test suite:

```bash
python -m unittest tests/TestDynamicBayesNet.py
```

Tests cover:

* Variable creation and deletion
* Arc management
* CPT assignment and value retrieval
* Unrolling 
* Custom separators
* Structure preservation through unrolling

---

## 🌦 Real-World Example: Weather Forecasting

In `experiments.ipynb`, we designed a realistic weather model with variables like:

* `SunnyWithClouds`
* `RainingCatsAndDogs`
* `StormOfTheCentury`
* `SnowingSoMuchThereMayBeAnAvalanche`

This use case allowed us to:

* Model a complex temporal structure
* Generate and inspect probabilistic transitions
* Plot variable evolution over time
* Confirm display and CPT formatting in dynamic conditions

---

## 🪠 Main Modules

### `DynamicBayesNet.py`

* `DynamicBayesNet`: Main kTBN builder (add vars, arcs, CPTs, erase, etc.)
* `dTensor`: Wrapper for CPTs, enabling intuitive indexed access like `cpt[{("A", 1): 0}]`

### `notebook.py`

* `showKTBN`, `showUnrolled`: Graphical display of temporal networks
* `unrollKTBN`: Transforms kTBN into a full BN over any time window
* `getPosterior`, `plotFollow`: Inference tools to visualize how probabilities evolve

---

## 📌 Design Philosophy

* **Modular**: Separated model logic, display tools, and tests.
* **Intuitive**: Follows pyAgrum’s API where possible, adds temporal extensions naturally.
* **User-friendly**: Suitable for both experimentation and educational use.
* **Fully documented**: Function-level docstrings provided throughout.

---

## 📄 License

MIT License 

---

## ✍️ Authors

* Doruk OZGENC & Haya MAMLOUK — Sorbonne University — M1 AI2D
* Special thanks to Pierre-Henri WUILLEMIN

---

## 📚 Future Additions

* [ ] Jupyter-based documentation
* [ ] Annexes for full technical specification & user manual
* [ ] Export to standard Bayesian Network formats (e.g. `.xdsl`, `.bif`)
