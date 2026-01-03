# 🧠 Chip-Clearing Games – Probabilities

An interactive project for analyzing chip allocation strategies with exact and numerical computations of winning probabilities.  
All computations are carried out in Python.

## 🚀 Start directly with Binder (may take some time)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/RVeh/Setzstrategien/HEAD)

---

## 📂 Notebook Structure

| Notebook | Content |
|----------|--------|
| `01a-one-die-winning-probabilities.ipynb` | Winning probabilities in the one-die game with input and output |
| `01b-one-die-winning-probabilities.ipynb` | Winning probabilities in the one-die game |
| `02-one-strategy-vs-all.ipynb` | All strategies $T$ with total chip count $n$ versus $S$ |
| `03a-search-better-strategy-one-die.ipynb` | One-die case: fixed strategy $V$ versus all possible strategies $W$, with optional restrictions on the number of chips per slot |
| `03b-search-better-strategy-two-dice.ipynb` | Two-dice case: fixed strategy $S$ versus all possible strategies $T$, with optional restrictions on the number of chips per slot |
| `03c-dominance-and-cycles.ipynb` | Exact pairwise comparison of all strategies; dominance relations and non-transitive 3-cycles |
| `04a-symbolic-winning-probabilities-one-die.ipynb` | Symbolic computations for two betting strategies |
| `04b-symbolic-exact-numerical-one-die.ipynb` | Symbolic, exact, and numerical computations |
| `05-exact-one-and-two-dice.ipynb` | Exact computations for one- and two-dice cases |
| `06-simulation-one-die.ipynb` | Simulations (one die) with 95% Wald confidence intervals |
| `07-simulation-one-and-two-dice.ipynb` | Simulations (one/two dice) with confidence intervals and exact results |
| `08-simulation-vs-exact-compact.ipynb` | Simulations with Wilson confidence intervals and exact probabilities |

---

## 📦 Additional Notes

All core functions are contained in:

```python
chip_strategies.py
```

---

## Usage Notes

- Notebooks can be opened locally in Jupyter or executed via Binder.
- In Binder or JupyterLab, use **Run All Cells** to reproduce all results.
- The project uses the Python standard library and `matplotlib`.
- When using JupyterLab, *chip_strategies.py* must be located in the same directory as the notebook.

---

## 📦 Additional File

| File | Content |
|----------|--------|
| `appendix-explicit-computation-P200001` | Manual computations  |
---

## 🧮 Requirements

- Python ≥ 3.7  
- matplotlib  
- pandas  

---

## ✍️ Author

Reimund Vehling  
(with AI-assisted development support)
