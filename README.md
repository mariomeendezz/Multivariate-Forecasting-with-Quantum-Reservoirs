# About the repository

This repository contains an experimental reproduction of selected results from the paper:

**Novel Approach to Multivariate Forecasting with Quantum Reservoirs - D. A. Aranda, J. Ballesteros, J. Bonilla, Elías F. Combarro, N. Monrio, S. Ranilla-Cortina & J. Ranilla**

The goal of this project is to reproduce and analyze the main experimental ideas presented in the paper using Python-based quantum computing and machine learning tools.

## Repository Structure

The repository is organized as follows:

- `data.xlsx`: csv file with SP500 data.
- `main.ipynb`: notebook with the Quantum Reservoir implementation.

## Data

The data is obtained from: https://www.kaggle.com/datasets/camnugent/sandp500?resource=download

## Project Requirements

First, create a virtual environment:

```bash
python -m venv .venv
```

Then activate it:

```bash
# Windows
.venv\Scripts\activate
```

```bash
# macOS/Linux
source .venv/bin/activate
```

Finally, install all required dependencies:

```bash
pip install -r requirements.txt
```


## Usage

The recommended workflow is:

1. Create and activate a virtual environment.
2. Install the dependencies from `requirements.txt`.
3. Open the notebook `main` with Jupyter Notebook, JupyterLab, VS Code, or another compatible environment.
4. Run each notebook cell by cell in order.


## Sources

- Novel Approach to Multivariate Forecasting with Quantum Reservoirs - D. A. Aranda, J. Ballesteros, J. Bonilla, Elías F. Combarro, N. Monrio, S. Ranilla-Cortina & J. Ranilla: https://ceur-ws.org/Vol-4153/abstract4.pdf
- QuTiP documentation: https://qutip.readthedocs.io/en/qutip-5.3.x/ 

## License

This project is licensed under the MIT License.