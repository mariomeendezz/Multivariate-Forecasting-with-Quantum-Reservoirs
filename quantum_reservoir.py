import numpy as np
from scipy.linalg import expm
from dataclasses import dataclass
from typing import Literal


# Single-qubit Pauli matrices.
_I2 = np.eye(2, dtype=complex)
_X = np.array(
    [[0, 1], 
     [1, 0]], dtype=complex)
_Y = np.array(
    [[0, -1j], 
     [1j, 0]], dtype=complex)
_Z = np.array(
    [[1, 0], 
     [0, -1]], dtype=complex)

@dataclass
class ReadoutConfig:
    """
    Configuration of the linear readout training method.

    Parameters
    ----------
    method:
        Method used to compute the readout weights.
        Options are "pinv", "svd", and "ridge".

    ridge:
        Ridge regularization coefficient used when method="ridge".
        It must be non-negative.

    rcond:
        Cutoff parameter used when method="svd".
        Singular values smaller than the specified threshold are discarded.
        It must be non-negative.
    """
    method: Literal["pinv", "ridge", "svd"] = "ridge"
    ridge: float | None = None
    rcond: float | None = None

    def __post_init__(self) -> None:
        if self.method == "ridge":
            if self.ridge is None:
                raise ValueError(
                    "ridge must be specified when method='ridge'"
                )

            if self.ridge < 0:
                raise ValueError("ridge must be non-negative")

        if self.method == "svd":
            if self.rcond is None:
                raise ValueError(
                    "rcond must be specified when method='svd'"
                )

            if self.rcond < 0:
                raise ValueError("rcond must be non-negative")
            
@dataclass
class HamiltonianConfig:
    """
    Configuration of the quantum reservoir Hamiltonian.

    Parameters
    ----------
    J_mode:
        Strategy used to generate the coupling coefficients.
        Options are "const", "uniform", and "normal".

    J_value:
        Constant coupling value used when J_mode="const".

    J_min, J_max:
        Lower and upper bounds used when J_mode="uniform".

    J_mean, J_var:
        Mean and variance used when J_mode="normal".

    h_mode:
        Strategy used to generate the local field coefficients.
        Options are "const", "uniform", and "normal".

    h_value:
        Constant field value used when h_mode="const".

    h_min, h_max:
        Lower and upper bounds used when h_mode="uniform".

    h_mean, h_var:
        Mean and variance used when h_mode="normal".
    """
    J_mode: Literal["const", "uniform", "normal"] = "const"
    J_value: float = 0.5
    J_min: float | None = None
    J_max: float | None = None
    J_mean: float | None = None
    J_var: float | None = None

    h_mode: Literal["const", "uniform", "normal"] = "uniform"
    h_value: float = 0
    h_min: float | None = 0.1
    h_max: float | None = 1
    h_mean: float | None = None
    h_var: float | None = None

    def __post_init__(self) -> None:
        if self.J_mode == "const":
            if self.J_value is None:
                raise ValueError(
                    "J_value must be specified when method='const'"
                )

            if self.J_value < 0:
                raise ValueError("J_value must be non-negative")

        elif self.J_mode == "uniform":
            if self.J_min is None or self.J_max is None:
                raise ValueError(
                    "J_min and J_max must be specified when method='uniform'"
                )

            if self.J_min > self.J_max:
                raise ValueError(
                    "J_min must be lower than or equal to J_max"
                )

        elif self.J_mode == "normal":
            if self.J_mean is None or self.J_var is None:
                raise ValueError(
                    "J_mean and J_var must be specified when method='normal'"
                )

            if self.J_var < 0:
                raise ValueError("J_var must be non-negative")

        if self.h_mode == "const":
            if self.h_value is None:
                raise ValueError(
                    "h_value must be specified when method='const'"
                )

            if self.h_value < 0:
                raise ValueError("h_value must be non-negative")

        elif self.h_mode == "uniform":
            if self.h_min is None or self.h_max is None:
                raise ValueError(
                    "h_min and h_max must be specified when method='uniform'"
                )

            if self.h_min > self.h_max:
                raise ValueError(
                    "h_min must be lower than or equal to h_max"
                )

        elif self.h_mode == "normal":
            if self.h_mean is None or self.h_var is None:
                raise ValueError(
                    "h_mean and h_var must be specified when method='normal'"
                )

            if self.h_var < 0:
                raise ValueError("h_var must be non-negative")

def _kron_op(
    op: np.ndarray,
    site: int,
    n: int
) -> np.ndarray:
    """
    Embed a single-qubit operator on `site` into the n-qubit Hilbert space.
    """
    mats = [op if k == site else _I2 for k in range(n)]
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out

class QuantumReservoir():
    def __init__(
        self,
        n_qubits: int = 6,
        t: float = 4.0,
        n_vnodes: int = 5,
        observables: str = "Z",
        hamiltonian: HamiltonianConfig | None = None,
        readout: ReadoutConfig | None = None,
        washout: int = 100,
        history_length: int = 120,
        seed: int = 42,
    ) -> None:
        """
        n_qubits   : number of qubits.
        t          : evolution time per input step under H.
        n_vnodes   : number of virtual nodes.
        observables: "Z" uses only Z measurements <Z>;
                     "full" uses the complete one- and two-body Pauli set
                     (<A_i B_j>, A,B in {X,Y,Z});
                     "diag" uses only same-axis correlators
                     (<X_iX_j>, <Y_iY_j>, <Z_iZ_j>).
        hamiltonian: Hamiltonian configuration, including the generation
                     method and parameters for the coupling matrix J and
                     local fields h.
        readout    : configuration of the linear readout training method.

        washout    : steps discarded at the start of training.
        history_length : history length used to re-seed rho at each forecast origin;
                     the reservoir has fading memory, so recent history suffices
                     and this keeps rolling-origin evaluation fast.
        seed       : RNG seed for the fixed random couplings.
        """
        if hamiltonian is None:
            hamiltonian = HamiltonianConfig()

        if readout is None:
            readout = ReadoutConfig()

        self.readout = readout
        self.n = n_qubits
        self.t = t
        self.n_vnodes = n_vnodes
        self.observables = observables
        self.washout = washout
        self.history_length = history_length
        self.seed = seed
        self.name = f"Quantum reservoir (n={n_qubits})"
        self._u_min = 0.0
        self._u_max = 1.0

        def fit(self):
            ...

        def forecast(self):
            ...