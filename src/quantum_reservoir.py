import numpy as np
from scipy.linalg import expm
from dataclasses import dataclass
from typing import Literal


# Single-qubit Pauli matrices.
_I2 = np.eye(2, dtype=complex)
_X = np.array(
    [[0, 1], 
     [1, 0]], dtype=complex)
_Z = np.array(
    [[1, 0], 
     [0, -1]], dtype=complex)

# Quantum states for one qubit
_ket0 = np.array([1.0, 0.0], dtype=complex)
_ket1 = np.array([0.0, 1.0], dtype=complex)

# |+> = (|0> + |1>) / sqrt(2)
_ket_plus = (_ket0 + _ket1) / np.sqrt(2)

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
    method: Literal["pinv", "ridge", "svd"] = "pinv"
    ridge: float | None = None
    rcond: float | None = None

    def __post_init__(self) -> None:
        # Readout parameters
        if self.method == "ridge":
            if self.ridge is None:
                raise ValueError(
                    "ridge must be specified when method='ridge'"
                )

            if self.ridge < 0:
                raise ValueError("ridge must be non-negative")

        elif self.method == "svd":
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
        Options are "const", "uniform", "normal" and "coupling".

    J_value:
        Constant coupling value used when J_mode="const".

    J_min, J_max:
        Lower and upper bounds used when J_mode="uniform".

    J_mean, J_var:
        Mean and variance used when J_mode="normal".

    J_k:
        Scaling exponent used when J_mode="coupling".

    h_mode:
        Strategy used to generate the local field coefficients.
        Options are "const", "uniform", and "normal".

    h_value:
        Constant field value used when h_mode="const".

    h_min, h_max:
        Lower and upper bounds used when h_mode="uniform".

    h_mean, h_var:
        Mean and variance used when h_mode="normal".

    initial_state:
        Initial reservoir state. "plus" uses the pure product state
        |+>^N and "mixed" uses the maximally mixed state I / 2^N.
    """
    J_mode: Literal["const", "uniform", "normal", "coupling"] = "const"
    J_value: float = 0.5
    J_min: float | None = None
    J_max: float | None = None
    J_mean: float | None = None
    J_var: float | None = None
    J_k: float | None = None

    h_mode: Literal["const", "uniform", "normal"] = "uniform"
    h_value: float | None = None
    h_min: float | None = 0.1
    h_max: float | None = 1
    h_mean: float | None = None
    h_var: float | None = None
    initial_state: Literal["plus", "mixed"] = "plus"

    def __post_init__(self) -> None:
        # Coupling configuration
        if self.J_mode == "const":
            if self.J_value is None:
                raise ValueError(
                    "J_value must be specified when method='const'"
                )

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

        elif self.J_mode == "coupling":
            if self.J_k is None:
                raise ValueError(
                    "J_k must be specified when method='coupling'"
                )

            if self.J_k < 0:
                raise ValueError("J_k must be non-negative")

        # Local-field configuration
        if self.h_mode == "const":
            if self.h_value is None:
                raise ValueError(
                    "h_value must be specified when method='const'"
                )

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

def _local_operator(
    operator: np.ndarray,
    site: int,
    n_qubits: int
) -> np.ndarray:
    """
    Embed a single-qubit operator on `site` into the n-qubit Hilbert space.
    """
    if not 0 <= site < n_qubits:
        raise ValueError("site must satisfy 0 <= site < n_qubits")

    # Tensor-product embedding
    mats = [operator if k == site else _I2 for k in range(n_qubits)]
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out

def _partial_trace(
    rho: np.ndarray,
    n_qubits: int,
    n_trace: int,
) -> np.ndarray:
    """
    Compute the partial trace over the first n_trace qubits.
    """
    if not 0 <= n_trace <= n_qubits:
        raise ValueError("n_trace must satisfy 0 <= n_trace <= n_qubits")

    # Subsystem dimensions
    dim_trace = 2**n_trace
    dim_keep = 2**(n_qubits - n_trace)

    rho = rho.reshape(
        dim_trace,
        dim_keep,
        dim_trace,
        dim_keep,
    )

    # Trace over the selected subsystem
    rho_reduced = np.trace(
        rho,
        axis1=0,
        axis2=2,
    )

    return rho_reduced

def _ridge_solve(
    X: np.ndarray,
    y: np.ndarray,
    ridge: float
)-> np.ndarray:
    """
    Closed-form ridge regression W = (X^T X + ridge I)^-1 X^T y.
    """
    # Regularized normal equations
    penalty = np.eye(X.shape[1], dtype=X.dtype)
    G = X.T @ X + ridge * penalty
    return np.linalg.solve(G, X.T @ y)

def _check_data(data: np.ndarray) -> None:
    """
    Validates the complete data series for the quantum reservoir.
    """
    # Shape and content checks
    if data.ndim != 1:
        raise ValueError("data must be one-dimensional")

    if len(data) == 0:
        raise ValueError("data cannot be empty")

    if not np.all(np.isfinite(data)):
        raise ValueError("data must contain only finite values")

    if data.max() == data.min():
        raise ValueError("data cannot be constant")

class QuantumReservoir():
    """
    Quantum reservoir computer for multi-step time-series forecasting.

    The model first normalizes the training prefix to `[0, 1]`. Each value
    is amplitude-encoded into the first qubit, after which the full system
    evolves under the configured Ising Hamiltonian. Measurements taken at the
    virtual nodes form the reservoir features.

    During method `fit`, consecutive feature vectors are grouped into windows
    and a linear readout is trained to predict the next `horizon` values.
    Method `forecast` applies that readout to the last available feature window
    and converts the result back to the original data scale.

    Typical use
    -----------
    `model.fit(data, train_end=1000)`
    `prediction = model.forecast()`
    """

    def __init__(
        self,
        n_qubits: int = 6,
        t: float = 0.3,
        n_vnodes: int = 2,
        window_size: int = 4,
        horizon: int = 25,
        observables: str = "Z",
        hamiltonian: HamiltonianConfig | None = None,
        readout: ReadoutConfig | None = None,
        washout_mode: Literal["discard", "train_pass"] = "discard",
        washout_steps: int = 100,
        seed: int = 42,
    ) -> None:
        """
        n_qubits     : number of qubits.
        t            : evolution time per input step under H.
        n_vnodes     : number of virtual nodes.
        window_size  : number of input steps used to compute the reservoir features.
        horizon      : number of output steps to predict.
        observables  : "Z" uses only Z measurements <Z>;
                       "full" uses the complete one- and two-body Pauli set
                       (<A_i B_j>, A,B in {X,Y,Z});
                       "diag" uses only same-axis correlators
                       (<X_iX_j>, <Y_iY_j>, <Z_iZ_j>).
        hamiltonian  : Hamiltonian configuration, including the generation
                       method and parameters for the coupling matrix J and
                       local fields h.
        readout      : configuration of the linear readout training method.
        washout_mode : "discard" discards the first washout_steps;
                       "train_pass" evolves through the complete training
                       series once and then replays it to collect features.
        washout_steps: number of initial steps discarded when
                       washout_mode="discard". It is ignored when
                       washout_mode="train_pass".
        seed         : RNG seed for the fixed random couplings.
        """
        # Parameter validation
        if n_qubits < 1:
            raise ValueError("n_qubits must be at least 1")

        if n_vnodes < 1:
            raise ValueError("n_vnodes must be at least 1")

        if window_size < 1 or horizon < 1:
            raise ValueError("window_size and horizon must be at least 1")

        if washout_steps is not None and washout_steps < 0:
            raise ValueError("washout_steps must be non-negative")
        
        if washout_mode not in ("discard", "train_pass"):
            raise ValueError("washout_mode must be either 'discard' or 'train_pass'")

        # Default configurations
        if hamiltonian is None:
            hamiltonian = HamiltonianConfig()

        if readout is None:
            readout = ReadoutConfig()

        # Public configuration
        self.n_qubits = n_qubits
        self.t = t
        self.n_vnodes = n_vnodes
        self.observables = observables
        self.window_size = window_size
        self.horizon = horizon
        self.hamiltonian = hamiltonian
        self.readout = readout
        self.washout_mode = washout_mode
        self.washout_steps = washout_steps
        self.seed = seed
        self.name = f"Quantum reservoir (n={n_qubits})"

        # Cached model state
        self._built_U = False
        self._built_features = False
        self._feature_input: np.ndarray | None = None
        self._W: np.ndarray | None = None
        self._train_end: int | None = None


    @property
    def _feature_offset(self) -> int:
        """
        Original data index represented by reservoir_features[0].
        """
        # Discarded prefix
        if self.washout_mode == "discard":
            return self.washout_steps

        return 0


    def _build(self) -> None:
        """
        Builds the Hamiltonian using the HamiltonianConfig data
        class and then builds the evolution operator.
        """
        rng = np.random.default_rng(self.seed)

        n = self.n_qubits
        dim = 2**n
        cfg = self.hamiltonian

        self._dim = dim

        # Local operators
        self._X_ops = np.stack([
            _local_operator(_X, i, n)
            for i in range(n)
        ])

        self._Z_ops = np.stack([
            _local_operator(_Z, i, n)
            for i in range(n)
        ])

        # Qubit pairs
        pairs = [
            (i, j)
            for i in range(n)
            for j in range(i + 1, n)
        ]

        # Coupling coefficients
        n_pairs = len(pairs)

        if cfg.J_mode == "const":
            couplings = np.full(
                n_pairs,
                float(cfg.J_value),
            )

        elif cfg.J_mode == "uniform":
            couplings = rng.uniform(
                float(cfg.J_min),
                float(cfg.J_max),
                size=n_pairs,
            )

        elif cfg.J_mode == "normal":
            couplings = rng.normal(
                float(cfg.J_mean),
                np.sqrt(float(cfg.J_var)),
                size=n_pairs,
            )
            
        elif cfg.J_mode == "coupling":
            k = float(cfg.J_k)

            total = 0.0

            for i in range(1, n + 1):
                for j in range(1, n + 1):
                    total += (i + j) ** k

            c_k = (2.0 / n**2) * total

            couplings = np.asarray([
                ((i + 1) + (j + 1)) ** k / c_k
                for i, j in pairs
            ])

        else:
            raise ValueError(
                f"Unknown J_mode: {cfg.J_mode}"
            )

        self._J = np.zeros((n, n), dtype=float)

        for (i, j), J_ij in zip(pairs, couplings):
            self._J[i, j] = J_ij
            self._J[j, i] = J_ij

        # Local fields
        if cfg.h_mode == "const":
            local_fields = np.full(
                n,
                float(cfg.h_value),
            )

        elif cfg.h_mode == "uniform":
            local_fields = rng.uniform(
                float(cfg.h_min),
                float(cfg.h_max),
                size=n,
            )

        elif cfg.h_mode == "normal":
            local_fields = rng.normal(
                float(cfg.h_mean),
                np.sqrt(float(cfg.h_var)),
                size=n,
            )

        else:
            raise ValueError(
                f"Unknown h_mode: {cfg.h_mode}"
            )

        self._local_fields = local_fields

        # Hamiltonian
        H = np.zeros((dim, dim), dtype=complex)

        for i, j in pairs:
            H += self._J[i, j] * (
                self._X_ops[i] @ self._X_ops[j]
            )

        for i in range(n):
            H += (
                self._local_fields[i]
                * self._Z_ops[i]
            )

        self._H = H

        # Evolution operator
        virtual_dt = self.t / self.n_vnodes
        self._U = expm(-1j * H * virtual_dt)
        self._U_dag = self._U.conj().T

        # Observables
        if self.observables == "Z":
            self._observables = self._Z_ops
        else:
            raise NotImplementedError(
                f"Observables {self.observables} not implemented"
            )

        self._built_U = True

    def _ensure_U_built(self) -> None:
        """
        Ensures that the evolution operator U is built. If not,
        it builds the Hamiltonian and the evolution operator.
        """
        # Lazy construction
        if not self._built_U:
            self._build()

    def _inject(
        self,
        rho: np.ndarray,
        xl: float,
    ) -> np.ndarray:
        """
        Injects the input value xl into the reservoir
        state rho using amplitude encoding.
        """
        # Input validation
        xl = float(xl)

        if not np.isfinite(xl):
            raise ValueError("Input must be finite")

        if not 0.0 <= xl <= 1.0:
            raise ValueError(
                "Encoded input must be in [0, 1]"
            )

        # Reduced reservoir state
        rho_rest = _partial_trace(
            rho,
            n_qubits=self.n_qubits,
            n_trace=1,
        )

        # Amplitude-encoded input state
        psi = (
            np.sqrt(1.0 - xl) * _ket0
            + np.sqrt(xl) * _ket1
        )

        rho_input = np.outer(
            psi,
            psi.conj(),
        )

        # Updated joint state
        rho_updated = np.kron(rho_input, rho_rest)

        return rho_updated

    def _expect(
        self,
        rho: np.ndarray,
    ) -> np.ndarray:
        """
        Computes the expectation values of the observables
        for the given density matrix rho.
        """
        # Batched observable expectations
        return np.einsum(
            "kij,ji->k",
            self._observables,
            rho,
            optimize=True,
        ).real

    def _step(
        self,
        rho: np.ndarray,
        u: float,
        collect: bool = True
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Performs a single time step of the reservoir dynamics,
        including input injection and evolution under the Hamiltonian.
        Returns the updated density matrix and the extracted features.
        """
        # Input injection
        rho = self._inject(rho, u)

        # Virtual-node evolution
        snapshots = []

        for _ in range(self.n_vnodes):
            rho = self._U @ rho @ self._U_dag
            if collect:
                snapshots.append(self._expect(rho))

        # Feature vector
        features = (
            np.concatenate(snapshots)
            if collect
            else None
        )

        return rho, features

    def _initial_rho(self) -> np.ndarray:
        # Pure product state
        if self.hamiltonian.initial_state == "plus":
            initial_ket = _ket_plus.copy()

            for _ in range(self.n_qubits - 1):
                initial_ket = np.kron(
                    initial_ket,
                    _ket_plus,
                )

            return np.outer(
                initial_ket,
                initial_ket.conj(),
            )

        # Maximally mixed state
        if self.hamiltonian.initial_state == "mixed":
            return np.eye(
                self._dim,
                dtype=complex,
            ) / self._dim

        raise ValueError(
            "initial_state must be either 'plus' or 'mixed'"
        )

    def _get_features(
        self,
        u: list | np.ndarray,
    ) -> None:
        """
        Gets the features of the reservoir for a given dataset
        """
        # Washout offset
        offset = self._feature_offset

        if len(u) <= offset:
            raise ValueError(
                f"data must contain more than {offset} samples"
            )

        # Reservoir initialization
        self._ensure_U_built()

        rho = self._initial_rho()

        if self.washout_mode == "train_pass":
            # Initial pass without measurements
            for u_t in u:
                rho, _ = self._step(
                    rho,
                    u_t,
                    collect=False,
                )

            collect_from = 0

        else:
            # Discard-mode collection start
            collect_from = offset


        reservoir_features = []

        # Feature collection
        for t, u_t in enumerate(u):
            collect = t >= collect_from

            rho, features = self._step(
                rho,
                u_t,
                collect=collect,
            )

            if collect:
                reservoir_features.append(features)

        # Feature cache
        self.reservoir_features = np.stack(
            reservoir_features
        )
        self._feature_input = np.asarray(u, float).copy()
        self._built_features = True

    def _ensure_features_built(self, u: np.ndarray) -> np.ndarray:
        """
        Ensures that the reservoir features match the provided data.
        """
        # Rebuild the cache when the input changes
        if (
            not self._built_features
            or self._feature_input is None
            or not np.array_equal(u, self._feature_input)
        ):
            self._get_features(u)

        return self.reservoir_features

    def _create_sequences(
            self,
            window_data: np.ndarray,
            horizon_data: np.ndarray,
            train_end: int | None = None
        ) -> tuple[np.ndarray, np.ndarray]:
        """
        Creates the sequences for the linear readout.
        """
        # Training range
        if train_end is None:
            train_end = len(horizon_data) - 1

        if not 0 <= train_end < len(horizon_data):
            raise ValueError("train_end is outside the data")

        # Sliding feature and target windows
        X_train = []
        y_train = []

        offset = self._feature_offset
        first_window_end = offset + self.window_size - 1

        last_window_end = train_end - self.horizon

        for data_end in range(
            first_window_end,
            last_window_end + 1
        ):
            feature_end = data_end - offset

            X_window = window_data[
                feature_end - self.window_size + 1:
                feature_end + 1
            ]

            y_window = horizon_data[
                data_end + 1:
                data_end + self.horizon + 1
            ]

            X_train.append(X_window.reshape(-1))
            y_train.append(y_window)

        return np.stack(X_train), np.stack(y_train)

    def _to_unit(self, x: np.ndarray) -> np.ndarray:
        """
        Min-max scaling.
        """
        # Forward scaling
        return (np.asarray(x, float) - self._u_min) / (self._u_max - self._u_min)

    def _from_unit(self, u: np.ndarray) -> np.ndarray:
        """
        Inverse min-max scaling.
        """
        # Inverse scaling
        return np.asarray(u, float) * (self._u_max - self._u_min) + self._u_min
    
    def fit(
        self,
        data: np.ndarray,
        train_end: int | None = None,
    ) -> None:
        """
        Build reservoir features and fit the linear readout.

        Only the prefix ending at train_end is used for normalization and
        reservoir feature generation. If train_end is not specified, it is
        set so that exactly self.horizon observations remain available for
        evaluation. train_end is an inclusive index. The fitted weights and
        generated features are stored in the instance for forecasting.
        """
        # Input validation
        data = np.asarray(data, float)
        _check_data(data)

        # Training limits
        max_train_end = len(data) - self.horizon - 1

        if train_end is None:
            train_end = max_train_end

        if train_end > max_train_end:
            raise ValueError(
                "train_end must leave at least horizon observations "
                "after it for forecasting"
            )

        min_train_end = self._feature_offset + self.window_size + self.horizon - 1
        
        if train_end < min_train_end:
            raise ValueError(
                f"train_end must be at least {min_train_end} "
                "to create one training sequence"
            )

        # Training prefix
        train_data = data[:train_end + 1]

        # Training-only normalization
        min_train = train_data.min()
        max_train = train_data.max()

        # Full-series scaling (introduces data leakage)
        # min_train = data.min()
        # max_train = data.max()

        self._u_min = float(min_train)
        self._u_max = float(max_train)

        # Reservoir features
        u_train = self._to_unit(train_data)

        features = self._ensure_features_built(u_train)

        # Supervised sequences
        X_train, y_train = self._create_sequences(
            window_data=features,
            horizon_data=u_train,
            train_end=train_end,
        )

        # Intercept term
        X_train = np.column_stack([
            np.ones(len(X_train)),
            X_train,
        ])

        self._X_train = X_train
        self._y_train = y_train

        # Linear readout
        if self.readout.method == "ridge":
            self._W = _ridge_solve(
                X_train,
                y_train,
                ridge=self.readout.ridge,
            )

        elif self.readout.method == "pinv":
            self._W = np.linalg.pinv(X_train) @ y_train

        elif self.readout.method == "svd":
            raise NotImplementedError(
                "SVD readout method is not implemented yet"
            )
        else:
            raise ValueError(
                f"Unknown readout method: {self.readout.method}"
            )

        # Fitted training origin
        self._train_end = train_end


    def forecast(self) -> np.ndarray:
        """
        Predict the next `horizon` values after the fitted training prefix.
        """
        # Fitted-state check
        if self._W is None or self._train_end is None:
            raise RuntimeError(
                "the model must be fitted before forecasting"
            )

        # Forecast origin
        origin = int(self._train_end)

        # Available feature range
        offset = self._feature_offset
        first_origin = offset + self.window_size - 1
        last_origin = offset + len(self.reservoir_features) - 1

        if not first_origin <= origin <= last_origin:
            raise ValueError(
                f"origin must be between {first_origin} "
                f"and {last_origin}"
            )

        # Final feature window
        feature_end = origin - offset
        feature_start = feature_end - self.window_size + 1

        feature_window = self.reservoir_features[
            feature_start:
            feature_end + 1
        ]

        # Readout input
        x = np.concatenate([
            np.ones(1),
            feature_window.reshape(-1),
        ])

        if len(x) != self._W.shape[0]:
            raise RuntimeError(
                "forecast features are incompatible with readout weights"
            )

        # Linear prediction
        prediction_unit = x @ self._W

        # Original data scale
        return self._from_unit(prediction_unit)
