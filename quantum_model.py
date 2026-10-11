"""Small Qiskit variational classifier; all execution is simulator based."""
from time import perf_counter

def train_quantum(X_train, y_train, X_test, max_rows=100):
    """Fit PCA and a four-qubit VQC on a bounded training sample.

    Returns a dictionary of measured results, or raises an import/API/runtime error
    for the UI to report. No placeholder metrics are generated.
    """
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import MinMaxScaler
    from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import SamplerV2
    from qiskit_machine_learning.algorithms.classifiers import VQC
    from qiskit_machine_learning.optimizers import COBYLA
    from qiskit_machine_learning.utils import algorithm_globals
    # PCA fit uses training data only; four components map to four qubits.
    pca = PCA(n_components=4, random_state=17)
    reduced_train = pca.fit_transform(X_train)
    reduced_test = pca.transform(X_test)
    scale = MinMaxScaler(feature_range=(0, 3.14159))
    quantum_train = scale.fit_transform(reduced_train)
    quantum_test = scale.transform(reduced_test)
    feature_map = ZZFeatureMap(feature_dimension=4, reps=1, entanglement="linear")
    ansatz = RealAmplitudes(num_qubits=4, reps=1, entanglement="linear")
    # SamplerV2 is Qiskit ML's expected interface and executes through Aer.
    # backend_options select Aer's simulator engine for statevector sampling.
    simulator = AerSimulator(method="statevector")
    sampler = SamplerV2(seed=17, options={"backend_options": {"method": "statevector"}})
    algorithm_globals.random_seed = 17
    # Reps=1 and a bounded sample keep this educational simulator experiment manageable.
    classifier = VQC(feature_map=feature_map, ansatz=ansatz, optimizer=COBYLA(maxiter=25), sampler=sampler)
    train_x = np.asarray(quantum_train[:max_rows], dtype=float)
    train_y = np.asarray(y_train[:max_rows], dtype=int)
    start = perf_counter()
    classifier.fit(train_x, train_y)
    training_seconds = perf_counter() - start
    return {"model": classifier, "pca": pca, "scale": scale, "feature_map": feature_map,
      "ansatz": ansatz, "simulator": simulator, "sampler": sampler, "test_x": quantum_test,
      "training_seconds": training_seconds, "explained_variance": float(pca.explained_variance_ratio_.sum()),
      "train_rows": len(train_x)}
