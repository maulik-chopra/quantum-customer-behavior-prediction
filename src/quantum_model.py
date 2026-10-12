"""Small Qiskit variational classifier; all execution is simulator based."""
# Quantum-model training uses a timer and imports the optional quantum packages only when needed.
from time import perf_counter

def train_quantum(X_train, y_train, X_test, max_rows=100):
    """Fit PCA and a four-qubit VQC on a bounded training sample.

    Returns a dictionary of measured results, or raises an import/API/runtime error
    for the UI to report. No placeholder metrics are generated.
    """
    # Load packages needed for feature reduction, quantum circuits, and the Aer simulator.
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import MinMaxScaler
    from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import SamplerV2
    from qiskit_machine_learning.algorithms.classifiers import VQC
    from qiskit_machine_learning.optimizers import COBYLA
    from qiskit_machine_learning.utils import algorithm_globals
    # PCA: compress the prepared customer features to four principal components using training rows only.
    pca = PCA(n_components=4, random_state=17)
    reduced_train = pca.fit_transform(X_train)
    reduced_test = pca.transform(X_test)
    # Scale PCA values to a 0-to-pi range used as circuit rotation angles.
    scale = MinMaxScaler(feature_range=(0, 3.14159))
    quantum_train = scale.fit_transform(reduced_train)
    quantum_test = scale.transform(reduced_test)
    # FEATURE MAP: encode the four PCA values into a quantum circuit; some samplers require this template to be transpiled first.
    feature_map = ZZFeatureMap(feature_dimension=4, reps=1, entanglement="linear")
    # ANSATZ: create the trainable four-qubit circuit whose weights the classifier learns.
    ansatz = RealAmplitudes(num_qubits=4, reps=1, entanglement="linear")
    # SamplerV2 is Qiskit ML's expected interface and executes through Aer.
    # backend_options select Aer's simulator engine for statevector sampling.
    # SIMULATOR: use local statevector simulation rather than physical quantum hardware.
    simulator = AerSimulator(method="statevector")
    sampler = SamplerV2(seed=17, options={"backend_options": {"method": "statevector"}})
    # Fix the random seed so repeated runs are more reproducible.
    algorithm_globals.random_seed = 17
    # Reps=1 and a bounded sample keep this educational simulator experiment manageable.
    # CLASSIFIER: combine the feature map, trainable circuit, optimizer, and sampler into one model.
    classifier = VQC(feature_map=feature_map, ansatz=ansatz, optimizer=COBYLA(maxiter=25), sampler=sampler)
    # Limit the training sample because variational-circuit simulation is computationally expensive.
    train_x = np.asarray(quantum_train[:max_rows], dtype=float)
    train_y = np.asarray(y_train[:max_rows], dtype=int)
    start = perf_counter()
    # FIT: adjust circuit weights on the selected training examples and record elapsed time.
    classifier.fit(train_x, train_y)
    training_seconds = perf_counter() - start
    # Return the fitted model and transformations so the app can score test/customer rows consistently.
    return {"model": classifier, "pca": pca, "scale": scale, "feature_map": feature_map,
      "ansatz": ansatz, "simulator": simulator, "sampler": sampler, "test_x": quantum_test,
      "training_seconds": training_seconds, "explained_variance": float(pca.explained_variance_ratio_.sum()),
      "train_rows": len(train_x)}
