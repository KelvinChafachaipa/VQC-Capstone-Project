# Import necessary libraries
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting decision boundaries
from sklearn.model_selection import train_test_split  # Splitting dataset
from sklearn.preprocessing import StandardScaler  # Normalizing data
from sklearn.datasets import make_classification  # Generating synthetic dataset

# Import Qiskit libraries
from qiskit import Aer  # Quantum simulator
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes  # Quantum feature map and ansatz
from qiskit.algorithms.optimizers import COBYLA  # Classical optimizer
from qiskit_machine_learning.algorithms.classifiers import VQC  # Variational Quantum Classifier
from qiskit_machine_learning.datasets import ad_hoc_data  # Example datasets
from qiskit.utils import algorithm_globals  # For setting a random seed

# Set a fixed seed for reproducibility
algorithm_globals.random_seed = 42

# Generate a synthetic classification dataset
X, y = make_classification(n_samples=100, n_features=2, n_classes=2, 
                           n_informative=2, n_redundant=0, random_state=42)
# - n_samples=100: Generates 100 data points
# - n_features=2: Each data point has 2 features
# - n_classes=2: Binary classification (0 or 1)
# - n_informative=2: Both features are useful for classification
# - n_redundant=0: No unnecessary features

# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data to improve performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Fit to training data and transform it
X_test = scaler.transform(X_test)  # Apply the same transformation to test data

# Define the quantum feature map (encodes classical data into quantum states)
feature_map = ZZFeatureMap(feature_dimension=2, reps=2, entanglement='linear')
# - feature_dimension=2: Uses 2 qubits since we have 2 features
# - reps=2: The feature map is repeated twice to enhance representation
# - entanglement='linear': Each qubit is entangled with its neighbor

# Define the variational ansatz (trainable quantum circuit)
ansatz = RealAmplitudes(num_qubits=2, reps=2, entanglement='linear')
# - num_qubits=2: Matches the number of qubits with feature dimensions
# - reps=2: The quantum layers are repeated twice
# - entanglement='linear': Linear entanglement between qubits

# Select the quantum simulator backend
backend = Aer.get_backend('qasm_simulator')  # Simulates quantum computations

# Define the optimizer for tuning the quantum circuit parameters
optimizer = COBYLA(maxiter=100)  
# - COBYLA: Constrained Optimization BY Linear Approximations (classical optimizer)
# - maxiter=100: Performs 100 iterations to optimize the circuit parameters

# Create the Variational Quantum Classifier (VQC)
vqc = VQC(optimizer=optimizer, feature_map=feature_map, ansatz=ansatz, quantum_instance=backend)
# - Uses the defined feature_map (quantum encoding) and ansatz (trainable circuit)
# - Uses COBYLA to optimize parameters
# - Runs on the selected backend (simulator)

# Train the classifier on the training dataset
vqc.fit(X_train, y_train)

# Evaluate the trained model on the test set
score = vqc.score(X_test, y_test)  # Calculates accuracy

# Print the accuracy of the VQC model
print(f"VQC Model Accuracy: {score * 100:.2f}%")  
# Multiplies score by 100 to show accuracy as a percentage

# Function to plot the decision boundary of the VQC model
def plot_decision_boundary(model, X, y):
    # Define the range of values for the plot
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    # Create a mesh grid to evaluate the model predictions
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    
    # Predict the class for each point in the grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)  # Reshape for visualization
    
    # Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3)  # Colored regions for classification areas
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', marker='o', cmap=plt.cm.Paired)  # Data points
    plt.title("VQC Decision Boundary")  # Title
    plt.xlabel("Feature 1")  # X-axis label
    plt.ylabel("Feature 2")  # Y-axis label
    plt.show()

# Call the function to plot the decision boundary
plot_decision_boundary(vqc, X_test, y_test)
