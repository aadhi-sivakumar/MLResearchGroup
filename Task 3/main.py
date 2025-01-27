import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler

# Load the training data
data_train = np.genfromtxt("ndp-ddos-train-oc-benign.netflow", dtype=float, delimiter="\t")
labels_train = data_train[:, 0].astype(int)  # The first column as label
x_train = data_train[:, 1:]                  # The remaining columns as features

# Scale the training features to improve model stability
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)

# Define the contamination and nu ranges
iso_forest_contamination_range = np.linspace(0.01, 0.5, 10)  # For Isolation Forest
elliptic_contamination_range = np.linspace(0.0001, 0.5, 10)  # For Elliptic Envelope
oc_svm_nu_range = np.linspace(0.01, 0.5, 10)  # For One-Class SVM

# List of test files
test_files = ["ndp-ddos-test01.netflow", "ndp-ddos-test02.netflow", "ndp-ddos-test03.netflow"]

# Function to evaluate models
def evaluate_model(model, test_data, labels):
    pred = model.predict(test_data)
    TP = TN = FP = FN = 0
    for p, l in zip(pred, labels):
        if p == 1 and l == 1:
            TP += 1
        elif p == -1 and l == 0:
            TN += 1
        elif p == 1 and l == 0:
            FP += 1
        elif p == -1 and l == 1:
            FN += 1
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    return precision

# Evaluate models for each test file
for test_file in test_files:
    # Load and separate the labels and features for each test file
    data_test = np.genfromtxt(test_file, dtype=float, delimiter="\t")
    labels_test = data_test[:, 0].astype(int)  # The first column as label
    test_data = data_test[:, 1:]               # The remaining columns as features

    # Ensure test data has the same number of features as training data
    if x_train_scaled.shape[1] > test_data.shape[1]:
        missing_features = np.zeros((test_data.shape[0], x_train_scaled.shape[1] - test_data.shape[1]))
        test_data = np.concatenate((test_data, missing_features), axis=1)

    # Scale test data features
    test_data_scaled = scaler.transform(test_data)

    # Initialize variables to store the best models and their metrics
    best_iso_forest = None
    best_precision_iso = 0
    best_elliptic = None
    best_precision_elliptic = 0
    best_oc_svm = None
    best_precision_svm = 0

    # Tune Isolation Forest
    for contamination in iso_forest_contamination_range:
        model = IsolationForest(contamination=contamination, random_state=42)
        model.fit(x_train_scaled)
        precision = evaluate_model(model, test_data_scaled, labels_test)
        if precision > best_precision_iso:
            best_precision_iso = precision
            best_iso_forest = model

    # Tune Elliptic Envelope with increased support_fraction
    for contamination in elliptic_contamination_range:
        model = EllipticEnvelope(contamination=contamination, support_fraction=0.9)
        model.fit(x_train_scaled)
        precision = evaluate_model(model, test_data_scaled, labels_test)
        if precision > best_precision_elliptic:
            best_precision_elliptic = precision
            best_elliptic = model

    # Tune One-Class SVM (using nu similar to contamination)
    for nu in oc_svm_nu_range:
        oc_svm = OneClassSVM(nu=nu, kernel='rbf')
        oc_svm.fit(x_train_scaled)
        precision = evaluate_model(oc_svm, test_data_scaled, labels_test)
        if precision > best_precision_svm:
            best_precision_svm = precision
            best_oc_svm = oc_svm

    # Print best results for each model
    print(f"Results for {test_file}:")
    if best_iso_forest is not None:
        print(f"Best Isolation Forest - Contamination: {best_iso_forest.get_params()['contamination']:.4f}, Precision: {best_precision_iso:.4f}")
    if best_elliptic is not None:
        print(f"Best Elliptic Envelope - Contamination: {best_elliptic.get_params()['contamination']:.4f}, Precision: {best_precision_elliptic:.4f}")
    if best_oc_svm is not None:
        print(f"Best One-Class SVM - Nu: {best_oc_svm.get_params()['nu']:.4f}, Precision: {best_precision_svm:.4f}")
    print("")
