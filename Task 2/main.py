from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
import numpy as np

# Load the training data
x_train = np.genfromtxt("ndp-ddos-train-oc-benign.netflow", dtype=float, delimiter="\t")

# Instantiate and train Isolation Forest
iso_forest = IsolationForest(contamination=0.05)
iso_forest.fit(x_train)

# Instantiate and train One-Class SVM
oc_svm = OneClassSVM(nu=0.1)
oc_svm.fit(x_train)

# Instantiate and train Elliptic Envelope
elliptic_envelope = EllipticEnvelope(contamination=0.0001)
elliptic_envelope.fit(x_train)

# List containing the names of the test files.
test_files = ["ndp-ddos-test01.netflow", "ndp-ddos-test02.netflow", "ndp-ddos-test03.netflow"]

# Function to evaluate models using zip and logical and
def evaluate_model(model, test_data, labels):
    pred = model.predict(test_data)

    # Initialize confusion matrix components
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    # Use zip to loop through each prediction and label pair
    # p = each indivisual prediction from the pred_binary array
    # l = each corresponding label in the label array
    for p, l in zip(pred, labels):
        if p == 1 and l == 1:  # True Positive
            TP += 1
        elif p == -1 and l == 0:  # True Negative
            TN += 1
        elif p == 1 and l == 0:  # False Positive
            FP += 1
        elif p == -1 and l == 1:  # False Negative
            FN += 1

    # Calculate precision and recall
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    return precision, recall, TP, TN, FP, FN

for test_file in test_files:
    # Load each test file
    test = np.genfromtxt(test_file, dtype=float, delimiter="\t")
    labels = test[:, 0].astype(int)  # takes the first column as labels
    test_data = test[:, 1:]  # the rest of the test set

    # Evaluate Isolation Forest
    precision_iso, recall_iso, TP_iso, TN_iso, FP_iso, FN_iso = evaluate_model(iso_forest, test_data, labels)

    # Evaluate One-Class SVM
    precision_svm, recall_svm, TP_svm, TN_svm, FP_svm, FN_svm = evaluate_model(oc_svm, test_data, labels)

    # Evaluate Elliptic Envelope
    precision_elliptic, recall_elliptic, TP_elliptic, TN_elliptic, FP_elliptic, FN_elliptic = evaluate_model(elliptic_envelope, test_data, labels)

    # Print results
    print("Results for {}:".format(test_file))
    print("Isolation Forest - Precision: {:.4f}, Recall: {:.4f}, TP: {}, TN: {}, FP: {}, FN: {}".format(
        precision_iso, recall_iso, TP_iso, TN_iso, FP_iso, FN_iso))
    print("OCC-SVM - Precision: {:.4f}, Recall: {:.4f}, TP: {}, TN: {}, FP: {}, FN: {}".format(
        precision_svm, recall_svm, TP_svm, TN_svm, FP_svm, FN_svm))
    print("Elliptic Envelope - Precision: {:.4f}, Recall: {:.4f}, TP: {}, TN: {}, FP: {}, FN: {}".format(
        precision_elliptic, recall_elliptic, TP_elliptic, TN_elliptic, FP_elliptic, FN_elliptic))
    print("")
