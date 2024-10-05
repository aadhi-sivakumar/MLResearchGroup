from sklearn.ensemble import IsolationForest
import numpy as np

# Download the dataset here:
# https://github.com/Ayush-sharma-py/Research/blob/main/datasets/ts2-occ-ddos.zip

# Note the underscores in the training file name.
x_train = np.genfromtxt("ddos_train_oc_benign.netflow", dtype=float, delimiter="\t")  # input csv file

# Instantiate learning algorithm/model
# We will use an Isolation Forest model for anomaly detection
clf = IsolationForest(contamination=0.0001)

# Method to actually train this model
clf.fit(x_train)

# List containing the names of the test files.
test_files = ["ts2-ddos-test01.netflow", "ts2-ddos-test02.netflow"]

for test_file in test_files:
    # Load each test file and test on our trained model.
    test = np.genfromtxt(test_file, dtype=float, delimiter="\t")
    test[:, 0] = test[:, 0].astype(int)
    label = test[:, 0]  # takes the first column and return it as label
    test = test[:, 1:]  # the rest of the test set

    # Classify the test examples
    pred = clf.predict(test)

    # Prediction: 1 is not anomaly, -1 is anomaly
    # Labeled Data: 0 is not anomaly, 1 is anomaly
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for p_value, l_value in zip(pred, label):
        if p_value == 1 and l_value == 0:
            # correct classification:  not an anomaly (benign traffic)
            true_negative = true_negative + 1
        elif p_value == -1 and l_value == 1:
            # correct classification:  anomaly (DDoS traffic)
            true_positive = true_positive + 1
        elif p_value == 1 and l_value == 1:
            # incorrect classification as benign traffic
            false_negative = false_negative + 1
        else:
            # incorrect classification as DDoS traffic
            false_positive = false_positive + 1  # p_value is -1, l_value is 0

    recall = (true_positive / (true_positive + false_positive))
    precision = (true_positive / (true_positive + false_negative))

    print("tp: " + str(true_positive))
    print("tn: " + str(true_negative))
    print("fn: " + str(false_negative))
    print("fp: " + str(false_positive))
    print("Recall: " + str(recall))
    print("Precision: " + str(precision))
    print("")

# Not happy with your recall?  Train your model with this kwarg passed to the constructor:  contamination=0.0001
# What do you think contamination does?
