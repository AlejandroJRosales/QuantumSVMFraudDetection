# Quantum Support Vector Machine Fraud Detection
Applying a Quantum Support Vector Machine for credit card fraud detection in Qiskit. The applied QSVM makes use of quantum enhanced feature spaces optimization based on the research paper, Supervised Learning with Quantum Enhanced Feature Spaces. The classifier used is a Variational Quantum Classifier with 29-dimensional ZZ feature mapping for a 2-qubit quantum kernel. Adjust parameters in code if need be. 

Read the amazing original work of the researchers on the paper, Supervised Learning with Quantum Enhanced Feature Spaces. Which you can find here: https://arxiv.org/pdf/1804.11326.pdf

You can read a summary and analysis of the math, circuitry, and quantum mechanics behind the support vector machine with quantum-enhanced feature space on my website here: https://www.contextswitching.org/tcs/quantummachinelearning/#quantum-support-vector-machine

## Installation

### Running IBM Quantum Experience Locally

**Install and Set Up Qiskit**

```
pip install qiskit
```

```
pip install qiskit-ibm-runtime
```

**Qiskit Quantum Algorithms Installation**

```
pip install qiskit-algorithms
```

**Qiskit Machine Learning Installation**

```
pip install qiskit-machine-learning
```

### Non-Qiskit Libraries

**Our 'Data' Libraries**

```
pip install numpy
```

```
pip install pandas
```

**SKLearn Machine Learning Library**

```
pip install -U scikit-learn
```

## Testing Example

Example output when using 20 samples with 2 features each, implementing 29-dimensional feature space mapping for 2-qubit quantum kernel. 

**Output**
```
                Time            V1            V2            V3            V4  ...           V26           V27           V28         Amount          Class
count  284807.000000  2.848070e+05  2.848070e+05  2.848070e+05  2.848070e+05  ...  2.848070e+05  2.848070e+05  2.848070e+05  284807.000000  284807.000000
mean    94813.859575  1.168375e-15  3.416908e-16 -1.379537e-15  2.074095e-15  ...  1.683437e-15 -3.660091e-16 -1.227390e-16      88.349619       0.001727
std     47488.145955  1.958696e+00  1.651309e+00  1.516255e+00  1.415869e+00  ...  4.822270e-01  4.036325e-01  3.300833e-01     250.120109       0.041527
min         0.000000 -5.640751e+01 -7.271573e+01 -4.832559e+01 -5.683171e+00  ... -2.604551e+00 -2.256568e+01 -1.543008e+01       0.000000       0.000000
25%     54201.500000 -9.203734e-01 -5.985499e-01 -8.903648e-01 -8.486401e-01  ... -3.269839e-01 -7.083953e-02 -5.295979e-02       5.600000       0.000000
50%     84692.000000  1.810880e-02  6.548556e-02  1.798463e-01 -1.984653e-02  ... -5.213911e-02  1.342146e-03  1.124383e-02      22.000000       0.000000
75%    139320.500000  1.315642e+00  8.037239e-01  1.027196e+00  7.433413e-01  ...  2.409522e-01  9.104512e-02  7.827995e-02      77.165000       0.000000
max    172792.000000  2.454930e+00  2.205773e+01  9.382558e+00  1.687534e+01  ...  3.517346e+00  3.161220e+01  3.384781e+01   25691.160000       1.000000

[8 rows x 31 columns]
Starting fit.
Done.
Starting test.
Callable kernel classification test score: 0.85
Done.
```

## Dataset

Credit Card Fraud Dataset provided by Kaggle.

Input Data:

source: https://www.kaggle.com/code/pierra/credit-card-dataset-svm-classification/input

filename: creditcard.csv

size: 150.83MB

type: csv

dimensions: ...

