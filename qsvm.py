import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# dataset libs
from qiskit_algorithms.utils import algorithm_globals
from qiskit_machine_learning.datasets import ad_hoc_data
# qml libs
from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import Sampler
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel
# SVClassification lib
from sklearn.svm import SVC

# same seed for reproducibility
algorithm_globals.random_seed = 12345

class QSVM:
    def __init__(self, data):
        # init quantum kernel
        self.data = data
        # 29 dimensional space
        self.adhoc_dimension = 29
        self.quantum_kernel()

    def fit(self, test=False):
        print("Starting fit.")
        self.fit_kernel()
        print("Done.")
        if test:
            print("Starting test.")
            self.test_fit()
            print("Done.")

    def quantum_kernel(self):
        # init kenrel and 2-qubit ZZ feature mapping
        self.adhoc_feature_map = ZZFeatureMap(feature_dimension=self.adhoc_dimension, reps=2, entanglement="linear")
        self.sampler = Sampler()
        self.fidelity = ComputeUncompute(sampler=self.sampler)
        self.adhoc_kernel = FidelityQuantumKernel(fidelity=self.fidelity, feature_map=self.adhoc_feature_map)

    def fit_kernel(self):
        self.adhoc_svc = SVC(kernel=self.adhoc_kernel.evaluate)
        self.adhoc_svc.fit(self.data.train_features, self.data.train_labels)

    def test_fit(self):
        adhoc_score_callable_function = self.adhoc_svc.score(self.data.test_features, self.data.test_labels)
        print(f"Callable kernel classification test score: {adhoc_score_callable_function}")


class Data:
    def __init__(self, f_in, summary=False):
        self.filename = f_in
        self.get_data(summary)
        # initialize training and testing dataset and labels
        self.set_datasets()

    def get_data(self, summary=False):
        # read in data and convert to data frame
        self.df = pd.DataFrame(pd.read_csv(self.filename))
        if summary:
            print(self.df.describe())

    def set_datasets(self):
        self.train_features, self.train_labels = self.set_dataset(0, 150000, clean=True)
        self.test_features, self.test_labels = self.set_dataset(150000, -1)

    def set_dataset(self, idx_start, idx_end, clean=False):
        df_subdata = self.clean_data(self.df[idx_start:idx_end]) if clean else self.df[idx_start:idx_end]
        # for testing uncomment line; below grab first 2 features and use 20 samples
        # return np.asarray(df_subdata[['V1', 'V2']])[0:20], np.asarray(df_subdata['Class'])[0:20]
        # for testing comment line below
        return np.asarray(df_subdata.drop(['Time', 'Class'], axis=1)), np.asarray(df_subdata['Class'])

    def clean_data(self, data):
        # seperate data by labels
        df_1 = data[data['Class'] == 1]
        df_0 = data[data['Class'] == 0]
        # take similar number of non-fraud to cover for oversampling
        self.sample_total = len(df_1)
        df_0 = df_0.sample(self.sample_total)
        # join and mix data
        return df_1.append(df_0).sample(frac=1)
        
    @staticmethod
    def plot_features(ax, features, labels, class_label, marker, face, edge, label):
        # A train plot
        ax.scatter(
            # x coordinate of labels where class is class_label
            features[np.where(labels[:] == class_label), 0],
            # y coordinate of labels where class is class_label
            features[np.where(labels[:] == class_label), 1],
            marker=marker,
            facecolors=face,
            edgecolors=edge,
            label=label,
        )

    def plot_dataset(self):
        plt.figure(figsize=(5, 5))
        plt.ylim(0, 2 * np.pi)
        plt.xlim(0, 2 * np.pi)
        plt.imshow(
            # if testing, use chosen sample size
            np.asmatrix(self.sample_total).T,
            interpolation="nearest",
            origin="lower",
            cmap="RdBu",
            extent=[0, 2 * np.pi, 0, 2 * np.pi],
        )

        # A train plot
        self.plot_features(plt, self.train_features, self.train_labels, 0, "s", "w", "b", "A train")
        # B train plot
        self.plot_features(plt, self.train_features, self.train_labels, 1, "o", "w", "r", "B train")
        # A test plot
        self.plot_features(plt, self.test_features, self.test_labels, 0, "s", "b", "w", "A test")
        # B test plot
        self.plot_features(plt, self.test_features, self.test_labels, 1, "o", "r", "w", "B test")

        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
        plt.title("Ad hoc dataset")

        plt.show()


# init credit card data set
credit_card_data = Data('input\creditcard.csv\creditcard.csv', summary=True)
# credit_card_data.plot_dataset()

# init quantum kernel
qsvm = QSVM(credit_card_data)
# fit quantum kernel on credit card data and test fit
qsvm.fit(test=True)

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.