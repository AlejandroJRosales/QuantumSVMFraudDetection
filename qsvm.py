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


class QSVM:
    def __init__(self, data):
        # init quantum kernel
        self.data = data
        self.quantum_kernel()

    def fit(self, test=False):
        self.fit_kernel()
        if test:
            self.test_fit()

    def quantum_kernel(self):
        # init kenrel and 2-qubit ZZ feature mapping
        self.adhoc_feature_map = ZZFeatureMap(feature_dimension=data.adhoc_dimension, reps=2, entanglement="linear")
        self.sampler = Sampler()
        self.fidelity = ComputeUncompute(sampler=self.sampler)
        self.adhoc_kernel = FidelityQuantumKernel(fidelity=self.fidelity, feature_map=self.adhoc_feature_map)

    def fit_kernel(self):
        self.adhoc_svc = SVC(kernel=self.adhoc_kernel.evaluate)
        self.adhoc_svc.fit(data.train_features, data.train_labels)

    def test_fit(self):
        adhoc_score_callable_function = self.adhoc_svc.score(data.test_features, data.test_labels)
        print(f"Callable kernel classification test score: {adhoc_score_callable_function}")


class Data:
    def __init__(self, f_in, summary=False):
        # initialize seed for reproducibility
        algorithm_globals.random_seed = 12345
        self.adhoc_dimension = 2

        self.filename = f_in
        self.get_data(summary)
        self.set_datasets()

        # self.train_features, self.train_labels, self.test_features, self.test_labels, self.adhoc_total = ad_hoc_data(
        #     training_size=20,
        #     test_size=5,
        #     n=self.adhoc_dimension,
        #     gap=0.3,
        #     plot_data=False,
        #     one_hot=False,
        #     include_sample_total=True,
        # )

    def get_data(self, summary):
        # read in data and convert to data frame
        self.df = pd.DataFrame(pd.read_csv(self.filename))
        if summary:
            # statistic features for data
            print(self.df.describe())

    def set_datasets(self):
        self.train_features, self.train_labels = self.set_data(0, 150000, clean=True)
        self.test_features, self.test_labels = self.set_data(15000, -1)

    def set_data(self, idx_start, idx_end, clean=False):
        df_subdata = self.clean_data(self.df[idx_start:idx_end]) if clean else self.df[idx_start:idx_end]
        return np.asarray(df_subdata.drop(['Time', 'Class'], axis=1)), np.asarray(df_subdata['Class'])

    def clean_data(self, data):
        # seperate the data by labels
        df_1 = data[data['Class'] == 1]
        df_0 = data[data['Class'] == 0]
        # take similar number of non-fraud to cover for oversampling
        self.sample_total = len(df_1)
        df_0 = df_0.sample(self.sample_total)
        # join and mix dataset
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
data = Data('input\creditcard.csv\creditcard.csv', summary=False)
# data.plot_dataset()