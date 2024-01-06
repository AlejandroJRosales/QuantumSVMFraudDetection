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

        self.filename = f_in
        self.get_data(summary)

        # initalize training and testing data
        self.get_datasets()

        # drop features
        self.drop_feauters(['Time', 'Class'])

        # self.adhoc_dimension = 2
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

        #  description of statistic features (Sum, Average, Variance, minimum, 1st quartile, 2nd quartile, 3rd Quartile and Maximum)
        if summary:
            print(self.df.describe())

    def get_datasets(self):
        # generic var names for code reusability
        # take subset of original data
        df_train_all = self.df[0:150000]
        # seperate the data into frauds and no frauds, 1 and 0 respectively
        df_train_1 = df_train_all[df_train_all['Class'] == 1]
        df_train_0 = df_train_all[df_train_all['Class'] == 0]
        df_sample = df_train_0.sample(len(df_train_1))
        # join and then mix dataset for training dataset
        self.df_train = df_train_1.append(df_sample).sample(frac=1)
        # print(self.df_train.head())

    def drop_feauters(self, feat_ls):
        self.df_train.drop(feat_ls,axis=1)
        

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
            np.asmatrix(self.adhoc_total).T,
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