import math
import pandas as pd
import numpy as np
import timeit
from itertools import product, combinations
from scipy.stats import ttest_ind
from termcolor import colored


def load_data(filename):
    """load the data from the CSV file"""
    return pd.read_csv(filename, sep="\t", index_col="ime")


class HierarchicalClustering:
    """Improved hierarchical clustering."""
    def __init__(self, data):
        """initialize"""
        self.data = data
        self.clusters = None
        self.name = "Hierarchical Clustering"

    def row_distance(self, r1, r2):
        """distance between rows"""
        return math.sqrt(sum((self.data.loc[r1] - self.data.loc[r2]) ** 2))

    def cluster_distance(self, c1, c2):
        """distance between two clusters"""
        s = sum(self.row_distance(r1, r2) for r1, r2 in product(c1, c2))
        return s / (len(c1) * len(c2))

    @staticmethod
    def closest_clusters(distances):
        """find the closest clusters"""
        dist, pair = min((d, c) for c, d in distances.items())
        return pair, dist

    def run(self, k):
        """hierarchical clustering"""
        self.clusters = [(k, ) for k in self.data.index]
        distances = {(c1, c2): self.cluster_distance(c1, c2)
                     for c1, c2 in combinations(self.clusters, 2)}

        while len(self.clusters) > k:
            (c1, c2), dist = self.closest_clusters(distances)
            # joins[(c1, c2)] = dist
            self.clusters.remove(c1)
            self.clusters.remove(c2)
            distances = {c: d for c, d in distances.items()
                         if (c1 not in c) and (c2 not in c)}
            new_c = c1 + c2
            new_distances = {(new_c, c): self.cluster_distance(new_c, c)
                             for c in self.clusters}
            distances.update(new_distances)
            self.clusters.append(new_c)

    def compute_silhouette(self, item):
        bs = []
        for c in self.clusters:
            if item in c:
                if len(c) == 1:
                    return 0
                else:
                    a = sum(self.row_distance(item, i)
                            for i in c if i != item) / (len(c) - 1)
            else:
                bs.append(sum(self.row_distance(item, i) for i in c) / len(c))
        b = min(bs)
        return (b - a) / max(a, b)

    def silhouettes(self):
        return {i: self.compute_silhouette(i) for i in self.data.index}

    def cluster_silhouette(self):
        return sum(self.silhouettes().values())/len(self.data)

    def explain(self, cluster):
        others = list(set(self.data.index).difference(cluster))
        scores = []
        for c in self.data.columns:
            p1 = self.data[c].loc[list(cluster)]
            p2 = self.data[c].loc[others]
            v, p = ttest_ind(p1, p2)
            scores.append((c, p, v))
        scores.sort(key=lambda x: x[1])
        return scores


def explain_pp(scores, t=0.01):
    print("\n".join(f"{i} {'+' if v > 0 else '-'} {p:.4f}"
                    for i, p, v in scores if p <= t))


class KMeans(HierarchicalClustering):
    def __init__(self, data, verbose=0):
        """initialize"""
        self.data = data
        self.centroids = None
        self.membership = None
        self.clusters = None
        self.verbose = verbose
        self.name = "K-Means Clustering"
        np.random.seed(42)

    def run(self, k=3, max_iter=100):
        iteration = 0
        self.centroids = self.data.sample(k).to_numpy()
        while True:
            iteration += 1
            dist = np.array([np.sqrt(np.sum((self.data - self.centroids[i]) ** 2, axis=1))
                             for i in range(k)])
            self.membership = np.argmin(dist, axis=0)
            new = np.array([np.average(self.data[self.membership == i], axis=0)
                            for i in range(k)])
            if iteration == max_iter or np.array_equal(self.centroids, new):
                break
            self.centroids = new
        self.clusters = [list(self.data.index[self.membership == i]) for i in range(k)]
        if self.verbose:
            print(f"iterations: {iteration}")


def cluster_analysis(data, clustering, t=0.001):
    clust = clustering(data)
    print(colored(clust.name, "green", attrs=["bold"]))

    print("Cluster silhouettes:")
    for k in range(2, 6):
        clust.run(k)
        s = clust.cluster_silhouette()
        print(f"{k}: {s:0.3}")
    print()

    print("Explanation for chosen number of clusters:")
    clust.run(k=3)

    for cluster in clust.clusters:
        print(", ".join(e for e in cluster))
        sc = clust.explain(cluster)
        explain_pp(sc, t=t)
        print()


def runtime(data, clustering, num_times=10, k=3):
    clust = clustering(data)
    t = timeit.timeit(stmt=lambda k=3: clust.run(k), number=num_times)/num_times
    print(f"Runtime of {clust.name}: {t:.2f} s")


dataset = load_data("grades.csv")

# run hierarchical clustering first, explain the clusters
cluster_analysis(dataset, HierarchicalClustering)
cluster_analysis(dataset, KMeans)

print(colored("Comparison of speed", "green"))
runtime(dataset, HierarchicalClustering)
runtime(dataset, KMeans)
