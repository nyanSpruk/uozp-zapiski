import math
import pandas as pd
from itertools import product, combinations


class Clustering:
    def __init__(self, filename):
        """load the data"""
        self.data = pd.read_csv(filename, sep="\t", index_col="ime")

    def row_distance(self, r1, r2):
        """distance between rows"""
        return math.sqrt(sum((self.data.loc[r1] - self.data.loc[r2]) ** 2))

    def cluster_distance(self, c1, c2):
        """distance between two clusters"""
        s = sum(self.row_distance(r1, r2) for r1, r2 in product(c1, c2))
        return s / (len(c1) * len(c2))

    def closest_clusters(self, clusters):
        """find closest clusters"""
        _, pair = min((self.cluster_distance(c1, c2), (c1, c2))
                      for c1, c2 in combinations(clusters, 2))
        return pair

    def run(self, k):
        """hierarchical clustering"""
        clusters = [[k] for k in self.data.index]
        while len(clusters) > k:
            closest = self.closest_clusters(clusters)
            clusters = [c for c in clusters if c not in closest] + \
                [closest[0] + closest[1]]
        return clusters

    def cluster_profiles(self, clusters):
        profiles = pd.DataFrame(data=None, columns=self.data.columns)
        for i, c in enumerate(clusters):
            z = sum(self.data.loc[id] for id in c) / len(c)
            name = "C" + "%d" % i
            profiles.loc[name] = (list(z))
        return profiles

    def q_profiles(self, profiles, delta=20):
        q = pd.DataFrame(data=None, columns=profiles.columns, index=profiles.index)
        mean = self.data.mean()
        positive = profiles - mean > delta
        negative = profiles - mean < -delta
        q[positive] = "+"
        q[negative] = "-"
        q[q.isna()] = ""
        return q


hc = Clustering("grades.csv")
cls = hc.run(3)
print("\n".join(["C%d: " % i + ", ".join(c) for i, c in enumerate(cls)]))
print()
p = hc.cluster_profiles(cls)
q = hc.q_profiles(p)
print(q)
