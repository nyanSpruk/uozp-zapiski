import glob
import os.path
import math
from itertools import combinations
from collections import Counter


def read_corpus(dir):
    """read the corpus of documents from the directory"""
    corpus = {}
    file_names = glob.glob(f"{dir}/*")
    for file_name in file_names:
        name = os.path.splitext(os.path.basename(file_name))[0]
        text = " ".join(open(file_name, "rt").readlines())
        text = text.replace("\n \n", " ")
        text = text.replace("\n", "")
        text = text.replace("  ", " ")
        corpus[os.path.splitext(name)[0]] = text
    return corpus


def kmers(s, k=3):
    """k-mers generator"""
    for i in range(len(s)-k+1):
        yield s[i:i+k]


def jaccard(data, k1, k2):
    """jaccard similarity"""
    s1 = data[k1]
    s2 = data[k2]
    return len(s1 & s2) / len(s1 | s2)


def length(vector):
    """vector length"""
    return math.sqrt(sum(v**2 for v in vector))


def cosine(data, k1, k2):
    """cosine similarity"""
    prod = sum(data[k1][key] * data[k2][key] for key in data[k1].keys() & data[k2].keys())
    return prod / (length(data[k1].values()) * length(data[k2].values()))


def compute_and_report(data, scorer):
    """compute pairwise document similarity and report"""
    sim = sorted([(scorer(data, a, b), a, b)
                  for a, b in combinations(data.keys(), 2)],
                 reverse=True)

    for s, a, b in sim:
        print("%.2f" % s, a, b)


docs = read_corpus("text-data")
substring_length = 4
kmer_data = {key: set(kmers(docs[key], k=substring_length)) for key in docs}
count_data = {key: Counter(kmers(docs[key], k=substring_length)) for key in docs}

print("Jaccard similarities:")
compute_and_report(kmer_data, jaccard)
print("\nCosine similarities")
compute_and_report(count_data, cosine)
