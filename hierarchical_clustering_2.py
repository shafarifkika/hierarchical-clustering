import csv
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, fcluster
from scipy.cluster.hierarchy import linkage as scipy_linkage
from scipy.spatial.distance import pdist

DATA_FILE = "data_uji_ipm_kemiskinan_jateng_2025.csv"
METRIC = "euclidean"  
METHODS = ["single", "complete", "average"]
NORMALIZE = False
N_CLUSTERS = 3
SHOW_TABLE = True

LINKAGES = {
    "single": lambda dak, dbk, na, nb: min(dak, dbk),
    "complete": lambda dak, dbk, na, nb: max(dak, dbk),
    "average": lambda dak, dbk, na, nb: (na * dak + nb * dbk) / (na + nb),
}
SCIPY_METRIC = {"euclidean": "euclidean", "manhattan": "cityblock"}


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        lines = f.read().splitlines()
    delimiter = csv.Sniffer().sniff(lines[0], delimiters=",;\t").delimiter
    rows = [r for r in csv.reader(lines, delimiter=delimiter) if r]
    names = [r[0] for r in rows[1:]]
    X = np.array([[float(v.replace(",", ".")) for v in r[1:]] for r in rows[1:]])
    return names, rows[0][1:], X


def zscore(X):
    std = X.std(axis=0)
    std[std == 0] = 1
    return (X - X.mean(axis=0)) / std


def distance_matrix(X, metric):
    diff = X[:, None, :] - X[None, :, :]
    if metric == "euclidean":
        return np.sqrt((diff ** 2).sum(axis=2))
    if metric == "manhattan":
        return np.abs(diff).sum(axis=2)
    raise ValueError("METRIC harus 'euclidean' atau 'manhattan'")


def pair(a, b):
    return (a, b) if a < b else (b, a)


def agglomerative(D, method):
    update = LINKAGES[method]
    n = len(D)
    sizes = {i: 1 for i in range(n)}
    dist = {(i, j): D[i, j] for i in range(n) for j in range(i + 1, n)}
    Z = []

    for new in range(n, 2 * n - 1):
        (a, b), h = min(dist.items(), key=lambda kv: kv[1])
        na, nb = sizes.pop(a), sizes.pop(b)
        dist.pop((a, b))
        for k in sizes:
            dist[pair(k, new)] = update(dist.pop(pair(a, k)), dist.pop(pair(b, k)), na, nb)
        sizes[new] = na + nb
        Z.append([a, b, h, na + nb])

    return np.array(Z)


def print_merges(Z, names):
    n = len(names)

    def label(i):
        return names[int(i)] if i < n else f"K{int(i) - n + 1}"

    rows = [(f"{label(a)} + {label(b)}", h, int(size)) for a, b, h, size in Z]
    width = max(len(text) for text, _, _ in rows) + 2
    print(f"{'Langkah':<9}{'Gabung':<{width}}{'Jarak':>10}{'Ukuran':>8}")
    for step, (text, h, size) in enumerate(rows, start=1):
        print(f"{step:<9}{text:<{width}}{h:>10.4f}{size:>8}")


def report(Z, X, names, method):
    print(f"=== {method.upper()} LINKAGE ===")
    print_merges(Z, names)

    Z_ref = scipy_linkage(pdist(X, SCIPY_METRIC[METRIC]), method=method)
    sama = np.allclose(np.sort(Z[:, 2]), np.sort(Z_ref[:, 2]))
    print(f"Validasi dengan scipy: {'SAMA' if sama else 'BEDA (cek jarak kembar/tie)'}")

    if N_CLUSTERS:
        groups = fcluster(Z, t=N_CLUSTERS, criterion="maxclust")
        print(f"Pembagian {N_CLUSTERS} kelompok:")
        for c in sorted(set(groups)):
            members = [name for name, g in zip(names, groups) if g == c]
            print(f"  Kelompok {c}: {', '.join(members)}")
    print()


def main():
    names, features, X = read_csv(DATA_FILE)
    print(f"Data: {len(names)} objek, {len(features)} fitur ({', '.join(features)})")
    if NORMALIZE:
        X = zscore(X)
        print("Data distandarisasi (z-score).")
    print(f"Ukuran jarak: {METRIC}\n")

    D = distance_matrix(X, METRIC)
    fig, axes = plt.subplots(1, len(METHODS), figsize=(5 * len(METHODS), 4.5), squeeze=False)

    for ax, method in zip(axes[0], METHODS):
        Z = agglomerative(D, method)
        if SHOW_TABLE:
            report(Z, X, names, method)
        dendrogram(Z, labels=names, ax=ax)
        ax.set_title(f"{method} linkage")
        ax.set_ylabel("Jarak")

    plt.tight_layout()
    plt.savefig("dendrogram.png", dpi=150)
    print("Dendrogram disimpan ke dendrogram.png")
    plt.show()


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError:
        sys.exit(f"File '{DATA_FILE}' tidak ditemukan. Taruh di folder yang sama dengan script ini.")