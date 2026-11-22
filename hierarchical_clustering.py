import csv
import math
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage as scipy_linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist


DATA_FILE = "data_uji_ipm_kemiskinan_jateng_2025.csv"
METRIC = "euclidean" # dapat diubah manhatta atau euclidean
METHODS = ["single", "complete", "average", "ward"] # dapat diubah single, complete, average, ward
NORMALIZE = False
N_CLUSTERS = 3
SMALL_DATA = 15


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        text = f.read()

    lines = text.splitlines()
    first = next(line for line in lines if line.strip(",;\t "))
    delimiter = max(",;\t", key=first.count)

    rows = [
        r for r in csv.reader(lines, delimiter=delimiter)
        if any(c.strip() for c in r)
    ]

    header, body = rows[0], rows[1:]
    names = [r[0].strip() for r in body]

    columns, features = [], []

    for j in range(1, len(header)):
        values = [r[j].strip() if j < len(r) else "" for r in body]
        label = header[j].strip() or f"kolom ke-{j + 1}"

        try:
            columns.append([float(v.replace(",", ".")) for v in values])
            features.append(label)
        except ValueError:
            print(f"Kolom '{label}' dilewati (ada sel kosong atau bukan angka).")

    if not columns:
        raise ValueError("Tidak ada kolom angka di file data.")

    return names, features, np.array(columns).T


def zscore(X):
    std = X.std(axis=0)
    std[std == 0] = 1
    return (X - X.mean(axis=0)) / std


def distance_matrix(X, metric):
    n = len(X)
    D = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            diff = X[i] - X[j]

            if metric == "euclidean":
                d = math.sqrt(np.sum(diff ** 2))
            elif metric == "manhattan":
                d = np.sum(np.abs(diff))
            else:
                raise ValueError("METRIC harus 'euclidean' atau 'manhattan'")

            D[i, j] = D[j, i] = d

    return D


def pair(a, b):
    return (a, b) if a < b else (b, a)


def agglomerative(D, method):
    n = len(D)
    sizes = {i: 1 for i in range(n)}
    dist = {(i, j): D[i, j] for i in range(n) for j in range(i + 1, n)}

    Z = []
    next_id = n

    while len(sizes) > 1:
        (a, b), h = min(dist.items(), key=lambda kv: kv[1])

        na, nb = sizes[a], sizes[b]
        new = next_id
        next_id += 1

        for k in sizes:
            if k in (a, b):
                continue

            dak, dbk, nk = dist[pair(a, k)], dist[pair(b, k)], sizes[k]

            if method == "single":
                dn = min(dak, dbk)

            elif method == "complete":
                dn = max(dak, dbk)

            elif method == "average":
                dn = (na * dak + nb * dbk) / (na + nb)

            elif method == "ward":
                dn = math.sqrt(
                    ((na + nk) * dak ** 2 + (nb + nk) * dbk ** 2
                     - nk * h ** 2) / (na + nb + nk)
                )

            else:
                raise ValueError(f"Linkage tidak dikenal: {method}")

            dist[pair(k, new)] = dn

        for k in list(sizes):
            dist.pop(pair(a, k), None)
            dist.pop(pair(b, k), None)

        del sizes[a], sizes[b]
        sizes[new] = na + nb

        Z.append([a, b, h, na + nb])

    return np.array(Z)


def print_merges(Z, names):
    n = len(names)

    def label(i):
        return names[int(i)] if i < n else f"K{int(i) - n + 1}"

    print(f"{'Langkah':<8}{'Gabung':<20}{'Jarak':>10}{'Ukuran':>8}")

    for step, (a, b, h, size) in enumerate(Z, start=1):
        print(
            f"{step:<8}{label(a) + ' + ' + label(b):<20}"
            f"{h:>10.4f}{int(size):>8}"
        )


def main():
    names, features, X = read_csv(DATA_FILE)

    print(f"Data: {len(names)} objek, {len(features)} fitur ({', '.join(features)})")

    if NORMALIZE:
        X = zscore(X)
        print("Data distandarisasi (z-score).")

    print(f"Ukuran jarak: {METRIC}\n")

    D = distance_matrix(X, METRIC)
    scipy_metric = "euclidean" if METRIC == "euclidean" else "cityblock"

    many = len(names) > SMALL_DATA

    if not many:
        fig, axes = plt.subplots(
            1, len(METHODS),
            figsize=(5 * len(METHODS), 4.5),
            squeeze=False
        )

    for i, method in enumerate(METHODS):

        if method == "ward" and METRIC != "euclidean":
            print(f"[{method}] dilewati: ward hanya cocok dengan euclidean.\n")

            if not many:
                axes[0][i].set_visible(False)

            continue

        Z = agglomerative(D, method)

        print(f"=== {method.upper()} LINKAGE ===")
        print_merges(Z, names)

        Z_ref = scipy_linkage(pdist(X, scipy_metric), method=method)

        cocok = np.allclose(np.sort(Z[:, 2]), np.sort(Z_ref[:, 2]))

        print(f"Validasi dengan scipy: {'SAMA' if cocok else 'BEDA'}")

        if N_CLUSTERS:
            labels = fcluster(Z, t=N_CLUSTERS, criterion="maxclust")

            print(f"Pembagian {N_CLUSTERS} kelompok:")

            for c in sorted(set(labels)):
                anggota = [
                    names[j]
                    for j in range(len(names))
                    if labels[j] == c
                ]

                print(f"  Kelompok {c}: {', '.join(anggota)}")

        print()

        if many:
            fig, ax = plt.subplots(
                figsize=(9, min(30, 0.2 * len(names) + 2))
            )

            dendrogram(
                Z,
                labels=names,
                ax=ax,
                orientation="left",
                leaf_font_size=8
            )

            ax.set_xlabel("Jarak")

        else:
            ax = axes[0][i]

            dendrogram(Z, labels=names, ax=ax)
            ax.set_ylabel("Jarak")

        ax.set_title(f"{method} linkage")

        if many:
            fig.tight_layout()
            fig.savefig(f"dendrogram_{method}.png", dpi=150)

            print(f"Dendrogram disimpan ke dendrogram_{method}.png")

    if not many:
        plt.tight_layout()
        plt.savefig("dendrogram.png", dpi=150)

        print("Dendrogram disimpan ke dendrogram.png")

    plt.show()


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError:
        sys.exit(f"File '{DATA_FILE}' tidak ditemukan. Taruh di folder yang sama dengan script ini.")
