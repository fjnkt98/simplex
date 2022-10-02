from typing import *
import collections
import itertools
import bisect
import math
import numpy as np


def main():
    N, M = map(int, input().split())
    C: np.ndarray = np.array(list(map(int, input().split())))
    A: np.ndarray = np.array(
        [list(map(int, input().split())) for i in range(M)], dtype=np.float64
    )

    n: int = N + M
    m: int = M

    N: np.ndarray = A[:, :N].copy()
    B: np.ndarray = np.eye(M, dtype=np.float64)
    b: np.ndarray = A[:, n - m].copy()

    non_basic_indices: List[int] = list(range(n - m))
    basic_indices: List[int] = list(range(n - m, n))

    c: np.ndarray = np.zeros(n, dtype=np.float64)
    c[: n - m] = C
    cn: np.ndarray = c[: n - m].copy()
    cb: np.ndarray = c[n - m : n].copy()

    xb: np.ndarray = np.zeros(m, dtype=np.float64)

    while True:
        b_bar: np.ndarray = np.dot(np.linalg.inv(B), b.T)
        c_bar: np.ndarray = cn - np.dot(N.T, np.dot(np.linalg.inv(B).T, cb.T))

        if (c_bar <= 0).all():
            xb = b_bar.copy()
            print("Stop")
            break

        k: int = np.where(c_bar > 0)[0][0]
        N_bar: np.ndarray = np.dot(np.linalg.inv(B), N)
        a_bar: np.ndarray = N_bar[:, k].copy()

        if (a_bar <= 0).all():
            print("No bound")
            return

        T: np.ndarray = np.where(a_bar > 0, b_bar / a_bar, np.inf)
        theta: float = np.min(T)
        i: int = np.argmin(T)

        xb = (b_bar - theta * a_bar).copy()
        basic_indices[i], non_basic_indices[k] = non_basic_indices[k], basic_indices[i]
        N[:, k], B[:, i] = B[:, i], N[:, k].copy()
        cn[k], cb[i] = cb[i], cn[k]

    x: np.ndarray = np.zeros(n)
    for i in range(m):
        x[basic_indices[i]] = xb[i]

    print(x)
    print(np.sum(x[: n - m] * C))


if __name__ == "__main__":
    main()
