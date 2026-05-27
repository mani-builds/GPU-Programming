#!/usr/bin/env python

import numpy as np

M = 1024
N = 2048
K = 1024

TILE = 32

def tiledMatmul(a, b, c, M, N, K):
    for row_out in range(0, TILE):
        for col_out in range(0, TILE):
            a_s = a[row_out][col_out]
    return

def matmul(a, b, c, M, N, K):
    for row_out in range(0, M):
        for col_out in range(0, N):
            sum = 0
            for i in range(0, K):
                sum += a[row_out][i] * b[i][col_out]
            c[row_out][col_out] = sum
    return c


a = np.ones((M, K))
b = np.ones((K, N))
c = np.zeros((M, N))

c = matmul(a,b,c,M,N,K)
