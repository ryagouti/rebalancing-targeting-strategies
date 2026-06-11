import numpy as np
import math
from utils import *

def general_neyman(true_pr, floor = 0.0):
    perm = sorted(range(len(true_pr)), key=lambda i: true_pr[i], reverse=True)
    p = np.array([true_pr[e] for e in perm])

    K = len(p)
    s = next((i for i, x in enumerate(p) if x != p[0]), K)
    g = next((i for i, x in enumerate(reversed(p)) if x != p[-1]), K)

    if s == K:
        return np.array([1.0 / K] * K)

    a = math.sqrt(p[0] * (1 - p[0]))
    b = math.sqrt(p[-1] * (1 - p[-1]))

    if floor > 0:
        inv_prod = 1 / (p * (1 - p))
        diff_k = p - p[-1]
        alpha_1 = (a / (a + b)) * (b ** 2 * sum(inv_prod[s:K - g])) - (K - s - g)
        alpha_2 = a * b / (diff_k[0]) * sum((diff_k * inv_prod)[s:K - g])
        alpha = alpha_1 - alpha_2

        rho1 = (alpha * floor + a / (a + b)) / s
        rhot = (1 - floor * (K - s - g) - s * rho1) / g
        rho = np.full(K, floor)

    else:
        rho1 = a / (s * (a + b))  # for first s entries
        rhot = (1 - s * rho1) / g  # for last g entries
        rho = [0.0] * K

    for i in range(s): rho[i] = rho1
    for i in range(K - g, K): rho[i] = rhot

    real_order_rho = np.zeros(K)
    for i in range(K):
        real_order_rho[perm[i]] = rho[i]

    return real_order_rho



def target_allocation(theta_est, name):
    if name == "Uniform":
        K = len(theta_est)
        return np.array([1/K] * K)
    if name == "GPU":
        est_q = 1 - theta_est
        return (1 / est_q) / np.sum(1 / est_q)
    if name == "RSIHR":
        denom = sum(np.sqrt(theta_est))
        if denom == 0:
            return np.array([1 / len(theta_est)] * len(theta_est))
        return np.sqrt(theta_est) / denom
    if name == "Neyman":
        est_q_sqrt = np.sqrt(1 - theta_est)
        denom = sum(est_q_sqrt * np.sqrt(theta_est))
        if denom == 0:
            return np.array([1 / len(theta_est)] * len(theta_est))
        return est_q_sqrt * np.sqrt(theta_est) / denom
    if name == "BAI-Gaussian":
        expfam = Gaussian(1.0)
        tstar, target_alloc = oracle(expfam, theta_est)
        return target_alloc
    if name == "Tymofyeyev":
        target_alloc = general_neyman(theta_est)
        return target_alloc    
    spl = name.split("-")
    if len(spl) > 1:
        if spl[0] == "Tymofyeyev":
            floor = float(spl[1])
            target_alloc = general_neyman(theta_est, floor)
            return target_alloc
    return None


def sigma_rsihr(p):
    q = 1 - p
    p1, p2, p3 = p
    q1, q2, q3 = q

    s1, s2, s3 = np.sqrt(p1), np.sqrt(p2), np.sqrt(p3)

    num = (s2 + s3)**2 * q1 * s2 * s3 + (p1**1.5) * (q2 * s3 + q3 * s2)
    denom = 4.0 * np.sqrt(p1 * p2 * p3) * (s1 + s2 + s3)**3

    return num / denom

def g_alloc(x, y, alpha, L):
    numerator = np.minimum(y * (y/x)**alpha, L)
    probabilities = numerator / numerator.sum()
    return probabilities


# W-star function from BAI paper
def oracle(expfam, mus):
    mu_star = max(mus)

    # if all means are equal
    if all(mu == mu_star for mu in mus):
        n = len(mus)
        return math.inf, np.ones(n) / n

    # determine upper range for binary search
    hi = min(
        d(expfam, mu_star, mu)
        for mu in mus
        if mu != mu_star
    )

    def f(z):
        s = 0.0
        for mu in mus:
            if mu == mu_star:
                continue
            mu_x = X(expfam, mu_star, mu, z)[1]
            s += d(expfam, mu_star, mu_x) / d(expfam, mu, mu_x)
        return s - 1.0

    val = binary_search(f, 0.0, hi)

    ws = [
        1.0 if mu == mu_star else X(expfam, mu_star, mu, val)[0]
        for mu in mus
    ]

    S = sum(ws)
    return S / val, np.array(ws) / S
