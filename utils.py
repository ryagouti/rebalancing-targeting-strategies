import numpy as np
import math
from scipy.stats import chi2, norm
import subprocess
from pathlib import Path


def check_probability(x):
    length = len(x)
    try:
        np.random.choice(length, p=x)
    except ValueError as e:
        if "probabilities do not sum to 1" in str(e):
            print("Bad probabilities:", x)
            print("Sum =", x.sum())
        if "probabilities contain NaN" in str(e):
            print("Bad probabilities:", x)
        raise
    
def gen_str(var_name, var_value, var_dict):
    s = ""
    for key in var_dict.keys():
        if key == "alloc_duo":
            var_splt = var_value.split("-")
            s += str(var_splt[0]) + " "
            s += str(var_splt[1]) + " "
        elif key == var_name:
            s += str(var_value) + " "
        else:
            s += str(var_dict[key][0]) + " "
    return s[:-1]

def norm_v(matrix):
    res = []
    for l in matrix:
        res.append(np.linalg.norm(l))
    return np.array(res)


def sign(x, y):
    if x > y:
        return 1
    if x < y:
        return -1
    return 0

sign_v = np.vectorize(sign)


def wald_reject(phat, n_list, alpha=0.05):
    if len(phat) > 2:
        return "Wald test cannot be conducted with more that 2 arms"
    phat0, phat1 = phat
    q0, q1 = 1 - phat0, 1 - phat1
    n0, n1 = n_list
    denom = math.sqrt((phat0 * q0) / max(n0, 1) + (phat1 * q1) / max(n1, 1))
    z = (phat1 - phat0) / denom

    zquant = norm.ppf(1 - alpha / 2)
    return abs(z) >= zquant


def pearson_reject(p_hat, phat_by_treat, n_assign_by_treat):
    nom = sum(n_assign_by_treat * (phat_by_treat - p_hat) ** 2)
    den = p_hat * (1 - p_hat)
    df = len(phat_by_treat) - 1
    cquant = chi2.ppf(0.95, df)
    return nom / den >= cquant


class Gaussian:
    def __init__(self, sigma2):
        self.sigma2 = sigma2

class Bernoulli:
    def __init__(self, proba):
        self.proba = proba

def d(expfam, mu1, mu2):
    # Gaussian KL: same variance
    if isinstance(expfam, Gaussian):
        return (mu1 - mu2) ** 2 / (2.0 * expfam.sigma2)

    # Bernoulli KL
    if isinstance(expfam, Bernoulli):
        eps = 1e-12  # numerical safety
        mu1 = min(max(mu1, eps), 1 - eps)
        mu2 = min(max(mu2, eps), 1 - eps)
        return (
            mu1 * math.log(mu1 / mu2)
            + (1 - mu1) * math.log((1 - mu1) / (1 - mu2))
        )

    raise NotImplementedError(f"KL not implemented for {type(expfam)}")

# def d(expfam, mu1, mu2):
#     sigma2 = expfam["sigma2"]   # variance
#     return (mu1 - mu2) ** 2 / (2.0 * sigma2)

def binary_search(f, lo, hi, eps=1e-10, maxiter=100):
    assert f(lo) < +eps, f"f({lo})={f(lo)} should be negative at low end"
    assert f(hi) > -eps, f"f({hi})={f(hi)} should be positive at high end"

    for _ in range(maxiter):
        mid = (lo + hi) / 2.0

        if mid == lo or mid == hi:
            # not going to get any better
            return mid

        fmid = f(mid)

        if fmid < -eps:
            lo = mid
        elif fmid > eps:
            hi = mid
        else:
            return mid

    # did not reach tolerance
    return (lo + hi) / 2.0


def X(expfam, mu1, mui, v):
    kl1i = d(expfam, mu1, mui)  # range of V(x) is [0, kl1i]
    assert 0 <= v <= kl1i, f"0 <= {v} <= {kl1i}"

    def f(z):
        muz = (1 - z) * mu1 + z * mui
        return ((1 - z) * d(expfam, mu1, muz)
                + z * d(expfam, mui, muz)
                - (1 - z) * v)

    alpha = binary_search(f, 0.0, 1.0, eps=kl1i * 1e-10)

    return alpha / (1 - alpha), (1 - alpha) * mu1 + alpha * mui


def generate_pdf(tex_file, directory_name="tables"):
    output_dir = Path(directory_name)
    subprocess.run(
        [
            "pdflatex",
            "-interaction=nonstopmode",
            "-output-directory",
            str(output_dir),
            str(tex_file),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
)