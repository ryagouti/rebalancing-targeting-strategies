from utils import *

def g_erade(x, y, alpha):
    p = y[0]
    if x[0] > y[0]:
        p = alpha * y[0]
    if x[0] < y[0]:
        p = 1 - alpha * (1 - y[0])
    return np.array([p, 1-p])

def g_uniform(x):
    K = len(x)
    return np.array([1/K] * K)

def g_erade_2025(x, y, alpha):
    K = len(x)
    if sum(np.abs(x - y)) <= 1e-10:
        return np.array([1/K] * K)

    probabilities = np.zeros(K)
    S = []
    T = []
    for i in range(K):
        if x[i] > y[i]:
            S.append(i)
        if x[i] < y[i]:
            T.append(i)

    for i in range(K):
        if x[i] == y[i]:
            probabilities[i] = y[i]
        elif x[i] > y[i]:
            probabilities[i] = alpha * y[i]
        else:
            probabilities[i] = (1 - alpha) * sum(y[j] for j in S) / len(T) + y[i]

    return probabilities


def g_erade_distance(x, y, alpha):
    K = len(x)
    probabilities = np.zeros(K)
    props = [max(e, 0) for e in y - x]
    if sum(props) == 0:
        props_arr = y
    else:
        props_arr = np.array(props) / sum(props)

    for k in range(K):
        probabilities[k] = alpha * y[k] + (1 - alpha) * props_arr[k]

    return probabilities

def g_tas(x, y, alpha):
    # K = len(x)
    # probabilities = np.zeros(K)
    # max_idx = max(enumerate(y - x), key=lambda x: x[1])[0]
    # probabilities[max_idx] = 1
    # return probabilities

    deficits = y - x
    k_star = np.argmax(deficits)

    probabilities = alpha * y.copy()
    probabilities[k_star] += (1 - alpha)

    return probabilities


def g_design(version, x, y, alpha):
    check_probability(x)
    if version == "ERADE" and len(x) == 2:
        return g_erade(x, y, alpha)

    if version == "Distance":
        return g_erade_distance(x, y, alpha)

    if version == "ERADE2025":
        return g_erade_2025(x, y, alpha)

    if version == "D-Tracking":
        return g_tas(x, y, alpha)

    if version == "Uniform":
        return g_uniform(x)
    
    if version == "rho":
        return y
    
    return "unknown design"