from utils import *
from adaptive_designs import g_design
from allocation_functions import target_allocation


def h_func_forced_exp(t, K, forced_exp_name = "BAI"):
    if forced_exp_name == "BAI":
        return (np.sqrt(t) - K / 2) 
    if forced_exp_name == "type-1":
        return (np.pow(t, 1/3) - K / 2)
    if "type-2" in forced_exp_name:
        # the forced exploration function here is h(s) = (s / K+1)^(1/A) where A is as : "type-2-A"
        gamma = float(forced_exp_name.split("-")[-1])
        return np.pow(t / (K + 1), 1 / gamma)
    if "x:" in forced_exp_name:                
        func_h = eval(f"lambda {forced_exp_name}")
        return func_h(t)
    return None

def forced_exploration(x, K, t, forced_exp_name = "BAI"):
    check_probability(x)
    probabilities = np.zeros(K)
    U_t = []
    for i in range(K):
        if x[i] < h_func_forced_exp(t, K, forced_exp_name) / t:
            U_t.append((x[i], i))

    if len(U_t) > 0:
        min_idx = min(U_t, key=lambda x: x[0])[1]
        probabilities[min_idx] = 1
        return probabilities
    return []


def get_reject_result(n_patients, n_treatments, phat_by_treat, n_assign_by_treat, n_success_by_treat):
    if n_treatments == 2:
        return wald_reject(phat_by_treat, n_assign_by_treat)
    else:
        p_hat = sum(n_success_by_treat) / n_patients
        return pearson_reject(p_hat, phat_by_treat, n_assign_by_treat)


def update_data(outcomes, n_treatments, alloc_name, 
                n_assign_traj, rho_estimate_traj, rejects_traj, pre_burn_in = False):
    curr_assignment_nb = len(outcomes)
    n_assign_by_treat = np.array([sum(1 for a, _ in outcomes if a == k) for k in range(n_treatments)])
    n_success_by_treat = np.array([sum(y for a, y in outcomes if a == k) for k in range(n_treatments)])
    if pre_burn_in:
        phat_by_treat = np.array([0.5] * n_treatments)
    else:
        phat_by_treat = n_success_by_treat / n_assign_by_treat  
    target_rho = target_allocation(phat_by_treat, alloc_name)
    check_probability(target_rho)
    
    reject = get_reject_result(curr_assignment_nb, n_treatments, phat_by_treat, n_assign_by_treat, n_success_by_treat)
    
    n_assign_traj.append(n_assign_by_treat / curr_assignment_nb)
    rho_estimate_traj.append(target_rho)
    rejects_traj.append(reject)

    return {"n_assign_by_treat": n_assign_by_treat,  
            "target_rho": target_rho,
            "n_success_by_treat": n_success_by_treat, 
            "phat_by_treat": phat_by_treat,
            "reject": reject}

def adaptive_design(burn_in, 
                    alpha, 
                    n_patients, 
                    true_probs,
                    version, 
                    alloc_name,
                    use_forced_exp,
                    forced_exp_name, 
                    config = 0):
    n_treatments = len(true_probs)
    n_assign_traj = []
    rho_estimate_traj = []
    assignments = []
    outcomes = []
    rejects_traj = []

    if config == 0:
        for treat in range(n_treatments):
            outcomes.append((treat, 0.5))
            assignments.append(treat)
            d_res = update_data(outcomes, n_treatments, alloc_name, n_assign_traj, rho_estimate_traj, rejects_traj, pre_burn_in=True)

    for _ in range(burn_in):
        for treat in range(n_treatments):
            assignments.append(treat)
            y = np.random.binomial(1, true_probs[treat])
            outcomes.append((treat, y))
            d_res = update_data(outcomes, n_treatments, alloc_name, n_assign_traj, rho_estimate_traj, rejects_traj)

    
    while len(assignments) < n_patients:
        n_assign_by_treat = d_res["n_assign_by_treat"]
        target_rho = d_res["target_rho"]
        
        if use_forced_exp:
            allocation_probs = forced_exploration(n_assign_by_treat / len(assignments), n_treatments, len(assignments), forced_exp_name)
            if len(allocation_probs) == 0:
                allocation_probs = g_design(version, n_assign_by_treat / len(assignments), target_rho, alpha)

        else:
            allocation_probs = g_design(version, n_assign_by_treat / len(assignments), target_rho, alpha)

        check_probability(allocation_probs)

        a_next = np.random.choice(n_treatments, p=allocation_probs)
        y_next = np.random.binomial(1, true_probs[a_next])

        assignments.append(a_next)
        outcomes.append((a_next, y_next))
        # d_res = update_data(outcomes, n_treatments, alloc_name, n_assign_traj, rho_estimate_traj)
        d_res = update_data(outcomes, n_treatments, alloc_name, n_assign_traj, rho_estimate_traj, rejects_traj)


    result = {}
    result["proportions_trajectory"] = n_assign_traj
    result["rho_trajectory"] = rho_estimate_traj
    result["reject_trajectory"] = rejects_traj

    # if n_treatments == 2:
    #     reject = wald_reject(phat_by_treat, n_assign_by_treat)
    #     rejects_traj.append(reject)
    #     result["reject"] = rejects_traj
    # else:
    #     p_hat = sum(n_success_by_treat) / n_patients
    #     reject = pearson_reject(p_hat, phat_by_treat, n_assign_by_treat)
    #     rejects_traj.append(reject)
    #     result["reject"] = rejects_traj

    result.update({"proportions": n_assign_by_treat / len(assignments), 
                "rho": target_rho,
                "assignments_number": n_assign_by_treat})
    return result
