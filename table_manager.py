from utils import *
from tqdm.notebook import tqdm
from main_algo import adaptive_design
from allocation_functions import target_allocation
from joblib import Parallel, delayed

latex_plot_path = "plots/"

d_path = {"fixed_target": ["version_list", 
                           "tables/simulations_fixed_target.tex",
                           "alloc_name_list", 
                          r"\caption{Type-1 error (\%) for different targeting designs with fixed target allocation "], 
          "fixed_targeting": ["alloc_name_list", 
                              "tables/simulations_fixed_targeting.tex", 
                              "version_list",
                              r"\caption{Type-1 error (\%) for different target allocations with fixed targeting design "],
            "fixed_target_diff": ["version_list", 
                           "tables/simulations_fixed_target_diff.tex",
                           "alloc_name_list", 
                          r"\caption{Type-1 error (\%) for different targeting designs with fixed target allocation "], 
          "fixed_targeting_diff": ["alloc_name_list", 
                              "tables/simulations_fixed_targeting_diff.tex", 
                              "version_list",
                              r"\caption{Type-1 error (\%) for different target allocations with fixed targeting design "],
            "sparse_target": ["alloc_name_list",
                            "tables/simulations_fe_sparse.tex", 
                            "version_list",
                           r"\caption{Simulated allocation proportions for different targeting designs with fixed target allocation "]
                           }



def get_var_mult_entries(var_dict, specific = ""):
    for ky in var_dict.keys():
        if len(var_dict[ky]) > 1:
            return ky
    return specific


def key_d(output_name, k):
    if output_name == "proportions":
        return "N_{}/n".format(k)
    if output_name == "rho":
        return "rho_{}".format(k)
    return ""


def key_p(n_treatments):
    tt = ["p{}".format(k) for k in range(n_treatments)]
    return "(" + ",".join(tt) +")"


def array_to_str(lst):
    tt = [f"{val}" for val in lst]
    return "(" + ",".join(tt) +")"


def add_caption_and_save(latex, d_data, setting_type, n_simulations, n_patients):
    latex += "\t" + d_path[setting_type][3]
    latex += "({}), ".format(d_data[d_path[setting_type][2]][0]) 
    latex += "{} simulations, n = {}".format(n_simulations, n_patients) + "}\n"
    latex += "\t" + r"\label{tab:" + "-".join(setting_type.split("_")) + "}" + "\n"
    latex += r"\end{table}"
    latex += r"\end{document}"
    with open(d_path[setting_type][1], "w") as f:
        f.write(latex)
    print("Check output in {}".format(d_path[setting_type][1]))


def write_table(power_res, d_data, true_probs_arr, true_probs_null, n_simulations, n_patients, setting_type = "fixed_target", return_text = False):
    n_treatments = len(true_probs_arr[0])
    n_scenarios = len(d_data[d_path[setting_type][0]])
    latex = r"\documentclass{article}" + "\n" + r"\usepackage{booktabs}" + "\n" + r"\usepackage{multirow}" + "\n" + r"\usepackage{graphicx}" + "\n" + r"\begin{document}" + "\n"
    latex += r"\begin{table}[h]" + "\n\t" + r"\centering" + "\n\t" + r"\small" + "\n\t" + r"\resizebox{\textwidth}{!}{%" + "\n\t"
    latex += r"\begin{tabular}{c|" + n_scenarios * "c" + "|" + n_scenarios * "c" + "}" + "\n" + "\t\t" + r"\hline" + "\n"
    latex += "\t\t" + r"& \multicolumn{" + str(n_scenarios) + r"}{c|}{\(\alpha.RTS\)} & \multicolumn{" + str(n_scenarios) + r"}{c}{\(\alpha.RTS-FE\)} \\" + "\n"
    latex += "\t\t$(" + ", ".join(["p_{}".format(k) for k in range(n_treatments)]) + ")$" + "\n"
    latex += "\t\t& " + " & ".join(d_data[d_path[setting_type][0]]) + "\n"
    latex += "\t\t& " + " & ".join(d_data[d_path[setting_type][0]]) + " \\\\" + "\n"
    latex += "\t\t" + r"\hline" + "\n"
    for en, true_probs in enumerate(true_probs_arr):    
        true_probs_str = str(tuple(float(x) for x in true_probs))
        line = "\t\t" + true_probs_str + " & "
        for burn_in in d_data["burn_in_list"]:
            for use_forced_exp in d_data["forced_exp_list"]:
                for alloc_name in d_data["alloc_name_list"]:
                    for version in d_data["version_list"]:
                        value = power_res["power"][true_probs_str][alloc_name][version][burn_in][use_forced_exp]
                        line += value[:-1] + r"\%" + " & "
        latex += line[:-2] + "\\\\" + "\n"
        if en == len(true_probs_null) - 1:
            latex += "\t\t" + r"\hline" + "\n"
    latex += "\t\t" + r"\hline" + "\n"
    latex += "\t" + r"\end{tabular}" + "\n"  
    latex += "}" + "\n"    
    add_caption_and_save(latex, d_data, setting_type, n_simulations, n_patients)
    generate_pdf(d_path[setting_type][1], "tables")
    if return_text:
        return latex
    return None


def write_table_sparse(power_res, d_data, true_probs_arr, true_probs_null, n_simulations, n_patients):
    n_treatments = len(true_probs_arr[0])
    str_prop = " & ".join([r"$\frac{N_" + str(k+1) + r"}{n}$" for k in range(n_treatments)])    
    latex = r"\documentclass{article}" + "\n" + r"\usepackage{booktabs}" + "\n" + r"\usepackage{multirow}" + "\n" + r"\usepackage{graphicx}" + "\n" + r"\begin{document}" + "\n"
    latex += r"\begin{table}[h]" + "\n\t" + r"\centering" + "\n\t" + r"\small" + "\n" 
    for version in d_data["version_list"]:
        latex += "\t" + r"\begin{tabular}{c|" + n_treatments*"c" + "|" + n_treatments*"c" + r"}" + "\n" + "\t\t" + r"\hline" + "\n"
        latex += "\t\t" + r"\textbf{" + version + r"}" + r" & \multicolumn{" + str(n_treatments) + r"}{c|}{\(\alpha.RTS\)} & \multicolumn{" + str(n_treatments) + r"}{c}{\(\alpha.RTS-FE\)} \\" + "\n"
        latex += "\t\t$(" + ", ".join(["p_{}".format(k+1) for k in range(n_treatments)]) + ")$" + "\n"
        latex += "\t\t& " + str_prop + "\n"
        latex += "\t\t& " + str_prop + " \\\\" + "\n"
        latex += "\t\t" + r"\hline" + "\n"
        for en, true_probs in enumerate(true_probs_arr):    
            true_probs_str = str(tuple(float(x) for x in true_probs))
            line = "\t\t" + true_probs_str + " & "
            for burn_in in d_data["burn_in_list"]:
                for use_forced_exp in d_data["forced_exp_list"]:
                    for alloc_name in d_data["alloc_name_list"]:                    
                        for k in range(n_treatments):
                            value = power_res["outputs"][true_probs_str][alloc_name][version][burn_in][use_forced_exp][k]
                            line += value["proportions"].split()[0] + " & "
            latex += line[:-2] + "\\\\" + "\n"
            if en == len(true_probs_null) - 1:
                latex += "\t\t" + r"\hline" + "\n"
        latex += "\t\t" + r"\hline" + "\n"
        latex += "\t" + r"\end{tabular}" + "\n"
        latex += "\t\n" + r"\vspace{0.4cm}" + "\n\n"

    add_caption_and_save(latex, d_data, "sparse_target", n_simulations, n_patients)
    generate_pdf(d_path["sparse_target"][1], "tables")

def run_one_simulation(sim, burn_in, alpha, n_patients, true_probs, version, alloc_name, forced_exp, forced_exp_name, config, observed_outputs):
    np.random.seed(42 + sim)
    result = adaptive_design(burn_in,
                    alpha,
                    n_patients,
                    true_probs,
                    version,
                    alloc_name,
                    forced_exp,
                    forced_exp_name,                    
                    config=config
                )
    
    dict_res = {"reject": np.array(result["reject_trajectory"][-1]),        
                "assignment": np.array(result["assignments_number"]),
            }
    for obs_output in observed_outputs:
        dict_res[obs_output] = result[obs_output]
    return dict_res


class TableManager:
    def __init__(self, d_data):
        self.true_probs_arr = d_data["true_probs_arr"]
        self.alloc_name_list = d_data["alloc_name_list"]
        self.version_list = d_data["version_list"]
        self.forced_exp_list = d_data["forced_exp_list"]
        self.forced_exp_names = d_data["forced_exp_names"]
        self.burn_in_list = d_data["burn_in_list"]
        self.K = len(d_data["true_probs_arr"][0])
    
    def generate_table(self, alpha, n_patients, n_simulations, config = 1, record_power = True, observed_outputs = ["proportions"], use_parallel = False):
        d_v_star = {}
        d_res_outputs = {}
        d_res_power = {}
        for true_probs in tqdm(self.true_probs_arr):
            # s = "(" + ", ".join(map(str, map(float, lst))) + ")"
            true_probs_str = str(tuple(float(x) for x in true_probs))
            d_res_power[true_probs_str] = {}
            d_res_outputs[true_probs_str] = {}
            d_v_star[true_probs_str] = {}
            for alloc_name in self.alloc_name_list:
                d_res_power[true_probs_str][alloc_name] = {}
                d_res_outputs[true_probs_str][alloc_name] = {}
                d_v_star[true_probs_str][alloc_name] = target_allocation(true_probs, alloc_name)
                for version in self.version_list:
                    d_res_power[true_probs_str][alloc_name][version] = {}
                    d_res_outputs[true_probs_str][alloc_name][version] = {}
                    for burn_in in self.burn_in_list:
                        d_res_power[true_probs_str][alloc_name][version][burn_in] = {}
                        d_res_outputs[true_probs_str][alloc_name][version][burn_in] = {}
                        for use_forced_exp in self.forced_exp_list:
                            for forced_exp_name in self.forced_exp_names:
                                rejects = []
                                proportions = {obs_output: [] for obs_output in observed_outputs}
                                assignments = []
                                np.random.seed(42)

                                ############################################
                                if not use_parallel:
                                    for _ in range(n_simulations):
                                        result = adaptive_design(burn_in, 
                                                                    alpha, 
                                                                    n_patients, 
                                                                    true_probs, 
                                                                    version, 
                                                                    alloc_name, 
                                                                    use_forced_exp, 
                                                                    forced_exp_name, 
                                                                    config=config)                                        
                                        rejects.append(result["reject_trajectory"][-1])
                                        for obs_output in observed_outputs:
                                            proportions[obs_output].append(result[obs_output])
                                        assignments.append(result["assignments_number"])

                                ###########################################
                                ######### PARALLEL 
                                ###########################################
                                else:
                                    results = Parallel(n_jobs=-1)(
                                        delayed(run_one_simulation)(
                                            sim, 
                                            burn_in, 
                                            alpha, 
                                            n_patients, 
                                            true_probs, 
                                            version, 
                                            alloc_name, 
                                            use_forced_exp, 
                                            forced_exp_name,
                                            config,                                             
                                            observed_outputs
                                            )
                                        for sim in range(n_simulations)
                                    )
                                    for res in results:                                    
                                        rejects.append(res["reject"])
                                        for obs_output in observed_outputs:
                                            proportions[obs_output].append(res[obs_output])
                                        assignments.append(res["assignment"])
                                    ###########################################
                                ###########################################

                                power = np.mean(rejects)
                                d_res_power[true_probs_str][alloc_name][version][burn_in][use_forced_exp] = f"{power:.2%}"
                                d_res_outputs[true_probs_str][alloc_name][version][burn_in][use_forced_exp] = {}

                                avg_prop = {obs_output: np.mean(proportions[obs_output], axis=0) for obs_output in observed_outputs}
                                var = {obs_output: np.mean([[(x[k] - avg_prop[obs_output][k]) ** 2 for k in range(self.K)] for x in proportions[obs_output]], axis=0) for obs_output in observed_outputs}
                                for k in range(self.K):
                                    d_res_outputs[true_probs_str][alloc_name][version][burn_in][use_forced_exp][k] = {}
                                    for obs_output in observed_outputs:
                                        d_res_outputs[true_probs_str][alloc_name][version][burn_in][use_forced_exp][k][obs_output] = f"{avg_prop[obs_output][k]:.4f} ({n_patients * var[obs_output][k]:.4f})"
        return {"power": d_res_power, "outputs": d_res_outputs, "allocation_targets": d_v_star}    