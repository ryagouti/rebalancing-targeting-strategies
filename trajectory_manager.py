from utils import *
from allocation_functions import target_allocation
from main_algo import adaptive_design
from joblib import Parallel, delayed
import matplotlib.pyplot as plt



# output_tex_map = {"prop_diff": r'$\frac{N_n}{n} - v$', "rho_diff": r'$\hat{\rho}_n - v$', "prop_rho_diff": r'$\frac{N_n}{n} - \hat{\rho}_n$', "power": "Power"}

latex_model_name = {False: r'$\alpha$RTS', True: r'$\alpha$RTS-FE'}
def output_tex_map(k, output_name, expectation = False):
    exp = ''
    exp_cl = ''
    if expectation:
        exp = r'\mathbb{E}\left['
        exp_cl = r'\right]'
    k += 1
    formula = r'$\times\sqrt{\frac{n}{loglog(n)}}$'
    formula_1 = r'$\times\sqrt{\frac{1}{n}}$'
    if output_name == "prop_diff":
        return r'$' + exp + r'\left|\frac{N_{n,' + str(k) + r'}}{n} - v_{' + str(k) + r'}\right|' + exp_cl + r'$'
    if output_name == "rho_diff":
        return r'$' + exp + r'\left|\hat{\rho}_{n,' + str(k) + r'} - v_{' + str(k) + r'}\right|' + exp_cl + r'$'
    if output_name == "prop_rho_diff":
        return r'$' + exp + r'\left|\frac{N_{n,' + str(k) + r'}}{n} - \hat{\rho}_{n,' + str(k) + r'}\right|' + exp_cl + r'$'
    
    if output_name == "prop_diff_scaled":
        return r'$\left|\frac{N_{n,' + str(k) + r'}}{n} - v_{' + str(k) + r'}\right|$' + formula
    if output_name == "rho_diff_scaled":
        return r'$\left|\hat{\rho}_{n,' + str(k) + r'} - v_{' + str(k) + r'}\right|$' + formula
    if output_name == "prop_rho_diff_scaled":
        return r'$\left|\frac{N_{n,' + str(k) + r'}}{n} - \hat{\rho}_{n,' + str(k) + r'}\right|$' + formula    
    
    if output_name == "prop_diff_scaled_1":
        return r'$\left|\frac{N_{n,' + str(k) + r'}}{n} - v_{' + str(k) + r'}\right|$' + formula_1
    if output_name == "rho_diff_scaled_1":
        return r'$\left|\hat{\rho}_{n,' + str(k) + r'} - v_{' + str(k) + r'}\right|$' + formula_1
    if output_name == "prop_rho_diff_scaled_1":
        return r'$\left|\frac{N_{n,' + str(k) + r'}}{n} - \hat{\rho}_{n,' + str(k) + r'}\right|$' + formula_1    
    
    if output_name == "power":
        return "Power"


def get_ci(bloc_mc_sq, bloc_mc, n_simulations):
    power_var = (bloc_mc_sq / n_simulations) - bloc_mc**2
    power_se = np.sqrt(power_var / n_simulations)                            
    power_ci_lower = bloc_mc - 1.96 * power_se
    power_ci_upper = bloc_mc + 1.96 * power_se

    return {"low": power_ci_lower, "up": power_ci_upper}
    
def run_one_simulation(sim, burn_in, alpha, n_patients, true_probs, version, alloc_name, forced_exp, forced_exp_name):
    np.random.seed(42 + sim)
    
    result = adaptive_design(burn_in,
                    alpha,
                    n_patients,
                    true_probs,
                    version,
                    alloc_name,
                    forced_exp,
                    forced_exp_name,
                    config=0
                )

    return {
        "prop": np.array(result["proportions_trajectory"]),
        "rho": np.array(result["rho_trajectory"]),
        "power": np.array(result["reject_trajectory"]),
    }


class TrajectoryManager:
    def __init__(self, var_dict, true_probs):
        self.true_probs = true_probs
        self.n_treatments = len(self.true_probs)
        self.var_dict = var_dict

    def core_computation(self, d_data, v_star, n_simulations, n_patients, alloc_name, version, burn_in, forced_exp, forced_exp_name, alpha, use_parallel):
        n_lst = [3, 3, 3] + list(range(3, n_patients)) 
        scale = np.sqrt(np.array(n_lst) / np.log(np.log(n_lst)))                    
        scale_1 = 1 / np.sqrt(np.array(n_lst))
        all_power = np.zeros((n_simulations, n_patients))                                
        bloc_mc_prop = np.array([[0. for k in range(self.n_treatments)] for _ in range(n_patients)])
        bloc_mc_rho = np.array([[0. for k in range(self.n_treatments)] for _ in range(n_patients)])
        bloc_mc_power = np.array([0. for _ in range(n_patients)])
        bloc_mc_power_sq = np.array([0. for _ in range(n_patients)])
        
        key_str = "{} {} {} {} {} {}".format(alloc_name, version, burn_in, forced_exp, forced_exp_name, alpha)
        ##################################################################################################################
        ##################################################################################################################    
        if not use_parallel:
            np.random.seed(42)
            for sim in range(n_simulations):
                np.random.seed(42 + sim)
                result = adaptive_design(burn_in, 
                                            alpha, 
                                            n_patients, 
                                            self.true_probs, 
                                            version, 
                                            alloc_name, 
                                            forced_exp, 
                                            forced_exp_name, 
                                            config=0)
                
                power_sim = np.array(result["reject_trajectory"])
                bloc_mc_prop += np.array(result["proportions_trajectory"])
                bloc_mc_rho += np.array(result["rho_trajectory"])
                bloc_mc_power += power_sim
                bloc_mc_power_sq += power_sim**2

        ##################################################################################################################
        ######### PARALLEL
        ##################################################################################################################                            
        else:
            results = Parallel(n_jobs=-1)(
                delayed(run_one_simulation)(
                    sim,
                    burn_in,
                    alpha,
                    n_patients,
                    self.true_probs,
                    version,
                    alloc_name,
                    forced_exp, 
                    forced_exp_name
                )
                for sim in range(n_simulations)
            )

            for res in results:
                bloc_mc_prop += res["prop"]                                                                
                bloc_mc_rho += res["rho"]                                
                bloc_mc_power += res["power"]
                bloc_mc_power_sq += res["power"]**2

        bloc_mc_prop /= n_simulations
        bloc_mc_rho /= n_simulations
        bloc_mc_power /= n_simulations                            
        
        ##################################################################################################################
        ##################################################################################################################

        # Monte Carlo standard error and 95% CI for power                                                        
        ci_power = get_ci(bloc_mc_power_sq, bloc_mc_power, n_simulations)
        power_ci_lower = ci_power["low"]
        power_ci_upper = ci_power["up"]

        # Percentile analysis
        power_pcts = {
            "5": np.percentile(all_power, 5, axis=0),
            "25": np.percentile(all_power, 25, axis=0),
            "50": np.percentile(all_power, 50, axis=0),
            "75": np.percentile(all_power, 75, axis=0),
            "95": np.percentile(all_power, 95, axis=0),
        }
        prop_diff = np.array([norm_v(bloc_mc_prop[:,k] - v_star[k]) for k in range(self.n_treatments)])
        rho_diff = np.array([norm_v(bloc_mc_rho[:,k] - v_star[k]) for k in range(self.n_treatments)])
        prop_rho_diff = np.array([norm_v(bloc_mc_prop[:,k] - bloc_mc_rho[:,k]) for k in range(self.n_treatments)])
        
        prop_diff_scaled = prop_diff * scale
        rho_diff_scaled = rho_diff * scale
        prop_rho_diff_scaled = prop_rho_diff * scale

        prop_diff_scaled_1 = prop_diff * scale_1
        rho_diff_scaled_1 = rho_diff * scale_1
        prop_rho_diff_scaled_1 = prop_rho_diff * scale_1

        d_data[key_str] = {"allocation_prop": bloc_mc_prop, 
                        "prop_diff": prop_diff, 
                        "rho_diff": rho_diff, 
                        "prop_rho_diff": prop_rho_diff, 
                        "prop_diff_scaled": prop_diff_scaled, 
                        "rho_diff_scaled": rho_diff_scaled, 
                        "prop_rho_diff_scaled": prop_rho_diff_scaled, 
                        "prop_diff_scaled_1": prop_diff_scaled_1, 
                        "rho_diff_scaled_1": rho_diff_scaled_1, 
                        "prop_rho_diff_scaled_1": prop_rho_diff_scaled_1, 
                        "power": bloc_mc_power, 
                        "power_ci_lower": power_ci_lower, 
                        "power_ci_upper": power_ci_upper, 
                        "percentiles": power_pcts}
        
    def generate_trajectory_data(self, n_patients, n_simulations, use_parallel = True):
        d_data = {}        
        for alloc_name in self.var_dict["alloc_name"]:
            v_star = target_allocation(self.true_probs, alloc_name)
            for version in self.var_dict["version"]:
                for burn_in in self.var_dict["burn_in"]:          
                    for forced_exp in self.var_dict["forced_exp"]:
                        for forced_exp_name in self.var_dict["forced_exp_names"]:
                            for alpha in self.var_dict["alpha"]:
                                self.core_computation(d_data, v_star, n_simulations, n_patients, alloc_name, version, burn_in, forced_exp, forced_exp_name, alpha, use_parallel)                              
        return d_data
    
    def generate_trajectory_data_2(self, n_patients, n_simulations, use_parallel = True):
        d_data = {}        
        for element in self.var_dict["alloc_duo"]:
            elt_sp = element.split("-")
            alloc_name = elt_sp[0]
            version = elt_sp[1]
            v_star = target_allocation(self.true_probs, alloc_name)            
            for burn_in in self.var_dict["burn_in"]:          
                for forced_exp in self.var_dict["forced_exp"]:
                    for forced_exp_name in self.var_dict["forced_exp_names"]:
                        for alpha in self.var_dict["alpha"]:
                            self.core_computation(d_data, v_star, n_simulations, n_patients, alloc_name, version, burn_in, forced_exp, forced_exp_name, alpha, use_parallel)                              
        return d_data
    
    
    def generate_plots(self, var_name, d_data, n_sim, alpha, path_name, outputs_list=["prop_diff", "rho_diff", "prop_rho_diff", "power"], start_idx_lst = [0, 0, 0, 0], line_width = 3, p_step = 1, f_s = 16, change_line_style = True, save_fig = False):
        d_line_style = ["-", "--", ":", "-."]
        if not change_line_style:
            d_line_style = ["-", "-", "-", "-"]
        for out_num, output_name in enumerate(outputs_list):
            start_idx = start_idx_lst[out_num]
            if output_name == "power":              
                fig, ax = plt.subplots(figsize=(10, 5))
                for en, var_value in enumerate(self.var_dict[var_name]):
                    key_str = gen_str(var_name, var_value, self.var_dict)
                    y = d_data[key_str][output_name][start_idx:]
                    y_lower = d_data[key_str]["power_ci_lower"][start_idx:]
                    y_upper = d_data[key_str]["power_ci_upper"][start_idx:]                    
                    x = np.arange(start_idx, start_idx + len(y))
                    if var_name == "alloc_duo":
                        label_plot = ''
                        if "Uniform" in var_value:
                            label_plot = "Uniform"
                        else:
                            label_plot = var_value.split("-")[0]

                        ax.plot(x[::p_step], y[::p_step], label="{}".format(label_plot), 
                                linestyle = d_line_style[en], 
                                linewidth = line_width)
                    elif var_name == "forced_exp":
                        ax.plot(x[::p_step], y[::p_step], label="{}".format(latex_model_name[var_value]), 
                                linestyle = d_line_style[en], 
                                linewidth = line_width)
                    else:
                        ax.plot(x[::p_step], y[::p_step], label="{}".format(var_value), 
                                linestyle = d_line_style[en], 
                                linewidth = line_width)
                    # ax.fill_between(x - start_idx, y_lower, y_upper, alpha=0.2)
                    ax.fill_between(x, y_lower, y_upper, alpha=0.2)
                    # ax.set_title('Power')
                    ax.set_xlabel('Number of Patients (n)')
                    ax.set_ylabel(output_tex_map(0, output_name))
                    ax.legend(title=r"$n_{sim}$" + r" = {}, $\alpha = {}$".format(n_sim, alpha))
                    # ax.legend(fontsize=f_s)
                    ax.grid(True)

                fig.canvas.draw()
                if save_fig:
                    extent = ax.get_tightbbox(fig.canvas.get_renderer())
                    fig.savefig(
                        f"{path_name}/power.pdf",
                        bbox_inches=extent.transformed(fig.dpi_scale_trans.inverted()),
                        transparent=True)
                    print("Check output in {}/power.pdf".format(path_name))
                plt.show()

            else:
                fig, axes = plt.subplots(1, self.n_treatments, figsize=(20, 5))
                for k in range(self.n_treatments):
                    for en, var_value in enumerate(self.var_dict[var_name]):
                        key_str = gen_str(var_name, var_value, self.var_dict)                        
                        y = d_data[key_str][output_name][k][start_idx:]
                        x = np.arange(start_idx, start_idx + len(y))                        
                        if var_name == "forced_exp":
                            axes[k].plot(x[::p_step], y[::p_step], label="{}".format(latex_model_name[var_value]), 
                                         linestyle = d_line_style[en], 
                                         linewidth = line_width)
                        else:                        
                            axes[k].plot(x[::p_step], y[::p_step], label="{}".format(var_value), 
                                         linestyle = d_line_style[en], 
                                         linewidth = line_width)
                        
                        axes[k].set_xlabel('Number of Patients (n)')
                        
                        if var_name == "forced_exp":
                            axes[k].set_ylabel(output_tex_map(k, output_name, True))
                            axes[k].legend(title=r"$n_{sim}$" + r" = {}, $\alpha = {}$".format(n_sim, alpha))
                        else:
                            axes[k].set_ylabel(output_tex_map(k, output_name))
                            axes[k].legend(title=r"$\alpha = {}$".format(alpha), fontsize=f_s)
                        axes[k].grid(True)

                if save_fig:
                    for k, ax in enumerate(axes):
                        extent = ax.get_tightbbox(fig.canvas.get_renderer())
                        fig.savefig(f"{path_name}/{output_name}_{k}.pdf", 
                                    bbox_inches=extent.transformed(fig.dpi_scale_trans.inverted()), 
                                    transparent=True)
                        print("Check output in {}/{}_{}.pdf".format(path_name, output_name, k))
                plt.show()

