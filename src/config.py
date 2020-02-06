
## training hyper parameters
config = {}

config["input_dir_path"] = "../data/a9a"
# config["input_dir_path"] = "../data/gisette"
config["num_workers"] = 2
config["output_dir_path"] = "../result"

# config["privacy_budget"] = 1.0
config["is_with_noise"] = False
config["espilon"] = 3
config["delta"] = 1
config["noise_scale"] = 0
config["noise_method"] = "variable" # variable or result, variable adds noise to x while result adds noise to Dmxm
config["noise_eval_method"] = "fixed" # computed or fixed, computed uses the gurantee in paper while fixed added noise directly with fixed level

config["max_iter"] = 100
config["rho"] = 1
config["lambda"] = 0.001

config["model"] = "lr" # lr ln svm 
# config["z_solver"] = "BFGS"
# config["x_solver"] = "Newton-CG"
config["x_solver"] = "L-BFGS-B" # best solver

config["is_parallel"] = False
config["is_verbose"] = True
config["num_cpus"] = 2

config["is_cache_inverse_and_speed"] = True 