
## training hyper parameters
config = {}

config["input_dir_path"] = "../data/a9a"
# config["input_dir_path"] = "../data/gisette"
config["num_workers"] = 2
config["output_dir_path"] = "../result"

config["lambda"] = 0.001
# config["privacy_budget"] = 1.0
config["is_with_noise"] = True
config["espilon"] = 3
config["delta"] = 1
config["noise_scale"] = 0

config["max_iter"] = 2
config["rho"] = 1

config["model"] = "lr" # lr ln svm 
config["z_solver"] = "BFGS" # BFGS
# config["x_solver"] = "Newton-CG"
config["x_solver"] = "L-BFGS-B"

config["is_parallel"] = False
config["is_verbose"] = True
config["num_cpus"] = 2
