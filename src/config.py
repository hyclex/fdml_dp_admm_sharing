
## training hyper parameters
config = {}
config["input_dir_path"] = "../data/a9a"
config["num_workers"] = 2

config["lambda"] = 0.0000001
config["privacy_budget"] = 1.0

config["max_iter"] = 50
config["rho"] = 1

config["model"] = "lr" # lr ln svm 
config["z_solver"] = "BFGS" # BFGS
config["x_solver"] = "Newton-CG"

config["is_parallel"] = False
config["is_verbose"] = True
config["num_cpus"] = 2
