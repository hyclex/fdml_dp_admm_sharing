
## training hyper parameters
config = {}
config["input_dir_path"] = "../data/a9a"
config["num_workers"] = 2

config["lambda"] = 0.000001
config["privacy_budget"] = 1.0

config["max_iter"] = 2
config["rho"] = 1

config["is_verbose"] = True
config["model"] = "lr" # lr ln svm 
config["z_solver"] = "Newton-CG" # BFGS
config["x_solver"] = "Newton-CG"
