import optunity

def make_PSO(Opt_Hyperparameter, ML_Hyperparameter):
    '''return optunity.make_solver(solver_name = 'particle swarm', 
                                num_particles = int(Opt_Hyperparameter['num_particles']), 
                                num_generations = int(Opt_Hyperparameter['num_generations']),
                                **ML_Hyperparameter)'''
    return optunity.solvers.ParticleSwarm(num_particles = int(Opt_Hyperparameter['num_particles']), 
                                  num_generations = int(Opt_Hyperparameter['num_generations']),
                                  **ML_Hyperparameter)

def run_PSO(solver, ML_algorithm):
    # recieve and maximize(run) solver
    # return accuracy or AUROC according to ML_algorithm (CNN, SVM, RF) code 
    return solver.maximize(ML_algorithm)
