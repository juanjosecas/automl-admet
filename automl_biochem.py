import argparse
import random

from src.grammar_boa_gp import GrammarBayesOptGeneticfProgAlgorithm
from src.bnf_grammar_parser import BNFGrammar
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AutoML for Biochemical Property Prediction"
    )

    # Mandatory arguments
    parser.add_argument("training_path", type=str, help="Path to the training dataset")
    parser.add_argument("testing_path", type=str, help="Path to the testing dataset")
    parser.add_argument("grammar_path", type=str, help="Path to the grammar defining the AutoML search space")
    parser.add_argument("output_dir", type=str, help="Output directory")
    
    # Optional arguments 
    parser.add_argument("-s", "--seed", type=int, default=1, help="The seed for reproducibility")
    parser.add_argument("-m", "--metric", type=str, default="auc", help="The metric for optimization")
    parser.add_argument("-e", "--exp_name", type=str, default="Exp_ADMET", help="The name of the experiment")
    parser.add_argument("-t", "--time", type=int, default=5, help="Time in minutes to run the method")
    parser.add_argument("-n", "--ncores", type=int, default=20, help="Number of cores")
    parser.add_argument("-ta", "--time_budget_minutes_alg_eval", type=int, default=1, help="Time to assess each individual ML pipeline")
    parser.add_argument("-p", "--population_size", type=int, default=100, help="Population size")
    parser.add_argument("-mr", "--mutation_rate", type=float, default=0.15, help="Mutation rate")
    parser.add_argument("-cr", "--crossover_rate", type=float, default=0.80, help="Crossover rate")
    parser.add_argument("-cmr", "--crossover_mutation_rate", type=float, default=0.05, help="Crossover and mutation rate")
    parser.add_argument("-es", "--elitism_size", type=int, default=1, help="Elitism size")

    # Parse arguments
    args = parser.parse_args()
    
    # Extract parameters
    training_path = args.training_path
    testing_path = args.testing_path
    grammar_path = args.grammar_path
    
    seed = args.seed
    metric = args.metric
    expname = args.exp_name
    maxtime = args.time
    ncores = args.ncores
    timebudgetminutesalgeval = args.time_budget_minutes_alg_eval
    populationsize = args.population_size
    mutationrate = args.mutation_rate
    crossoverrate = args.crossover_rate
    crossovermutationrate = args.crossover_mutation_rate
    elitismsize = args.elitism_size

    # Set random seed for reproducibility
    random.seed(seed)

    # Load grammar from file
    with open(grammar_path, "r") as file:
        grammar_text = file.read()

    # Parse grammar
    grammar = BNFGrammar()
    grammar.load_grammar(grammar_text)

    # Run evolutionary AutoML
    ggp = GrammarBayesOptGeneticfProgAlgorithm(
        grammar, training_path, testing_path, 
        fitness_metric=metric, experiment_name=expname, seed=seed, 
        max_time=maxtime, num_cores=ncores, 
        time_budget_minutes_alg_eval=timebudgetminutesalgeval, 
        population_size=populationsize, mutation_rate=mutationrate, 
        crossover_rate=crossoverrate, 
        crossover_mutation_rate=crossovermutationrate,
        elitism_size=elitismsize
    )
    best_program = ggp.evolve()
