!pip install deap
import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score
from deap import base, creator, tools, algorithms
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
bounded_run=True
# CONSTS #
RESULTS_FOLDER_NAME = "results"
epsilon = 1e-6  # Small constant to prevent division by zero
filter_cols = ["country"]
initial_guess = [
    1,        # bui_const1
    0.8,      # bui_const2
    0.4,      # bui_const3
    0.92,     # bui_const4
    0.0114,   # bui_const5
    1.7,      # bui_const6
    0.434,    # fwi_const1
    0.537,    # fwi_const2
    -1.233,   # fwi_const3
    0.208,    # isi_const1
    0.05039,  # f_U_const1
    91.9,     # f_F_const1
    -0.1386,  # f_F_const2
    5.31,     # f_F_const3
    4.93e7    # f_F_const4
]  # Initial guess for the parameters
# END - CONSTS #

bounds_param1=0.9
bounds_param2=1.1

def print_red(text):
    """Print text in red."""
    print(f"\033[91m{text}\033[0m")

def print_green(text):
    """Print text in green."""
    print(f"\033[92m{text}\033[0m")

#FULL FORMULA                            

def model(inputs, bui_const1, bui_const2, bui_const3, bui_const4, bui_const5, bui_const6,
          fwi_const1, fwi_const2, fwi_const3, isi_const1, f_U_const1, f_F_const1,
          f_F_const2, f_F_const3, f_F_const4):
    wind_speed, ffmc_values, dmc_values, dc_values = inputs
    DMC_DC_THRESHOLD=0.4
    bui_values = np.where(
        dmc_values <= DMC_DC_THRESHOLD * dc_values,
        bui_const2 * dmc_values * dc_values / (dmc_values + DMC_DC_THRESHOLD * dc_values),
        dmc_values - (bui_const1 - bui_const2 * dc_values / (dmc_values + bui_const3 * dc_values)) * (bui_const4 + (bui_const5 * dmc_values) ** bui_const6)
    )
    return 1 / (1 + np.exp(-1 * np.exp(fwi_const1 * np.log(np.maximum(epsilon,
            isi_const1 * np.exp(f_U_const1 * wind_speed) * f_F_const1 * np.exp(
                f_F_const2 * 147.2 * (101 - ffmc_values) / (59.5 + ffmc_values)
            ) * (1 + (147.2 * (101 - ffmc_values) / (59.5 + ffmc_values)) ** f_F_const3) / f_F_const4
            + epsilon
        )) + fwi_const2 * np.log(np.maximum(epsilon,
            bui_values)
        ) + fwi_const3)))


# Fitness function for the Genetic Algorithm (AUC)
def fitness_function(individual, wind_speed, ffmc_values, dmc_values, dc_values, y_values):
    fitted_y_values = model(
        (np.array(wind_speed), np.array(ffmc_values), np.array(dmc_values), np.array(dc_values)),
        *individual
    )

    # Calculate AUC
    try:
        auc = roc_auc_score(y_values, fitted_y_values)
        return auc,  # Return as tuple
    except ValueError:
        return 0,  # Return 0 AUC if calculation fails


def run_ga(wind_speed, ffmc_values, dmc_values, dc_values, y_values):
    # Define the individual (parameters to optimize) and fitness function
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Maximize AUC
    creator.create("Individual", list, fitness=creator.FitnessMax)
    if bounded_run==True:
    # Define the toolbox for evolutionary algorithms
        toolbox = base.Toolbox()
    
        # Compute bounds for each parameter as ±X% of their initial values
        lower_bounds = [val * bounds_param1 if val >= 0 else val * bounds_param2 for val in initial_guess]
        upper_bounds = [val * bounds_param2 if val >= 0 else val * bounds_param1 for val in initial_guess]
    
        # Register attributes with constrained bounds
        def bounded_random():
            return [np.random.uniform(lb, ub) for lb, ub in zip(lower_bounds, upper_bounds)]
    
        toolbox.register("attr_float", bounded_random)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", tools.cxBlend, alpha=0.25)  # Crossover method
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)  # Mutation with small perturbations
        toolbox.register("select", tools.selTournament, tournsize=10)  # Selection method
        toolbox.register("evaluate", fitness_function, wind_speed=wind_speed, ffmc_values=ffmc_values,
                         dmc_values=dmc_values, dc_values=dc_values, y_values=y_values)
    else:
        # Define the toolbox for evolutionary algorithms
        toolbox = base.Toolbox()
        toolbox.register("attr_float", np.random.uniform, -2, 2)  # Random float between -2 and 2 for initial population
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=len(initial_guess))
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", tools.cxBlend, alpha=0.25)  # Crossover method
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1.5, indpb=0.2)  # Mutation
        toolbox.register("select", tools.selTournament, tournsize=10)  # Selection method
        toolbox.register("evaluate", fitness_function, wind_speed=wind_speed, ffmc_values=ffmc_values,
                         dmc_values=dmc_values, dc_values=dc_values, y_values=y_values)

    # Create the population
    population = toolbox.population(n=100)  # Population size

    # Replace the first individual with the initial_guess
    population[0][:] = initial_guess

    # Run the Genetic Algorithm
    algorithms.eaSimple(population, toolbox, cxpb=0.75, mutpb=0.25, ngen=100, stats=None, halloffame=None)

    # Extract the best individual
    best_individual = tools.selBest(population, 1)[0]
    return best_individual

def remove_nan_inf(list1, list2):
  # Convert lists to NumPy arrays for efficient operations
  array1 = np.array(list1)
  array2 = np.array(list2)
  # Create a boolean mask to identify valid indices
  valid_mask = ~(np.isnan(array1) | np.isinf(array1) | np.isnan(array2) | np.isinf(array2))
  # Filter the lists using the mask
  filtered_list1 = array1[valid_mask]
  filtered_list2 = array2[valid_mask]
  return filtered_list1, filtered_list2


def main():

    try:
        os.mkdir(os.path.join(os.path.dirname(__file__), RESULTS_FOLDER_NAME))
        print("Created results folder")
    except:
        pass

    print("Loading data...")

    data = pd.read_csv('fire_data.csv')
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    data=train_data
    data['burned'] = (data['burned'] > 0).astype(int)

    start_time = time.time()  # Start the timer

    for filter_col in filter_cols:
        print(f"\n\n\nWorking on {filter_col}\n\n")
        unique_values = data[filter_col].unique()

        # Prepare results list to store the results for CSV
        results = []

        value_index = 1
        for i, value in enumerate(unique_values):
            subset = data[data[filter_col] == value]

            # Ensure data consistency
            wind_speed = np.sqrt(subset['u']**2 + subset['v']**2) * 3.6
            ffmc_values = subset['FFMC']
            dmc_values = subset['duff_moisture']
            dc_values = subset['drought']
            y_values = subset['burned'].values

            try:
                # Perform GA optimization for the model
                best_individual = run_ga(wind_speed, ffmc_values, dmc_values, dc_values, y_values)
            except Exception as e:
                print(f"Error fitting model for {filter_col}={value}: {e}")
                best_individual = initial_guess

            fitted_y_values = model((np.array(wind_speed), np.array(ffmc_values), np.array(dmc_values), np.array(dc_values)), *best_individual)
            fitted_y_values = np.nan_to_num(fitted_y_values, nan=0.0)

            try:
                auc_score = roc_auc_score(y_values, fitted_y_values)
            except ValueError:
                try:
                    fitted_y_values = model(
                        (np.array(wind_speed), np.array(ffmc_values), np.array(dmc_values), np.array(dc_values)),
                        *initial_guess)
                    #y_values, fitted_y_values = remove_nan_inf(y_values, fitted_y_values)
                    fitted_y_values = np.nan_to_num(fitted_y_values, nan=0.0)
                    auc_score = roc_auc_score(y_values, fitted_y_values)
                except Exception as error:
                    print(f"Error {error} in country {value}")
                    auc_score = 0.5  # Set to 0 if AUC calculation fails


            # Train a linear regression model
            try:
                y_values = np.array(y_values).astype(int)
                fitted_y_values = np.array(fitted_y_values.reshape(-1, 1))

                # Filter out pairs with NaN values
                cleaned_data = [
                    (y, fitted_y) for y, fitted_y in zip(y_values, fitted_y_values)
                    if not (np.isnan(y) or np.isnan(fitted_y))
                ]

                # Unzip the cleaned data
                y_values, fitted_y_values = zip(*cleaned_data)

                logistic_reg = LogisticRegression(max_iter=1000)
                logistic_reg.fit(fitted_y_values, y_values)

                # Predict probabilities using the logistic regression model
                predicted_probabilities = logistic_reg.predict_proba(fitted_y_values)[:, 1]

                # Calculate AUC for the logistic regression
                auc_score_logistic = roc_auc_score(y_values, predicted_probabilities)

            except Exception as ex:
                auc_score_logistic = 0.5
                print_red(f"Error plotting results for {filter_col}={value}: {ex}")

            if auc_score > 1:
                auc_score = 1
            if auc_score_logistic > 1:
                auc_score_logistic = 1

            print(f"filter = {filter_col}, value= {value}, auc={auc_score:.3f}, auc_logistic:{auc_score_logistic:.3f}")
            vals = [filter_col, value, len(y_values), auc_score, auc_score_logistic, len(np.unique(fitted_y_values)) == 1]
            vals.extend(best_individual)
            results.append(vals)

            # Save results to CSV
            cols = ["filter_col", "value", "data_size", "AUC", "AUC_logistic", "is_all_same_predict"]
            cols.extend("bui_const1, bui_const2, bui_const3, bui_const4, bui_const5, bui_const6, fwi_const1, fwi_const2, fwi_const3, isi_const1, f_U_const1, f_F_const1, f_F_const2, f_F_const3, f_F_const4".split(", "))
            results_df = pd.DataFrame(results, columns=cols)
            results_df.to_csv('constants_results_files.csv', index=False)
            print(f"Results saved for {filter_col}.")

    print(f"Total time taken: {time.time() - start_time:.2f} seconds")


if __name__ == '__main__':
    main()
