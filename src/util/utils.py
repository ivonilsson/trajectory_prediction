from tabulate import tabulate
import os

def save_results(file_path, results):
    # save results
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    table = tabulate(results, headers="keys", tablefmt="fancy_grid")
    with open(file_path, 'w') as file:
        file.write(table)