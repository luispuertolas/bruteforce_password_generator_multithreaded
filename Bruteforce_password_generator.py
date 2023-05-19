from multiprocessing import Pool, cpu_count
import itertools

# Define your characters and numbers
characters = list('abcefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ._-1234567890')

def get_combinations(r):
    # Get all combinations of the current length
    combinations = itertools.product(characters, repeat=r)

    # Return combinations as a list of strings
    return [''.join(combo) for combo in combinations]

if __name__ == '__main__':
    with Pool(cpu_count()) as p:
        all_combinations = p.map(get_combinations, range(1, 6))

    # Write all combinations to a single file
    with open('combinations.txt', 'w') as f:
        for combos in all_combinations:
            for combo in combos:
                f.write(combo + '\n')
