import math
import numba
from numba import cuda, types
import numpy as np
import sys

# --------------------------------------------------------------
# Configuration
# --------------------------------------------------------------
CHARS = "abcefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"
n = len(CHARS)  # 58 characters
min_len = 1
max_len = 10

# Chunk size for GPU processing
CHUNK_SIZE = 10_000_000

# --------------------------------------------------------------
# Utility: permutations_count(n, r) = n * (n-1) * ... * (n-r+1)
# --------------------------------------------------------------
def permutations_count(n, r):
    total = 1
    for i in range(r):
        total *= (n - i)
    return total

# --------------------------------------------------------------
# A small helper to display a progress bar
# --------------------------------------------------------------
def print_progress_bar(current, total, bar_length=40):
    """
    Prints a text-based progress bar in one line, overwriting itself each call.
    """
    progress = current / total
    filled_length = int(bar_length * progress)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    percent = progress * 100
    # \r moves the cursor back to the start of the line (overwrite), and flush=True updates immediately.
    print(f"\rProgress: |{bar}| {percent:6.2f}%  ", end='', flush=True)

# --------------------------------------------------------------
# GPU Kernel: Decode a global permutation index into an integer array
# --------------------------------------------------------------
@cuda.jit
def decode_permutation_kernel(results, base_idx, n, r):
    """
    results: device array of shape (chunk_size, r) storing integer indices
    base_idx: the first permutation index for this chunk
    n: number of distinct characters
    r: permutation length
    """
    global_idx = base_idx + cuda.grid(1)
    if global_idx >= (base_idx + results.shape[0]):
        return

    # Local boolean array to track used characters
    # Allocate size at least n=58, but let's keep 63 to avoid index issues. 
    used = cuda.local.array(63, types.boolean)
    for i in range(n):
        used[i] = False

    cur_index = global_idx
    remaining = n
    thread_id = cuda.grid(1)  # row in 'results'

    for position in range(r):
        # permutations for next positions with (remaining-1) items
        perms_of_rest = 1
        for k in range(remaining - 1, remaining - (r - position), -1):
            perms_of_rest *= k

        choice = cur_index // perms_of_rest
        cur_index %= perms_of_rest

        # find the 'choice'-th unused item
        found_count = 0
        char_idx = 0
        while True:
            if not used[char_idx]:
                if found_count == choice:
                    used[char_idx] = True
                    results[thread_id, position] = char_idx
                    break
                found_count += 1
            char_idx += 1

        remaining -= 1

# --------------------------------------------------------------
# Generate & Save permutations for a given length 'r', optionally printing
# --------------------------------------------------------------
def generate_permutations_gpu(r, print_flag, chunk_size=CHUNK_SIZE):
    """
    Enumerate all permutations of length r using the CHARS set, in chunks.
    
    - Always saves permutations to 'combinations.txt'.
    - Prints permutations to console if print_flag is True.
      If print_flag is False, shows a progress bar instead.
    """
    total_perms = permutations_count(n, r)
    print(f"Generating permutations of length {r} -> {total_perms} total.")

    threads_per_block = 256
    current_offset = 0

    # Open the file in append mode once
    with open("combinations.txt", "a", encoding="utf-8") as f_out:
        # While we still have permutations to process
        while current_offset < total_perms:
            # limit the chunk size for the remaining permutations
            current_chunk_size = min(chunk_size, total_perms - current_offset)

            # allocate device array
            results_device = cuda.device_array((current_chunk_size, r), dtype=np.int32)

            # compute number of blocks
            blocks = (current_chunk_size + threads_per_block - 1) // threads_per_block

            # launch kernel
            decode_permutation_kernel[blocks, threads_per_block](results_device, current_offset, n, r)
            cuda.synchronize()

            # copy back to host
            results_host = results_device.copy_to_host()

            # Convert each row to a string, then write to file (and print if requested)
            for row_idx in range(current_chunk_size):
                perm_indices = results_host[row_idx]
                perm_str = "".join(CHARS[idx] for idx in perm_indices)

                # Always save to file
                f_out.write(perm_str + "\n")

                # If user wants to see permutations, print them
                if print_flag:
                    print(perm_str)

            current_offset += current_chunk_size

            # Show either "chunk done" or update progress bar
            if print_flag:
                print(f"Chunk of {current_chunk_size} done -> offset: {current_offset}/{total_perms}")
            else:
                # Update progress bar
                print_progress_bar(current_offset, total_perms)

    # If we used the progress bar, ensure we move to a new line at the end
    if not print_flag:
        print()  # finish the progress bar line

    print(f"Finished generating permutations of length {r}.\n")

# --------------------------------------------------------------
# Main
# --------------------------------------------------------------
if __name__ == "__main__":
    # Ask user if they want to see (print) all permutations
    choice = input("Do you want to see all generated permutations? (Y/N): ").strip().upper()
    see_passwords = (choice == "Y")

    # Optional: Clear the file at the start
    # with open("combinations.txt", "w", encoding="utf-8") as f:
    #     pass

    # Generate permutations for lengths 4 through 10
    for r in range(min_len, max_len + 1):
        generate_permutations_gpu(r, print_flag=see_passwords, chunk_size=CHUNK_SIZE)
        print("=" * 60)
