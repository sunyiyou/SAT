from collections import defaultdict
from functools import partial

def is_subsequence(subseq, sequence):
    """
    Check if 'subseq' is a non-contiguous subsequence of 'sequence'.
    Returns True if subseq is contained in sequence, False otherwise.
    """
    i, j = 0, 0
    while i < len(subseq) and j < len(sequence):
        if subseq[i] == sequence[j]:
            i += 1
        j += 1
    return i == len(subseq)


def count_occurrences(sequences, candidate):
    """
    Count how many sequences in 'sequences' contain 'candidate' as a subsequence.
    """
    count = 0
    for seq in sequences:
        if is_subsequence(candidate, seq):
            count += 1
    return count


def find_subsequence_occurrences(c, s):
    """
    Returns a list of all index-lists in s that match the subsequence c.
    E.g. s = (A,B,A,D), c = (A,D) => [[0,3], [2,3]]
    """
    results = []

    def backtrack(start_idx, c_idx, current_indices):
        """
        start_idx: where we can start searching in s
        c_idx: which element of c we are trying to match next
        current_indices: the indices chosen so far
        """
        # If we've matched the entire c, we record the indices
        if c_idx == len(c):
            results.append(current_indices[:])  # copy
            return

        # If we still have elements in c to match but no more in s, just return
        if start_idx >= len(s):
            return

        # We can either skip s[start_idx]...
        backtrack(start_idx + 1, c_idx, current_indices)

        # ... or if s[start_idx] matches c[c_idx], we use it
        if s[start_idx] == c[c_idx]:
            current_indices.append(start_idx)
            backtrack(start_idx + 1, c_idx + 1, current_indices)
            current_indices.pop()  # backtrack

    backtrack(0, 0, [])
    return results


def expand_by_one(s, subseq_indices):
    """
    Insert exactly one element into the subsequence (defined by subseq_indices)
    at any possible gap:
      - before the first item
      - between consecutive items
      - after the last item
    preserving the original order in 's'.

    s              : the original sequence (tuple or list)
    subseq_indices : a list (or tuple) of sorted indices (strictly increasing)
                     that define the current subsequence
    Returns        : a list of subsequences (as tuples of items) of size len(subseq_indices)+1
    """
    expansions = []
    k = len(subseq_indices)

    # Extended indices: let -1 be "before the start", and len(s) be "after the end".
    extended_indices = [-1] + list(subseq_indices) + [len(s)]

    # We'll consider each adjacent pair in extended_indices.
    # That means (extended_indices[i], extended_indices[i+1]) for i in range(k+1).
    # Each pair defines a "gap" in which we can insert a new index.
    for i in range(k + 1):
        start_idx = extended_indices[i]  # could be -1 if i=0
        end_idx = extended_indices[i + 1]  # could be len(s) if i=k

        # Possible insertion points are m where start_idx < m < end_idx
        for m in range(start_idx + 1, end_idx):
            # Build a new index list by inserting m at position i
            new_subseq_indices = subseq_indices[:i] + [m] + subseq_indices[i:]
            # Convert indices to the actual items
            new_subseq_items = tuple(s[idx] for idx in new_subseq_indices)

            expansions.append(new_subseq_items)

    return expansions

def generate_candidates(prev_level_candidates, s):
    new_candidates = set()
    for cand in prev_level_candidates:
        """
            1) Find all ways c can occur in s
            2) For each occurrence, expand by one
            3) Return all unique expansions
            """
        # Step 1: All occurrences
        all_occurrences = find_subsequence_occurrences(cand, s)

        # Step 2 & 3: Expand and deduplicate
        for occ_indices in all_occurrences:
            # Expand
            expansions = expand_by_one(s, occ_indices)
            # Add expansions to a global set
            for exp in expansions:
                new_candidates.add(exp)
    return sorted(new_candidates)

def most_frequent_subsequence(sequences, k, s, ignore_items=None, return_all=False, beam=5, scoring_func=None):
    """
    Find the most frequent (highest probability) non-contiguous subsequence of length k
    among the given list of sequences. Optionally ignore a specific element.

    :param sequences: list of lists (each is one sequence)
    :param k: integer length of the target subsequence
    :param s: reference sequence
    :param ignore_items: optional value to exclude from all sequences
    :param return_all: boolean, if True, return frequencies of all subsequences at each length
    :return: a tuple (best_subsequence, best_count) or a tuple (best_subsequence, best_count, all_frequencies)
    """
    # 1) If requested, remove the ignore_item from all sequences
    # if ignore_items is not None:
    #     filtered_sequences = []
    #     for seq in sequences:
    #         filtered_seq = [x for x in seq if x not in ignore_items]
    #         filtered_sequences.append(filtered_seq)
    #     sequences = filtered_sequences

    if scoring_func is None:
        scoring_func = partial(count_occurrences, sequences)

    # 2) Extract all unique items from all (possibly filtered) sequences
    all_items = set(s)

    # 3) Create initial 1-length candidates
    level_candidates = [(item,) for item in all_items]

    # We'll store (subsequence, frequency) for the best found so far.
    best_subsequence = None
    best_count = 0

    # Dictionary to store frequencies of all subsequences for return_all
    all_frequencies = {} if return_all else None

    current_length = 1
    while current_length <= k and level_candidates:
        new_level_candidates = []
        freq_dict = {}

        # Count occurrences of each candidate
        for cand in level_candidates:
            freq = scoring_func(cand)
            # freq = count_occurrences(sequences, cand)

            freq_dict[cand] = freq

            # Update best if we are at length k
            if freq > best_count and len(cand) == k:
                best_count = freq
                best_subsequence = cand

        # Store frequencies for the current length if return_all is enabled
        if return_all:
            all_frequencies[current_length] = freq_dict

        if current_length == k:
            # We only need to find the best among length-k candidates
            break

        # 4) Generate next-level candidates
        #    (You might prune candidates with freq = 0 if you want.)
        # frequent_candidates = [c for c, f in freq_dict.items() if f > 0]
        # level_candidates = generate_candidates(frequent_candidates, all_items)

        # 4) Generate next-level candidates
        # Keep only the best-5 current level candidates
        frequent_candidates_kvs = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)[:beam]
        frequent_candidates = [c[0] for c in frequent_candidates_kvs]
        # freqs = [c[1] for c in frequent_candidates_kvs]
        # scores = [scoring_func(c[0]) for c in frequent_candidates_kvs]
        level_candidates = generate_candidates(frequent_candidates, s)

        current_length += 1

    if return_all:
        return best_subsequence, best_count, all_frequencies
    return best_subsequence, best_count


if __name__ == "__main__":
    # Example usage
    sequences = [
        [1, 2, 3, 4],
        [2, 1, 4, 3],
        [1, 2, 2, 3, 4],
        [1, 3, 2, 4]
    ]
    k = 2

    # Without ignoring anything
    best_subseq, best_freq = most_frequent_subsequence(sequences, k)
    print(f"Best subsequence of length {k} (no ignore): {best_subseq} (freq = {best_freq})")

    # Example: ignoring the element '2'
    best_subseq_ignored, best_freq_ignored = most_frequent_subsequence(
        sequences, k, ignore_item=1
    )
    print(f"Best subsequence of length {k} (ignoring '1'): {best_subseq_ignored} (freq = {best_freq_ignored})")

    # Example: ignoring the element '2'
    best_subseq_ignored, best_freq_ignored, all = most_frequent_subsequence(
        sequences, k, ignore_item=1, return_all=True
    )
