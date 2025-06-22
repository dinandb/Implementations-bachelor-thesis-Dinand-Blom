
from help_functions import random_invertible_matrix, n, random_tensor, new_tensor_from_tensor_and_isomorphism
from prover import NaiveProver
from verifier import NaiveVerifier
import time
import statistics


prover = None
verifier = None

def init():
    prover = NaiveProver([random_invertible_matrix(n), random_invertible_matrix(n), random_invertible_matrix(n)], random_tensor(n))
    verifier = NaiveVerifier(prover.C, prover.D)

    return prover, verifier
def n_iterations(prover, verifier, n):
    successful_iterations = 0
    times_0 = []
    times_1 = []
    for i in range(n):
        start_time = time.perf_counter_ns()
        # gen commit
        commit = prover.generate_commit()
        # send commit to verifier
        challenge = verifier.accept_and_generate_challenge(commit)
        # prover generates response
        response = prover.generate_challenge_response(challenge)
        # verifier verifies response
        result = verifier.verify_response(response)
        elapsed_time = time.perf_counter_ns() - start_time
        print(f"succesful iteration {i} = {result}")
        if result:
            successful_iterations += 1
            if challenge == 0:
                times_0.append(elapsed_time)
            else:
                times_1.append(elapsed_time)

    return successful_iterations, times_0, times_1



def main(n_its):
    prover, verifier = init()
    
    successes, times_0, times_1 = n_iterations(prover, verifier, n_its)
    # print("Total successful iterations:", successes)
    # print(f"Percentage of successful iterations: {n_iterations(prover, verifier, n) / n * 100:.2f}%")
    median_0 = statistics.median(times_0) if times_0 else None
    median_1 = statistics.median(times_1) if times_1 else None
    # print(f"Median time for challenge 0: {median_0}")
    # print(f"Median time for challenge 1: {median_1}")
    return successes, median_0, median_1

