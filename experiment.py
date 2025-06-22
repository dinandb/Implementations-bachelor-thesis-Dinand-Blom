

def main():
    from smart_signature import main as smart_main
    from naive_signature import main as naive_main

    n = 5  # Number of iterations for testing

    print("Running Smart Signature Experiment...")
    smart_successes, median_0, median_1 = smart_main(n)
    print(f"Smart Signature: {smart_successes} successful iterations out of {n}")
    print(f"Median time for challenge 0: {median_0}")
    print(f"Median time for challenge 1: {median_1}")

    print("Running Naive Signature Experiment...")
    naive_successes, median_0, median_1 = naive_main(n)
    print(f"Naive Signature: {naive_successes} successful iterations out of {n}")
    print(f"Median time for challenge 0: {median_0}")
    print(f"Median time for challenge 1: {median_1}")

main()