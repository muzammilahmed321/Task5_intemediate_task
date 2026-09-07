import numpy as np


# Build the payoff matrices for both agents
def build_payoff_matrix():
    strategies = ["Use Cache", "Call API"]

    payoff_A = np.array([
        [-1, -1],
        [-5, -5],
    ])

    payoff_B = np.array([
        [-1, -5],
        [-1, -5],
    ])

    return strategies, payoff_A, payoff_B


# Find pure-strategy Nash equilibria
def find_pure_nash_equilibria(strategies, payoff_A, payoff_B):
    n = len(strategies)
    equilibria = []

    # Check every possible strategy combination
    for i in range(n):
        for j in range(n):
            # Check whether Agent A is choosing a best response
            a_best = all(
                payoff_A[i, j] >= payoff_A[k, j]
                for k in range(n)
            )

            # Check whether Agent B is choosing a best response
            b_best = all(
                payoff_B[i, j] >= payoff_B[i, k]
                for k in range(n)
            )

            # Store strategy pair if both agents have best responses
            if a_best and b_best:
                equilibria.append(
                    (
                        strategies[i],
                        strategies[j],
                        payoff_A[i, j],
                        payoff_B[i, j]
                    )
                )

    return equilibria


# Display the payoff matrix in a readable format
def print_matrix(strategies, payoff_A, payoff_B):
    print(f"{'':15}" + "".join(f"{s:15}" for s in strategies))

    for i, s in enumerate(strategies):
        row = f"{s:15}"

        for j in range(len(strategies)):
            row += f"({payoff_A[i,j]}, {payoff_B[i,j]})".ljust(15)

        print(row)


# Run the game-theory analysis
if __name__ == "__main__":
    # Create payoff matrices
    strategies, payoff_A, payoff_B = build_payoff_matrix()

    print("Payoff matrix (Agent A payoff, Agent B payoff):\n")
    print_matrix(strategies, payoff_A, payoff_B)

    # Calculate Nash equilibria
    equilibria = find_pure_nash_equilibria(
        strategies,
        payoff_A,
        payoff_B
    )

    # Display discovered equilibria
    print("\nPure-strategy Nash Equilibria:")

    for eq in equilibria:
        print(
            f"  A: {eq[0]}, B: {eq[1]} "
            f"-> payoffs ({eq[2]}, {eq[3]})"
        )