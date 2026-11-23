import numpy as np

def expected_losses(posterior, loss):
    posterior = np.asarray(posterior, dtype=float)
    loss = np.asarray(loss, dtype=float)

    return posterior @ loss


def bayes_policy(posterior, loss):
    losses = expected_losses(posterior, loss)

    return np.argmin(losses, axis=1)


def covid_table(prevalence, sens, spec, drug_cost,
                qaly_young=60.0, qaly_old=10.0):

    # Bayes' theorem
    p_negative = (
        prevalence * (1 - sens)
    ) / (
        prevalence * (1 - sens) + (1 - prevalence) * spec
    )

    p_positive = (
        prevalence * sens
    ) / (
        prevalence * sens + (1 - prevalence) * (1 - spec)
    )

    results = []

    for test, age, qaly, p_covid in [
        (0, 0, qaly_young, p_negative),
        (0, 1, qaly_old, p_negative),
        (1, 0, qaly_young, p_positive),
        (1, 1, qaly_old, p_positive)
    ]:

        posterior = np.array([[1 - p_covid, p_covid]])

        loss = np.array([
            [0, drug_cost],
            [qaly, drug_cost]
        ])

        # Calculate expected losses
        costs = expected_losses(posterior, loss)[0]

        # Select optimal action
        action = int(bayes_policy(posterior, loss)[0])

        results.append([
            int(test),
            int(age),
            float(p_covid),
            float(costs[0]),
            float(costs[1]),
            action
        ])

    return results
