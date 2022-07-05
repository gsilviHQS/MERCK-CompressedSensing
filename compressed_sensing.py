""" generate random Poisson-distributed numbers as given
by Donald E. Knuth (1969). Seminumerical Algorithms.
The Art of Computer Programming, Volume 2. Addison Wesley  """

import math
import random
import numpy as np


def poisson(lmbd: float) -> int:  # labmda is the average number of events per unit time
    """Generate a Poisson-distributed random number.

    Parameters:
    lmbd (float): The average number of events per unit time.
    Returns:
    int: The number of events.
    """
    L = math.exp(-lmbd)  # The probability of the first event (k=0)
    k = 0           # The number of events
    p = 1        # The probability of the current event
    while p >= L:  # Loop until the probability is less than the probability of the first event
        u = random.random()  # Generate a random number
        p *= u  # Update the probability of the current event
        k += 1  # Increment the number of events = gap size
    return k - 1


def PoissonNumbers(seed_value: int, number_of_samples: int, total_number_of_indices: int, usenumpy=False):
    """
    Generate random Poisson-distributed numbers
    total_number_of_indices: the total number of indices e.g. 256
    i: Fourier grid index, e.g. 1 through 256
    k: generate gap size
    n: temporary # index
    v: temporary storage vector
    """
    v = [0] * total_number_of_indices  # Initialize the vector
    ld = total_number_of_indices / number_of_samples  # Number of indices per sample,establish 1/fraction
    adj = 2 * (ld - 1)  # initial guess of adjustment, useful to get the right number of samples
    random.seed(seed_value)  # Set the seed
    n = 0
    while n != number_of_samples:  # if not at first, try, try again until you get the right number of samples
        i = 0
        n = 0
        while i < total_number_of_indices:  # Loop over all indices
            v[n] = i  # Save the index
            i += 1  # Increment the index
            # The lambda (average number of events) for the current index
            lambd = adj * math.sin((i + 0.5) / (total_number_of_indices + 1) * np.pi / 2)
            
            if usenumpy:
                k = np.random.poisson(lambd)
            else:
                k = poisson(lambd)  # this produce the gap size for the next index
            i += k  # Increment the index by the gap size
            n += 1  # Increment the number of samples
        if n > number_of_samples:  # If the number of samples is greater than the number of samples asked
            adj *= 1.02  # too many points created, so adjust the adjustment
        if n < number_of_samples:  # If the number of samples is less than the number of samples asked
            adj /= 1.02  # too few points created, so adjust the adjustment

    for j in range(number_of_samples):
        print(v[j], end='\t')
    print()
    return v[0:number_of_samples]
