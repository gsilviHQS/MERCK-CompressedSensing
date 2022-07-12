""" generate random Poisson-distributed numbers as given
by Donald E. Knuth (1969). Seminumerical Algorithms.
The Art of Computer Programming, Volume 2. Addison Wesley  """

import math
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as spfft


class ISTrecontruction(object):  # object is a class, which is a blueprint for an object,it has attributes and methods
    """
    class for the IST recontruction procedure
    """
    def __init__(self, decrease_factor=0.7, max_iter=100, tol=1e-3):
        self.decrease_factor = decrease_factor
        self.max_iter = max_iter
        self.tol = tol

    def run(self, signal, non_sampled_points, verbose=False):
        self.signal = signal
        self.length = len(signal)
        final_spectrum = np.zeros(self.length)  # create a new array with size of the signal
        spectrum = spfft.dct(self.signal, norm='ortho', axis=0)  # perform DCT on the signal
        steps = 0
        while True:
            steps += 1
            max_value = np.max(np.abs(spectrum))  # find the max value of the spectrum
            threshold = self.decrease_factor * max_value  # define the threshold
            if verbose:
                print('threshold:', threshold, end='\t')
            for i, data_point in enumerate(spectrum):  # loop through the spectrum
                # sign of data_point
                sign = np.sign(data_point)  # sign of data_point
                if abs(data_point) > threshold:  # if the absolute value of data_point is greater than threshold
                    final_spectrum[i] += sign * (abs(data_point) - threshold)  # add the "excess" to the final spectrum
                    spectrum[i] = sign * threshold  # set the value of spectrum to the sign of data_point times threshold
                # convert the spectrum to a time-domain signal using the idct
                signal = spfft.idct(spectrum, norm='ortho', axis=0)
                signal[non_sampled_points] = 0  # set the non_sampled_points to 0
                spectrum = spfft.dct(signal, norm='ortho', axis=0)
            if threshold < self.tol:
                print('Reason to stop: Threshold<' + str(self.tol))
                break

            if steps > self.max_iter:
                print('Reason to stop: Iterations>' + str(self.max_iter))
                break
        return final_spectrum



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


def PoissonNumbers(number_of_samples: int, total_number_of_indices: int, seed_value: int = 42, usenumpy=False):
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

    # for j in range(number_of_samples):
    #     print(v[j], end='\t')
    # print()
    return v[0:number_of_samples]


class MyPlot:
    
    def find_between(self,set1,start,end):
        return [i for i in set1 if start <= i < end]

    def split_list(self,l, n, max_value):
        divisor = max_value // n
        return [self.find_between(l,divisor*i, divisor*(i+1)) for i in range(n)]
    def subtract_list(self,l, n):
        return [x-n for x in l]

    def plot_list(self,set1, max_value, rows=5):
        splitset = self.split_list(set1, rows, max_value)
        divisor = max_value // rows
        #plt.gca().axes.get_yaxis().set_visible(False)
        plt.axis('off')
        #plt.ylim(-1,rows+1)
        for i,subset in enumerate(splitset):
            y = len(splitset)-i
            plt.plot(np.arange(0,divisor), [y]*(divisor), 'o',color='white', markeredgewidth=0.2, markeredgecolor='black', markersize=rows-2)
            plt.plot(self.subtract_list(subset,divisor*i),[y]*len(subset) , 'o',color='black', markersize=rows-2)
            #make a list of number between divisor*i and divisor*(i+1)
            
            #move the text further to the left: 
            plt.text(0,y,str(i*divisor)+"  ",fontsize=rows-1 , ha='right',va='center', color='black',fontweight='bold')
            plt.text(divisor,y,str(divisor*(i+1)),fontsize=rows-1,va='center',color='black',fontweight='bold')

            for j,entry in enumerate(subset):
                # have a bolder font for text, use the following:

                plt.text(entry-divisor*i,y,str(entry),color='white',ha='center', va='center', fontsize=-1+rows-len(str(entry)))
