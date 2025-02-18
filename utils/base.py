import timeit
import numpy as np

def compare_runtime(func1, func2, input_sequence, num_runs=10):
    """
    Compare the execution time of two functions with the same input.
    
    :param func1: First function to test
    :param func2: Second function to test
    :param input_sequence: The input sequence (1D array) to be used in both functions
    :param num_runs: Number of runs to average the execution time
    :return: Dictionary with execution times for both functions
    """
    
    # Define wrapper functions to measure time
    def wrapper_1():
        func1(np.copy(input_sequence))  # Copy to avoid side effects
    
    def wrapper_2():
        func2(np.copy(input_sequence))

    # Measure execution time using timeit (best of num_runs runs)
    time_func1 = timeit.timeit(wrapper_1, number=num_runs) / num_runs
    time_func2 = timeit.timeit(wrapper_2, number=num_runs) / num_runs

    # Print results
    # print(f"Average execution time of {func1.__name__}: {time_func1:.6f} sec")
    # print(f"Average execution time of {func2.__name__}: {time_func2:.6f} sec")
    
    return time_func1, time_func2



def sineWavGen(frequency=440, duration=2.0, sr=44100, iBlockLength=1024, iHopLength=512):
    # Create time vector for the entire signal
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    wav = np.sin(2 * np.pi * frequency * t)
    
    # Compute the number of blocks (frames) that will be used in pitch analysis.
    n_frames = int(np.floor((len(wav)) / iHopLength)) + 1
    # Create a constant ground truth pitch array.
    gt_qfreq = np.full(n_frames, frequency)
    
    return wav, sr, gt_qfreq