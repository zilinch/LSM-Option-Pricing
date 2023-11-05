import numpy as np
import matplotlib.pyplot as plt

# Generate exercise boundary using all exercise points
def genScatterExerciseBound(t, exe_bound):
    x, y = [], []
    for dt in t:
        if exe_bound[dt]:
            x.extend([dt]*len(exe_bound[dt]))
            y.extend(exe_bound[dt])
    
    return x, y

# Generate exercise boundary using MAX of the exercise points
def genLineExerciseBound(t, exe_bound):
    vals = exe_bound.values()
    return t, [max(e, default=None) for e in vals]

def plotExerciseBoundary(T, K, S0, exe_bound, hist_prices = None, ptype='all', opt_type='Put') -> None:
    t = list(exe_bound.keys())

    if ptype == 'all':
        x, y = genScatterExerciseBound(t, exe_bound)
    elif ptype == 'max':
        x, y = genLineExerciseBound(t, exe_bound)
    else:
        raise ValueError ("Only ptype only supports 'all' or 'max'")


    # Plotting
    plt.figure(figsize=(8, 4))
    plt.axhline(y=K, color='gray', linestyle='--', label=f'K = {K}')
    plt.axhline(y=S0, color='orange', linestyle='--', label=f'S0 = {S0}')
    if ptype == 'all':
        plt.scatter(x, y, color='blue', marker='.', label = 'LSM') #plot points
    elif ptype == 'max':
        plt.plot(x, y, color='blue', label = 'LSM')
    if hist_prices:
        plt.plot(t, list(hist_prices.values()))
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.title(f'Exercise Boundary for an American {opt_type} Option')
    plt.legend()
    plt.xlim(0, T)
    plt.ylim(0, 120)
    plt.show()

    return


def plot_exe_times(time_steps, exe_times):
    plt.figure(figsize=(5,4))
    for k in exe_times:
        plt.plot(time_steps, exe_times[k], label=f'{k} basis function')
    plt.title("Execution time for S0 = 90")
    plt.xlabel("Number of time steps")
    plt.ylabel("Execution time (seconds)")
    plt.legend()
    plt.show()