# flow_metrics.py
import numpy as np
import matplotlib.pyplot as plt 

def calculate_average_flow(u, v):
    mean_u = np.mean(u)
    mean_v = np.mean(v)
    return mean_u, mean_v

def calculate_magnitude(u, v):
    magnitude = np.sqrt(u**2 + v**2)
    return magnitude

def calculate_angle(u, v):
    angle = np.arctan2(v, u) * 180 / np.pi
    return angle

def calculate_histogram(angle, bins=30):
    hist, bin_edges = np.histogram(angle, bins=bins, range=(-180, 180))
    return hist, bin_edges

def calculate_variance(u, v):
    var_u = np.var(u)
    var_v = np.var(v)
    return var_u, var_v

def plot_average_flow(mean_u, mean_v):
    plt.figure()
    plt.quiver(0, 0, mean_u, mean_v, angles='xy', scale_units='xy', scale=1, color='r')
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.xlabel('U (horizontal component)')
    plt.ylabel('V (vertical component)')
    plt.title('Average Flow Vector')
    plt.grid()
    plt.show()

def plot_magnitude_distribution(magnitude):
    plt.figure()
    plt.hist(magnitude, bins=50, range=(0, np.max(magnitude)), color='b', alpha=0.75)
    plt.xlabel('Magnitude')
    plt.ylabel('Frequency')
    plt.title('Magnitude Distribution')
    plt.grid()
    plt.show()

def plot_angle_distribution(angle):
    plt.figure()
    plt.hist(angle, bins=50, range=(-180, 180), color='g', alpha=0.75)
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Frequency')
    plt.title('Angle Distribution')
    plt.grid()
    plt.show()

def plot_flow_direction_histogram(angle, hist, bin_edges):
    plt.figure()
    plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), edgecolor='black', align='edge')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Frequency')
    plt.title('Flow Direction Histogram')
    plt.grid()
    plt.show()

def plot_flow_variance(u, v):
    plt.figure()
    plt.scatter(u, v, color='c', alpha=0.5)
    plt.xlabel('U (horizontal component)')
    plt.ylabel('V (vertical component)')
    plt.title('Flow Variance')
    plt.grid()
    plt.show()
