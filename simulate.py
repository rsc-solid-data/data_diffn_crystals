import numpy as np
from typing import Sequence, Callable


# These functions will calculate a peak using a Gaussian or Lorentzian function 
# as defined above.

def gaussian(x_array, ampl, centre, width):
    """Generate a signal with a Gaussian shape."""
    sigma = width/np.sqrt(8*np.log(2))
    return ampl*(1/(sigma*np.sqrt(2*np.pi)))*np.exp(-(x_array-centre)**2/(2*sigma**2))

def lorentzian(x_array, ampl, centre, width):
    """Generate a signal with a Lorentzian shape."""
    h_width = width/2
    return ampl/np.pi * h_width/((x_array-centre)**2 + h_width**2)

def pseudo_voigt(x_array, ampl, centre, width, fraction=0.7):
    """Generate a signal with a pseudo-Voigt shape."""
    return (1-fraction)*gaussian(x_array, ampl, centre, width) + fraction*lorentzian(x_array, ampl, centre, width)



def simulate_peak(peak_fn, x_array, ampl, centre, width):
    calc_peak = peak_fn(x_array, ampl, centre, width)
    sim_peak = [y+np.random.normal(0, 0.05) for y in calc_peak]
    return sim_peak

def simulate_spectrum(peak_fn: Callable, x_array: Sequence[float], peaks: list[dict], add_noise=False) -> tuple[np.ndarray, np.ndarray]:
    #x_array = np.linspace(x_range[0], x_range[1], abs(x_range[0]-x_range[1])*100)
    y_array = np.zeros(len(x_array))
    for peak in peaks:
        calc_peak = peak_fn(x_array, peak["ampl"], peak["centre"], peak["width"])
        y_array = y_array + calc_peak
    if add_noise:    
        y_array = [y+np.random.normal(0, 8) for y in y_array]
    return x_array, y_array

