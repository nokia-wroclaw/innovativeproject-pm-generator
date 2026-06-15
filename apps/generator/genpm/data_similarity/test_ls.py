import time

import numpy as np
from astropy.timeseries import LombScargle

N = 750000
ts_hours = np.linspace(0, N, N)
values = np.random.randn(N)

freq_min = 1.0 / (24 * 14)
freq_max = 1.0 / 2.0
frequency = np.linspace(freq_min, freq_max, 2000)

start = time.time()
power = LombScargle(ts_hours, values).power(frequency)
print(f"Lomb Scargle took {time.time() - start:.2f} seconds")
