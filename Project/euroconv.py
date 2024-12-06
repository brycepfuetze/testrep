import cmath
import numpy as np
import matplotlib.pyplot
import scipy.fft
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})
import plotly.express as px
import plotly.graph_objects as go

i = 1j

def test_dft():

    for _ in range(25):
        x = np.random.random(16) + 1j * np.random.random(16)  
        assert np.allclose(naive_dft(x), scipy.fft.fft(x))
        assert np.allclose(fft(x), scipy.fft.fft(x))
        assert np.allclose(naive_ift(x), scipy.fft.ifft(x))
    

def naive_dft(f):
    
    n = len(f)

    hat_f = [0 for _ in range(n)]

    omega_n = np.exp(-2 * np.pi * i / n)
    
    if n <= 1: return f

    # O(n^2) Complexity
    for k in range(n):

        hat_f[k] = sum([f[j] * omega_n ** (j * k) for j in range(n)])

    return hat_f

def naive_ift(hat_f):
    
    n = len(hat_f)

    f = [0 for _ in range(n)]

    omega_n_bar = np.exp(2 * np.pi * i / n)
    
    # O(n^2) Complexity
    for k in range(n):

        f[k] = 1 / n * sum([hat_f[j] * omega_n_bar ** (j * k) for j in range(n)])

    return f

def fft(f):

    n = len(f)

    hat_f = [0 for _ in range(n)]

    if n <= 1: return f

    omega, omega_n = 1, np.exp(-2*np.pi*i/n)

    f_even = [f[i] for i in range(0, n, 2)]
    f_odd = [f[i] for i in range(1, n, 2)]

    hat_f_even = fft(f_even)
    hat_f_odd = fft(f_odd)

    for k in range(n // 2):

        hat_f[k] = hat_f_even[k] + omega * hat_f_odd[k]
        hat_f[k + n // 2] = hat_f_even[k] - omega * hat_f_odd[k]
        omega = omega * omega_n

    return hat_f

# Reference one of Sriram's FFT Lectures
def keep_k_lowest_freqs(hat_f, k):

    n = len(hat_f)

    # We want the 0th frequency, the first k, and the last k (since the first k and last k correspond to the same frequencies since the FFT is ``mirrored'')
    # 0 out the remaining middle elements
    return [hat_f[0]] + hat_f[1:k+1] + [0] * (n-(2*k+1)) + hat_f[n-k:n]


## Data Testing
import pandas as pd

# CSV From https://data.ecb.europa.eu/data/datasets/EXR/EXR.D.USD.EUR.SP00.A?chart_props=W3sibm9kZUlkIjoiMzIyNDUxIiwicHJvcGVydGllcyI6W3siY29sb3JIZXgiOiIiLCJjb2xvclR5cGUiOiIiLCJjaGFydFR5cGUiOiJsaW5lY2hhcnQiLCJsaW5lU3R5bGUiOiJTb2xpZCIsImxpbmVXaWR0aCI6IjEuNSIsImF4aXNQb3NpdGlvbiI6ImxlZnQiLCJvYnNlcnZhdGlvblZhbHVlIjpmYWxzZSwiZGF0ZXMiOlsiMjAwMC0wMS0wMVQwNzowMDowMC4wMDBaIiwiMjAyNC0wMS0wMVQwNzowMDowMC4wMDBaIl0sImlzVGRhdGEiOmZhbHNlLCJtb2RpZmllZFVuaXRUeXBlIjoiIiwieWVhciI6ImRhdGV3aXNlIiwic3RhcnREYXRlIjoiMTk5OS0xMi0zMSIsImVuZERhdGUiOiIyMDIzLTEyLTMxIiwic2V0RGF0ZSI6dHJ1ZSwic2hvd1RhYmxlRGF0YSI6ZmFsc2UsImNoYW5nZU1vZGUiOmZhbHNlLCJzaG93TWVudVN0eWxlQ2hhcnQiOmZhbHNlLCJkaXNwbGF5TW9iaWxlQ2hhcnQiOnRydWUsInNjcmVlblNpemUiOiJtYXgiLCJzY3JlZW5XaWR0aCI6MTUxMiwic2hvd1RkYXRhIjpmYWxzZSwidHJhbnNmb3JtZWRGcmVxdWVuY3kiOiJub25lIiwidHJhbnNmb3JtZWRVbml0Ijoibm9uZSIsImZyZXF1ZW5jeSI6Im5vbmUiLCJ1bml0Ijoibm9uZSIsIm1vZGlmaWVkIjoiZmFsc2UiLCJzZXJpZXNLZXkiOiJkYWlseSIsInNob3d0YWJsZVN0YXRlQmVmb3JlTWF4U2NyZWVuIjpmYWxzZSwiaXNkYXRhY29tcGFyaXNvbiI6ZmFsc2UsInNlcmllc0ZyZXF1ZW5jeSI6ImRhaWx5IiwiaW50aWFsU2VyaWVzRnJlcXVlbmN5IjoiZGFpbHkiLCJtZXRhZGF0YURlY2ltYWwiOiI0IiwiaXNUYWJsZVNvcnRlZCI6ZmFsc2UsImlzWWVhcmx5VGRhdGEiOmZhbHNlLCJyZXNwb25zZURhdGFFbmREYXRlIjoiIiwiaXNpbml0aWFsQ2hhcnREYXRhIjp0cnVlLCJpc0RhdGVzRnJvbURhdGVQaWNrZXIiOnRydWUsImRhdGVQaWNrZXJFbmREYXRlIjoiMjAyNC0wMS0wMSIsImlzRGF0ZVBpY2tlckVuZERhdGUiOnRydWUsInNlcmllc2tleVNldCI6IiIsImRhdGFzZXRJZCI6IjE4IiwiaXNDYWxsYmFjayI6ZmFsc2UsImlzU2xpZGVyVGRhdGEiOnRydWUsImlzU2xpZGVyRGF0YSI6dHJ1ZSwiaXNJbml0aWFsQ2hhcnREYXRhRnJvbUdyYXBoIjp0cnVlLCJjaGFydFNlcmllc0tleSI6IkVYUi5ELlVTRC5FVVIuU1AwMC5BIiwidHlwZU9mIjoiIn1dfV0%3D

us_euro = pd.read_csv("us_euro.csv")
exchange_rate = list(us_euro.iloc[:, 2].fillna(method='ffill').to_numpy()) # fill missing with the next day
exchange_rate = list(map(float, exchange_rate))
X = [i for i in range(len(exchange_rate))]
exchange_ratefft = fft(exchange_rate)

p = 100
fig = px.scatter(x = X, y=exchange_rate)
fig.add_scatter(x=X, y=np.real(scipy.fft.ifft(keep_k_lowest_freqs(exchange_ratefft,p))))
#fig.ylabel("U.S. Dollar/Euro Exchange Rate")
#fig.xlabel("Days Since Jan. 4, 1999")
#fig.legend(loc='best')
#fig.title("U.S. Dollar-Euro Exchange Rate")
fig.show()