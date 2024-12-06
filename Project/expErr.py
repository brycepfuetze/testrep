import cmath
import numpy as np
import matplotlib.pyplot
import scipy.fft
import matplotlib.pyplot as plt
# matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': True,
#     'pgf.rcfonts': False,
# })
i = 1j


def keep_k_lowest_freqs(hat_f, k):

    n = len(hat_f)

    # We want the 0th frequency, the first k, and the last k (since the first k and last k correspond to the same frequencies since the FFT is ``mirrored'')
    # 0 out the remaining middle elements
    return [hat_f[0]] + hat_f[1:k+1] + [0] * (n-(2*k+1)) + hat_f[n-k:n]

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

# f = lambda x: -2*x**3+3*x**2+5*x-2
f = lambda x: np.exp(np.cos(x)**3)
# f = lambda x: np.exp(np.cos(x))
# f = lambda x: x/np.exp(x**2)
X = np.linspace(-2,2,1024)

fftsample = fft(f(X))

k = 2
filterk = keep_k_lowest_freqs(fftsample, k)

# plt.plot(X, f(X), color='k', linewidth=2)
# plt.plot(X, scipy.fft.ifft(filterk), color='r', linewidth=2, label=f'After Keeping {k} Lowest Frequencies')
# plt.title("g(x) = exp(cos^3(x))")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.legend(loc='best')
# plt.savefig("assets/expcossin.png")
# plt.show()

# fourier_coeffs = fftsample[1:int(len(fftsample)/2 + 1)]
# plt.scatter(range(1,len(fourier_coeffs)+1), np.abs(fourier_coeffs), s=2, color='k')
# plt.ylabel("Fourier Coefficient Magnitude")
# plt.xlabel("Coefficient Number")
# plt.legend(loc='best')
# plt.title("Decay of Fourier Coefficient Magnitude of g(x)=exp(cos^3(x))")
# plt.yscale('log')
# plt.savefig(f"assets/fourier_coeff_decay_periodic.png")
# plt.show()

noisy_normal_f = f(X) + np.random.choice([-1, 1])*0.1*np.random.normal(size=1024)
noisy_uniform_f = f(X) + np.random.choice([-1, 1])*0.1*np.random.random(1024)

# move outside of for loop, don't need to calc every time
fftsample = fft(noisy_normal_f)

for k in [1,2,5,15,30,50,100]:
    filterk = keep_k_lowest_freqs(fftsample, k)

    ## Error analysis 
    error = noisy_normal_f / (scipy.fft.ifft(filterk)).real - 1

    # Calculate inf
    maxerror = np.max(noisy_normal_f / (scipy.fft.ifft(filterk)).real - 1)

    # Calculate RMS
    square = np.square(error)
    mean = np.mean(square)
    root = np.sqrt(mean)
    print('RMS Error = ', root*100, '%',f' for {k} lowest frequencies')
    print('Max Error = ', maxerror*100, '%',f' for {k} lowest frequencies')

    # plt.scatter(X, noisy_normal_f, color='k', linewidth=2, s=2)
    # plt.plot(X, scipy.fft.ifft(filterk), color='r', linewidth=2, label=f'After Keeping {k} Lowest Frequencies')
    # plt.title("g(x) = exp(cos^3(x)) With Normally Distributed Noise")
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.legend(loc='best')
    # plt.savefig(f"assets/expcossinnormnoise{k}.png")
    # plt.show()

# fourier_coeffs = fftsample[1:int(len(fftsample)/2 + 1)]
# plt.scatter(range(1,len(fourier_coeffs)+1), np.abs(fourier_coeffs), s=2, color='k')
# plt.ylabel("Fourier Coefficient Magnitude")
# plt.xlabel("Coefficient Number")
# plt.legend(loc='best')
# plt.title("Decay of Fourier Coefficient Magnitude of exp(cos^3(x))\nWith Normally Distributed Noise")
# plt.yscale('log')
# plt.savefig(f"assets/fourier_coeff_decay_normnoise.png")
# plt.show()

