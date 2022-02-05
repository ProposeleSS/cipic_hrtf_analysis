import pdb
import matplotlib.pyplot as plt
import json
import numpy as np
from numpy.fft import fft, fftfreq

from generate_data import gen_data

DATADIR = 'standard_hrir_database'
Fs = 44100
N_SAMPLES = 200 # length of fft

def main():
    hrir_raw = gen_data(DATADIR)['data']
    # hrir array is 3d, but we care only about 60 deg azimuth and 0 deg elevation (perfect listening position):
    # this corresponds to:
    # hrir_r[2][8] for az -65 and 0 el  <left side
    # hrir_r[3][8] for az -55 and 0 el  <left side
    # hrir_r[23][8] for az 55 and 0 el  <right side
    # hrir_r[24][8] for az 65 and 0 el  <right side
    # same for left
    hrir_filtered = []
    for s in hrir_raw:
        left_raw = {
            'right_ear_1': s['hrir_r'][2][8],
            'right_ear_2': s['hrir_r'][3][8],
            'left_ear_1': s['hrir_l'][2][8],
            'left_ear_2': s['hrir_l'][3][8],
        }
        right_raw = {
            'right_ear_1': s['hrir_r'][23][8],
            'right_ear_2': s['hrir_r'][24][8],
            'left_ear_1': s['hrir_l'][23][8],
            'left_ear_2': s['hrir_l'][24][8],
        }
        left_avg = {
            'right': [(r1 + r2) / 2 for r1, r2 in zip (left_raw['right_ear_1'], left_raw['right_ear_2'])],
            'left': [(l1 + l2) / 2 for l1, l2 in zip (left_raw['left_ear_1'], left_raw['left_ear_2'])]
        }
        right_avg = {
            'right': [(r1 + r2) / 2 for r1, r2 in zip (right_raw['right_ear_1'], right_raw['right_ear_2'])],
            'left': [(l1 + l2) / 2 for l1, l2 in zip (right_raw['left_ear_1'], right_raw['left_ear_2'])]
        }
        name = s['name'][0]

        hrir_filtered.append({
            'left': left_raw,
            'right': right_raw,
            'left_avg': left_avg,
            'right_avg': right_avg,
            'left_avg_fft': {
                'right': fft(left_avg['right']),
                'left': fft(left_avg['left'])
            },
            'right_avg_fft': {
                'right': fft(right_avg['right']),
                'left': fft(right_avg['left'])
            },
            'name': name
        })
    
    yf = hrir_filtered[0]['left_avg_fft']['left']
    xf = fftfreq(N_SAMPLES, 1/Fs)[:N_SAMPLES//2]
    plt.plot(xf, 2.0/N_SAMPLES * np.abs(yf[0:N_SAMPLES//2]))
    plt.grid()
    plt.show()

if __name__ == '__main__':
    main()