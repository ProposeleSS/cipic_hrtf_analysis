import pdb
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, fftfreq
from scipy.stats import pearsonr, spearmanr, kendalltau

from generate_data import gen_data_hrir, gen_data_anthro

DATADIR_HRIR = 'standard_hrir_database'
DATADIR_PHY = 'anthropometry'
Fs = 44100
N_SAMPLES = 200 # length of fft


def rms(value):
    return np.sqrt(np.mean(np.absolute(value)**2))


def extract_data():
    hrir_raw = gen_data_hrir(DATADIR_HRIR)['data']
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
            # raw data from measurements
            'left': left_raw,      
            'right': right_raw,
            # averaged data of 55 and 65 degrees as we only care about 60
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
    
    anthro = gen_data_anthro(DATADIR_PHY)
    # we only care about head circumference and horizontal ear offset from center.
    # circumference: x16, offset: x5 (Don't forget -1 as index starts from 0, not 1)
    for s in hrir_filtered:
        s_id = int(s['name'].split('_')[1])
        for i in range(len(anthro['id'])):
            if anthro['id'][i] == s_id:
                break
        x = anthro['X'][i]
        s['X'] = x
        s['circumference'] = x[15]
        s['offset'] = x[4]

    # since anthropometric data is assuming human head is symetric, we will only check one side.
    for s in hrir_filtered:
        left = s['left_avg_fft']['left']
        right = s['left_avg_fft']['right']
        diff = abs(left - right)
        s['fft_diff'] = diff

        # yf = s['fft_diff']
        xf = fftfreq(N_SAMPLES, 1/Fs)[:N_SAMPLES//2]
        # fig, ax = plt.subplots()
        # ax.semilogx(xf, 2.0/N_SAMPLES * np.abs(left[0:N_SAMPLES//2]), label='left')
        # ax.semilogx(xf, 2.0/N_SAMPLES * np.abs(right[0:N_SAMPLES//2]), label='right')
        # ax.semilogx(xf, 2.0/N_SAMPLES * np.abs(yf[0:N_SAMPLES//2]), label='diff')
        # plt.grid()
        # plt.legend()
        # plt.show()

        # get low frequency attenuation
        for i in range(len(diff)):
            if diff[i+1] < diff[i]:
                peak_diff = diff[i]
                break
    
        # get frequency mapped to fft index
        xfeed_freq = xf[i]

        # get attenuation in dB
        left_rms = rms(left[:i])
        right_rms = rms(right[:i])
        diff_rms = left_rms - right_rms
        xfeed_att_db = 20 * np.log10(diff_rms)

        s['xfeed_att'] = diff_rms
        s['xfeed_att_db'] = xfeed_att_db
        s['xfeed_freq'] = xfeed_freq

        # print(f"circumference: {s['circumference']}, offset: {s['offset']:.2f}, att_db: {xfeed_att_db:.2f}, freq: {xfeed_freq}")
        
    return hrir_filtered


def statistical_analysis(data):
    # search for corelations within physical parameters and xfeed:

    # first, collect data in a matrix:
    stats_data = np.zeros((len(data[0]['X']), len(data)))
    xfeed_freq = np.zeros(len(data))
    xfeed_db = np.zeros(len(data))
    xfeed_att = np.zeros(len(data))

    for i in range(len(data)):
        for j in range(len(data[i]['X'])):
            stats_data[j][i] = data[i]['X'][j]
        xfeed_freq[i] = data[i]['xfeed_freq']
        xfeed_db[i] = data[i]['xfeed_att_db']
        xfeed_att[i] = data[i]['xfeed_att']
            
    # Run tests: 
    results = []
    meta = []
    for x in [xfeed_freq, xfeed_db, xfeed_att]:
        for sd in range(len(stats_data)):
            # check for nans and skip them! 
            x_test = []
            sd_test = []
            for i in range(len(stats_data[sd])):
                if not np.isnan(stats_data[sd][i]):
                    x_test.append(x[i])
                    sd_test.append(stats_data[sd][i])
            p = pearsonr(x_test, sd_test)
            s = spearmanr(x_test, sd_test)
            k = kendalltau(x_test, sd_test)
            p_values = [i[1] for i in [p,s,k]]
            if all([i<0.001 for i in p_values]):
                print(f'EUREKA! statistically significant x index: {sd}')
            results.append((p, s, k))
            meta.append(sd)

    # for r in range(len(results)):
    #     p_values = [p[1] for p in results[r]]
    #     if all([i<0.05 for i in p_values]):
    #         print(f'EUREKA!: {p_values}: meta: {meta[r]}')
    # pdb.set_trace()

if __name__ == '__main__':
    data = extract_data()
    statistical_analysis(data)
