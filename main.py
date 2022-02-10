import pdb
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, fftfreq
from scipy.stats import pearsonr, spearmanr, kendalltau

from generate_data import gen_data_hrir, gen_data_anthro
from x_names import x_params_map

DATADIR_HRIR = 'standard_hrir_database'
DATADIR_PHY = 'anthropometry'
Fs = 44100
N_SAMPLES = 200 # length of fft


def rms(value):
    return np.sqrt(np.mean(np.absolute(value)**2))


def extract_data():
    hrir_raw = gen_data_hrir(DATADIR_HRIR)['data']
    # hrir array is 3d, but we care only about +-30 deg azimuth and 0 deg elevation (perfect listening position):
    # this corresponds to:
    # hrir_r[7][8] for az -65 and 0 el  <left side
    # hrir_r[19][8] for az 55 and 0 el  <right side
    # same for left
    hrir_filtered = []
    for s in hrir_raw:
        left = {
            'right': s['hrir_r'][7][8],
            'left': s['hrir_l'][7][8],
        }
        right = {
            'right': s['hrir_r'][19][8],
            'left': s['hrir_l'][19][8],
        }
        name = s['name'][0]


        hrir_filtered.append({
            # raw data from measurements
            'left': left,      
            'right': right,
            'left_fft': {
                'right': fft(left['right']),
                'left': fft(left['left'])
            },
            'right_avg_fft': {
                'right': fft(right['right']),
                'left': fft(right['left'])
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
        left = s['left_fft']['left']
        right = s['left_fft']['right']
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
        for i in range(1, len(diff)):
            if diff[i+1] < diff[i]:
                peak_diff = diff[i]
                break
    
        # get frequency mapped to fft index
        xfeed_freq = xf[i]
        if xfeed_freq == 0:
            pdb.set_trace()

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
    x_params = {
        'x_freq': xfeed_freq,
        'x_att_db': xfeed_db,
        'x_att': xfeed_att
    }
    for name, x in x_params.items():
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

            results.append((p, s, k))
            meta.append((name, sd))

    for r in range(len(results)):
        p_values = [p[1] for p in results[r]]
        significance = [0.05, 0.01, 0.001]
        for s in significance:
            if all([i<s for i in p_values]):
                print(f"{meta[r][0]} correlates to {x_params_map[str(meta[r][1])]} with {s} significance")

if __name__ == '__main__':
    data = extract_data()
    statistical_analysis(data)
    # TODO: plot statistically significant data
    # TODO: create usable results
