import scipy.io as sio

from pathlib import Path

def gen_data(datadir):
    hrir_all = []
    failed_to_open = []
    for path in Path(datadir).iterdir():
        if path.is_dir():
            for p in path.iterdir():
                if p.suffix == '.mat':
                    try:
                        hrir_all.append(sio.loadmat(p))
                    except:
                        failed_to_open.append(p)
                    
    print(f'collected {len(hrir_all)} .mat files')
    return {
        'data': hrir_all,
        'failed': failed_to_open
    }
