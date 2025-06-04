# ------------ 라이브러리 ------------
import os, glob, math
import numpy as np
import librosa
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score

# ------------ 설정값 ------------
SR = 16000
N_MFCC = 20
N_MIX  = 8      # GMM 컴포넌트
N_FOLD = 4
WIN = int(0.025 * SR)
HOP = int(0.010 * SR)

# ------------ 함수 ------------
def mfcc(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    m = librosa.feature.mfcc(y, sr=SR, n_mfcc=N_MFCC,
                             n_fft=WIN, hop_length=HOP)
    return m.T

def split(arr, n=4):
    step = math.ceil(len(arr)/n)
    return [arr[i*step:(i+1)*step] for i in range(n)]

# ------------ 파일 불러오기 ------------
#wav파일 경로 설정. wav 파일 만들어서 불러오기 했어요.
#음성 파일 이름도 바꿔서 다시 첨부했습니다...!
wav_dir = 'wav'
fs = [os.path.join(wav_dir, f"f{i}.wav") for i in range(1,6)]
ms = [os.path.join(wav_dir, f"m{i}.wav") for i in range(1,6)]
paths = fs + ms
labs  = [1]*5 + [0]*5     # 1=f, 0=m

# ------------ 교차검증 ------------
acc_all = []

for fold in range(N_FOLD):
    f_tr, m_tr = [], []
    y_true, y_pred = [], []

    # 학습셋 만들기
    for p, lb in zip(paths, labs):
        parts = split(mfcc(p), N_FOLD)
        train = np.vstack([parts[i] for i in range(N_FOLD) if i!=fold])
        (f_tr if lb else m_tr).append(train)
        y_true.append(lb)

    g_f = GaussianMixture(N_MIX, covariance_type='diag',
                          random_state=0).fit(np.vstack(f_tr))
    g_m = GaussianMixture(N_MIX, covariance_type='diag',
                          random_state=0).fit(np.vstack(m_tr))

    # 테스트
    for p in paths:
        feat = split(mfcc(p), N_FOLD)[fold]
        pred = 1 if g_f.score(feat) > g_m.score(feat) else 0
        y_pred.append(pred)

    acc = accuracy_score(y_true, y_pred)
    acc_all.append(acc*100)
    print(f"fold{fold+1}: {acc*100:.1f}%")

print(f"avg: {np.mean(acc_all):.1f}  std: {np.std(acc_all):.1f}")
