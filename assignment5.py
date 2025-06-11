# 과제5  ─  10명 음성 -> 10-개 GMM, 4-fold 시간 분할 CV
import os, glob, math
import numpy as np
import librosa
from sklearn.mixture import GaussianMixture
from sklearn.metrics  import accuracy_score

# ---------- 파라미터 ----------
SR      = 16000
N_MFCC  = 20
N_MIX   = 8
FOLD    = 4
WIN     = int(0.025*SR)   # 25 ms
HOP     = int(0.010*SR)   # 10 ms

# ---------- 함수 ----------
def get_mfcc(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    m = librosa.feature.mfcc(y, sr=SR, n_mfcc=N_MFCC,
                             n_fft=WIN, hop_length=HOP)
    return m.T                        # (frame, 20)

def split4(arr):
    step = math.ceil(len(arr)/4)
    return [arr[i*step:(i+1)*step] for i in range(4)]

# ---------- 데이터 읽기 ----------
wav_dir = 'wav'                       # 폴더 안에 10 개 wav
paths   = sorted(glob.glob(os.path.join(wav_dir, '*.wav')))
assert len(paths) == 10, 'wav 폴더에 파일이 10 개여야 합니다'

mfcc_parts = [split4(get_mfcc(p)) for p in paths]  # 미리 계산 · 캐시
labels     = list(range(10))                       # 0~9 = 학생 ID

# ---------- 교차검증 ----------
fold_acc = []

for fd in range(FOLD):
    # 1) 학생별 학습용 프레임 모으기
    train_sets = []
    for idx in range(10):
        parts = [mfcc_parts[idx][i] for i in range(4) if i != fd]
        train_sets.append(np.vstack(parts))        # (총프레임, 20)

    # 2) 10 개 GMM 학습
    models = []
    for tr in train_sets:
        g = GaussianMixture(N_MIX, covariance_type='diag',
                            random_state=0).fit(tr)
        models.append(g)

    # 3) 테스트 & 예측
    y_true, y_pred = [], []
    for idx in range(10):
        test_feat = mfcc_parts[idx][fd]            # ¼ 구간
        scores = [m.score(test_feat) for m in models]
        y_pred.append(int(np.argmax(scores)))
        y_true.append(idx)

    acc = accuracy_score(y_true, y_pred)
    fold_acc.append(acc)
    print(f'fold{fd+1}: {acc*100:.1f}%  preds={y_pred}')

# ---------- 결과 ----------
print(f'avg : {np.mean(fold_acc)*100:.1f}%')
print(f'std : {np.std(fold_acc)*100:.1f}%')
