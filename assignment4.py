"""
gender_gmm_cv.py  ─  과제 4
wav/ 폴더 안에
   f1.wav … f5.wav   # female
   m1.wav … m5.wav   # male
총 10 개 파일이 있다고 가정
"""

# ───────── 라이브러리 ─────────
import os, glob, math, warnings
import numpy as np
import librosa
from sklearn.mixture import GaussianMixture
from sklearn.metrics  import accuracy_score
warnings.filterwarnings("ignore")

# ───────── 하이퍼파라미터 ─────────
SR          = 16_000        # 고정(16 kHz)
N_MFCC      = 20
N_COMPONENT = 8             # GMM mixture 수
FRAME_LEN   = int(0.025*SR) # 25 ms
HOP_LEN     = int(0.010*SR) # 10 ms
N_FOLDS     = 4             # 7.5 s 학습 / 2.5 s 테스트

# ───────── 공용 함수 ─────────
def extract_mfcc(path: str) -> np.ndarray:
    y, _ = librosa.load(path, sr=SR, mono=True)
    mfcc = librosa.feature.mfcc(y, sr=SR, n_mfcc=N_MFCC,
                                n_fft=FRAME_LEN, hop_length=HOP_LEN)
    return mfcc.T                       # (frames, 20)

def time_chunks(feat: np.ndarray, n=4):
    stride = math.ceil(len(feat)/n)
    return [feat[i*stride:(i+1)*stride] for i in range(n)]

# ───────── wav 목록 수집 ─────────
def load_filelist(root="wav"):
    # 1) 파일 패턴에 맞춰 읽어오기
    female = sorted(glob.glob(os.path.join(root, "f[1-5].wav")))
    male   = sorted(glob.glob(os.path.join(root, "m[1-5].wav")))

    # 2) 리스트 합치기 + 레이블 생성 (1=female, 0=male)
    files  = female + male
    labels = [1] * len(female) + [0] * len(male)

    # 3) 유효성 검사
    assert len(files) == 10, "wav 폴더에 f1~f5, m1~m5 10개가 모두 있어야 합니다."
    return files, labels


wav_files, wav_labels = load_filelist()

# ───────── 4-fold CV ─────────
fold_acc = []
for fold in range(N_FOLDS):
    Xtr_f, Xtr_m = [], []
    y_test, y_pred = [], []

    # ── ① train / test split ──
    for path, lab in zip(wav_files, wav_labels):
        chunks = time_chunks(extract_mfcc(path), N_FOLDS)
        train  = np.vstack([c for i,c in enumerate(chunks) if i != fold])
        (Xtr_f if lab == 1 else Xtr_m).append(train)
        y_test.append(lab)              # GT for this file (파일 단위)

    # ── ② 두 GMM 학습 ──
    gmm_f = GaussianMixture(N_COMPONENT, covariance_type='diag',
                            max_iter=200, random_state=0).fit(np.vstack(Xtr_f))
    gmm_m = GaussianMixture(N_COMPONENT, covariance_type='diag',
                            max_iter=200, random_state=0).fit(np.vstack(Xtr_m))

    # ── ③ fold 평가 ──
    print(f"\n── Fold {fold+1} ─────────────────────────────────────────")
    print(f"{'File':<8} {'logF':>10} {'logM':>10}  Pred  True  Hit")
    y_pred.clear()        # 비워 준 뒤 다시 채우기
    for path, lab in zip(wav_files, wav_labels):
        test_feat = time_chunks(extract_mfcc(path), N_FOLDS)[fold]
        log_f = gmm_f.score(test_feat)        # frame-avg log-likelihood
        log_m = gmm_m.score(test_feat)

        pred  = 1 if log_f > log_m else 0
        y_pred.append(pred)

        hit   = '✔' if pred == lab else '✘'
        fname = os.path.basename(path)
        print(f"{fname:<8} {log_f:10.2f} {log_m:10.2f}    {pred:^3}   {lab:^3}   {hit}")

    # ── ④ fold 정확도 ──
    acc = accuracy_score(y_test, y_pred)
    fold_acc.append(acc)
    print(f"Fold {fold+1} accuracy = {acc*100:.2f} %")

# ───────── 결과 요약 ─────────
print("\n=== 4-fold 평균 성능 ===")
print(f"평균 정확도 : {np.mean(fold_acc)*100:.2f} %")
print(f"표준편차    : {np.std(fold_acc)*100:.2f} %")
