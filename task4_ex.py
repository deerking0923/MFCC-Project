##############################################################
#  과제 5 : 10명의 음성(wav)으로 10개의 GMM 모델을 만든 뒤
#           4-fold(=시간을 4등분) 교차검증으로
#           어느 학생(화자)의 목소리인지 맞혀 보는 실험
#
#           └ wav/ 폴더에 10개의 파일이 준비되어 있다고 가정
#              예) f1.wav ~ f5.wav, m1.wav ~ m5.wav
##############################################################

# ┌─────────────────────────────────────────────┐
# │  1. 라이브러리 불러오기                     │
# └─────────────────────────────────────────────┘
import os, glob, math          # 파일 경로 / 리스트업 / 간단한 계산
import numpy as np             # 숫자 배열(행렬) 계산
import librosa                 # 음성 파일 로드 + MFCC 추출
from sklearn.mixture import GaussianMixture   # GMM 모델
from sklearn.metrics import accuracy_score    # 정확도 계산 함수

# ┌─────────────────────────────────────────────┐
# │  2. 실험에 쓸 ‘상수’(파라미터) 설정          │
# └─────────────────────────────────────────────┘
SR      = 16000     # ① 샘플링레이트(Hz) – 음성을 16kHz 로 통일하여 읽기
N_MFCC  = 20        # ② MFCC 차원 – 프레임마다 20개의 숫자를 뽑겠다
N_MIX   = 8         # ③ GMM 컴포넌트 개수 – ‘정규분포 8개’를 섞어서 모델링
FOLD    = 4         # ④ 데이터를 4등분 → 4번 교차검증
WIN     = int(0.025 * SR)   # ⑤ 창 길이 : 25ms → 0.025×16000=400 샘플
HOP     = int(0.010 * SR)   # ⑥ hop 길이 : 10ms 간격으로 MFCC 계산

# ┌─────────────────────────────────────────────┐
# │  3. 함수 정의                               │
# └─────────────────────────────────────────────┘
def get_mfcc(path):
    """
    (1) wav 파일을 16kHz, 모노로 로드
    (2) MFCC(20차원) 시퀀스로 변환
    (3) 반환 형태는 (프레임 수, 20)
    """
    y, _ = librosa.load(path, sr=SR, mono=True)        # 음성 로드
    m = librosa.feature.mfcc(y=y,
                             sr=SR,
                             n_mfcc=N_MFCC,
                             n_fft=WIN,
                             hop_length=HOP)           # MFCC 추출
    return m.T    # librosa 결과는 (20, T)이므로 전치 → (T, 20)

def split4(arr):
    """
    MFCC 배열을 시간순으로 4조각(list)으로 나눠 돌려준다.
    ─ 실제 길이가 4로 딱 나누어떨어지지 않아도
      마지막 조각이 ‘남은 프레임 전부’를 포함하므로 문제 없음.
    """
    step = math.ceil(len(arr) / 4)     # 한 조각당 프레임 수
    return [arr[i*step : (i+1)*step] for i in range(4)]

# ┌─────────────────────────────────────────────┐
# │  4. 데이터 읽어오기                         │
# └─────────────────────────────────────────────┘
wav_dir = 'wav'                                   # wav 폴더 이름
paths = sorted(glob.glob(os.path.join(wav_dir, '*.wav')))
#  ↑ 사전식 정렬 → 예) f1, f2, …, f5, m1, …, m5

assert len(paths) == 10, "✖ wav 폴더에 파일이 10개가 있어야 합니다!"

labels = list(range(10))      # [0,1,2,3,4,5,6,7,8,9]  => 학생 ID

# ◆ 모든 wav를 한 번씩만 읽어 MFCC로 변환해 메모리에 저장(캐싱)
mfcc_parts = [split4(get_mfcc(p)) for p in paths]   # 길이 10 × 4

# ┌─────────────────────────────────────────────┐
# │  5. 4-fold 교차검증                         │
# └─────────────────────────────────────────────┘
fold_acc = []  # fold별 정확도 담을 리스트

for fd in range(FOLD):        # fd = 0,1,2,3
    # ── (1) 학습 데이터 준비 ──────────────────
    train_sets = []           # 학생 0~9 각각 학습용 프레임 모음
    for stu in range(10):
        # stu 학생 wav:   [전체 4조각] - [테스트용 1조각(fd)] = 학습용 3조각
        parts = [mfcc_parts[stu][i] for i in range(4) if i != fd]
        train_sets.append(np.vstack(parts))   # 세 조각 세로로 이어 붙이기

    # ── (2) 10개 GMM 학습 ────────────────────
    models = []
    for tr in train_sets:
        gmm = GaussianMixture(n_components=N_MIX,
                              covariance_type='diag',
                              max_iter=200,
                              random_state=0)      # seed 고정 → 재현 가능
        gmm.fit(tr)
        models.append(gmm)

    # ── (3) 테스트 & 예측 ────────────────────
    y_true, y_pred = [], []
    for stu in range(10):
        test_feat = mfcc_parts[stu][fd]            # stu 학생의 테스트 조각
        scores = [mdl.score(test_feat) for mdl in models]  # 10개 모델 우도
        y_pred.append(int(np.argmax(scores)))      # 최고 점수 모델 = 예측 ID
        y_true.append(stu)                         # 정답은 자기 ID

    # ── (4) fold 정확도 계산 ─────────────────
    acc = accuracy_score(y_true, y_pred)
    fold_acc.append(acc)
    print(f'fold{fd+1}: {acc*100:.1f}%   preds={y_pred}')

# ┌─────────────────────────────────────────────┐
# │  6. 최종 평균/표준편차                      │
# └─────────────────────────────────────────────┘
print("\n============ 결과 요약 ============")
print(f'평균 정확도 : {np.mean(fold_acc)*100:.1f}%')
print(f'표준편차    : {np.std(fold_acc)*100:.1f}%')
