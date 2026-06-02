# PCA-KRR 기반 거리 Fingerprint 위치 추정 알고리즘 - 12243072 윤서연

## 1. 모티베이션 & 인트로

본 프로젝트의 목표는 18개의 기지국에서 측정된 거리 추정값을 이용하여 사용자의 2차원 위치를 예측하는 것이다. 제공된 데이터는 각 사용자에 대해 18개 기지국이 측정한 RTT 기반 거리 추정값 d_hat, 실제 사용자 위치 p, 그리고 기지국 좌표 BS_positions로 구성되어 있다. 따라서 이 문제는 단순히 하나의 거리값으로 위치를 계산하는 문제가 아니라, 여러 기지국에서 동시에 관측된 거리값의 조합을 이용하여 사용자의 위치를 추정하는 문제로 볼 수 있다.

중간 단계에서는 기지국 좌표와 거리값의 기하학적 관계를 이용하는 WLS 기반 위치 추정 방식을 검토하였다. WLS는 기지국 위치와 거리값이 주어졌을 때 물리적으로 해석하기 쉽고, 별도의 학습 과정 없이 위치를 계산할 수 있다는 장점이 있다. 그러나 실제 거리 측정값에는 반사, 차폐, multipath, 측정 오차와 같은 요인이 포함될 수 있다. 이 경우 특정 기지국의 거리값이 크게 왜곡되면 위치 추정 결과도 크게 흔들릴 수 있다. 또한 WLS에 correction이나 clipping과 같은 보정 규칙을 추가하면 제공된 학습 데이터에는 잘 맞을 수 있지만, hidden test set에 대해서는 과적합될 위험이 있다.

이번 final project에서는 이러한 한계를 줄이기 위해 18개의 거리값 전체를 하나의 distance fingerprint로 해석하였다. 즉, 한 사용자의 위치는 18개 기지국까지의 거리값 조합으로 표현되며, 서로 가까운 위치에 있는 사용자는 비슷한 거리 패턴을 가질 것이라고 보았다. 이 관점에서는 개별 거리값 하나하나의 정확도보다 전체 거리 패턴과 실제 위치 사이의 관계를 학습하는 것이 중요하다.

초기 알고리즘 설계에서는 PCA-Kernel Ridge Regression을 최종 후보로 선택하였다. 먼저 각 기지국 거리값의 스케일 차이를 줄이기 위해 standardization을 적용하고, PCA를 통해 18차원 distance fingerprint의 중복 정보를 줄인 뒤, RBF kernel 기반 Kernel Ridge Regression으로 위치 좌표를 예측하는 구조를 계획하였다. 이후 실제 구현 및 검증 과정에서는 거리값의 범위가 넓다는 점을 고려하여 log1p 변환을 추가하였고, PCA whitening과 hyperparameter 탐색을 함께 적용하였다.

최종 모델은 log1p 변환, StandardScaler, PCA whitening, RBF kernel 기반 Kernel Ridge Regression으로 구성하였다. 이 구조는 거리값의 넓은 분포를 완화하고, feature의 스케일을 정리하며, 거리 fingerprint와 실제 위치 좌표 사이의 비선형 관계를 학습하는 것을 목표로 한다. 또한 hyperparameter는 단순히 학습 데이터에 잘 맞는 값이 아니라, 5-Fold cross-validation에서 P90 위치 오차가 낮게 나타나는 조합을 선택하였다.

## 2. 알고리즘 설명

입력 데이터 d_hat은 18개의 기지국이 각 사용자에 대해 측정한 거리 추정값이다. 전체 사용자 수를 N이라고 하면 d_hat의 크기는 18 × N이고, 실제 위치 좌표 p의 크기는 2 × N이다. 본 알고리즘에서는 각 사용자의 18개 거리값을 하나의 입력 벡터로 사용하기 위해 데이터를 N × 18 형태로 변환하고, 실제 위치 좌표는 N × 2 형태로 변환하여 학습하였다. 즉, 한 행은 한 명의 사용자 샘플을 의미하고, 18개의 열은 각 기지국까지의 거리 추정값을 의미한다.

첫 번째 단계는 log1p 변환이다. 제공된 d_hat 거리값은 최소값과 최대값의 차이가 비교적 크게 나타났다. 이처럼 입력값의 범위가 넓으면 RBF kernel에서 샘플 사이의 거리 계산이 일부 큰 값에 과도하게 영향을 받을 수 있다. 따라서 각 거리값에 log1p 변환을 적용하여 큰 거리값의 영향을 완화하였다. log1p는 log(1 + d)의 형태로 계산되며, 거리값이 양수일 때 안정적으로 적용할 수 있다.

두 번째 단계는 feature scaling이다. 18개의 거리 feature는 모두 거리값이라는 동일한 의미를 가지지만, 기지국의 위치에 따라 평균과 분산이 서로 다르게 나타날 수 있다. 예를 들어 중앙에 가까운 기지국과 가장자리에 있는 기지국은 사용자 위치 분포에 따라 거리값의 범위가 다를 수 있다. 만약 scaling을 적용하지 않으면 값의 크기가 큰 feature가 모델 학습에 더 큰 영향을 줄 수 있다. 따라서 각 기지국별 거리 feature에 대해 평균을 빼고 표준편차로 나누는 standardization을 적용하였다.

| 처리 단계 | 식 | 의미 |
|---|---|---|
| Log transform | z = log(1 + d) | 큰 거리값의 영향을 완화 |
| Standardization | z = (d - μ) / σ | 각 기지국 거리값의 평균과 표준편차 차이를 보정 |

세 번째 단계는 PCA 기반 feature transformation이다. 18개의 거리값은 완전히 독립적인 정보라고 보기 어렵다. 사용자가 특정 방향으로 이동하면 여러 기지국까지의 거리값이 동시에 증가하거나 감소할 수 있기 때문이다. 따라서 18개의 거리 feature 안에는 서로 연관된 정보가 포함되어 있을 가능성이 있다. PCA는 원래 feature들의 주요 변화 방향을 찾고, 데이터를 principal component 공간으로 변환하는 방법이다.

초기에는 PCA를 feature reduction 목적으로 사용하려고 하였으나, 실제 hyperparameter 탐색 결과 최종 모델에서는 n_components가 18로 선택되었다. 이는 본 데이터셋에서는 18개 기지국 거리값 대부분이 위치 추정에 유효한 정보를 포함하고 있으며, 과도한 차원 축소가 오히려 정보 손실을 만들 수 있음을 의미한다. 따라서 최종 모델에서 PCA는 단순한 차원 축소보다는 whitening을 포함한 feature transformation 단계로 사용되었다.

PCA whitening은 각 principal component의 분산을 정규화하는 역할을 한다. 이를 통해 RBF kernel이 특정 분산이 큰 성분에만 과도하게 영향을 받는 것을 줄이고, 여러 성분을 더 균일하게 반영할 수 있다. 따라서 최종 모델에서는 PCA n_components를 18로 유지하면서 whiten=True를 적용하였다.

네 번째 단계는 Kernel Ridge Regression이다. 일반적인 Ridge Regression은 입력 feature와 출력 위치 좌표 사이의 선형 관계를 학습한다. 그러나 거리 fingerprint와 실제 위치 사이의 관계는 단순한 선형 관계로 보기 어렵다. 같은 거리 변화라도 공간상의 위치와 기지국 배치에 따라 실제 좌표 변화가 다르게 나타날 수 있기 때문이다. 따라서 본 알고리즘에서는 RBF kernel을 사용하는 Kernel Ridge Regression을 적용하였다.

RBF kernel은 두 distance fingerprint 사이의 유사도를 계산한다. 두 샘플의 feature가 가까우면 kernel 값이 크게 나타나고, 멀리 떨어져 있으면 kernel 값이 작게 나타난다. 이를 통해 새로운 사용자의 거리 패턴이 학습 데이터의 어떤 패턴과 유사한지 반영하여 위치를 예측할 수 있다. Kernel Ridge Regression은 이러한 kernel 기반 비선형 회귀에 ridge 정규화 항을 포함하므로, 제한된 데이터에 과도하게 맞춰지는 문제를 줄이는 데 도움이 된다.

| 항목 | 식 | 의미 |
|---|---|---|
| RBF Kernel | K(xᵢ, xⱼ) = exp(-γ × \|\|xᵢ - xⱼ\|\|²) | 두 distance fingerprint 사이의 유사도 |
| Ridge 정규화 | Loss = error + α × regularization | 예측 오차를 줄이면서 과적합 억제 |
| 위치 오차 | eᵢ = \|\|pᵢ - p̂ᵢ\|\|₂ | 실제 위치와 예측 위치 사이의 2차원 거리 |
| P90 위치 오차 | percentile(e, 90) | 전체 샘플 중 90%가 이 값 이하의 오차를 갖는 기준 |

학습 과정에서는 5-Fold cross-validation을 사용하였다. 각 fold에서 training data에 대해서만 log 변환, scaling, PCA 변환, KRR 학습이 이루어지고, validation data에는 training data에서 학습된 변환만 적용된다. 이를 통해 validation data의 정보가 전처리 과정에 미리 반영되는 data leakage를 방지하였다.

최종 구현에서 train.py는 제공된 700개 데이터를 이용하여 baseline 모델과 최종 PCA-KRR 모델을 비교하고, RandomizedSearchCV를 통해 hyperparameter를 탐색한다. 탐색 대상에는 log 변환 적용 여부, scaler 종류, PCA component 수, PCA whitening 여부, KRR의 alpha, KRR의 gamma가 포함된다. 최종적으로 5-Fold cross-validation에서 P90 위치 오차가 가장 낮은 모델을 선택하고, 전체 700개 데이터로 다시 학습한 뒤 model.pkl로 저장하였다.

main.py는 학습을 다시 수행하지 않고, 저장된 model.pkl을 불러와 hidden test set의 d_hat에 대해 위치를 예측한다. 채점 기준에 맞게 최종 출력 p_hat은 2 × num_user 형태로 반환되며, 사용자 수는 입력 데이터의 shape에서 동적으로 가져오도록 구성하였다.

본 알고리즘은 Le et al.의 PCA-KRR 기반 fingerprinting indoor positioning 구조를 참고하였다. 해당 논문에서는 실내 위치 추정을 위해 RSS fingerprint를 구성하고, PCA를 이용하여 feature를 줄인 뒤, Kernel Ridge Regression을 통해 위치를 예측하였다. 본 프로젝트에서는 이 흐름 중 여러 신호 또는 거리값을 하나의 fingerprint로 해석한다는 점, PCA를 통해 feature 공간을 변환한다는 점, Kernel Ridge Regression을 이용해 비선형 위치 관계를 학습한다는 점을 참고하였다.

그러나 본 프로젝트의 입력 데이터는 논문과 다르다. 참고 논문은 RSS 값을 fingerprint로 사용하였지만, 본 프로젝트에서는 18개 기지국이 측정한 RTT 기반 거리 추정값 d_hat을 distance fingerprint로 사용하였다. 또한 참고 논문은 radio map 구성, fingerprint point selection, PCA 기반 feature reduction을 포함하지만, 본 구현에서는 제공된 18개 기지국 거리값 전체를 사용하고, cross-validation을 통해 log1p 변환, StandardScaler, PCA whitening, KRR의 alpha와 gamma를 선택하였다. 특히 최종 결과에서 PCA n_components가 18로 선택되었기 때문에, 본 구현에서 PCA는 단순한 차원 축소보다는 whitening을 포함한 feature transformation 단계로 사용되었다.

## 3. Agent AI 활용 방안

본 프로젝트에서는 Agent AI로 ChatGPT를 활용하였다. 다만 AI를 알고리즘을 대신 결정하거나 결과를 자동으로 생성하는 도구로 사용한 것이 아니라, 알고리즘 후보를 비교하고 구현 과정에서 발생할 수 있는 오류를 점검하는 보조 도구로 활용하였다.

알고리즘 선정 단계에서는 WLS, KNN fingerprinting, Ridge Regression, Kernel Ridge Regression의 차이를 비교하는 데 AI를 활용하였다. 이전 중간 과제에서는 WLS 기반 위치 추정을 사용한 경험이 있었기 때문에, 이번 final project에서는 같은 구조를 반복하기보다 18개 거리값 전체를 하나의 distance fingerprint로 해석하는 접근을 적용하고자 하였다. 이 과정에서 각 알고리즘의 장단점, hidden test set에서의 일반화 가능성, baseline으로 비교할 수 있는 방법을 정리하는 데 AI의 도움을 받았다.

데이터 구조 분석 단계에서는 d_hat, p, BS_positions의 shape을 확인하고, scikit-learn 모델에 맞는 입력 형태로 변환하는 과정을 점검하였다. 제공된 d_hat은 18 × 700 형태이고, p는 2 × 700 형태이므로, 학습 과정에서는 샘플이 행 방향이 되도록 각각 700 × 18, 700 × 2 형태로 변환해야 한다. AI는 이 과정에서 shape error가 발생하지 않도록 입력과 출력 구조를 확인하는 데 활용되었다.

구현 단계에서는 train.py와 main.py의 역할을 분리하는 방향을 점검하였다. train.py는 제공된 train data를 이용해 모델을 학습하고 validation 성능을 확인한 뒤 model.pkl을 저장하는 역할을 하도록 구성하였다. 반면 main.py는 hidden test data가 들어왔을 때 저장된 model.pkl을 불러와 예측 결과만 반환하도록 구성하였다. 이때 최종 출력 p_hat이 채점 규격에 맞게 2 × num_user 형태가 되는지 확인하는 과정에서도 AI를 활용하였다.

성능 개선 단계에서는 기존 PCA-KRR 구조에 log1p 변환, PCA whitening, gamma와 alpha 탐색 범위 확장, RandomizedSearchCV, P90 기준 scorer를 적용하는 방향을 검토하였다. 또한 baseline 비교가 불공정해지지 않도록 KNN과 Ridge Regression에도 최종 모델과 동일한 전처리를 적용한 matched baseline을 구성하였다. 이를 통해 성능 차이가 단순히 전처리 때문인지, 또는 RBF Kernel Ridge Regression의 비선형 학습 능력 때문인지 더 공정하게 비교할 수 있도록 하였다.

직접 수행한 부분은 데이터셋 구조 확인, distance fingerprint 관점 설정, 최종 알고리즘 선택, train.py와 main.py 실행, validation 결과 확인, baseline 비교 및 최종 결과 해석이다. AI는 알고리즘 설명 정리, 코드 구조 점검, shape 및 파일 구조 관련 오류 확인, 보고서 문장 구성 보조에 활용하였다. 따라서 본 프로젝트에서 AI는 설계와 구현을 보조하는 도구로 사용되었으며, 최종 알고리즘 선택과 결과 해석은 직접 수행하였다.

## 4. 결과 도출 & 디스커션

성능 평가는 제공된 700개 데이터를 이용하여 5-Fold cross-validation 방식으로 수행하였다. 전체 데이터를 학습에만 사용하면 모델이 실제로 일반화되는지 확인하기 어렵기 때문에, 동일한 fold split을 사용하여 WLS, KNN-matched, Ridge-matched, PCA-KRR을 비교하였다. 본 프로젝트에서는 평균 위치 오차뿐만 아니라 일부 큰 위치 오차가 발생하는 경우도 중요하다고 판단하여, hyperparameter 선택 기준으로 P90 위치 오차를 사용하였다.

P90 위치 오차는 전체 샘플 중 90%가 해당 오차 이하에 들어온다는 의미이다. 평균 오차만 사용하면 일부 큰 오차가 발생하는 상황이 충분히 드러나지 않을 수 있다. 위치 추정 문제에서는 평균적으로 잘 맞는 것도 중요하지만, 특정 사용자에 대해 매우 큰 오차가 발생하는 것을 줄이는 것도 중요하므로 P90 위치 오차를 주요 평가 기준으로 사용하였다. 또한 최종 선택된 모델에 대해서는 CV 평균 위치 오차도 함께 확인하였다.

| 평가 항목 | 설정 |
|---|---|
| 데이터 수 | 700개 제공 샘플 |
| 입력 feature | 18개 기지국 거리 추정값 |
| 출력 | 사용자 위치 좌표 x, y |
| 검증 방식 | 5-Fold Cross Validation |
| hyperparameter 선택 기준 | P90 위치 오차 |
| 추가 확인 지표 | 평균 위치 오차, RMSE, 최대 오차 |
| 최종 출력 형태 | 2 × num_user |

Baseline은 WLS, KNN-matched, Ridge-matched로 설정하였다. WLS는 기지국 좌표와 거리값의 기하학적 관계를 이용하는 물리 기반 baseline이다. KNN-matched는 최종 모델과 동일한 전처리 조건에서 거리 fingerprint의 유사도를 이용하는 fingerprinting baseline이다. Ridge-matched는 동일한 전처리 조건에서 18개 거리 feature와 위치 좌표 사이의 선형 관계를 학습하는 회귀 baseline이다. PCA-KRR은 이들과 비교하여 distance fingerprint의 비선형 관계를 학습하고, log transform, PCA whitening, ridge 정규화를 통해 안정성을 높인 최종 제안 모델이다.

Baseline 비교의 fairness를 높이기 위해 KNN과 Ridge Regression에는 최종 PCA-KRR 모델과 동일한 log1p 변환, StandardScaler, PCA whitening 전처리를 적용한 matched baseline을 추가하였다. 이를 통해 성능 차이가 단순히 전처리 방식의 차이에서만 발생한 것인지, 아니면 최종 회귀 모델인 RBF Kernel Ridge Regression의 비선형 학습 능력에서 발생한 것인지 더 공정하게 비교하고자 하였다.

WLS의 경우에는 기지국 좌표와 거리값의 기하학적 관계를 직접 이용하는 물리 기반 알고리즘이다. 따라서 log 변환된 거리값을 사용하면 원래 거리의 물리적 의미가 깨질 수 있으므로, WLS는 원래 거리값을 사용하는 물리 기반 baseline으로 두었다. 반면 KNN-matched, Ridge-matched, PCA-KRR은 동일한 전처리 조건에서 비교하였다.

| 모델 | 사용 정보 | 전처리 | 핵심 아이디어 | 5-Fold CV P90 위치 오차 |
|---|---|---|---|---:|
| WLS | 기지국 좌표, 거리값 | Raw distance | 거리 원의 기하학적 관계를 이용하여 위치 추정 | 22.8088 m |
| KNN-matched | 거리값 | log1p, StandardScaler, PCA whitening | 동일 전처리 후 유사한 distance fingerprint를 가진 학습 샘플의 위치 이용 | 17.4219 m |
| Ridge-matched | 거리값 | log1p, StandardScaler, PCA whitening | 동일 전처리 후 18개 거리 feature와 위치 좌표 사이의 선형 관계 학습 | 19.7575 m |
| PCA-KRR | 거리값 | log1p, StandardScaler, PCA whitening | 동일 전처리 후 RBF kernel 회귀를 이용하여 비선형 위치 관계 학습 | 13.0973 m |

실험 결과 PCA-KRR 모델의 5-Fold CV P90 위치 오차는 13.0973 m로 나타났다. 이는 WLS의 22.8088 m, KNN-matched의 17.4219 m, Ridge-matched의 19.7575 m보다 낮은 값이다. 따라서 본 데이터셋에서는 전처리만으로 성능이 개선된 것이 아니라, 동일한 전처리 조건에서도 RBF kernel 기반 Kernel Ridge Regression이 거리 fingerprint와 실제 위치 좌표 사이의 비선형 관계를 더 잘 반영한 것으로 판단하였다.

KNN-matched는 동일한 전처리를 적용한 뒤 거리 패턴이 가까운 학습 샘플을 이용하여 위치를 예측하였다. 그러나 KNN은 가까운 일부 샘플에 크게 의존하기 때문에, hidden test sample이 학습 데이터의 특정 위치와 정확히 유사하지 않을 경우 오차가 커질 수 있다. Ridge-matched는 동일한 전처리를 적용했지만 선형 회귀 모델이므로 거리 fingerprint와 위치 좌표 사이의 비선형 관계를 충분히 반영하기 어렵다. 반면 PCA-KRR은 RBF kernel을 통해 여러 학습 샘플과의 유사도를 종합적으로 반영할 수 있어 더 낮은 P90 위치 오차를 보인 것으로 해석하였다.

| 항목 | 선택 값 |
|---|---:|
| Log transform | log1p |
| Scaling | StandardScaler |
| PCA n_components | 18 |
| PCA whiten | True |
| KRR alpha | 0.23969194076200354 |
| KRR gamma | 0.03738463468988912 |
| Kernel | RBF |
| KNN-matched best n_neighbors | 9 |
| KNN-matched best weights | distance |
| Ridge-matched best alpha | 1.0 |

Hyperparameter 탐색 결과 log1p 변환과 PCA whitening을 적용한 모델이 선택되었다. d_hat의 거리값은 최소값과 최대값의 차이가 크기 때문에, log1p 변환을 통해 큰 거리값의 영향을 완화하였다. 또한 PCA whitening을 적용하여 주성분별 분산을 정규화함으로써 RBF kernel의 거리 계산이 특정 성분에 과도하게 영향을 받지 않도록 하였다.

PCA의 component 수는 18이 선택되었다. 이는 본 데이터셋에서는 18개 기지국 거리값 대부분이 위치 추정에 유효한 정보를 포함하고 있으며, 과도한 차원 축소가 오히려 정보 손실을 만들 수 있음을 의미한다. 따라서 최종 모델에서 PCA는 단순한 feature reduction보다는 whitening을 포함한 feature transformation 단계로 사용되었다.

최종 모델은 선택된 hyperparameter를 이용하여 전체 700개 train data로 다시 학습하였다. 이후 학습 데이터 기준의 진단 오차와 cross-validation 오차를 함께 확인하였다. Train error는 모델이 학습 데이터에 얼마나 잘 맞는지를 보여주고, CV error는 fold를 나누었을 때의 일반화 성능을 보여준다.

| 평가 지표 | 결과 |
|---|---:|
| Train 평균 위치 오차 | 4.1743 m |
| Train RMSE | 5.0963 m |
| Train P90 오차 | 7.8028 m |
| Train 최대 오차 | 22.8649 m |
| 5-Fold CV 평균 위치 오차 | 7.4553 m |
| 5-Fold CV P90 위치 오차 | 13.0973 m |

Train 평균 위치 오차는 4.1743 m이고, 5-Fold CV 평균 위치 오차는 7.4553 m로 나타났다. Train 성능이 validation 성능보다 좋게 나타나는 것은 자연스러운 결과이지만, baseline 모델들과 비교했을 때 CV P90 위치 오차가 가장 낮게 나타났기 때문에 최종 모델이 hidden test set에서도 비교적 안정적으로 작동할 가능성이 있다고 판단하였다.

본 알고리즘의 장점은 거리값 하나하나를 독립적인 원의 반지름으로만 해석하지 않고, 18개 거리값 전체를 하나의 fingerprint로 사용한다는 점이다. 또한 log transform과 standardization을 통해 입력 분포를 정리하고, PCA whitening과 RBF kernel을 이용하여 거리 패턴과 위치 좌표 사이의 비선형 관계를 반영할 수 있다. 반면 단점으로는 hyperparameter 선택에 따라 성능이 달라질 수 있고, hidden test set의 분포가 train data와 크게 다를 경우 성능이 떨어질 수 있다는 점이 있다.

향후 개선 방향으로는 기지국별 거리 오차 특성을 분석하여 불안정한 기지국에 낮은 가중치를 부여하는 방법을 고려할 수 있다. 또한 평균 위치 오차와 P90 위치 오차를 함께 최적화하는 방식, 또는 여러 모델의 예측을 ensemble하는 방식을 적용하면 일부 큰 오차를 더 줄일 수 있을 것으로 보인다.

## 5. Reference

| 번호 | 참고문헌 | 참고한 내용 | 본 프로젝트에서 직접 수행한 부분 |
|---|---|---|---|
| [1] | Yanfen Le, Shijialuo Jin, Hena Zhang, Weibin Shi, and Heng Yao, “Fingerprinting Indoor Positioning Method Based on Kernel Ridge Regression with Feature Reduction,” Wireless Communications and Mobile Computing, vol. 2021, Article ID 6631585, pp. 1–12, 2021. | 이 논문에서는 실내 측위 문제에서 측정 신호값을 fingerprint로 구성하고, PCA를 이용하여 feature를 변환 또는 축소한 뒤, Kernel Ridge Regression으로 위치를 추정하는 구조를 제안하였다. 본 프로젝트에서는 이 논문의 전체 흐름 중 fingerprint 기반 위치 추정, PCA 기반 feature transformation, RBF kernel 기반 KRR 회귀 모델을 참고하였다. | 본 프로젝트에서는 논문에서 사용한 RSS fingerprint 대신 18개 기지국의 RTT 기반 거리 추정값 d_hat을 distance fingerprint로 사용하였다. 또한 논문과 달리 제공된 18개 기지국 거리값 전체를 사용하였고, log1p 변환, StandardScaler, PCA whitening, RandomizedSearchCV를 적용하여 본 데이터셋에 맞게 모델을 구성하였다. |
| [2] | scikit-learn documentation, “Kernel Ridge Regression.” | Kernel Ridge Regression의 기본 구조, ridge 정규화 항, kernel 기반 비선형 회귀의 개념을 참고하였다. 특히 RBF kernel을 사용하면 입력 fingerprint 사이의 유사도를 기반으로 비선형적인 위치 관계를 학습할 수 있다는 점을 참고하였다. | 본 프로젝트에서는 KRR을 최종 위치 예측 모델로 사용하였고, alpha와 gamma를 고정하지 않고 cross-validation 기반 RandomizedSearchCV로 탐색하였다. 최종적으로 alpha = 0.23969194076200354, gamma = 0.03738463468988912가 선택되었다. |
| [3] | scikit-learn documentation, “Principal Component Analysis.” | PCA의 n_components 설정, principal component 변환, whitening 옵션의 의미를 참고하였다. PCA whitening은 각 주성분의 분산을 정규화하여 이후 RBF kernel의 거리 계산이 특정 성분에 과도하게 치우치지 않도록 하는 데 활용할 수 있다. | 초기에는 PCA를 feature reduction 목적으로 고려하였으나, validation 결과 n_components = 18이 선택되었다. 따라서 본 프로젝트에서는 PCA를 차원 축소보다는 whitening을 포함한 feature transformation 단계로 사용하였다. |
| [4] | scikit-learn documentation, “StandardScaler, FunctionTransformer, Pipeline.” | 거리 feature의 scale 차이를 줄이기 위한 StandardScaler, log1p 변환을 pipeline에 포함하기 위한 FunctionTransformer, 전처리와 모델을 하나로 묶어 data leakage를 방지하는 Pipeline 구조를 참고하였다. | 본 프로젝트에서는 log1p 변환, StandardScaler, PCA, KRR을 하나의 Pipeline으로 구성하였다. 이를 통해 cross-validation 과정에서 validation data가 scaler나 PCA fitting에 미리 사용되지 않도록 하였다. |
| [5] | scikit-learn documentation, “RandomizedSearchCV.” | 제한된 시간 안에서 alpha와 gamma처럼 연속적인 hyperparameter 공간을 효율적으로 탐색하는 방법을 참고하였다. GridSearchCV는 모든 조합을 탐색하므로 log 변환 여부, PCA whitening 여부, alpha, gamma 범위를 모두 넓히면 계산량이 커질 수 있다. | 본 프로젝트에서는 RandomizedSearchCV를 사용하여 150개의 후보 조합을 5-Fold cross-validation으로 평가하였다. hyperparameter 선택 기준은 평균 오차가 아니라 P90 위치 오차로 설정하여, 일부 큰 위치 오차가 발생하는 경우까지 고려하였다. |
| [6] | scikit-learn documentation, “KNeighborsRegressor” and “Ridge Regression.” | KNN 기반 fingerprinting baseline과 선형 회귀 기반 Ridge baseline을 구현하는 데 참고하였다. KNN은 거리 fingerprint가 유사한 학습 샘플의 위치를 이용하는 방식이고, Ridge는 정규화된 선형 회귀 모델이다. | 본 프로젝트에서는 baseline 비교의 fairness를 높이기 위해 KNN과 Ridge에도 최종 모델과 동일한 log1p, StandardScaler, PCA whitening 전처리를 적용한 matched baseline을 구성하였다. 이를 통해 단순 전처리 차이가 아니라 최종 회귀 방식의 차이를 비교하고자 하였다. |
