# XGBoost from Scratch

이 프로젝트는 XGBoost 알고리즘을 처음부터 Python으로 구현한 것입니다. 이 구현에서는 기본적인 Decision Tree, Gradient Boosting, Column Subsampling 등을 포함합니다. 이 README 파일은 프로젝트의 구조, 설치 방법, 사용법 등을 설명합니다.

## XGBoost 알고리즘 설명

`XGBoost(Extreme Gradient Boosting)`는 Gradient Boosting 알고리즘을 기반으로 한 향상된 기계 학습 라이브러리입니다. 주로 회귀와 분류 문제를 해결하는 데 사용되며, 높은 성능과 효율성으로 잘 알려져 있습니다. XGBoost는 다음과 같은 기능을 제공합니다:

1. **Gradient Boosting**: 각 트리가 순차적으로 학습되며 이전 트리의 오차를 줄이기 위해 새로운 트리가 추가됩니다. Gradient Boosting은 여러 개의 약한 학습기를 순차적으로 학습시키고 이들을 결합하여 강한 학습기를 만드는 앙상블 학습 방법입니다. 각 단계에서 모델은 이전 단계의 오차를 줄이기 위해 새로운 모델을 추가합니다. XGBoost는 Gradient Boosting을 효율적으로 구현하여 높은 예측 성능을 제공합니다.

> 본 프로젝트에서는 다음 코드를 통해 구현됨

```python
## GradientBoostingRegressor Class
for booster in range(boosting_rounds):
    pseudo_residuals = self.negativeMeanSquaredErrorDerivitive(y, self.base_pred)
    boosting_tree = DecisionTreeRegressor().fit(X, pseudo_residuals, depth=depth, min_leaf=min_leaf, gamma=gamma, lambda_=lambda_, colsample_bytree=colsample_bytree)
    self.base_pred += self.learning_rate * boosting_tree.predict(X)
    self.estimators.append(boosting_tree)
```

2. **Regularization**: L1 및 L2 정규화를 통해 모델의 복잡성을 줄이고 과적합을 방지합니다.

> 본 프로젝트는 다음 코드를 통해 구현됨

```python
self.gamma = gamma
self.lambda_ = lambda_
```

3. **Column Subsampling**: 트리를 만들 때마다 무작위로 선택된 하위 특성 집합을 사용하여 모델의 다양성을 증가시킵니다.

> 본 프로젝트에서는 다음 코드를 통해 구현됨

```python
## Node Class
self.selected_cols = random.sample(range(self.col_count), int(self.col_count * self.colsample_bytree))
```

4. **Tree Pruning**: 최대 깊이에 도달한 후에도 분할을 시도하고, 이득이 충분하지 않으면 가지를 잘라내어 최적의 트리 구조를 유지합니다.

> 본 프로젝트에서는 다음 코드를 통해 구현됨

```python
## Node Class
@property
def is_leaf(self):
    return self.score == float('-inf') or self.depth <= 0

def gain(self, lhs, rhs):
    gain = ((lhs_gradient**2 / (lhs_n_intances + self.lambda_)) + 
            (rhs_gradient**2 / (rhs_n_intances + self.lambda_)) - 
            ((lhs_gradient + rhs_gradient)**2 / (lhs_n_intances + rhs_n_intances + self.lambda_)) - self.gamma)
    return gain
```

5. **Parallel Processing**: 병렬 처리를 통해 트리 노드를 효율적으로 구축합니다.

### XGBoost의 장점

- **고성능**: 최적화된 알고리즘과 병렬 처리를 통해 빠른 학습과 예측을 제공합니다.
- **유연성**: 다양한 손실 함수와 사용자 정의 목적 함수를 지원합니다.
- **확장성**: 대규모 데이터셋에 대해 효율적으로 작동하며, 분산 환경에서도 실행 가능합니다.
- **과적합 방지**: 정규화 및 조기 종료(pruning) 기술을 사용하여 과적합을 방지합니다.

## 프로젝트 구조

### `__init__.py`
- 패키지 초기화 파일로, 각 모듈을 임포트합니다.

### `node.py`
- 결정 트리의 노드(Node) 클래스를 정의합니다.
  - `__init__`: 노드를 초기화합니다.
  - `find_varsplit`: 최적의 분할 지점을 찾습니다.
  - `find_greedy_split`: 주어진 특성에 대해 분할 지점을 찾습니다.
  - `gain`: 분할 지점의 이득을 계산합니다.
  - `compute_gamma`: 리프 노드 값을 계산합니다.
  - `split_col`: 분할된 열의 값을 반환합니다.
  - `is_leaf`: 리프 노드 여부를 확인합니다.
  - `predict`: 예측 값을 반환합니다.
  
### `tree.py`
- 결정 트리 회귀 모델(DecisionTreeRegressor) 클래스를 정의합니다.
  - `fit`: 데이터를 사용하여 모델을 훈련시킵니다.
  - `predict`: 입력 데이터에 대해 예측 값을 반환합니다.
  
### `gradient_boosting.py`
- Gradient Boosting 회귀 모델(GradientBoostingRegressor) 클래스를 정의합니다.
  - `MeanSquaredError`: 평균 제곱 오차를 계산합니다.
  - `negativeMeanSquaredErrorDerivitive`: 평균 제곱 오차의 음의 도함수를 계산합니다.
  - `fit`: 데이터를 사용하여 Gradient Boosting 모델을 훈련시킵니다.
  - `predict`: 입력 데이터에 대해 예측 값을 반환합니다.
  
## 설치 방법

이 프로젝트는 별도의 라이브러리 설치가 필요하지 않지만, Python 3.x와 Pandas, NumPy가 필요합니다. 필요한 라이브러리를 설치하려면 다음 명령어를 사용하세요:

```bash
pip install pandas numpy
```

## 사용법

main.py를 실행합니다.