---
title:  "Kernel PCA"
categories: [jekyll]
tags: [Kernel PCA, PCA, Kernel]
---

*본 포스팅의 주요 내용은 고려대학교 강필성 교수님의 강의를 정리하여 작성하였습니다.*

　이번 포스팅에서는 비선형 차원축소 기법 중 하나인 **Kernel PCA**에 대하여 알아보겠습니다. 이름에서 알 수 있듯이 Kernel trick과 PCA가 함께 사용되는 기법이기 때문에, 먼저 **Kernel Trick**과 **PCA**에 대해 간단히 살펴보겠습니다.

#### Kernel Trick
![](https://github.com/jieunchoi1120/jieunchoi1120.github.io/blob/master/images/post/kernel_trick.png?raw=true" alt="kernel_trick.png)
　위 그림과 같이 input space에서 선형 분류가 불가능한 데이터를 mapping function Φ를 통해 고차원 공간(feature space)상에 mapping하면, 데이터를 선형 분류하는 hyperplane을 찾을 수 있습니다. 하지만 고차원 mapping은 많은 연산량이 소요된다는 문제가 있습니다. 이런 문제를 해결하면서, 고차원의 이점을 취하는 방법이 바로 **Kernel Trick**입니다.
　kernel trick은 input space의 두 벡터 xi, xj를 받아서 고차원 상에서의 내적 값을 출력하는 kernel fucntion K를 찾습니다. 다시 말해 데이터를 고차원 상에 mapping하지 않고도 데이터를 고차원 상에 mapping한 것과 같은 효과를 얻는 것인데, 이를 수식으로 표현하면 다음과 같습니다.

$$K(x_i,x_j)=\phi(x_i)^T\phi(x_i)$$


#### PCA(Principal Component Analysis)
　PCA는 차원 축소 기법 중 하나로, 주어진 데이터의 분산을 최대한 보존하면서 고차원 상의 데이터를 저차원 데이터로 변환하는 기법입니다. 아래 그림([출처](https://learnche.org/pid/latent-variable-modelling/principal-component-analysis/geometric-explanation-of-pca))에서와 같이 데이터의 분산을 최대한 보존하는, 서로 직교(orthogonal)하는 축(component)을 찾고 그 축에 데이터를 projection함으로써 데이터의 차원을 줄이는 동시에 데이터에 포함되어 있는 noise를 제거할 수 있습니다. 아래 예시에서는 3차원이었던 데이터를 2개의 축(1st component와 2nd component)에 projection함으로써 2차원 데이터로 변환하는 과정을 보여주고 있습니다.

![](https://github.com/jieunchoi1120/jieunchoi1120.github.io/blob/master/images/post/geometric-PCA-5-and-6-first-component-with-projections-and-second-component.png?raw=true" alt="geometric-PCA-5-and-6-first-component-with-projections-and-second-component.png)
![](https://github.com/jieunchoi1120/jieunchoi1120.github.io/blob/master/images/post/geometric-PCA-7-and-8-second-component-and-both-components.png?raw=true" alt="geometric-PCA-7-and-8-second-component-and-both-components.png)

　PCA에서 데이터를 projection하는 축(compoenet)에 대하여 좀 더 자세히 살펴보겠습니다. 이 축은 '데이터의 분산을 최대한 보존하는 특성을 갖는다'라고 앞서 언급한 바 있는데요. 이는 곧 '데이터와 preojected data의 거리(residual)를 최소화하는 특성을 갖는다'는 말과 같다고 볼 수 있습니다. 원데이터의 분산(D3), 축에 의해 보존되는 분산(D1)과 projection 과정에서 손실되는 분산(D2)은 다음([출처](http://alexhwilliams.info/itsneuronalblog/2016/03/27/pca/))과 같은 관계에 있기 때문입니다.

![](https://github.com/jieunchoi1120/jieunchoi1120.github.io/blob/master/images/post/projection_intuition.png?raw=true" alt="projection_intuition.png)

　다시 말해 PCA의 목적은 데이터의 분산을 최대한 보존하는, 데이터와 preojected data의 거리를 최소화하는 linear subspace를 찾는 것입니다. 그런데 PCA를 비선형 데이터에 적용하면 어떻게 될까요? 아래 그림([출처](https://www.analyticsvidhya.com/blog/2017/03/questions-dimensionality-reduction-data-scientist/))은 PCA와 비선형 차원 축소 기법인 Self Organizing Map(SOM)에 비선형 데이터를 적용한 결과를 비교하여 보여주고 있습니다. 그림에서 볼 수 있듯이, 아래 데이터를 파란 색의 축에 projection하면 많은 양의 분산(D2)이 손실될 것입니다. 따라서 **PCA는 비선형 데이터에 적합하지 않은 한계점을 갖습니다.**

![](https://github.com/jieunchoi1120/jieunchoi1120.github.io/blob/master/images/post/pca_linear.png?raw=true" alt="pca_linear.png)

#### Kernel PCA
 ![](https://github.com/jieunchoi1120/jieunchoi1120.github.io/blob/master/images/post/kpca.png?raw=true" alt="kpca.png)
　이와 같은 한계점의 대안으로, Kenel PCA를 사용할 수 있습니다. Kernel PCA의 핵심 아이디어는 비선형 kernel function Φ을 통해 데이터를 고차원 공간(F)에 mapping한 뒤, 고차원 공간(F)에서 PCA를 수행함으로써 다시 저차원 공간에 projection한다는 것입니다. Kernel PCA의 수행 과정을 수식으로 나타내면 다음과 같습니다.
　먼저, 고차원 공간(feature space) 상에 mapping된 데이터가 centering되어 있어 평균이 0이라고 가정합니다.

$$m^\phi={1 \over N}\sum_{i=1}^N\phi(x_i)=0$$

이 데이터의 평균이 0이기 때문에, 공분산행렬 C를 구하면 다음과 같습니다.

$$C^\phi={1 \over N}\sum_{i=1}^N(\phi(x_i)-m^\phi)(\phi(x_i)-m^\phi)^T={1 \over N}\sum_{i=1}^N\phi(x_i)\phi(x_i)^T$$

공분산행렬 C의 eigenvalue λ와 eigenvector v는 다음과 같이 구할 수 있습니다.

$$C^\phi v_k=\lambda_k v_k$$

위 식에 공분산행렬 C의 값을 대입합니다.

$${1 \over N}\sum_{i=1}^N\phi(x_i)\phi(x_i)^Tv_k=\lambda_k v_k$$

여기에서 공분산행렬 C의 eigenvector v는 고차원 상에 mapping된 데이터들의 선형 결합으로 표현이 가능하며, 이를 위 식에 대입하면 아래와 같습니다. (because Φ(xi)vk is just a scalar)

$$v_k={1 \over \lambda N}\sum_{i=1}^N\phi(x_i)\phi(x_i)^Tv_k={1 \over \lambda N}\sum_{i=1}^N\phi(x_i)v_k\phi(x_i)^T$$

$$=\sum_{i=1}^N\alpha_{ki}\phi(x_i)$$

$${1 \over N}\sum_{i=1}^N\phi(x_i)\phi(x_i)^T\sum_{j=1}^N\alpha_{kj}\phi(x_j)=\lambda_k \sum_{i=1}^N\alpha_{kj}\phi(x_i)$$

$${1 \over N}\sum_{i=1}^N\phi(x_i)\sum_{j=1}^N\alpha_{kj}\phi(x_i)^T\phi(x_j)=\lambda_k \sum_{i=1}^N\alpha_{kj}\phi(x_i)$$

앞서 Kernel Trick 파트에서 말씀드린 바와 같이, 고차원 mapping은 많은 연산량이 소요된다는 문제가 있기 때문에 Kernel PCA에서도 Kernel Trick을 사용하게 됩니다. 이를 위해 먼저 Kernel function을 정의합니다.

$$K(x_i,x_j)=\phi(x_i)^T\phi(x_j)$$

양 변에 Φ(xl)을 곱하여 위에서 정의한 kernel function으로 치환합니다.

$${1 \over N}\sum_{i=1}^N\phi(x_l)^T\phi(x_i)\sum_{j=1}^N\alpha_{kj}\phi(x_i)^T\phi(x_j)=\lambda_k \sum_{i=1}^N\alpha_{kj}\phi(x_l)^T\phi(x_i)$$

$${1 \over N}\sum_{i=1}^NK(x_l, x_i)\sum_{j=1}^N\alpha_{kj}K(x_i, x_l)=\lambda_k \sum_{i=1}^N\alpha_{kj}K(x_l, x_i)$$

위 식을 matirix notation을 이용하여 정리하면 다음과 같습니다. (Finding the eigenvector vk is equivalent to finding the coefficients alphak)

$$K^2\alpha_k=\lambda_k N K \alpha_k$$

$$K\alpha_k=\lambda_k N \alpha_k$$
　따라서, Kernel PCA의 수행 결과는 다음과 같이 정리할 수 있습니다.

$$y_k(x)=\phi(x)^Tv_k=\sum_{i=1}^N\alpha_{ki}K(x,x_i)$$


#### Python을 이용한 Kernel PCA 구현
　아이리스 데이터를 이용하여 linear PCA와 Kernel PCA의 결과를 비교해 보았습니다. 파이썬 코드는 아래와 같습니다.
``` ruby
import pandas as pd
import numpy as np

# 아이리스 데이터를 불러옵니다.
df = pd.read_csv(
    filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
    header=None,
    sep=',')

df.columns = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']

#X, y 변수를 지정합니다.
X = df.iloc[:,0:4].values
y = df.iloc[:,4].values
```
　polynomial kernel을 이용한 Kernel PCA 결과입니다. 
``` ruby
from sklearn.decomposition import KernelPCA
kpca0 = KernelPCA(n_components=2, kernel='poly')
Y = kpca0.fit_transform(X)

with plt.style.context("seaborn-darkgrid"):
    for l in label:
        plt.scatter(Y[y==l,0], Y[y==l,1],label=l)
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.legend()
    plt.show()
```
　linear PCA 결과입니다.
``` ruby
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
Y_ = pca.fit_transform(X)

with plt.style.context("seaborn-darkgrid"):
    for l in label:
        plt.scatter(Y_[y==l,0], Y_[y==l,1],label=l)
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.legend()
    plt.show()
```
