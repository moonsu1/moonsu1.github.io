---
title:  "Kernel PCA"
categories: [jekyll]
tags: [Kernel PCA, PCA, Kernel]
use_math: true
---

*본 포스팅의 주요 내용은 고려대학교 강필성 교수님의 강의를 정리하여 작성하였습니다.*

　이번 포스팅에서는 비선형 차원축소 기법 중 하나인 **Kernel PCA**에 대하여 알아보겠습니다. 이름에서 알 수 있듯이 Kernel trick과 PCA가 함께 사용되는 기법이기 때문에, 먼저 **Kernel Trick**과 **PCA**에 대해 간단히 살펴보겠습니다.

#### Kernel Trick
![](https://github.com/jieunchoi1120/jieunchoi1120.github.io/blob/master/images/post/kernel_trick.png?raw=true" alt="kernel_trick.png)
　위 그림과 같이 input space에서 선형 분류가 불가능한 데이터를 mapping function Φ를 통해 고차원 공간(feature space)상에 mapping하면, 데이터를 선형 분류하는 hyperplane을 찾을 수 있습니다. 하지만 고차원 mapping은 많은 연산량이 소요된다는 문제가 있습니다. 이런 문제를 해결하면서, 고차원의 이점을 취하는 방법이 바로 **Kernel Trick**입니다.
　kernel trick은 input space의 두 벡터 xi, xj를 받아서 고차원 상에서의 내적 값을 출력하는 kernel fucntion K를 찾습니다. 다시 말해 데이터를 고차원 상에 mapping하지 않고도 데이터를 고차원 상에 mapping한 것과 같은 효과를 얻는 것인데, 이를 수식으로 표현하면 다음과 같습니다.
$$K(x_i,x_j)=\phi(x_i)^T\phi(x_i)$$

#### PCA(Principal Component Analysis)
　PCA는 차원 축소 기법 중 하나로, 주어진 데이터의 분산을 최대한 보존하면서 고차원 상의 데이터를 저차원 데이터로 변환하는 기법입니다. 아래 그림([출처](https://learnche.org/pid/latent-variable-modelling/principal-component-analysis/geometric-explanation-of-pca))에서와 같이, 데이터의 분산을 최대한 보존하는 축(component)을 찾고 그 축에 데이터를 projection시킴으로써 데이터의 차원을 줄이는 동시에 데이터에 포함되어 있는 noise를 제거할 수 있습니다.
![](https://github.com/jieunchoi1120/jieunchoi1120.github.io/blob/master/images/post/geometric-PCA-5-and-6-first-component-with-projections-and-second-component.png?raw=true" alt="geometric-PCA-5-and-6-first-component-with-projections-and-second-component.png)
![](https://github.com/jieunchoi1120/jieunchoi1120.github.io/blob/master/images/post/geometric-PCA-7-and-8-second-component-and-both-components.png?raw=true" alt="geometric-PCA-7-and-8-second-component-and-both-components.png)

#### Kernel PCA
　Kernel PCA는 데이터를 선형 분류가 가능한 고차원 공간에 mapping한 뒤, PCA를 통해 다시 저차원 공간에 projection하는 기법입니다. 아래 그림(출처: Pattern recognition and Machine Learning, Bishop)
![](https://github.com/jieunchoi1120/jieunchoi1120.github.io/blob/master/images/post/kpca.png?raw=true" alt="kpca.png)