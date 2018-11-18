---
title:  "Kernel PCA"
categories: [jekyll]
tags: [Kernel PCA, PCA, Kernel]
---

 이번 포스팅에서는 **Kernel PCA**에 대하여 알아보겠습니다. 본 포스팅의 주요 내용은 고려대학교 강필성 교수님의 강의를 정리하여 작성하였습니다.

 Kernel PCA에 대해 알아보기 전에, **PCA**와 **Kernel Trick**에 대해 간단히 살펴보겠습니다. 먼저 PCA(Principal Component Analysis)는 차원 축소 기법 중 하나로, 주어진 데이터의 분산을 최대한 보존하면서 고차원 상의 데이터를 저차원 데이터로 변환하는 기법입니다.
![](https://github.com/jieunchoi1120/jieunchoi1120.github.io/blob/master/images/post/geometric-PCA-5-and-6-first-component-with-projections-and-second-component.png?raw=true" alt="geometric-PCA-5-and-6-first-component-with-projections-and-second-component.png)
![](https://github.com/jieunchoi1120/jieunchoi1120.github.io/blob/master/images/post/geometric-PCA-7-and-8-second-component-and-both-components.png?raw=true" alt="geometric-PCA-7-and-8-second-component-and-both-components.png)