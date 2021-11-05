<h1 align="center">
<p>semantic-segmentation-level2-cv-10
</h1>

<h3 align="center">
<p>권예환_T2012, 권용범_T2013, 김경민_T2018, 방누리_T2098, 심재빈_T2124, 정현종_T2208
</h3>

## Overview
자원의 대부분을 수입하는 우리나라는 특히 자원 재활용이 중요합니다. 국내에서 버려지는 쓰레기종량제 봉투 속을 살펴보면 70%는 재활용이 가능한 자원입니다. 분리수거에 대한 접근성 향상을 위해 딥러닝 모델을 이용해 분리수거를 돕고자 합니다.

## 시연 결과
<p float="left">
  <img src="/images/0000.png" width="500" />
</p>
<p float="left">
  <img src="/images/0001.png" width="500" />
</p>
<p float="left">
  <img src="/images/0002.png" width="500" />
</p>

## Prerequisite
```
python>=3.7
torch
mmsegmentation
mmdetection
```

## Model

1. [Swin-B + UperNet](/mmseg) (based on mmsegmentation)

2. [Swin-L + HTC++](/mmdet) (based on mmdetection)
