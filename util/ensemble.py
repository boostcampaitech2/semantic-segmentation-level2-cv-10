#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
from tqdm import tqdm


# model 1: swin-L HTC model, Instance segmentation
csv_1 = pd.read_csv('output_1.csv')
# model 2: swin-B UperNet
csv_2 = pd.read_csv('output_2.csv')


# submission file의 PredictionString이 한 str에 한 image의 모든 pixel의 category를 담고있다.
# 따라서 pixel by pixel로 split 후 array로 저장
pred_1 = np.array([csv_1['PredictionString'][i].split() for i in range(len(pred_1))], dtype='int8')
pred_2 = np.array([csv_2['PredictionString'][i].split() for i in range(len(pred_1))], dtype='int8')

# csv : ensemble한 output을 담는다.
csv = csv_1.copy()



pred = [0 for _ in range(len(pred_1[0]))]

# 2개만 ensemble: pred_1의 pixel 값이 0이 아니면 그대로 적용
# pred_1이 0이면 pred_2의 pixel 값 사용
for index, i in tqdm(enumerate(range(len(pred_1)))):
    for j in range(len(pred_1[i])):
        if pred_1[i][j] == 0:
            pred[j] = pred_2[i][j]
        else:
            pred[j] = pred_1[i][j]

    pred = [str(pred[i]) for i in range(len(pred))]
    csv['PredictionString'][i] = ' '.join(pred)


csv.to_csv("../submission/ensemble.csv", index=False)

