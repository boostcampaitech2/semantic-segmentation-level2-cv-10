# util 사용법

## cocoDataSave.py

COCO data format의 json을 이용하여, mmsegmentation input data format(image + mask)로 변경합니다.

## ensemble.py

각 모델에서 생성된 결과물 파일들을 이용해 최종 결과물 파일을 생성합니다.

사용법은 아래와 같습니다.

```
python3 ensemble.py -i1 htc_output.csv -i2 upernet_output.csv -o final_output.csv
```