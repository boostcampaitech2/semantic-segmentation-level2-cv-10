# Configs

---

```markdown
configs
│  └─default_runtime.py
│─models
│  ├─upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_22K.py
│  ├─upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_1K.py
│  └─upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K.py  
├─schedules
│  └─schedule_50k.py
├─datasets
│  └─dataset.py
└─inference
   └─MMSEGinference.py
```

# Model

---

MMSegmentation에는 1개의 모델을 이용하여 학습을 진행하였습니다.

- upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_22K.py

# Backbone - Swin Transformer

---

Backbone모델은 swin transformer를 기반으로 학습을 진행하였습니다.

# Train

---

- train는 mmsegmentation github 내부 디렉토리에서 진행한다.
- tools/train.py 를 사용한다.

python tools/test.py {model directory} 

```markdown
python tools/train.py ..mmseg/configs/upernet_swin_base_patch4_windows12_512x512_160k_ade20k_pretrain_384x384_22K.py 
```

# Test

---

- test는 mmsegmentation github 내부 디렉토리에서 진행한다.
- tools/test.py 를 사용한다.

python tools/test.py {model directory} {pth link} —show-dir {model directory}

```markdown
python tools/test.py ..mmseg/configs/upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_22K.py ../mmseg/latest.pth --show-dir../mmseg/configs/upernet_swin_base_patch4_windows12_512x512_160k_ade20k_pretrain_384x384_22K
```