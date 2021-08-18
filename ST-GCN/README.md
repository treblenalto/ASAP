# ST-GCN
해당 모델은 스켈레톤을 기반으로 애완동물의 동작을 예측할 시 활용한 모델입니다.

학습시 사용한 array_train.npy의 용량이 큰 관계로 해당 데이터는 [여기](https://drive.google.com/file/d/10ucSBZS_Jg8nvGTIO8uv12JjyZBUPnx6/view?usp=sharing)에서 다운받아 사용하시기 바랍니다.

## Structure
```
ST-GCN
├── README.md
├── requirements.txt
├── main.py
├── config
│   └───st_gcn
│       └───kinetics-skeleton
│           └───train.yaml
├── data
│   ├───test
│   │   ├───array_test.npy
│   │   └───test_class_changed.pkl
│   └───train
│       ├───array_train.npy         // 위 링크에서 다운받으실 수 있습니다.
│       └───train_class_changed.pkl
├── feeder
│   ├───feeder.py
│   └───tools.py
├── net
│   ├───st_gcn.py
│   └───utils
│       ├───graph.py
│       └───tgcn.py
├── processor
│   ├───processor.py
│   ├───recognition.py
│   └───io.py
├── torchlight
│   ├───torchlight
│   │   ├───io.py
│   │   └───gpu.py
│   └───setup.py
│ 
```