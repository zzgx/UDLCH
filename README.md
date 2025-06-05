# UDLCH
Unsupervised Deep Lifelong Cross-modal Hashing

## Environment
You can use the following command to deploy the environment：  
```
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

## Datasets
We release the three experimental datasets as follows:  
[Baidu Pan](https://pan.baidu.com/s/1vO638X91H4UT7IP1nEPSuQ?pwd=zzgx)  
After downloading these three datasets, please modify the relative path in ```src/Mat_5.py```.
## Usage
Our model can be trained and verified by the following command:
```
Python main_MIRFLICKR.py
Python main_NUSWIDE.py
Python main_MSCOCO.py
```
You can get outputs as follows:
```
epoch=7
Base hash representation Avg Loss: 14.100 | LR: 0.0001
patience_counter=1
Base mAP: i2t=0.762   t2i=0.764   avg=0.763
epoch=8
Base hash representation Avg Loss: 14.067 | LR: 0.0001
patience_counter=0
Base mAP: i2t=0.763   t2i=0.765   avg=0.764
Saving Base Max mAP : i2t=0.763   t2i=0.765   avg=0.764
epoch=9
Base hash representation Avg Loss: 14.058 | LR: 0.0001
patience_counter=0
Base mAP: i2t=0.765   t2i=0.769   avg=0.767
Saving Base Max mAP : i2t=0.765   t2i=0.769   avg=0.767
epoch=10
Base hash representation Avg Loss: 14.060 | LR: 0.0001
patience_counter=1
Base mAP: i2t=0.765   t2i=0.763   avg=0.764
epoch=11
Base hash representation Avg Loss: 14.050 | LR: 0.0001
patience_counter=0
Base mAP: i2t=0.769   t2i=0.769   avg=0.769
Saving Base Max mAP : i2t=0.769   t2i=0.769   avg=0.769
epoch=12
Base hash representation Avg Loss: 14.030 | LR: 0.0001
patience_counter=0
Base mAP: i2t=0.769   t2i=0.772   avg=0.771
Saving Base Max mAP : i2t=0.769   t2i=0.772   avg=0.771
```
