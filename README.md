# UDLCH
Unsupervised Deep Lifelong Cross-modal Hashing

## Environment
You can use the following command to deploy the environment：  
```pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116```

## Datasets
We release the three experimental datasets as follows:  
  
After downloading these three datasets, please modify the relative path in src/Mat_5.py.
## Usage
Our model can be trained and verified by the following command:
```
Python main_MIRFLICKR.py
Python main_NUSWIDE.py
Python main_MSCOCO.py
```
