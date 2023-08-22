## SIAST: A Slot Imbalance-Aware Self-Training Scheme for Semi-Supervised Slot Filling

This repository is used to store source code for paper "[SIAST: A Slot Imbalance-Aware Self-Training Scheme for Semi-Supervised Slot Filling](https://ieeexplore.ieee.org/document/10096302)" accepted by ICASSP2023. Welcome to cite our paper when using the model.



## Model train 

1. modify the config file, like `configs/by_count/classicSelfTraining_snips_by_count_10.json`  

2. model train

```
python train_main.py /data1/xcc/Projects/SIAST/configs/by_count/classicSelfTraining_snips_by_count_10.json
```

