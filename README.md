# CLIPScope

## Description of the files

- mining.py
  > Mining OOD candidate labels.

- CLIPScope.py
  > Our method

## Implementation of the files

### Setup device (GPU or CPU)
Modify
> cuda.py

### Train a backdoored model
Run

```
python trainmnist.py
```

### Implement the defense

```
python rob_sens_check.py
```
## Modification of the code

### Different regions and noise

Modify 
> config.py

### Load pre-trained models

Modify the *load()* function in
> rob_sense_check.py

### Cite the Work

```
@ARTICLE{10187163,
  author={Fu, Hao and Krishnamurthy, Prashanth and Garg, Siddharth and Khorrami, Farshad},
  journal={IEEE Transactions on Information Forensics and Security}, 
  title={Differential Analysis of Triggers and Benign Features for Black-Box DNN Backdoor Detection}, 
  year={2023},
  volume={18},
  number={},
  pages={4668-4680},
  doi={10.1109/TIFS.2023.3297056}}
```
