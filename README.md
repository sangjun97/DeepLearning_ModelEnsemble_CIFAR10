# DeepLearning_ModelEnsemble_CIFAR10

For the best performance to use PyTorch on the CIFAR10 dataset.
## Prerequisites

→ Python 3.6+</br>
→ PyTorch 1.0+
## Dataset

[CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html) is used to benchmark. CIFAR-10 contains 60000 color images of 32X32 pixels, each labeled with 10 classes. The images are further divided into training and test set with each set having 50000 and 10000 images.
![1_r8S5tF_6naagKOnlIcGXoQ](https://user-images.githubusercontent.com/77375401/173234298-95ff7347-bd06-436e-810d-293eef3ae216.png)

## Data Augmentation

→ RandomCrop</br>
→ ColorJitter</br>
→ RandomHorizontalFrlip</br>
→ RandAugment</br>
## Learning Rate

I started the learning rate at 0.01 and lowered it down to 0.0001 using a learning rate scheduler called CosineAnnealingLR.
## Training
```Python
# Start training each models with: 
python dense.py
python mobilenetv2.py
python resnext.py

# Start ensembling 3 models with:
python main_ens.py
```

## Result

|Data|[DenseNet](https://arxiv.org/abs/1608.06993)||[MobileNetv2](https://arxiv.org/abs/1801.04381)||[ResNet](https://arxiv.org/abs/1611.05431)||Ensembled||
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
||Loss|Accuracy|Loss|Accuracy|Loss|Accuracy|Loss|Accuracy|
|Train data|0.01|99.714%|0.032|99.004%|0.025|99.59%|-|-|
|Test data|0.173|95.95%|0.179|94.9%|0.176|95.64%|0.169|96.51|

![1](https://user-images.githubusercontent.com/77375401/173234426-5ec98ff2-a4c3-4312-8102-4d856b9d0851.png)</br>
             **Train/Test accuracy & loss of DenseNet**</br></br>
![1](https://user-images.githubusercontent.com/77375401/173234469-43f31c8e-08a5-41ea-89a9-7bb94bf60aa8.png)</br>
             **Train/Test accuracy & loss of Mobilenetv2**</br></br>
![1](https://user-images.githubusercontent.com/77375401/173234491-6fde6b02-0883-4da0-898c-9a90363afc89.png)</br>
             **Train/Test accuracy & loss of ResNext**</br></br>
![1](https://user-images.githubusercontent.com/77375401/173234509-18b5512e-e61e-4321-9257-910128933d74.png)</br>
             **Test accuracy & loss of Ensembled**</br>


