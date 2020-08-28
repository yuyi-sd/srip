## Usage
#### Search on Omniglot(N way k shot)

To run our code, you only need one GPU.
```
python train_search_few_shot.py --classes_per_set N --samples_per_class k
```
#### Search on miniImageNet(N way k shot)

```
# Data preparation: 
sh download_miniimagenet.sh
```
```
python train_miniimagenet_search_few_shot.py  --classes_per_set N --samples_per_class k
```
After searching, you need to add the searched architecture in the document genotypes.py

##### Here is the evaluation on Omniglot(N way k shot and use searches arch A stored in genotypes):

```
python train_few_shot.py --classes_per_set N --samples_per_class k --arch A
```

##### Here is the evaluation on ImageNet (N way k shot and use searches arch A stored in genotypes):
```
python train_miniimagenet.py --classes_per_set N --samples_per_class k --arch A
```

