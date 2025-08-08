# Gradient-Aware Token Injection Transformer for Fine-Grained Visual Classification

* install `Pytorch and torchvision`
```
pip install torch==1.13.1 torchvision==0.14.1
```
* install `timm`
```
pip install timm==0.4.5
```
* install `Apex`
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
* install other requirements
```
pip install opencv-python==4.5.1.48 yacs==0.1.8
```
#### data preparation
Download [iNaturalist 18](https://github.com/visipedia/inat_comp),[CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html),[NABirds](https://dl.allaboutbirds.org/nabirds),[Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html).

## Trained Model Checkpoints

We provide the trained model checkpoints on the following datasets. You can download and evaluate them directly

- **Gradient-Aware Token Injection Transformer for Fine-Grained Visual Classification trained on Datasets**  
  🔗 [Baidu Pan Link](https://pan.baidu.com/s/19sEDgygYqXA03vmBfsX0Vw?pwd=i25h)  
  🔐 Password: `i25h`  

## Acknowledgment

We would like to thank the authors of [MetaFormer](https://github.com/dqshuai/MetaFormer)for their publicly available code. Part of this work is built upon their implementations.

