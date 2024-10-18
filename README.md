# distirbuted_inference
learning about distributed inference, prepare to take resnet50 as a case to get to know how to use distributed inference, including one node muti card and muti node

## prepare model and dataset
1. download resnet50 from torchvision
```bash
mkdir -p ~/.cache/torch/hub/checkpoints
cp -r inference_models/resnet50-0676ba61.pth ~/.cache/torch/hub/checkpoints
```

2. download imagenet dataset(a sample is provided by demo)
```bash
# classes and labels

# get authorization from imageNet first
# download ILSVRC2012_img_val.tar from https://image-net.org/challenges/LSVRC/2012/2012-downloads.php
mkdir -p images
tar -xvf ILSVRC2012_img_val.tar -C images/
```

