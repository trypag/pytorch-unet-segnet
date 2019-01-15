# pytorch-unet-segnet

These architectures have shown good results in semantic segmentation, image reconstruction (denoising, super-resolution).

|Model|File|Figure|
|-------|--------|----------|
|[Unet](https://arxiv.org/abs/1505.04597)|[unet.py](https://github.com/trypag/pytorch-unet-segnet/blob/master/unet.py)|![Unet](https://github.com/trypag/pytorch-unet-segnet/blob/master/docs/unet/u-net-architecture.png)|
|[SegNet](https://arxiv.org/abs/1511.00561)|[segnet.py](https://github.com/trypag/pytorch-unet-segnet/blob/master/segnet.py)|![SegNet](https://github.com/trypag/pytorch-unet-segnet/blob/master/docs/segnet/segnet.png)|
|[ModSegNet](https://link.springer.com/chapter/10.1007/978-3-030-00931-1_68)|[modsegnet.py](https://github.com/trypag/pytorch-unet-segnet/blob/master/unet.py)|![ModSegNet](https://github.com/trypag/pytorch-unet-segnet/blob/master/docs/modsegnet/1.png)|


**I would encourage you to use SegNet if you don't see any major performance decrease with Unet: SegNet will be lighter and faster !**. SegNet uses maximum unpooling during the upsampling step, reusing the maximum pooling indices from the encoding step. Making the upsampling procedure parameter free, where Unet makes use of transpose convolution (filters) to learn how to upsample. 
We provide an updated version of SegNet, which was designed for medical image segmentation: [Semi-supervised Learning for Segmentation Under Semantic Constraint, MICCAI 2018](https://link.springer.com/chapter/10.1007/978-3-030-00931-1_68)
