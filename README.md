#### Gatys' Style Transfer and Universal Feature Transform

_This is a personal mini-project. I tried to reimplement the Style Transfer methods described in the
Gatys' [paper](https://arxiv.org/pdf/1508.06576.pdf) and Adobe's [paper](https://arxiv.org/pdf/1705.08086.pdf).
The code is written with Pytorch_

##### Image Preprocessing phase for VGG Nets
Typically for the pretrained `vgg19` (without `batch_normalization`) model of Pytorch, before passing to
the network, the image should be:
* rescaled (to size minimum `224x224`)
* transformed from `BGR` to `RGB`
* normalized with `mean=[0.40760392, 0.45795686, 0.48501961]` and `std=[1, 1, 1]`
* the matrix should then be upscaled by a factor `255.0`

The _postprocess_ should be a complete inverse of the preprocess above.

##### Original Gatys' method
You can try the following command:
```bash
python extract.py -c landscape -s <image-filename> --size <height> <width> --lambd 0
```

Here are some results:

![](./results/cnt2vangogh_lambd0.03_epochs50.jpg =400x)
![](./results/img.jpg =400x)

compared to original pictures:
![](./images/cnt.jpg =400x)
![](./images/landscape.jpg =400x)
