## Deeply Supervised Salient Object Detection with Short Connections

### Network architecture and more details
Please refer to [our paper](https://arxiv.org/abs/1611.04849).

### Usage
Please install [Caffe](https://github.com/BVLC/caffe) first. I think you may find a great number of tutorials talking about how to install it.
```bash
cd <caffe_root>/examples
git clone https://github.com/Andrew-Qibin/DSS.git
```
Before you start, you also need our pretrained model.
```bash
wget
```
If you want to train the model, please prepare your own training dataset first.

### Visual comparison with previous start-of-the-arts
![](https://github.com/Andrew-Qibin/DSS/blob/master/Compares.png)
From left to right: Source, Groundtruth, Ours, DCL, DHS, RFCN, DS, MDF, ELD, MC, DRFI, DSR.

### If you think this work is helpful, please cite
```latex
@article{hou2016deeply,
  title={Deeply supervised salient object detection with short connections},
  author={Hou, Qibin and Cheng, Ming-Ming and Hu, Xiaowei and Borji, Ali and Tu, Zhuowen and Torr, Philip},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2017}
}
```
