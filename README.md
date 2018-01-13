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
wget http://mftp.mmcheng.net/Andrew/dss_model_released.caffemodel
```
You can also download it from [here (google drive)](https://drive.google.com/file/d/0B21WWgRw0U1uMFVKN3NPT0VHMW8/view?usp=sharing). The results produced by this model are slightly different from the ones we reported in our paper (with higher F-measure score and also higher MAE score).

If you want to train the model, please prepare your own training dataset first. The data layer we used here is similar to the one used in [HED](https://github.com/s9xie/hed). You can also refer to the data layer used in [Deeplab](https://bitbucket.org/aquariusjay/deeplab-public-ver2) or write your own one. 

You may also find our data layer [here](https://github.com/Andrew-Qibin/caffe_dss). Notice that if you use caffe, please cite their paper.

Then, run
```bash
python run_saliency.py
```

If you want to test the model, you can run
```bash
ipython notebook DSS-tutorial.ipynb
```  

About the CRF code we used, you can find it [here](https://github.com/Andrew-Qibin/dss_crf). Notice that please provide a link to the original code as a footnote or a citation if you plan to use it.

### Visual comparison with previous start-of-the-arts
![](https://github.com/Andrew-Qibin/DSS/blob/master/Compares.png)
From left to right: Source, Groundtruth, Ours, DCL, DHS, RFCN, DS, MDF, ELD, MC, DRFI, DSR.

### Useful links that you might want
* [MSRAB](https://people.cs.umass.edu/~hzjiang/drfi/index.html): including 2500 training, 500 validation, and 2000 test images. (This is also our training set.) The source images can be found [here](http://mmcheng.net/msra10k/).
* [MSRA10K](http://mmcheng.net/msra10k/): You can also use this dataset for training as some works did.
* [Evaluation Code (Windows)](https://github.com/MingMingCheng/CmCode/tree/master/CmLib/Illustration): The cold is based on MS Visual Studio.
* [Evaluation Code (Ubuntu)](https://github.com/Andrew-Qibin/SalMetric): This code is based on C++ and with a python wrapper for python users.

### If you want to compare your results with ours, you may download them from here [(baidu drive)](https://pan.baidu.com/s/1o8r90Ga).

### If you think this work is helpful, please cite
```latex
@article{hou2016deeply,
  title={Deeply supervised salient object detection with short connections},
  author={Hou, Qibin and Cheng, Ming-Ming and Hu, Xiaowei and Borji, Ali and Tu, Zhuowen and Torr, Philip},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2017}
}
```
