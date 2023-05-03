# ResFAUnet-pytorch
The official version of ResFAUnet, based on the pytorch implementation.

The official code for : High precision segmentation of small sample buildings based on transfer learning and multi-scale fusion


Network structure：

![总结构图](https://user-images.githubusercontent.com/88971302/235640036-c5dcfa51-5860-4c01-94cc-7aaa7a305ab4.png)


Network performance：
We selected 500 images from Inria Aeriallmage Labeling Dataset as data sets to evaluate the performance of ResFAUnet
you can find Inria Dataset from : https://project.inria.fr/aerialimagelabeling/


![image](https://user-images.githubusercontent.com/88971302/235640874-5845729d-096e-49c6-8a10-86415365f679.png)




Test environment：
PyTorch 1.7.1;
TorchVision 0.8.2


How to use:
cd  ResFAUnet-pytorch;
python train.py;

