# PyTorch-MSTN

A pytorch implementation for [[Moving Semantic Transfer Network]](https://github.com/Mid-Push/Moving-Semantic-Transfer-Network)

    @inproceedings{xie2018learning,
      title={Learning Semantic Representations for Unsupervised Domain Adaptation},
      author={Xie, Shaoan and Zheng, Zibin and Chen, Liang and Chen, Chuan},
      booktitle={International Conference on Machine Learning},
      pages={5419--5428},
      year={2018}
    }

## Environment

- Python 2.7
- PyTorch 1.0.0

## Note

- Amazon-Webcam实验复现成功，使用的是[[pytorch_imagenet]](https://github.com/jiecaoyu/pytorch_imagenet)中提供的pretrained weights和LRN实现。
- MSTN作者使用的AlexNet来自于[[Finetuning AlexNet with Tensorflow]](https://github.com/kratzert/finetune_alexnet_with_tensorflow/)（和pytorch_imagenet的模型几乎相同）。我尝试[1]将这个模型及weights经过转换应用到PyTorch中；[2]使用torchvision提供的预训练的AlexNet（和[1]架构不同）。但这两种方式结果都只能到达67-70%。
- 尝试了SGD和Adam，目前实验中momentum=0.9,init_lr=0.01的SGD的效果更好。
- Office31上的其它迁移任务效果同样会或多或少差一些。
- 在将代码从TF迁移到PyTorch时可能需要注意的问题：[1]OpenCV默认的图像通道是BGR，而PyTorch（PIL）使用的通道一般是RGB（这个Repo没有转换通道）；[2]npy文件中的模型参数需要转置才能赋给PyTorch模型；[3]LRN层PyTorch已有官方实现，但它的参数size似乎和TF有所不同（这个Repo没有使用PyTorch的LRN层）；[4]PyTorch一般认为输入数据是0-1的，而caffe是0-255（这个Repo没有/255）。代码的迁移过程中还有很多没有理解的问题。
- 在train.py中import model或PretrainedAlexnet来使用上面提到的[1][2]两种预训练AlexNet。如果要使用[2]的模型及weights，请在train.py中注释掉model.load_state_dict一行。

## Result

|                        | Amazon-Webcam |
| :--------------------: | :-----------: |
| (paper)Source Only     |   0.616       |
| (paper)MSTN            |   0.805       |
| (this repo) Source Only|   not test    |
| (this repo) MSTN       |   0.805       |

## Reference

- [[Moving Semantic Transfer Network]](https://github.com/Mid-Push/Moving-Semantic-Transfer-Network)
- [[pytorch_imagenet]](https://github.com/jiecaoyu/pytorch_imagenet)
- [[Finetuning AlexNet with Tensorflow]](https://github.com/kratzert/finetune_alexnet_with_tensorflow/)
