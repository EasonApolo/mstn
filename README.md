# PyTorch-MSTN

A pytorch implementation for Moving Semantic Transfer Network [[MSTN]](https://github.com/Mid-Push/Moving-Semantic-Transfer-Network)

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

- 没有复现成功，从实验结果来看，可能是因为我使用的base net精度太低，我将作者提供的TF代码改写为source-only设置，精度确实可以达到61%。
- 作者使用的AlexNet来自于[[Finetuning AlexNet with Tensorflow]](https://github.com/kratzert/finetune_alexnet_with_tensorflow/)。我尝试了[1]将这个模型及weights经过转换应用到PyTorch中;[2]使用torchvision提供的预训练的AlexNet（和[1]架构不同）；最终两种方法的精度最优都会收敛到70左右，也很容易收敛到67左右，似乎是陷入了局部最优解。
- 尝试了不同的优化器，使用Adam(lr=1e-3)相较于SGD(lr=1e-2)收敛的更快，但对结果没有影响(括号中是收敛速度较快的lr)。
- Office31上的其它迁移任务效果同样会或多或少差一些。
- 在将代码从TF迁移到PyTorch时可能需要注意的问题：[1]Caffe和OpenCV默认的图像通道是BGR，而PyTorch默认的通道是RGB，对于第一个卷积层可能影响较大;[2]npy文件中的模型参数需要转置才能赋给PyTorch模型;[3]LRN层PyTorch已有官方实现，但它的参数size似乎和TF有所不同，可能要注意一下;[4]PyTorch一般认为输入数据是0-1的，而caffe是0-255。
- 在train.py中import MyModel或PretrainedAlexnet来切换上面提到的[1][2]两种预训练AlexNet。如果使用MyModel需要将model.load_state_dict取消注释。

## Result

|                        | Amazon-Webcam |
| :--------------------: | :-----------: |
| (paper)Source Only     |   0.616       |
| (paper)MSTN            |   0.805       |
| (this repo) Source Only|   0.486       |
| (this repo) MSTN       |   0.707       |