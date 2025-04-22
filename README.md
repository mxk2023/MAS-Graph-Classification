# 面向多应用场景的图分类算法研究
## 1.概述

本仓库存放着面向多应用场景的图分类算法代码，其包含了面向有标注数据的基于深度自适应的图混合专家算法代码DA-MoE和面向无标注数据的基于隐式对比学习的图掩码自编码器算法代码GraphMAE-CT。

## 2.运行方法
### 1.配置python环境
```shell
$ conda create -n damoe python=3.11
$ conda activate damoe
$ conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
$ conda install pyg -c pyg
$ pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
$ pip install -r requirements.txt
```
### 2.面向有标注数据的基于深度自适应的图混合专家算法
```shell
$ cd DA-MoE
$ sh ./scripts/run_proteins.sh 
$ sh ./scripts/run_bbbp.sh
$ sh ./scripts/run_arxiv.sh
$ sh ./scripts/run_ddi.sh
```

### 3.面向无标注数据的基于隐式对比学习的图掩码自编码器算法
```shell
$ cd GraphMAE-CT
$ sh ./scripts/run_proteins.sh 
```

