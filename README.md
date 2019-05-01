# Single Image Deraining using a Recurrent Multi-scale Aggregation and Enhancement Network (ICME'19)
   Youzhao Yang, [Hong Lu](http://homepage.fudan.edu.cn/honglu/en); School of Computer Science, Fudan University

# Abstract
   Single image deraining is an ill-posed inverse problem due to the presence of non-uniform rain shapes, directions, and densities in images. In this paper, we propose a novel progressive single image deraining method named Recurrent Multiscale Aggregation and Enhancement Network (ReMAEN). Differing from previous methods, ReMAEN contains a symmetric structure where recurrent blocks with shared channel attention are applied to select useful information collaboratively and remove rain streaks stage by stage. In ReMAEN, a Multi-scale Aggregation and Enhancement Block (MAEB) is constructed to detect multi-scale rain details. Moreover, to better leverage the rain details from rainy images, ReMAEN enables a symmetric skipping connection from low level to high level. Extensive experiments on synthetic and real-world datasets demonstrate that our method outperforms the state-of-the-art methods tremendously. Furthermore, ablation studies are conducted to show the improvements obtained by each module in ReMAEN.

# Citation
```
   @inproceedings{yang2019single,
      title={Single Image Deraining using a Recurrent Multi-scale Aggregation and Enhancement Network},
      author={Yang, Youzhao and Lu, Hong},   
      booktitle={IEEE International Conference on Multimedia and Expo (ICME)},
      year={2019}
   }
```

# Dataset
## Synthetic Datasets
   | Datasets | #train | #test | label |
   | :------- | -----: | ----: | ----- |
   | [Rain100L]() | 200    | 100   | rain mask & rain map|
   | [Rain100H]() | 1800   | 100   | rain mask & rain map|
   | [Rain800]()  | 700    | 100   | -     |
   | [Rain1200]() | 12000  | 1200  | rain density |
   | [Rain1400]() | 12600  | 1400  | - |
   
## Real-World Datasets
   | Datasets | #train | #test | label |
   | :------- | -----: | ----: | ----- |
   | [Real-World]() | - | 67 | - |
   
# Requirements
# Usages
# Results
# Acknowledgement
   - This work was supported in part by National Natural Science Foundation of China (No. U1509206).
**The pre-trained models and the usages of this source codes are available soon!!!**
