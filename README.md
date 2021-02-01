# Top-Related-Meta-Learning-Method-for-Few-Shot-Detection
code about https://arxiv.org/pdf/2007.06837.pdf

loss.py: TCL-C and category-based grouping mechanism


Implementation for the paper:

Top-Related-Meta-Learning-Method-for-Few-Shot-Detection,2020

Qian Li*, Nan Guo, Xiaochun Ye, Duo Wang, Dongrui Fan and Zhimin Tang (* main contribution)

Our code is based on https://github.com/bingykang/Fewshot_Detection and developed with Python 2.7 & PyTorch 0.3.1.

# Contribution

![image](https://github.com/futureisatyourhand/Top-Related-Meta-Learning-Method-for-Few-Shot-Detection/blob/main/%E5%9B%BE%E7%89%87/1.png)



# TCL-C and category-based grouping mechanism applying for https://github.com/bingykang/Fewshot_Detection
![image](https://github.com/futureisatyourhand/Top-Related-Meta-Learning-Method-for-Few-Shot-Detection/blob/main/%E5%9B%BE%E7%89%87/2.png)
Overall structure of our the TCL for classification and category-based grouping mechanism to help meta-learning model learn the related features between categories. The input of the meta-model M is an image and a mask of only an object selected randomly. The value of the mask within object is 1, otherwise, it is 0. The number of the meta-model input is divisible by the number of all categories about training. For the Pascal VOC dataset, when training the base model, the inputs of the meta-model are $15n$ samples, while fine-tuning, which are $20n$ samples, $n$ is the number of GPUs. The meta-model extracts meta-feature vectors about classes as the weight for dynamic convolution, then, the classifier and detector complete classification and regression task. During training, we use the TCL to training classification. According to category grouping mechanism, we split category-based meta-feature vectors into groups to learning better meta-features.


# Abstract
Many meta-learning methods are proposed for few-shot detection. However, previous most methods have two main problems, poor detection APs, and strong bias because of imbalance datasets. Previous works mainly alleviate these issues by additional datasets, multi-relation attention mechanisms and sub-modules. However, they require more cost. In this work, for meta-learning, we find that the main challenges focus on related or irrelevant semantic features between different categories, and poor distribution of category-based meta-features. Therefore, we propose a Top-C classification loss (i.e. TCL-C) for classification task and a category-based grouping mechanism. The TCL exploits true-label and the most similar class to improve detection performance on few-shot classes. According to appearance and environment, the category-based grouping mechanism groups categories into different groupings to make similar semantic features more compact for different categories, alleviating the strong bias problem and further improving detection APs. The whole training consists of the base model and the fine-tuning phase. During training detection model, the category-related meta-features are regarded as the weights to convolve dynamically, exploiting the meta-features with a shared distribution between categories within a group to improve the detection performance. According to grouping mechanism, we group the meta-features vectors, so that the distribution difference between groups is obvious, and the one within each group is less. Extensive experiments on Pascal VOC dataset demonstrate that ours which combines the TCL with category-based grouping significantly outperforms previous state-of-the-art methods for few-shot detection. Compared with previous competitive baseline, ours improves detection AP by almost 4% for few-shot detection

# Help
If you have any questions,please contact us at liqian18s@ict.ac.cn

# Train and Test
our novel set1 is "boat,cat,mbike,sheep,sofa"  
novel set2 is "bird,bus,cow,mbike,sofa"  
novel set3 is "aero,bottle,cow,horse,sofa"

Model training and testing and configuration are the same as https://github.com/bingykang/Fewshot_Detection

# Citation

If you use these methods in you research, please cite:
```
@misc{li2020toprelated,
      title={Top-Related Meta-Learning Method for Few-Shot Detection}, 
      author={Qian Li and Nan Guo and Xiaochun Ye and Duo Wang and Dongrui Fan and Zhimin Tang},
      year={2020},
      eprint={2007.06837},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```



