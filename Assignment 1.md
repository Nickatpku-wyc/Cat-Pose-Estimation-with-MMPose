### Assignment 1 -- Cat Pose_Estimation

#### Introduction

##### Background

The task of this assignment is mainly forcus on developing a 2D cat pose estimation system based on MMPose. Actually, pose estimation is a very important issue in deep learning, especially  2D Human Pose Estimation(2D HPE).

2D Human Pose Estimation (2D HPE) aims to predict the 2D spatial position coordinates of human joints (or key points, such as head, left hand, right foot, etc.) from images or videos. 2D HPE is used in a wide range of applications, including motion recognition, animation generation, augmented reality, etc. 

Current mainstream 2D HPE methods can be divided into bottom up and top down approaches. 

Bottom up methods predict all key points in the image simultaneously, and then combine different types of key points into a human body. The top down approach first detects one or more individuals in the input image and then predicts the keypoints for each individual individually. 

##### Related works

Some early top-down deep learning methods used neural networks to directly predict the 2D coordinates of key points on the human body. **DeepPose** [1] is a classic example of this type of approach. 


In recent years, **heatmap-based methods** for human pose estimation have become mainstream. The heatmap-based approach better preserves the spatial location information and is more in line with the design characteristics of Convolutional Neural Network (CNN), thus achieving better prediction accuracy.

**CPM (Convolutional Pose Machines)** [2] uses serialised convolutional neural networks to learn texture and spatial information for 2D human pose estimation. 

**Stacked Hourglass** [3] integrates structured spatial relationships of the human body through multi-scale feature fusion. The method performs successive upsampling and downsampling and uses heat map supervision in the middle of the network to improve the performance of the network.

**SBL (Simple Baseline)** [4] provides a benchmarking framework for human pose estimation. 

**HRNet** [5] designed an efficient network structure for human pose estimation. Unlike previous methods that use low resolution features to predict high resolution heatmaps, HRNet is designed with multiple parallel branches of different resolutions. 

#### Methods

Here I've listed what technique I've used in my experiments, more details can be found there.

**Data Augmentation**: For data augmentation, Besides the two of data augmentation techniques the baseline has used, Random-Flip and Random-ScaleRotation for top_down_transform model. We also tried the others for top_down_transform -- HalfBody-Transform and Random-Translation for our dataset, and some methods of Albumentation and Photometric-Distortion:

**Hyper-parameter tuning**: We first tried different No of epochs, since the baseline has trained too less. Based on that, we changed the learning rate and batch size together when we found that these two hyper-parameters will effect the weight adjustment together.  Except these, we also considered the learning policy and the parameter in warm-up strategy. Finally, we tried the last two hyper-parameters, image size and the heat-map size.

**Weight initialization**: Mmpose provides the pre-trained model for ResNet and HRNet, so we can just use them to do our weight initialization, 

**NN architectures**: We tried two NN architectures which having good performance for animal 2D pose estimation-- ResNet and HRNet.

**Optimizers**: We tried different optimizers supported by Pytorch and mmpose.

#### Experiments & Analysis

We got the performance the baseline model on validation set is $mAP=0.1437$.

Now we want to improve our model. The idea of the improvement is a top-down structure. 

##### Data Augmentation 

We first tried to add some more data augmentation operation for our data.

| Experiment No. |           Policy           | Parameter | Final mAP(epoch:mAP) | Best mAP(epoch:mAP) |
| :------------: | :------------------------: | :-------: | :------------------: | :-----------------: |
|       1        |   Add HalfBody-Transform   |  Default  |      20: 0.150       |      20: 0.150      |
|       2        |   Add Random-Translation   |  Default  |      20: 0.147       |      20: 0.147      |
|       3        |         Add 1 & 2          |  Default  |      20: 0.156       |      20: 0.156      |
|       4        |     Add Albumentation      |  Default  |      20: 0.136       |      20: 0.136      |
|       5        | Add Photometric-Distortion |  Default  |      20: 0.154       |      20: 0.154      |

From the record, we can see that the performance seems not improved a lot. Actually, most time we using data augmentation is to prevent over-fitting. Here, even we don't know what the performance will be in test set, it's still worth to try it. 

And also, we only trained for 20 epochs with no parameter tuning, so actually our model is far from good enough.

##### Hyper-Parameters tuning

As we mentioned before, the very first problem is, there is too less epochs for training. So, without changing the other parameters, we try to increase the number of epochs by more and, accordingly, change the corresponding parameters.

| Experiment No. |   Policy    |     Parameter      | Final mAP(epoch:mAP) | Best mAP(epoch:mAP) |
| :------------: | :---------: | :----------------: | :------------------: | :-----------------: |
|       6        | More epochs | total_epochs = 100 |      100: 0.285      |     100: 0.314      |
|       7        | More epochs | total_epochs = 200 |      200: 0.333      |     155: 0.352      |
|       8        | Less epochs | total_epochs = 150 |      150: 0.341      |     135: 0.359      |

The chosen of No of epochs is binary search. And when we get total_epochs = 150, It seems like there is no significant difference than the last experiment, so we stop here about epoch temporarily..

Now let's do with learning rate. We already know something about learning rate:

* small learning rate: Many iterations till convergence, and may trapped in local minimum
* large learning rate: Overshooting, No convergence 

We used warmup to adjust our learning rate, which is a learning rate warm-up method mentioned in the ResNet paper, which starts with a small learning rate, trains a number of epochs or steps (e.g. 4 epochs, 10,000 steps), and then modifies the training to a pre-set learning.

So the learning policy of baseline model is: The learning rate was decayed in steps, decreasing the learning rate (by default to 0.1 times each time) in the 15th and 20th epochs, for a total of 20 epochs of training. And using warmup strategy, initially setting a very small learning rate (0.001 times the initial learning rate lr) and increasing it linearly to lr=1e-5 over the first 10 iterations.

And actually, there are also another hyper-parameter connected with learning rate -- batch size. In practice, we usually tuning these two parameters together.

About the batch size, we clear know what effect it has to the performance of our model:

* small batch size: Improving the generalization of models, the noise from a small batch size helps to escape the sharp minimum
* large batch size: Reduced training time and improved stability of our model. Meanwhile, large batch size gradients are more stable to compute as the model training curve will be smoother

It's worth to mention that large batchsize performance degrades because the training time is not long enough[6], it is not essentially a problem of batchsize, the parameters are updated less at the same epochs and therefore require longer iterations.

Let we take a look at the SGD algorithm we used:

<img src="C:\Users\Nicka\AppData\Roaming\Typora\typora-user-images\image-20220317162736534.png" alt="image-20220317162736534" style="zoom:33%;" />

We can write it as(without Adaptive Learning Rate):
$$
w\leftarrow w - \alpha J(\theta),\ \ 
J(\theta)=\frac{1}{B}\sum_{i=1}^B\nabla e(x_i,y_i)
$$
So we can find that the adjustment of weight is depending on learning rate and batch size simultaneously. And these two are in turn directly related to each other as numerators and denominators. 

Which means usually when we increase the batch size to N times the original[7], to ensure that the updated weights are equal after the same samples, the learning rate should be increased to N times the original according to the linear scaling rule, and vice versa.

And Smith S L and Le Q V in 2017[8] showed that for a fixed learning rate, there exists an optimal batch size that maximise test accuracy, which is positively related to the learning rate and the size of the training set.

So our aim is to find this optimal combination, and the specific experimental strategy is that if the learning rate is increased, then the batch size should preferably increase as well, so that the convergence is more stable.

To verify our assuming, we adjust the learning rate first, then preferably increase batch size.

| Experiment No. |      Policy       |        Parameter         | Final mAP(epoch:mAP) | Best mAP(epoch:mAP) |
| :------------: | :---------------: | :----------------------: | :------------------: | :-----------------: |
|       9        |    Increase lr    |  lr=1e-4, batch size=16  |      150: 0.456      |     145: 0.459      |
|       10       | change batch size |  lr=1e-4, batch size=32  |      150: 0.460      |     125: 0.468      |
|       11       | change batch size |  lr=1e-4, batch size=24  |      150: 0.456      |     145: 0.460      |
|       12       | change batch size |  lr=1e-4, batch size=64  |      150: 0.432      |     150: 0.432      |
|       13       | change batch size |  lr=1e-4, batch size=48  |      150: 0.446      |     135: 0.452      |
|       14       |    Increase lr    | lr = 1e-3, batch size=32 |     not converge     |          —          |
|       15       |    decrease lr    | lr = 54-4, batch size=32 |      150: 0.485      |     150: 0.485      |
|       16       | change batch size | lr = 54-4, batch size=64 |      150: 0.452      |     150: 0.452      |
|       17       | change batch size | lr = 54-4, batch size=48 |      150: 0.471      |     150: 0.471      |
|       18       | change batch size | lr = 54-4, batch size=24 |      150: 0.466      |     115: 0.470      |

So we tried different combinations of lr and batch size, it seems that lr = 54-4, batch size=32 is the best for now.

Because we set larger learning rate, for better performance, we may try to add the No of epochs again:

| Experiment No. |   Policy    |     Parameter      | Final mAP(epoch:mAP) | Best mAP(epoch:mAP) |
| :------------: | :---------: | :----------------: | :------------------: | :-----------------: |
|       19       | More epochs | total_epochs = 200 |      200: 0.459      |     200: 0.459      |
|       20       | More epochs | total_epochs = 175 |      175: 0.473      |     155: 0.477      |

It seems not work. So we throw it up. Now, let's try other hyper-parameters, image size and the heat-map size.

Intuitively, the higher the resolution of our image, the more pixels it contains, and thus the more updates to the weight in the filter in the convolution layer, which will improve the accuracy of the image, and at the same time, the more information we get about the feature after the vector transformation of the feature map in the fully connected layer, without changing the network structure.

Of course, the higher the resolution of the image, the more computationally intensive the image will be and the longer the training time.

Since the original image is 128 * 128 in size, to avoid distortion of the image. We use a X * X size to ensure that the recognisability of the image is not affected as much as possible.

| Experiment No. |      Policy       |                       Parameter                       | Final mAP(epoch:mAP) | Best mAP(epoch:mAP) |
| :------------: | :---------------: | :---------------------------------------------------: | :------------------: | :-----------------: |
|       21       | Larger image size |      image_size=[256,256], heatmap_size=[65,64]       |      150: 0.542      |     150: 0.542      |
|       22       | Larger image size |      image_size=[192,192], heatmap_size=[48,48]       |      150: 0.499      |     150: 0.499      |
|       23       | change batch size |          image_size=[256.256], batch size=64          |      150: 0.548      |     120: 0.551      |
|       24       | change batch size |          image_size=[256.256], batch size=48          |      150: 0.501      |     135: 0.532      |
|       25       |    More epochs    | image_size=[256.256], batch size=64, total_epochs=210 |      210: 0.585      |     210: 0.585      |

We can see the performance has Greatly improved, which verified our idea. And based on the new image size and heat-map size, we also tried new changes of batch size and epochs, since larger image size need more training, so we tried larger batch size to neutralise its effects and tr	y more epochs to get accuracy, and as the experiments showed, the performance improved further.

Since we have larger image size, so the data augmentation may help us again, so we change some parameters of Data Augmentation to get better augmentation.

| Experiment No. |      Policy       | Parameter | Final mAP(epoch:mAP) | Best mAP(epoch:mAP) |
| :------------: | :---------------: | :-------: | :------------------: | :-----------------: |
|       26       | Data augmentation |     -     |      210: 0.593      |     190: 0.610      |

<img src="C:\Users\Nicka\AppData\Roaming\Typora\typora-user-images\image-20220319114244283.png" alt="image-20220319114244283" style="zoom: 50%;" />

And it works. So far, we have determined all hyper-parameters. Now let's move to next part.

##### NN architectures

For animal 2D pose estimation, actually, there are two NN architectures having good performance -- Resnet and Hrnet.

We already tried Resnet before, now let's try to use Hrnet. And for this new NN, we also need to find out the best lr + batch size.

| Experiment No. |      Policy       |   Parameter   | Final mAP(epoch:mAP) | Best mAP(epoch:mAP) |
| :------------: | :---------------: | :-----------: | :------------------: | :-----------------: |
|       26       |       HRNet       | batch size=64 |  CUDA out of memory  |          —          |
|       27       | Change batch size | batch size=32 |      210: 0.666      |     205: 0.674      |
|       28       | Change batch size | batch size=48 |  CUDA out of memory  |          —          |
|       29       | Change batch size | batch size=40 |  CUDA out of memory  |          —          |
|       30       | Change batch size | batch size=36 |      210: 0.660      |     210: 0.660      |

It seems that there are not very large improvement than we set batch size to 32.

##### Optimizers

The default optimizer baseline model used is Adam, which is an adaptive learning rate optimization method that dynamically adjusts the learning rate using first-order moment estimation and second-order moment estimation of the gradient. 

Adam is a combination of Momentum and RMSprop with bias correction. This means that in theory we do not need to try the optimizers like SGD, ASGD, AdaGrad, AdaDelta, RMSprop.

However, it is worth to try Adamax, which is an addition to Adam with the notion of an upper limit on the learning rate. This is likely to have further optimization for the performance of our model.

| Experiment No. |  Policy   |  Parameter  | Final mAP(epoch:mAP) | Best mAP(epoch:mAP) |
| :------------: | :-------: | :---------: | :------------------: | :-----------------: |
|       31       | Optimizer | type=Adamax |      210: 0.673      |     125: 0.685      |

We can see that performance changed better. Till now, we have down all experiments to improve the model. 



##### Visulization

We can see the contrast between the baseline model and the final model we use. Baseline V.S. Final model:

<img src="C:\Users\Nicka\AppData\Roaming\Typora\typora-user-images\image-20220320143221922.png" alt="image-20220320143221922" style="zoom: 67%;" /><img src="C:\Users\Nicka\AppData\Roaming\Typora\typora-user-images\image-20220320152855095.png" alt="image-20220320152855095" style="zoom: 67%;" />

We get obvious better pose estimation results. And try other images:

<img src="C:\Users\Nicka\AppData\Roaming\Typora\typora-user-images\image-20220320153201277.png" alt="image-20220320153201277" style="zoom: 33%;" /><img src="C:\Users\Nicka\AppData\Roaming\Typora\typora-user-images\image-20220320153227318.png" alt="image-20220320153227318" style="zoom: 33%;" /><img src="C:\Users\Nicka\AppData\Roaming\Typora\typora-user-images\image-20220320153317110.png" alt="image-20220320153317110" style="zoom: 33%;" /><img src="C:\Users\Nicka\AppData\Roaming\Typora\typora-user-images\image-20220320153447809.png" alt="image-20220320153447809" style="zoom: 33%;" />

<img src="C:\Users\Nicka\AppData\Roaming\Typora\typora-user-images\image-20220320153821150.png" alt="image-20220320153821150" style="zoom: 33%;" /><img src="C:\Users\Nicka\AppData\Roaming\Typora\typora-user-images\image-20220320153932836.png" alt="image-20220320153932836" style="zoom: 33%;" /><img src="C:\Users\Nicka\AppData\Roaming\Typora\typora-user-images\image-20220320154046745.png" alt="image-20220320154046745" style="zoom: 33%;" /><img src="C:\Users\Nicka\AppData\Roaming\Typora\typora-user-images\image-20220320154128557.png" alt="image-20220320154128557" style="zoom: 33%;" />

So actually, our model performs good generally, but for some images like cat with a woman, or with a dog, or half body, or some confusing background, our model still exists some problems on them.

#### Reference

[1] Toshev, A., & Szegedy, C. (2014). Deeppose: Human pose estimation via deep neural networks. In *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 1653-1660).

[2] Wei, S. E., Ramakrishna, V., Kanade, T., & Sheikh, Y. (2016). Convolutional pose machines. In *Proceedings of the IEEE conference on Computer Vision and Pattern Recognition* (pp. 4724-4732).

[3] Newell, A., Yang, K., & Deng, J. (2016, October). Stacked hourglass networks for human pose estimation. In *European conference on computer vision* (pp. 483-499). Springer, Cham.

[4] Xiao, B., Wu, H., & Wei, Y. (2018). Simple baselines for human pose estimation and tracking. In *Proceedings of the European conference on computer vision (ECCV)* (pp. 466-481).

[5] Sun, K., Xiao, B., Liu, D., & Wang, J. (2019). Deep high-resolution representation learning for human pose estimation. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 5693-5703).

[6] Hoffer E, Hubara I, Soudry D. Train longer, generalize better: closing the generalization gap in large batch training of neural networks[C]//Advances in Neural Information Processing Systems. 2017: 1731-1741.

[7] Goyal P, Dollar P, Girshick R B, et al. Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour.[J]. arXiv: Computer Vision and Pattern Recognition, 2017.

[8] Smith S L, Le Q V. A bayesian perspective on generalization and stochastic gradient descent[J]. arXiv preprint arXiv:1710.06451, 2017.
