# DeepLearning_FaceGen

Face Generation with Attributes.

## Introdoction
Normally humans do not memorize things in pixel-level. Instead, it is much easier for us to describe representations of objects in our memory. Our goal is to generate a facial image as close as a person we want to describe based on the given attributes.

## Problem Formulation
Given the input of different attributes, we produce an output image corresponding to those facial attributes. Our dataset has about 40 attributes which includes basics like Male, Female. Facial characteristics like, arched eyebrows, high cheekbones, bags under eyes, big nose, black hair, blond hair etc and also cosmetic attributes like eyeglasses, goatee, heavy makeup, wearing hat etc. Here 1 indicates the presence of a certain feature and -1 indicates the absence as the table shown in figure 1(a). In the future, we want to be able to allow users to interactively refine the first guess by providing the improved attributes. As you can see in figure 1(b).
<img alt="face generate" src="https://github.com/chiaoyuntu/DeepLearning_FaceGen/blob/master/figures/problem_formulation/problem_formulation_1.png" width="70%" height="70%">
		figure 1(a)
<img alt="face refine" src="https://github.com/chiaoyuntu/DeepLearning_FaceGen/blob/master/figures/problem_formulation/problem_formulation_2.png" width="70%" height="70%">
		figure 1(b)

## Dataset
CelebFaces Attributes Dataset (CelebA) [2] is a large- scale face attributes dataset with more than 200K celebrity images showed as figure 3, each with 40 attribute annotations. The images in this dataset cover large pose variations and background clutter. CelebA has large diversities, large quantities, and rich annotations, including 10,177 number of identities, 202,599 number of face images, and 5 landmark locations, 40 binary attributes annotations per image. 

### Data preprocessing - Attribute selection
<img alt="All Attributes" src="https://github.com/chiaoyuntu/DeepLearning_FaceGen/blob/master/figures/dataset/all_attr.png" width="80%" height="80%">
<img alt="23 Selected Attributes" src="https://github.com/chiaoyuntu/DeepLearning_FaceGen/blob/master/figures/dataset/23_attr.png" width="80%" height="80%">
We analyzed the 40 attributes showed as figure 4 to find the non sparse attributes, and discard the sparse ones. In order to experiment,  we picked 23 attributes as shown in the figure 5 which we feel would distinguish faces in a meaningful way, like hair colors, eyes shapes, nose etc. But for the further experiments we want to consider reducing the attributes to around 5. 

## Models

### Baseline Model - GAN with FC
#### Architecture
In the baseline model, we use 2 layer Fully-Connected Network for both generator and discriminator as in figure 6. The input of generator is a noise vector of size 100 concatenated with the attribute vector of size 23. We used 178 × 218 × 3 flattened image (without cropping) with 23 attributes vector as the input of discriminator. The parameter setting is showed in table 1.

#### Result
<img src="https://github.com/chiaoyuntu/DeepLearning_FaceGen/blob/master/figures/model/fcgan1.png" width="80%" height="80%">
<img src="https://github.com/chiaoyuntu/DeepLearning_FaceGen/blob/master/figures/model/fcgan2.png" width="80%" height="80%">
The generated female face image is shown in figure 7 and male image shown in figure 8. Although the face images do not look great and are blurred, we can still recognize the face is female or male with attributes given.

#### Training Details
In this phase, we use 10000 to 40000 images to train our model because we consider it is a reasonable number for not overfitting and is able to get results quickly. After several tries on different number of layers and parameters, we find the limit of current architecture. The fully-connected GAN model works best around 100 epoch. Even though we increase the number of epoch to 1000, the results get worse and become to noise again. 

### DCGAN
To design the face generation DCGAN model, we reference a stable DCGAN model [3] proposed by Radford et al.  In this project, we designed five different DCGAN models to achieve the goal we set: generate faces by given attributes and interactively modify face by given image and attributes. In each subsection, we will discuss the architecture and approach we choose, what we observe from result and how we improve from current design. 
Before we feed our raw images of human faces into our model for training, we first preprocess it through resizing the image to 32 pixels * 32 pixels in size for the consideration of training time. However, to obtain high resolution output image, we also try different size such as 64 pixels * 64 pixels.

#### Loss calculation
- D_fake: collect all generated faces and their attr
- D_real: a real face with its attr
- D_mistach; a real face with a wrong attr
```
D_loss = D_fake + D_real + D_mistach
```
Compared to the DCGAN model [3] proposed by Radford et al. For the discriminator optimization problem, we add a mismatch loss which calculate the loss for giving a real image with a wrong attributes to train the discriminator can map image with corresponding described attribute.
- Recon_loss: collect all the n pairs of a current image its refined image

- DCGAN model 1, 2, 3:
```
	G_loss = cross_entropy(D_fake, ones)
```

- DCGAN model 4, 5:
```
	G_loss = cross_entropy(D_fake, ones) + Recon_loss
```
Compared to the DCGAN model [3] proposed by Radford et al. For the generator optimization problem, besides the cross entropy loss passing from the discriminator, we also add a reconstruction loss between input image and generated image for model 4,5 in order to interactively refine the attributes based on input images. <br />
Next, we will going to introduce our five face generation DCGAN model. In each model, we will illustrate the architecture, hyperparameter settings, loss curves with respect to generators and discriminators, output values of discriminators, generated images.

### DCGAN 1 
In this DCGAN model, the discriminator uses 3 convolution layer with leaky relu activation and finally a fully connected layer. The generator is almost the exact opposite. Moreover, the attributes are given to the discriminator at the last fully connected layer.

#### Architecture
<img src="https://github.com/chiaoyuntu/DeepLearning_FaceGen/blob/master/figures/model/dcgan1-1.png" width="80%" height="80%">

| Parameter               | Value            |
| ----------------------- | ---------------- |
| Layers in discriminator | 3 conv + 1 fc    |
| Layers in generator     | 1 fc + 3 deconv  |
| generator input dim     | 100 + 1          |
| discriminator input dim | 32 * 32 * 3      |
| Batch size              | 32               |
| Noise vector size       | 10               |
| Attribute size          | 1                |
| Number of Images        | 50000            |
| Number of epoch         | 10               |
| Optimizer               | RMSPropOptimizer |
| Optimizer learning rate | 1e-4             |

- Result
face + loss + discriminator detailed prob <br />
<img src="https://github.com/chiaoyuntu/DeepLearning_FaceGen/blob/master/figures/model/dcgan1-2.png" width="50%" height="50%">
<img src="https://github.com/chiaoyuntu/DeepLearning_FaceGen/blob/master/figures/model/dcgan1-3.png" width="30%" height="30%">
<img src="https://github.com/chiaoyuntu/DeepLearning_FaceGen/blob/master/figures/model/dcgan1-4.png" width="30%" height="30%">

#### Training Details
The output image doesn’t show the male attribute we give. Perhaps because there is a considerable disparity in the ratio of number for attribute (1) and flattened image (2048). To improve the model, we adopt the architecture shown below that the attributes are given as a cube concatenated with the generated image to discriminator.

### DCGAN 2 with 32 * 32 image-size
#### Architecture
<img src="https://github.com/chiaoyuntu/DeepLearning_FaceGen/blob/master/figures/model/dcgan2-1(32_32).png" width="80%" height="80%">
<img src="https://github.com/chiaoyuntu/DeepLearning_FaceGen/blob/master/figures/model/dcgan2-2(32_32).png" width="80%" height="80%">
<img src="https://github.com/chiaoyuntu/DeepLearning_FaceGen/blob/master/figures/model/dcgan2-3(32_32).png" width="80%" height="80%">
<img src="https://github.com/chiaoyuntu/DeepLearning_FaceGen/blob/master/figures/model/dcgan2-4(32_32).png" width="80%" height="80%">

| Parameter               | Value                   |
|-------------------------|-------------------------|
| Layers in discriminator | 3 conv + 1 fc           |
| Layers in generator     | 1 fc + 3 deconv         |
| generator input dim     | 100 + attribute size    |
| discriminator input dim | (32, 32, 3 + attr size) |
| Batch size              | 32                      |
| Noise vector size       | 100                     |
| Number of Images        | 202599                  |

##### 2 attributes

| Parameter               | Value                   |
|-------------------------|-------------------------|
| Layers in discriminator | 3 conv + 1 fc           |
| Attribute size          | 2                       |
| Number of epoch         | 300                     |
| Optimizer               | AdamOptimizer           |
| Optimizer learning rate | 2e-4                    |

##### 8 attributes

| Parameter               | Value                   |
|-------------------------|-------------------------|
| Layers in discriminator | 3 conv + 1 fc           |
| Attribute size          | 8                       |
| Number of epoch         | 50                      |
| Optimizer               | AdamOptimizer           |
| Optimizer learning rate | 2e-4                    |

##### 23 attributes

| Parameter               | Value                   |
|-------------------------|-------------------------|
| Layers in discriminator | 4 conv + 1 fc           |
| Attribute size          | 23                      |
| Number of epoch         | 44                      |
| Optimizer               | AdamOptimizer           |
| Optimizer learning rate | 2e-4                    |

#### Result
##### 2 attributes
<img src="https://github.com/chiaoyuntu/DeepLearning_FaceGen/blob/master/figures/model/dcgan2-5(32_32)(2attr).png" width="80%" height="80%">

##### 8 attributes
<img src="https://github.com/chiaoyuntu/DeepLearning_FaceGen/blob/master/figures/model/dcgan2-6(32_32)(8attr).png" width="80%" height="80%">
<img src="https://github.com/chiaoyuntu/DeepLearning_FaceGen/blob/master/figures/model/dcgan2-7(32_32).png" width="80%" height="80%">
<img src="https://github.com/chiaoyuntu/DeepLearning_FaceGen/blob/master/figures/model/dcgan2-8(32_32).png" width="80%" height="80%">
<img src="https://github.com/chiaoyuntu/DeepLearning_FaceGen/blob/master/figures/model/dcgan2-9(32_32).png" width="80%" height="80%">

##### 23 attributes
<img src="https://github.com/chiaoyuntu/DeepLearning_FaceGen/blob/master/figures/model/dcgan2-10(32_32)(23attr).png" width="80%" height="80%">
<img src="https://github.com/chiaoyuntu/DeepLearning_FaceGen/blob/master/figures/model/dcgan2-11(32_32)(23attr).png" width="80%" height="80%">
<img src="https://github.com/chiaoyuntu/DeepLearning_FaceGen/blob/master/figures/model/dcgan2-13(32_32)(23attr).png" width="80%" height="80%">
<img src="https://github.com/chiaoyuntu/DeepLearning_FaceGen/blob/master/figures/model/dcgan2-12(32_32)(23attr).png" width="80%" height="80%">

#### Training Details
After we put the attributes as a cube to our discriminator, feature with male can show clearly in the output results. Then, we do experiments on our model with different numbers of attributes as our goal is to add as many attributes as possible. <br />
First, we train with two attributes, male and smiling, for hundreds of epoch. The results are good when we test on (male, smiling) as (0, 0), (1, 0), (0, 1), (1, 1) where 0 means no, 1 means yes. <br />
Second, we pick 8 attributes. Besides previous two attributes, we also pick eye-glasses, hair color, etc. We could see the eye-glasses and selected hair color in our pictures prominently. <br />
Next, we try to use 23 attributes. We have conclusion that adding more attributes makes our model unstable. We notice that it produces images whose attributes are not as we specified.

### DCGAN 2 with 64 * 64 image-size
#### Architecture
<img src="https://github.com/chiaoyuntu/DeepLearning_FaceGen/blob/master/figures/model/dcgan2(64_64)-1.png" width="80%" height="80%">
<img src="https://github.com/chiaoyuntu/DeepLearning_FaceGen/blob/master/figures/model/dcgan2(64_64)-2.png" width="80%" height="80%">

| Parameter               | Value                   |
|-------------------------|-------------------------|
| Number of Images        | 50000                   |
| Attribute size          | 5                       |
| Number of epoch         | 10                      |
| Optimizer               | RMSPropOptimizer        |
| Optimizer learning rate | 2e-4                    |

#### Result
<img src="https://github.com/chiaoyuntu/DeepLearning_FaceGen/blob/master/figures/model/dcgan2(64_64)-3.png" width="80%" height="80%">
<img src="https://github.com/chiaoyuntu/DeepLearning_FaceGen/blob/master/figures/model/dcgan2(64_64)-4.png" width="80%" height="80%">
<img src="https://github.com/chiaoyuntu/DeepLearning_FaceGen/blob/master/figures/model/dcgan2(64_64)-5.png" width="80%" height="80%">
<img src="https://github.com/chiaoyuntu/DeepLearning_FaceGen/blob/master/figures/model/dcgan2(64_64)-6.png" width="80%" height="80%">

### DCGAN 3
#### Architecture
<img src="https://github.com/chiaoyuntu/DeepLearning_FaceGen/blob/master/figures/model/dcgan3-1.png" width="80%" height="80%">
<img src="https://github.com/chiaoyuntu/DeepLearning_FaceGen/blob/master/figures/model/dcgan3-2.png" width="80%" height="80%">

| Parameter               | Value                |
| ----------------------- | -------------------  |
| generator input dim     | 32 * 32 * 3 + 10 * 5 |
| discriminator input dim | 32 * 32 * (3 + 5)    |
| Batch size              | 32                   |
| Noise cube size         | 32 * 32 * 3          |
| Attribute size          | 10 * 5               |
| Number of Images        | 15000                |
| Number of epoch         | 20                   |
| Optimizer               | RMSPropOptimizer     |
| Optimizer learning rate | 8e-5                 |

#### Result

#### Training Details


### Interactive DCGAN
#### Architecture
We put `noise` as initial input image.
<img src="https://github.com/chiaoyuntu/DeepLearning_FaceGen/blob/master/figures/model/dcgan4-1.png" width="80%" height="80%">

| Parameter               | Value                |
| ----------------------- | -------------------- |
| generator input dim     | 32 * 32 * 3 + 10 * 5 |
| discriminator input dim | 32 * 32 * (3 + 5)    |
| Batch size              | 32                   |
| Noise cube size         | 32 * 32 * 3          |
| Attribute size          | 10 * 5               |
| Number of Images        | 20000                |
| Number of epoch         | 20                   |
| Optimizer               | RMSPropOptimizer     |
| Optimizer learning rate | 8e-5                 |

#### Result
<img src="https://github.com/chiaoyuntu/DeepLearning_FaceGen/blob/master/figures/model/dcgan4-2.png" width="80%" height="80%">
<img src="https://github.com/chiaoyuntu/DeepLearning_FaceGen/blob/master/figures/model/dcgan4-3.png" width="80%" height="80%">
<img src="https://github.com/chiaoyuntu/DeepLearning_FaceGen/blob/master/figures/model/dcgan4-4.png" width="80%" height="80%">
<img src="https://github.com/chiaoyuntu/DeepLearning_FaceGen/blob/master/figures/model/dcgan4-5.png" width="80%" height="80%">
<img src="https://github.com/chiaoyuntu/DeepLearning_FaceGen/blob/master/figures/model/dcgan4-6.png" width="80%" height="80%">

#### Training Details


### Interactive DCGAN
#### Architecture
We put `image` as initial input image.
<img src="https://github.com/chiaoyuntu/DeepLearning_FaceGen/blob/master/figures/model/dcgan5-1.png" width="80%" height="80%">
<img src="https://github.com/chiaoyuntu/DeepLearning_FaceGen/blob/master/figures/model/dcgan5-2.png" width="80%" height="80%">

| Parameter               | Value                |
| ----------------------- | -------------------- |
| generator input dim     | 32 * 32 * 3 + 10 * 5 |
| discriminator input dim | 32 * 32 * (3 + 5)    |
| Batch size              | 32                   |
| Noise cube size         | 32 * 32 * 3          |
| Attribute size          | 10 * 5               |
| Number of Images        | 20000                |
| Number of epoch         | 20                   |
| Optimizer               | RMSPropOptimizer     |
| Optimizer learning rate | 8e-5                 |

#### Result
<img src="https://github.com/chiaoyuntu/DeepLearning_FaceGen/blob/master/figures/model/dcgan5-3.png" width="80%" height="80%">
<img src="https://github.com/chiaoyuntu/DeepLearning_FaceGen/blob/master/figures/model/dcgan5-4.png" width="80%" height="80%">
<img src="https://github.com/chiaoyuntu/DeepLearning_FaceGen/blob/master/figures/model/dcgan5-5.png" width="80%" height="80%">
<img src="https://github.com/chiaoyuntu/DeepLearning_FaceGen/blob/master/figures/model/dcgan5-6.png" width="80%" height="80%">
<img src="https://github.com/chiaoyuntu/DeepLearning_FaceGen/blob/master/figures/model/dcgan5-7.png" width="80%" height="80%">
<img src="https://github.com/chiaoyuntu/DeepLearning_FaceGen/blob/master/figures/model/dcgan5-8.png" width="80%" height="80%">

#### Training Details

## Experiment Results and Comparisons
In this section, we present our evaluation results for the above five models, mainly focusing on their compared quality of output images as well as their realism to human face, the stability of output image and the learning speed of model. The dataset we used for validation and testing is CelebA dataset, and sample outputs are presented in the following subsections.
### Comparison of applying attributes
#### DCGAN model 2 with 8 attributes

#### DCGAN model 3
<img src="https://github.com/chiaoyuntu/DeepLearning_FaceGen/blob/master/figures/experiments/exp_dcgan3-1.png" width="80%" height="80%">
<img src="https://github.com/chiaoyuntu/DeepLearning_FaceGen/blob/master/figures/experiments/exp_dcgan3-2.png" width="80%" height="80%">

### Comparison of our implemented models
#### Image quality
The base model started off on a positive note but the images had a lot of noise in them and would turn in random noise at higher epochs. Since the fully connected model could not eliminate noise we used the DC GAN model.  The DC GAN is more powerful model hence removes the noise.

#### Feature showing
We observed that though the basic DC GAN could produce relatively clearer images, the attributes were not stable. When we specify ‘male’ as one of the attributes we found that it sometimes incorrectly produces female faces too, we suspect that this is because of the fact that in DC-GAN-1 the ratio of the noise to attributes is too big, and therefore the attributes don’t seem to have much say on the resultant image. Therefore we try to give attributes also as a cube. This produces better results. Also we see that some attributes work well whereas some don’t. We think, this could be due to the fact that not all attributes are present in the same proportion in our dataset. Prominent attributes like, smile, gender, glasses etc perform well.

#### Stability
- Amount of data: We have observed that even though we keep the output image size same(32 X 32 X3) we can achieve results which seem to look like a higher resolution (clearer images), if we use a large dataset (200,000 images)
- Size of attributes: Appending attributes along with input image cannot represent enough information of attributes. Hence in this case using cube of attributes gives better results
- Learning rate: Too large values do not produce good 

## Conclusion
