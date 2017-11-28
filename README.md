# DeepLearning_FaceGen

Face Generation with Attributes.

## Instruction
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
face + loss + discriminator detailed prob <br />
[]
[]
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
(G: input noise vector + attr vector, D: input image + attr vector)  -> small attr info (compared to flattened image) in D <br />
In this DCGAN model, the discriminator uses 3 convolution layer with leaky relu activation and finally a fully connected layer. The generator is almost the exact opposite. Moreover, the attributes are given to the discriminator at the last fully connected layer.
#### Architecture
<img src="https://github.com/chiaoyuntu/DeepLearning_FaceGen/blob/master/figures/model/dcgan1-1.png" width="80%" height="80%">
| Parameter               | Value            |
|-------------------------|------------------|
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
<img src="https://github.com/chiaoyuntu/DeepLearning_FaceGen/blob/master/figures/model/dcgan1-2.png" width="80%" height="80%">
<img src="https://github.com/chiaoyuntu/DeepLearning_FaceGen/blob/master/figures/model/dcgan1-3.png" width="80%" height="80%">
<img src="https://github.com/chiaoyuntu/DeepLearning_FaceGen/blob/master/figures/model/dcgan1-4.png" width="80%" height="80%">



