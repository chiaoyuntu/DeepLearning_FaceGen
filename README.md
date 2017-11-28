# DeepLearning_FaceGen

Face Generation with Attributes.

## Instruction
Normally humans do not memorize things in pixel-level. Instead, it is much easier for us to describe representations of objects in our memory. Our goal is to generate a facial image as close as a person we want to describe based on the given attributes.

## Problem Formulation
Given the input of different attributes, we produce an output image corresponding to those facial attributes. Our dataset has about 40 attributes which includes basics like Male, Female. Facial characteristics like, arched eyebrows, high cheekbones, bags under eyes, big nose, black hair, blond hair etc and also cosmetic attributes like eyeglasses, goatee, heavy makeup, wearing hat etc. Here 1 indicates the presence of a certain feature and -1 indicates the absence as the table shown in figure 1(a). In the future, we want to be able to allow users to interactively refine the first guess by providing the improved attributes. As you can see in figure 1(b).

## Dataset
CelebFaces Attributes Dataset (CelebA) [2] is a large- scale face attributes dataset with more than 200K celebrity images showed as figure 3, each with 40 attribute annotations. The images in this dataset cover large pose variations and background clutter. CelebA has large diversities, large quantities, and rich annotations, including 10,177 number of identities, 202,599 number of face images, and 5 landmark locations, 40 binary attributes annotations per image. 

### Data preprocessing - Attribute selection
![All Attributes](https://github.com/chiaoyuntu/DeepLearning_FaceGen/blob/master/figures/dataset/all_attr.png?raw=true =600x400)

![23 Selected Attributes](https://github.com/chiaoyuntu/DeepLearning_FaceGen/blob/master/figures/dataset/23_attr.png?raw=true)

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
