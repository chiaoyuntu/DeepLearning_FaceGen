# DeepLearning_FaceGen

Face Generation with Attributes

## Instruction
Normally humans do not memorize things in pixel-level. Instead, it is much easier for us to describe representations of objects in our memory. Our goal is to generate a facial image as close as a person we want to describe based on the given attributes.

## Problem Formulation
Given the input of different attributes, we produce an output image corresponding to those facial attributes. Our dataset has about 40 attributes which includes basics like Male, Female. Facial characteristics like, arched eyebrows, high cheekbones, bags under eyes, big nose, black hair, blond hair etc and also cosmetic attributes like eyeglasses, goatee, heavy makeup, wearing hat etc. Here 1 indicates the presence of a certain feature and -1 indicates the absence as the table shown in figure 1(a). In the future, we want to be able to allow users to interactively refine the first guess by providing the improved attributes. As you can see in figure 1(b).

## Dataset
CelebFaces Attributes Dataset (CelebA) [2] is a large- scale face attributes dataset with more than 200K celebrity images showed as figure 3, each with 40 attribute annotations. The images in this dataset cover large pose variations and background clutter. CelebA has large diversities, large quantities, and rich annotations, including 10,177 number of identities, 202,599 number of face images, and 5 landmark locations, 40 binary attributes annotations per image. 

### Data preprocessing - Attribute selection
[<img src="https://drive.google.com/file/d/1accFe_XqPT1yJTPnZ6IvNZAU_zQ0Op5M/view?usp=sharing">]
We analyzed the 40 attributes showed as figure 4 to find the non sparse attributes, and discard the sparse ones. In order to experiment,  we picked 23 attributes as shown in the figure 5 which we feel would distinguish faces in a meaningful way, like hair colors, eyes shapes, nose etc. But for the further experiments we want to consider reducing the attributes to around 5. 

## Models

### Baseline Model - GAN with FC
- Architecture