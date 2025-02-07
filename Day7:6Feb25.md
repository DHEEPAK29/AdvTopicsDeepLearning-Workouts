Latent Diffusion Models (LDMs)  

Forward Diffusion: Addition of noise + Shrinking  
Then, Inverting the noise, step by step. Bringing the image back to its original form. 

Step by Step: Bunch of tiny changes. Want things that are close to reversible. Generative while Denoising. Fixed while Forward process [Noising].   

Learning Denoising  
Tracrable posterior distribution   

Reparametrization  

Noise Scheduling:  Control how much noise is to be added in each step.  

x input [Pixel space] -> Encode -> Latent space [Diffusion process -> De-noising U-Net -> Cross attention] -> Decode -> x dash output  

Loss function pushing into a Gaussian Distribution 

Goes to marginal Space. Not a pure gaussian. 


DALL-E 2: Pretrain text encoder with CLIP and freeze it   

Interpolation: in the embedding space. 
