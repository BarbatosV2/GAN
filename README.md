# GAN (Generative adversarial network)

# Install Requirements
```
pip install -r requirements.txt
```
# Create a directory for the dataset
```
mkdir -p stanford_dogs
cd stanford_dogs
```

# Download the dataset
Depends on what dataset u want to train. For me, I am training dog dataset.
```
wget http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar
wget http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar
```

if not working, can directly download at http://vision.stanford.edu/aditya86/ImageNetDogs/ 

# Extract the images
```
tar -xvf images.tar
tar -xvf annotation.tar
```
# Read Me

**cuda.py** is to check whether GPU is working or not. There are other way to check as well like ```nvidia-smi``` and ```nvcc --version```.

This GAN model use GPU which save the RAM usage and train data faster than CPU training.

Using RAM makes not only slowdown PC, also crash and makes memory insufficient while in training process.

Just run the python file and it will generate the image. Have Fun.

To start train the model, just run the **GANV2.py**. This will save the image of how the training process is looks like and it will also save epoch models in the models folder.

The **checkpoint.py** will start from the epoch where you stopped from. For example, your computer just crashed while training and stops at 300 epoch, this code will restart the training process from that epoch. **If it does not work, the last pth file from models folder might be correpted so please delete the last epoch pth file.**

**generator.py** will generate the image, fix the epoch number form the code to use the perfect trained model.

**Note**

For continuous training without saving or making checkpoint change to ```utils_nosave``` in **GANV2.py**. For saving checkpoint and model, use ```utils```

PS - use **GANV2.py** file to save the train data and use **generator.py** to generate the images using the saved model file

****Updated Files is better Neural Network Structure for Unsupervised Learning****

# Initial Result
**epoch 1**

![gan_image_1](https://github.com/BarbatosV2/GAN/assets/63419320/d011ad1d-5aba-4906-afd3-916c55fde0a2)

# Final Result
**epoch 30**

![gan_image_25740](https://github.com/BarbatosV2/GAN/assets/63419320/76da3070-8d35-4cf4-b2e9-300c030acfe6)

Looks like dogs to me XD (PS, Will change some neural network for more performance and training process)

