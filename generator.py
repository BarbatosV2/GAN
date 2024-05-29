import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image parameters
img_height, img_width = 64, 64

# Define the Generator (same as the one used during training)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.init_size = 64 // 4
        self.l1 = nn.Sequential(nn.Linear(100, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# Function to load the model
def load_model(model, epoch):
    model_dir = os.path.expanduser('~/Documents/GAN/models')
    model_path = f"{model_dir}/generator_epoch_{epoch}.pth"
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['generator_state_dict'])
    model.eval()
    return model

# Function to generate and save one image
def generate_images(model, num_images=1):
    with torch.no_grad():
        noise = torch.randn(num_images, 100, device=device)
        gen_imgs = model(noise).cpu()
        gen_imgs = 0.5 * gen_imgs + 0.5  # Rescale to 0-1

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(gen_imgs[0].permute(1, 2, 0))
        ax.axis('off')

        # Ensure the save directory exists
        save_dir = os.path.join(os.path.dirname(__file__), 'generatedImages')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Find the next available filename
        file_count = len([name for name in os.listdir(save_dir) if os.path.isfile(os.path.join(save_dir, name))])
        file_name = f'generated_image_{file_count + 1}.png'

        # Save the figure
        fig.savefig(os.path.join(save_dir, file_name))
        plt.close()

# Load the trained model
generator_loaded = Generator().to(device)
generator_loaded = load_model(generator_loaded, epoch=200)  # Adjust epoch number as needed

# Generate and save one image
generate_images(generator_loaded, num_images=1)
