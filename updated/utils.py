import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

def save_model(generator, discriminator, optimizer_G, optimizer_D, epoch):
    model_dir = 'C:/Users/zawwi/Documents/MachineLearning/GAN-main/models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save({
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
    }, f"{model_dir}/generator_epoch_{epoch}.pth")

def save_images(generator, epoch, device):
    generator.eval()  # Set the generator to evaluation mode
    with torch.no_grad():
        noise = torch.randn(1, 100, device=device)
        gen_img = generator(noise).cpu()
        gen_img = 0.5 * gen_img + 0.5  # Rescale 0-1

        fig, ax = plt.subplots()
        ax.imshow(gen_img[0].permute(1, 2, 0))
        ax.axis('off')

        save_dir = 'C:/Users/zawwi/Documents/MachineLearning/GAN-main/dog_gan_images'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fig.savefig(f"{save_dir}/dog_gan_image_{epoch}.png")
        plt.close()
    generator.train()  # Set the generator back to training mode

def train_gan(generator, discriminator, criterion, optimizer_G, optimizer_D, train_loader, device, epochs, start_epoch=0):
    # Initialize lists to store metrics
    G_losses = []
    D_losses = []
    Acc_real_list = []
    Acc_fake_list = []

    for epoch in range(start_epoch, epochs):
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
        epoch_G_loss = 0.0
        epoch_D_loss = 0.0
        epoch_acc_real = 0.0
        epoch_acc_fake = 0.0
        count = 0

        for i, data in loop:
            real_images, _ = data
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            count += batch_size

            label_real = torch.full((batch_size, 1), 1, dtype=torch.float, device=device)
            label_fake = torch.full((batch_size, 1), 0, dtype=torch.float, device=device)

            # Train Discriminator
            discriminator.zero_grad()
            output_real = discriminator(real_images)
            loss_real = criterion(output_real, label_real)
            loss_real.backward()

            noise = torch.randn(batch_size, 100, device=device)
            fake_images = generator(noise)
            output_fake = discriminator(fake_images.detach())
            loss_fake = criterion(output_fake, label_fake)
            loss_fake.backward()
            optimizer_D.step()

            D_loss = loss_real + loss_fake
            epoch_D_loss += D_loss.item()

            # Train Generator
            generator.zero_grad()
            output = discriminator(fake_images)
            loss_G = criterion(output, label_real)
            loss_G.backward()
            optimizer_G.step()

            G_loss = loss_G.item()
            epoch_G_loss += G_loss

            # Calculate accuracy
            output_real_acc = output_real >= 0.5
            output_fake_acc = output_fake < 0.5
            acc_real = (output_real_acc.sum().float() / batch_size) * 100
            acc_fake = (output_fake_acc.sum().float() / batch_size) * 100

            epoch_acc_real += acc_real.item() * batch_size
            epoch_acc_fake += acc_fake.item() * batch_size

            # Update progress bar
            loop.set_description(f"Epoch [{epoch + 1}/{epochs}]")
            loop.set_postfix(D_loss=D_loss.item(), G_loss=G_loss, Acc_real=acc_real.item(), Acc_fake=acc_fake.item())

        # Save metrics for the epoch
        G_losses.append(epoch_G_loss / count)
        D_losses.append(epoch_D_loss / count)
        Acc_real_list.append(epoch_acc_real / count)
        Acc_fake_list.append(epoch_acc_fake / count)
  
        # Save generated images
        save_images(generator, epoch + 1, device)
        # Save the Generator model
        #save_model(generator, discriminator, optimizer_G, optimizer_D, epoch + 1)
        
        # Save generated images and model every 500 epochs
        if (epoch + 1) % 10 == 0:
        #    save_images(generator, epoch + 1, device)
            save_model(generator, discriminator, optimizer_G, optimizer_D, epoch + 1)

    # Plot training graphs
    plt.figure(figsize=(12, 8))

    # Plot Generator and Discriminator Loss
    plt.subplot(2, 1, 1)
    plt.plot(range(start_epoch + 1, epochs + 1), G_losses, label="Generator Loss")
    plt.plot(range(start_epoch + 1, epochs + 1), D_losses, label="Discriminator Loss")
    plt.title("Generator and Discriminator Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Plot Real and Fake Accuracy
    plt.subplot(2, 1, 2)
    plt.plot(range(start_epoch + 1, epochs + 1), Acc_real_list, label="Real Accuracy")
    plt.plot(range(start_epoch + 1, epochs + 1), Acc_fake_list, label="Fake Accuracy")
    plt.title("Discriminator Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    plt.tight_layout()
    
    # Save the plot as an image
    plot_dir = 'C:/Users/zawwi/Documents/MachineLearning/GAN-main/training_plots'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    #plt.savefig(f"{plot_dir}/training_plot_epoch_{epochs}.png")
    
    # Get current date and time in a formatted string
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Save the file with the date and time in the name
    plt.savefig(f"{plot_dir}/training_plot_{current_time}.png")

    plt.show()
