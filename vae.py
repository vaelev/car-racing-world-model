import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import wandb

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygame")

class Encoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        # Input: 86x96x3 (cropped CarRacing image)
        self.conv1 = nn.Conv2d(3, 32, 4, 2, 1)   # 43x48x32
        self.conv2 = nn.Conv2d(32, 64, 4, 2, 1)  # 21x24x64
        self.conv3 = nn.Conv2d(64, 128, 4, 2, 1) # 10x12x128
        self.conv4 = nn.Conv2d(128, 256, 4, 2, 1) # 5x6x256
        
        self.fc_mean = nn.Linear(256 * 5 * 6, latent_dim)
        self.fc_logvar = nn.Linear(256 * 5 * 6, latent_dim)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        z_mean = self.fc_mean(x)
        z_logvar = self.fc_logvar(x)
        return z_mean, z_logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        # Reconstruct 86x96x3 images
        self.fc = nn.Linear(latent_dim, 256 * 5 * 6)
        self.deconv1 = nn.ConvTranspose2d(256, 128, 4, 2, 1) # 10x12x128
        self.deconv2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)  # 20x24x64
        self.deconv3 = nn.ConvTranspose2d(64, 32, 4, 2, 1)   # 40x48x32
        self.deconv4 = nn.ConvTranspose2d(32, 3, (10, 4), 2, 1) # 86x96x3
        
    def forward(self, z):
        x = F.relu(self.fc(z))
        x = x.view(x.size(0), 256, 5, 6)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = torch.sigmoid(self.deconv4(x))
        return x

class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

def preprocess_image(img):
    # Crop bottom 10% to remove status bar: 96x96 → 86x96
    img_cropped = img[:86, :]
    return img_cropped

def sample_action(env):
    # Bias toward movement: gas and steering more likely than brake/do-nothing
    # Actions: [0: do nothing, 1: left, 2: right, 3: gas, 4: brake]
    action_probs = [0.1, 0.25, 0.25, 0.35, 0.05]  # favor gas and steering
    return np.random.choice(5, p=action_probs)

def collect_images(num_images=40000):
    env = gym.make("CarRacing-v3", render_mode="rgb_array", continuous=False)
    images = []
    
    while len(images) < num_images:
        obs, _ = env.reset()
        
        # Skip the initial zoom-in phase (50 no-op steps)
        for i in range(50):
            obs, _, terminated, truncated, _ = env.step(0)  # 0 = do nothing
            if terminated or truncated:
                break
        
        done = terminated or truncated
        
        while not done and len(images) < num_images:
            action = sample_action(env)  # Biased sampling
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Crop bottom status bar
            cropped_obs = preprocess_image(obs)
            images.append(cropped_obs)
            
            if len(images) % 1000 == 0:
                print(f"Collected {len(images)} images")
    
    env.close()
    return np.array(images[:num_images])

class ImageDataset(Dataset):
    def __init__(self, images):
        # Images are 86x96x3 (cropped CarRacing frames)
        self.images = torch.FloatTensor(images).permute(0, 3, 1, 2) / 255.0
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        return self.images[idx]

def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss

def train_vae(model, train_loader, val_loader, epochs=200, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_recon_loss = 0
        train_kl_loss = 0
        
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss, recon_loss, kl_loss = vae_loss(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            train_recon_loss += recon_loss.item()
            train_kl_loss += kl_loss.item()
            optimizer.step()
            
        avg_loss = train_loss / len(train_loader.dataset)
        avg_recon_loss = train_recon_loss / len(train_loader.dataset)
        avg_kl_loss = train_kl_loss / len(train_loader.dataset)
        
        print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}, Recon: {avg_recon_loss:.4f}, KL: {avg_kl_loss:.4f}')
        
        # Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "train_recon_loss": avg_recon_loss,
            "train_kl_loss": avg_kl_loss,
        })
        
        if (epoch + 1) % 10 == 0:
            image_path = f'reconstruction_epoch_{epoch+1}.png'
            visualize_reconstruction(model, val_loader, device, save_path=image_path)
            
            wandb.log({f"reconstruction_epoch_{epoch+1}": wandb.Image(image_path)})
            
    return model

def visualize_reconstruction(model, val_loader, device, num_images=10, save_path=None):
    model.eval()
    with torch.no_grad():
        data = next(iter(val_loader))[:num_images].to(device)
        recon, _, _ = model(data)
        
        fig, axes = plt.subplots(2, num_images, figsize=(15, 3))
        for i in range(num_images):
            axes[0, i].imshow(data[i].cpu().permute(1, 2, 0))
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(recon[i].cpu().permute(1, 2, 0))
            axes[1, i].set_title('Reconstructed')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        else:
            plt.show()
        plt.close()

def save_model(model, filepath):
    torch.save({
        'model_state_dict': model.state_dict(),
        'latent_dim': 128,  # Save architecture info
        'model_class': 'VAE'
    }, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath, latent_dim=128, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(filepath, map_location=device)
    
    model = VAE(latent_dim=checkpoint.get('latent_dim', latent_dim))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded from {filepath}")
    return model

def encode_image(model, image, device=None):
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    with torch.no_grad():
        if isinstance(image, np.ndarray):
            # Convert numpy array to tensor
            if len(image.shape) == 3:  # Single image
                image = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0) / 255.0
            else:  # Batch of images
                image = torch.FloatTensor(image).permute(0, 3, 1, 2) / 255.0
        
        image = image.to(device)
        mu, logvar = model.encoder(image)
        # Use mean for deterministic encoding
        return mu.cpu().numpy()

def decode_latent(model, latent_vector, device=None):
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    with torch.no_grad():
        if isinstance(latent_vector, np.ndarray):
            latent_vector = torch.FloatTensor(latent_vector)
        
        if len(latent_vector.shape) == 1:
            latent_vector = latent_vector.unsqueeze(0)  # Add batch dimension
        
        latent_vector = latent_vector.to(device)
        decoded = model.decoder(latent_vector)
        
        # Convert back to numpy and proper image format
        decoded = decoded.cpu().numpy()
        if decoded.shape[0] == 1:  # Single image
            decoded = decoded[0].transpose(1, 2, 0)  # CHW -> HWC
        else:  # Batch
            decoded = decoded.transpose(0, 2, 3, 1)  # BCHW -> BHWC
        
        return np.clip(decoded, 0, 1)

def interpolate_in_latent_space(model, image1, image2, steps=10, device=None):
    if device is None:
        device = next(model.parameters()).device
    
    z1 = encode_image(model, image1, device)
    z2 = encode_image(model, image2, device)
    
    interpolations = []
    for i in range(steps):
        alpha = i / (steps - 1)
        z_interp = (1 - alpha) * z1 + alpha * z2
        decoded = decode_latent(model, z_interp, device)
        interpolations.append(decoded[0] if len(decoded.shape) == 4 else decoded)
    
    return interpolations

def test_inference_example(model, val_loader, device):
    test_batch = next(iter(val_loader))
    test_image = test_batch[0].cpu().permute(1, 2, 0).numpy()  # Convert to HWC
    
    print("Testing encoding...")
    latent = encode_image(model, test_image, device)
    print(f"Encoded to latent vector of shape: {latent.shape}")
    
    print("Testing decoding...")
    reconstructed = decode_latent(model, latent, device)
    print(f"Decoded to image of shape: {reconstructed.shape}")
    
    print("Inference test completed successfully!")

if __name__ == "__main__":
    wandb.init(
        project="car-racing-vae",
        config={
            "latent_dim": 128,
            "batch_size": 256,
            "epochs": 200,
            "learning_rate": 1e-3,
            "num_images": 40000,
            "train_split": 0.8
        }
    )
    
    print("Collecting images...")
    images = collect_images(40000)
    
    print("Creating datasets...")
    # Randomly sample 80% for training, 20% for validation
    indices = np.random.permutation(len(images))
    train_size = int(0.8 * len(images))
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_images = images[train_indices]
    val_images = images[val_indices]
    
    train_dataset = ImageDataset(train_images)
    val_dataset = ImageDataset(val_images)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    
    print("Creating model...")
    model = VAE(latent_dim=128)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    print(f"Encoder parameters: {encoder_params:,}")
    print(f"Decoder parameters: {decoder_params:,}")
    
    wandb.log({
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "encoder_parameters": encoder_params,
        "decoder_parameters": decoder_params
    })
    
    print("Training...")
    model = train_vae(model, train_loader, val_loader)
    
    print("Saving model...")
    save_model(model, 'vae_model.pth')
    
    print("Final visualization...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    visualize_reconstruction(model, val_loader, device, save_path='final_reconstruction.png')
    
    print("Saving artifacts to wandb...")
    wandb.log_artifact('vae_model.pth', name='trained_vae_model', type='model')
    wandb.log_artifact('final_reconstruction.png', name='final_reconstruction', type='image')
    
    print("Testing model loading...")
    loaded_model = load_model('vae_model.pth', latent_dim=128)
    test_inference_example(loaded_model, val_loader, device)
    
    wandb.finish()
    print("Training complete! Check wandb dashboard for detailed logs and artifacts.")

#apt-get update
#apt-get install -y swig