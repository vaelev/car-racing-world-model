import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

class Encoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 4, 2, 1)
        self.conv2 = nn.Conv2d(32, 64, 4, 2, 1)
        self.conv3 = nn.Conv2d(64, 128, 4, 2, 1)
        self.conv4 = nn.Conv2d(128, 256, 4, 2, 1)
        
        self.fc_mean = nn.Linear(256 * 6 * 6, latent_dim)
        self.fc_logvar = nn.Linear(256 * 6 * 6, latent_dim)
        
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
        self.fc = nn.Linear(latent_dim, 256 * 6 * 6)
        self.deconv1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.deconv4 = nn.ConvTranspose2d(32, 3, 4, 2, 1)
        
    def forward(self, z):
        x = F.relu(self.fc(z))
        x = x.view(x.size(0), 256, 6, 6)
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

def collect_images(num_images=10000):
    env = gym.make("CarRacing-v3", render_mode="rgb_array", continuous=False)
    images = []
    
    while len(images) < num_images:
        obs, _ = env.reset()
        done = False
        
        while not done and len(images) < num_images:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            images.append(obs)
            
            if len(images) % 1000 == 0:
                print(f"Collected {len(images)} images")
    
    env.close()
    return np.array(images[:num_images])

class ImageDataset(Dataset):
    def __init__(self, images):
        self.images = torch.FloatTensor(images).permute(0, 3, 1, 2) / 255.0
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        return self.images[idx]

def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss

def train_vae(model, train_loader, val_loader, epochs=50, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss, recon_loss, kl_loss = vae_loss(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            
        avg_loss = train_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')
        
        if (epoch + 1) % 10 == 0:
            visualize_reconstruction(model, val_loader, device)

def visualize_reconstruction(model, val_loader, device, num_images=10):
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
        plt.show()

if __name__ == "__main__":
    print("Collecting images...")
    images = collect_images(10000)
    
    print("Creating datasets...")
    train_images = images[:8000]
    val_images = images[8000:]
    
    train_dataset = ImageDataset(train_images)
    val_dataset = ImageDataset(val_images)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    print("Creating model...")
    model = VAE(latent_dim=128)
    
    print("Training...")
    train_vae(model, train_loader, val_loader)
    
    print("Final visualization...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    visualize_reconstruction(model, val_loader, device)