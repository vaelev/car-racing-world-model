import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from vae import load_model, encode_image, decode_latent, sample_action, preprocess_image
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygame")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MixtureOfGaussiansHead(nn.Module):
    def __init__(self, input_dim, latent_dim, num_components=8):
        super().__init__()
        self.num_components = num_components
        self.latent_dim = latent_dim
        
        self.means = nn.Linear(input_dim, num_components * latent_dim)
        self.logvars = nn.Linear(input_dim, num_components * latent_dim)
        self.weights = nn.Linear(input_dim, num_components)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        means = self.means(x).view(batch_size, self.num_components, self.latent_dim)
        logvars = self.logvars(x).view(batch_size, self.num_components, self.latent_dim)
        weights = F.softmax(self.weights(x), dim=-1)
        
        return means, logvars, weights

class TransformerWorldModel(nn.Module):
    def __init__(self, latent_dim=128, action_dim=5, hidden_dim=256, 
                 num_layers=6, num_heads=8, num_components=8, seq_len=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        
        self.state_proj = nn.Linear(latent_dim, hidden_dim // 2)
        self.action_embed = nn.Embedding(action_dim, hidden_dim // 2)
        self.input_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.pos_encoding = PositionalEncoding(hidden_dim, seq_len)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)
        
        self.mog_head = MixtureOfGaussiansHead(hidden_dim, latent_dim, num_components)
        
        self.register_buffer('causal_mask', 
                           torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1))
    
    def forward(self, states, actions):
        batch_size, seq_len = states.size(0), states.size(1)
        
        state_emb = self.state_proj(states)
        action_emb = self.action_embed(actions)
        
        x = torch.cat([state_emb, action_emb], dim=-1)
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        
        mask = self.causal_mask[:seq_len, :seq_len]
        x = self.transformer(x, x, tgt_mask=mask)
        
        means, logvars, weights = self.mog_head(x)
        return means, logvars, weights

def mog_loss(pred_means, pred_logvars, pred_weights, target):
    batch_size, seq_len, num_components, latent_dim = pred_means.shape
    target = target.unsqueeze(2).expand(-1, -1, num_components, -1)
    
    log_probs = -0.5 * ((target - pred_means) ** 2 / torch.exp(pred_logvars) + pred_logvars + np.log(2 * np.pi))
    log_probs = torch.sum(log_probs, dim=-1)
    
    weighted_log_probs = log_probs + torch.log(pred_weights + 1e-8)
    loss = -torch.logsumexp(weighted_log_probs, dim=-1)
    
    return loss.mean()

def collect_trajectories(vae_model, num_trajectories=1000, seq_len=64):
    env = gym.make("CarRacing-v3", render_mode="rgb_array", continuous=False)
    device = next(vae_model.parameters()).device
    
    trajectories = []
    
    for traj_idx in range(num_trajectories):
        obs, _ = env.reset()
        
        for i in range(50):
            obs, _, terminated, truncated, _ = env.step(0)
            if terminated or truncated:
                break
        
        states = []
        actions = []
        done = terminated or truncated
        
        while not done and len(states) < seq_len + 1:
            action = sample_action(env)
            cropped_obs = preprocess_image(obs)
            
            latent = encode_image(vae_model, cropped_obs, device)
            states.append(latent.flatten())
            actions.append(action)
            
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        
        if len(states) >= seq_len + 1:
            trajectories.append({
                'states': np.array(states),
                'actions': np.array(actions)
            })
        
        if (traj_idx + 1) % 100 == 0:
            print(f"Collected {traj_idx + 1} trajectories")
    
    env.close()
    return trajectories

class TrajectoryDataset(Dataset):
    def __init__(self, trajectories, seq_len=64):
        self.data = []
        
        for traj in trajectories:
            states = traj['states']
            actions = traj['actions']
            
            for i in range(len(states) - seq_len):
                input_states = states[i:i+seq_len]
                input_actions = actions[i:i+seq_len]
                target_states = states[i+1:i+seq_len+1]
                
                self.data.append({
                    'states': torch.FloatTensor(input_states),
                    'actions': torch.LongTensor(input_actions),
                    'targets': torch.FloatTensor(target_states)
                })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def train_world_model(model, train_loader, val_loader, vae_model, epochs=100, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            states = batch['states'].to(device)
            actions = batch['actions'].to(device)
            targets = batch['targets'].to(device)
            
            optimizer.zero_grad()
            means, logvars, weights = model(states, actions)
            loss = mog_loss(means, logvars, weights, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        scheduler.step()
        avg_loss = train_loss / len(train_loader)
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    states = batch['states'].to(device)
                    actions = batch['actions'].to(device)
                    targets = batch['targets'].to(device)
                    
                    means, logvars, weights = model(states, actions)
                    loss = mog_loss(means, logvars, weights, targets)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            print(f'Epoch {epoch+1}: Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
            
            if (epoch + 1) % 20 == 0:
                test_prediction(model, vae_model, device, epoch + 1)
                save_checkpoint(model, optimizer, epoch + 1, f'world_model_epoch_{epoch+1}.pth')
    
    return model

def sample_from_mog(means, logvars, weights):
    batch_size, num_components, latent_dim = means.shape
    
    component_idx = torch.multinomial(weights, 1).squeeze(-1)
    
    selected_means = means.gather(1, component_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, latent_dim)).squeeze(1)
    selected_logvars = logvars.gather(1, component_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, latent_dim)).squeeze(1)
    
    std = torch.exp(0.5 * selected_logvars)
    eps = torch.randn_like(std)
    
    return selected_means + eps * std

def test_prediction(model, vae_model, device, epoch):
    model.eval()
    env = gym.make("CarRacing-v3", render_mode="rgb_array", continuous=False)
    
    obs, _ = env.reset()
    for _ in range(50):
        obs, _, terminated, truncated, _ = env.step(0)
        if terminated or truncated:
            break
    
    seq_len = 32
    states = []
    actions = []
    
    for i in range(seq_len):
        action = sample_action(env)
        cropped_obs = preprocess_image(obs)
        latent = encode_image(vae_model, cropped_obs, device)
        
        states.append(latent.flatten())
        actions.append(action)
        
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    
    if len(states) < seq_len:
        env.close()
        return
    
    input_states = torch.FloatTensor(states).unsqueeze(0).to(device)
    input_actions = torch.LongTensor(actions).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred_states = []
        current_states = input_states.clone()
        current_actions = input_actions.clone()
        
        for step in range(16):
            means, logvars, weights = model(current_states, current_actions)
            next_state = sample_from_mog(means[:, -1:], logvars[:, -1:], weights[:, -1:])
            
            pred_states.append(next_state.squeeze(1))
            
            current_states = torch.cat([current_states[:, 1:], next_state], dim=1)
            next_action = torch.randint(0, 5, (1, 1)).to(device)
            current_actions = torch.cat([current_actions[:, 1:], next_action], dim=1)
    
    predicted_latents = torch.cat(pred_states, dim=0).cpu().numpy()
    decoded_images = decode_latent(vae_model, predicted_latents)
    
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    for i in range(8):
        if i < len(decoded_images):
            axes[0, i].imshow(decoded_images[i])
            axes[0, i].set_title(f'Pred {i+1}')
            axes[0, i].axis('off')
            
            if i < 8:
                axes[1, i].imshow(decoded_images[i + 8] if i + 8 < len(decoded_images) else decoded_images[-1])
                axes[1, i].set_title(f'Pred {i+9}')
                axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'prediction_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    env.close()

def save_checkpoint(model, optimizer, epoch, filepath):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, filepath)
    print(f"Checkpoint saved: {filepath}")

if __name__ == "__main__":
    print("Loading VAE model...")
    vae_model = load_model('vae_model.pth', latent_dim=128)
    
    print("Collecting trajectories...")
    trajectories = collect_trajectories(vae_model, num_trajectories=800, seq_len=64)
    
    print(f"Collected {len(trajectories)} trajectories")
    
    train_size = int(0.8 * len(trajectories))
    train_trajectories = trajectories[:train_size]
    val_trajectories = trajectories[train_size:]
    
    print("Creating datasets...")
    train_dataset = TrajectoryDataset(train_trajectories, seq_len=64)
    val_dataset = TrajectoryDataset(val_trajectories, seq_len=64)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"Train dataset: {len(train_dataset)} sequences")
    print(f"Val dataset: {len(val_dataset)} sequences")
    
    print("Creating transformer model...")
    model = TransformerWorldModel(
        latent_dim=128,
        action_dim=5,
        hidden_dim=256,
        num_layers=6,
        num_heads=8,
        num_components=8,
        seq_len=64
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    print("Training world model...")
    model = train_world_model(model, train_loader, val_loader, vae_model, epochs=200, lr=1e-4)
    
    print("Saving final model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'latent_dim': 128,
            'action_dim': 5,
            'hidden_dim': 256,
            'num_layers': 6,
            'num_heads': 8,
            'num_components': 8,
            'seq_len': 64
        }
    }, 'world_model_final.pth')
    
    print("Training complete!")