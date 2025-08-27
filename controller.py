import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import wandb
import warnings
import cv2
import os
from vae import load_model as load_vae_model, encode_image, preprocess_image, sample_action
from transformer_world_model import TransformerWorldModel

warnings.filterwarnings("ignore", category=UserWarning, module="pygame")

class Controller(nn.Module):
    def __init__(self, input_dim=256, action_dim=5):
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        # Single linear layer with bias - now takes world model features
        self.linear = nn.Linear(input_dim, action_dim)
        
    def forward(self, world_model_features):
        # world_model_features: (batch_size, input_dim) -> action_logits: (batch_size, action_dim)
        return self.linear(world_model_features)
    
    def get_action_probs(self, world_model_features, temperature=1.0):
        logits = self.forward(world_model_features)
        return F.softmax(logits / temperature, dim=-1)
    
    def sample_action(self, world_model_features, temperature=1.0):
        probs = self.get_action_probs(world_model_features, temperature)
        return torch.multinomial(probs, 1).item()

def load_world_model(filepath, device):
    """Load the trained transformer world model"""
    checkpoint = torch.load(filepath, map_location=device)
    config = checkpoint['config']
    
    model = TransformerWorldModel(
        latent_dim=config['latent_dim'],
        action_dim=config['action_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        num_components=config['num_components'],
        seq_len=config['seq_len']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"World model loaded from {filepath}")
    return model

def get_world_model_features(world_model, states_sequence, actions_sequence, device):
    """
    Extract features from world model's transformer layers
    
    Args:
        world_model: Trained transformer world model
        states_sequence: Sequence of latent states (seq_len, latent_dim)
        actions_sequence: Sequence of actions (seq_len,)
        device: Device to run on
    
    Returns:
        features: World model internal representation (hidden_dim,)
    """
    with torch.no_grad():
        # Ensure inputs are tensors and have batch dimension
        if isinstance(states_sequence, np.ndarray):
            states_sequence = torch.FloatTensor(states_sequence).to(device)
        if isinstance(actions_sequence, np.ndarray):
            actions_sequence = torch.LongTensor(actions_sequence).to(device)
        
        if len(states_sequence.shape) == 2:
            states_sequence = states_sequence.unsqueeze(0)  # Add batch dim
        if len(actions_sequence.shape) == 1:
            actions_sequence = actions_sequence.unsqueeze(0)  # Add batch dim
        
        # Get embeddings and run through transformer
        state_emb = world_model.state_proj(states_sequence)
        action_emb = world_model.action_embed(actions_sequence)
        
        x = torch.cat([state_emb, action_emb], dim=-1)
        x = world_model.input_proj(x)
        x = world_model.pos_encoding(x)
        
        # Run through transformer layers
        seq_len = x.size(1)
        mask = world_model.causal_mask[:seq_len, :seq_len]
        transformer_output = world_model.transformer(x, x, tgt_mask=mask)
        
        # Use the last timestep's representation as features for controller
        features = transformer_output[0, -1, :]  # (hidden_dim,)
        
        return features

def rollout_in_real_env(controller, world_model, vae_model, max_steps=1000, 
                       temperature=1.0, seq_len=32, device='cuda'):
    """
    Perform a trajectory rollout in the real environment using world model features
    
    Args:
        controller: Controller network
        world_model: Trained transformer world model (for feature extraction)
        vae_model: Trained VAE model
        max_steps: Maximum steps in rollout
        temperature: Temperature for action sampling
        seq_len: Sequence length for world model context
        device: Device to run on
    
    Returns:
        trajectory: Dict with states, actions, rewards, log_probs, world_model_features
    """
    env = gym.make("CarRacing-v3", render_mode="rgb_array", continuous=False)
    
    states = []  # Latent states
    actions = []
    log_probs = []
    rewards = []
    world_model_features = []
    
    # Initialize
    obs, _ = env.reset()
    
    # Skip initial zoom-in phase
    for _ in range(50):
        obs, _, terminated, truncated, _ = env.step(0)
        if terminated or truncated:
            env.close()
            return None  # Failed initialization
    
    # Build initial sequence by taking random actions
    states_sequence = []
    actions_sequence = []
    
    for _ in range(seq_len):
        action = sample_action(env)
        cropped_obs = preprocess_image(obs)
        latent = encode_image(vae_model, cropped_obs, device)
        
        states_sequence.append(latent.flatten())
        actions_sequence.append(action)
        
        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            env.close()
            return None  # Episode ended too early
    
    # Convert to arrays
    states_sequence = np.array(states_sequence)
    actions_sequence = np.array(actions_sequence)
    
    # Main rollout loop
    for _ in range(max_steps):
        # Get world model features from current sequence
        wm_features = get_world_model_features(world_model, states_sequence, 
                                             actions_sequence, device)
        
        # Controller predicts action from world model features
        action_probs = controller.get_action_probs(wm_features.unsqueeze(0), temperature)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        # Step in real environment
        obs, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        
        # Store trajectory data
        states.append(states_sequence[-1])  # Current state
        actions.append(action.item())
        rewards.append(reward)
        log_probs.append(log_prob.item())
        world_model_features.append(wm_features.cpu().numpy())
        
        if done:
            break
        
        # Update sequence for next step
        cropped_obs = preprocess_image(obs)
        next_latent = encode_image(vae_model, cropped_obs, device)
        
        # Shift sequences
        states_sequence = np.roll(states_sequence, -1, axis=0)
        states_sequence[-1] = next_latent.flatten()
        
        actions_sequence = np.roll(actions_sequence, -1, axis=0)
        actions_sequence[-1] = action.item()
    
    env.close()
    
    return {
        'states': np.array(states),
        'actions': np.array(actions),
        'rewards': np.array(rewards),
        'log_probs': np.array(log_probs),
        'world_model_features': np.array(world_model_features)
    }

def collect_trajectories_real_env(controller, world_model, vae_model, num_trajectories, 
                                 max_steps=500, seq_len=32, device='cuda'):
    """
    Collect trajectories from real environment using current controller policy
    """
    trajectories = []
    successful_trajectories = 0
    
    for traj_idx in range(num_trajectories):
        trajectory = rollout_in_real_env(controller, world_model, vae_model, 
                                       max_steps=max_steps, seq_len=seq_len, device=device)
        
        if trajectory is not None:
            trajectories.append(trajectory)
            successful_trajectories += 1
        
        if (traj_idx + 1) % 100 == 0:
            print(f"Collected {traj_idx + 1} trajectories ({successful_trajectories} successful)")
    
    print(f"Successfully collected {successful_trajectories}/{num_trajectories} trajectories")
    return trajectories

def create_validation_video(controller, world_model, vae_model, epoch, max_steps=500, 
                           seq_len=32, device='cuda'):
    """
    Create a validation video showing controller performance
    """
    env = gym.make("CarRacing-v3", render_mode="rgb_array", continuous=False)
    
    frames = []
    total_reward = 0
    
    obs, _ = env.reset()
    
    # Skip initial zoom-in phase
    for _ in range(50):
        obs, _, terminated, truncated, _ = env.step(0)
        if terminated or truncated:
            env.close()
            return None
        frames.append(obs.copy())
    
    # Build initial sequence by taking random actions
    states_sequence = []
    actions_sequence = []
    
    for _ in range(seq_len):
        action = sample_action(env)
        cropped_obs = preprocess_image(obs)
        latent = encode_image(vae_model, cropped_obs, device)
        
        states_sequence.append(latent.flatten())
        actions_sequence.append(action)
        
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        frames.append(obs.copy())
        
        if terminated or truncated:
            break
    
    if len(states_sequence) < seq_len or terminated or truncated:
        env.close()
        return None
    
    states_sequence = np.array(states_sequence)
    actions_sequence = np.array(actions_sequence)
    
    # Main rollout using trained controller
    step_count = len(frames)
    done = False
    
    while not done and step_count < max_steps:
        # Get world model features from current sequence
        wm_features = get_world_model_features(world_model, states_sequence, 
                                             actions_sequence, device)
        
        # Controller predicts action from world model features
        with torch.no_grad():
            action = controller.sample_action(wm_features.unsqueeze(0), temperature=0.5)
        
        # Step in real environment
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        frames.append(obs.copy())
        
        if not done:
            # Update sequence for next step
            cropped_obs = preprocess_image(obs)
            next_latent = encode_image(vae_model, cropped_obs, device)
            
            # Shift sequences
            states_sequence = np.roll(states_sequence, -1, axis=0)
            states_sequence[-1] = next_latent.flatten()
            
            actions_sequence = np.roll(actions_sequence, -1, axis=0)
            actions_sequence[-1] = action
        
        step_count += 1
    
    env.close()
    
    # Create video from frames
    if len(frames) > 0:
        video_filename = f'controller_validation_epoch_{epoch}.mp4'
        create_video_from_frames(frames, video_filename, fps=30)
        
        return {
            'video_filename': video_filename,
            'total_reward': total_reward,
            'num_steps': len(frames),
            'completed': done
        }
    
    return None

def create_video_from_frames(frames, filename, fps=30):
    """
    Create MP4 video from a list of frames
    """
    if len(frames) == 0:
        return
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    for frame in frames:
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr)
    
    video_writer.release()
    print(f"Validation video saved: {filename}")

def run_validation_episodes(controller, world_model, vae_model, num_episodes=3, 
                           max_steps=500, seq_len=32, device='cuda'):
    """
    Run multiple validation episodes and return statistics
    """
    episode_rewards = []
    episode_steps = []
    successful_episodes = 0
    
    for _ in range(num_episodes):
        env = gym.make("CarRacing-v3", render_mode="rgb_array", continuous=False)
        
        obs, _ = env.reset()
        
        # Skip initial zoom-in phase
        for _ in range(50):
            obs, _, terminated, truncated, _ = env.step(0)
            if terminated or truncated:
                break
        
        if terminated or truncated:
            env.close()
            continue
        
        # Build initial sequence
        states_sequence = []
        actions_sequence = []
        
        for _ in range(seq_len):
            action = sample_action(env)
            cropped_obs = preprocess_image(obs)
            latent = encode_image(vae_model, cropped_obs, device)
            
            states_sequence.append(latent.flatten())
            actions_sequence.append(action)
            
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        
        if len(states_sequence) < seq_len:
            env.close()
            continue
        
        states_sequence = np.array(states_sequence)
        actions_sequence = np.array(actions_sequence)
        
        total_reward = 0
        done = terminated or truncated
        steps = 0
        
        while not done and steps < max_steps:
            # Get world model features
            wm_features = get_world_model_features(world_model, states_sequence, 
                                                 actions_sequence, device)
            
            with torch.no_grad():
                action = controller.sample_action(wm_features.unsqueeze(0), temperature=0.5)
            
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            
            if not done:
                # Update sequences
                cropped_obs = preprocess_image(obs)
                next_latent = encode_image(vae_model, cropped_obs, device)
                
                states_sequence = np.roll(states_sequence, -1, axis=0)
                states_sequence[-1] = next_latent.flatten()
                
                actions_sequence = np.roll(actions_sequence, -1, axis=0)
                actions_sequence[-1] = action
        
        env.close()
        
        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        successful_episodes += 1
    
    if successful_episodes > 0:
        return {
            'avg_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'avg_steps': np.mean(episode_steps),
            'successful_episodes': successful_episodes,
            'total_episodes': num_episodes
        }
    else:
        return None

def train_controller(controller, world_model, vae_model, num_trajectories=10000, 
                    epochs=100, lr=1e-3, trajectories_per_epoch=50, device='cuda'):
    """
    Train controller using REINFORCE (policy gradient) on real environment data
    """
    optimizer = optim.Adam(controller.parameters(), lr=lr)
    best_avg_reward = float('-inf')
    
    for epoch in range(epochs):
        controller.train()
        
        # Collect trajectories from real environment using current policy
        print(f"Epoch {epoch+1}/{epochs}: Collecting {trajectories_per_epoch} trajectories...")
        trajectories = collect_trajectories_real_env(
            controller, world_model, vae_model, 
            num_trajectories=trajectories_per_epoch, device=device
        )
        
        if len(trajectories) == 0:
            print("No successful trajectories collected, skipping epoch")
            continue
        
        epoch_rewards = []
        epoch_losses = []
        
        # Collect all trajectory data for batch processing
        all_features = []
        all_actions = []
        all_returns = []
        
        for trajectory in trajectories:
            # Calculate returns (cumulative rewards)
            rewards = trajectory['rewards']
            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + 0.99 * G  # Discount factor
                returns.insert(0, G)
            
            # Store trajectory data for batch processing
            features = trajectory['world_model_features']
            actions = trajectory['actions']
            
            all_features.extend(features)
            all_actions.extend(actions)
            all_returns.extend(returns)
            
            epoch_rewards.append(np.sum(rewards))
        
        # Convert to tensors
        features_tensor = torch.FloatTensor(np.array(all_features)).to(device)
        actions_tensor = torch.LongTensor(np.array(all_actions)).to(device)
        returns_tensor = torch.FloatTensor(np.array(all_returns)).to(device)
        
        # Global baseline (mean return across all trajectories)
        baseline = torch.mean(returns_tensor)
        advantages = returns_tensor - baseline
        
        # Normalize advantages for stability (but preserve relative scale!)
        if len(advantages) > 1 and torch.std(advantages) > 1e-8:
            advantages = advantages / torch.std(advantages)
        
        # Recompute log probabilities with current policy (this is actually correct for on-policy methods)
        action_probs = controller.get_action_probs(features_tensor)
        action_dist = torch.distributions.Categorical(action_probs)
        log_probs = action_dist.log_prob(actions_tensor)
        
        # Policy gradient loss with baseline
        loss = -torch.mean(log_probs * advantages)  # Use mean instead of sum for better scaling
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(controller.parameters(), 1.0)
        optimizer.step()
        
        epoch_losses.append(loss.item())
        
        avg_reward = np.mean(epoch_rewards)
        avg_loss = np.mean(epoch_losses)
        
        print(f'Epoch {epoch+1}/{epochs}: Avg Reward: {avg_reward:.4f}, Loss: {avg_loss:.4f}')
        
        wandb.log({
            "epoch": epoch + 1,
            "avg_reward": avg_reward,
            "loss": avg_loss,
            "best_avg_reward": max(best_avg_reward, avg_reward),
            "num_successful_trajectories": len(trajectories)
        })
        
        # Run validation every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Running validation for epoch {epoch + 1}...")
            
            # Create validation video
            video_result = create_validation_video(controller, world_model, vae_model, 
                                                 epoch + 1, device=device)
            
            # Run validation episodes for statistics
            val_stats = run_validation_episodes(controller, world_model, vae_model, 
                                              num_episodes=5, device=device)
            
            if video_result is not None:
                wandb.log({
                    f"validation_video_epoch_{epoch+1}": wandb.Video(video_result['video_filename']),
                    "validation_video_reward": video_result['total_reward'],
                    "validation_video_steps": video_result['num_steps'],
                    "validation_video_completed": video_result['completed']
                })
                
                # Save video as artifact
                wandb.log_artifact(video_result['video_filename'], 
                                 name=f'validation_video_epoch_{epoch+1}', type='video')
            
            if val_stats is not None:
                wandb.log({
                    "validation_avg_reward": val_stats['avg_reward'],
                    "validation_std_reward": val_stats['std_reward'],
                    "validation_max_reward": val_stats['max_reward'],
                    "validation_min_reward": val_stats['min_reward'],
                    "validation_avg_steps": val_stats['avg_steps'],
                    "validation_success_rate": val_stats['successful_episodes'] / val_stats['total_episodes']
                })
                
                print(f"Validation: Avg Reward = {val_stats['avg_reward']:.2f} ± {val_stats['std_reward']:.2f}, "
                      f"Success Rate = {val_stats['successful_episodes']}/{val_stats['total_episodes']}")
        
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            # Save best model
            torch.save({
                'model_state_dict': controller.state_dict(),
                'input_dim': controller.input_dim,
                'action_dim': controller.action_dim,
                'avg_reward': avg_reward,
                'epoch': epoch + 1
            }, 'controller_best.pth')
            print(f"New best model saved with reward: {avg_reward:.4f}")
    
    return controller

def test_controller_in_real_env(controller, world_model, vae_model, num_episodes=10, render=True, seq_len=32):
    """Test the trained controller in the real environment"""
    env = gym.make("CarRacing-v3", render_mode="rgb_array" if not render else "human", continuous=False)
    device = next(controller.parameters()).device
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        
        # Skip initial zoom-in
        for _ in range(50):
            obs, _, terminated, truncated, _ = env.step(0)
            if terminated or truncated:
                break
        
        if terminated or truncated:
            continue
        
        # Build initial sequence
        states_sequence = []
        actions_sequence = []
        
        for _ in range(seq_len):
            action = sample_action(env)
            cropped_obs = preprocess_image(obs)
            latent = encode_image(vae_model, cropped_obs, device)
            
            states_sequence.append(latent.flatten())
            actions_sequence.append(action)
            
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        
        if len(states_sequence) < seq_len:
            continue
        
        states_sequence = np.array(states_sequence)
        actions_sequence = np.array(actions_sequence)
        
        total_reward = 0
        done = terminated or truncated
        
        while not done:
            # Get world model features
            wm_features = get_world_model_features(world_model, states_sequence, actions_sequence, device)
            
            with torch.no_grad():
                action = controller.sample_action(wm_features.unsqueeze(0), temperature=0.8)
            
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            if not done:
                # Update sequences
                cropped_obs = preprocess_image(obs)
                next_latent = encode_image(vae_model, cropped_obs, device)
                
                states_sequence = np.roll(states_sequence, -1, axis=0)
                states_sequence[-1] = next_latent.flatten()
                
                actions_sequence = np.roll(actions_sequence, -1, axis=0)
                actions_sequence[-1] = action
        
        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1}: Reward = {total_reward:.2f}")
    
    env.close()
    
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    
    print(f"\nTest Results:")
    print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Best Episode: {max(episode_rewards):.2f}")
    print(f"Worst Episode: {min(episode_rewards):.2f}")
    
    return episode_rewards

def save_controller(controller, filepath):
    """Save the trained controller"""
    torch.save({
        'model_state_dict': controller.state_dict(),
        'input_dim': controller.input_dim,
        'action_dim': controller.action_dim,
        'model_class': 'Controller'
    }, filepath)
    print(f"Controller saved to {filepath}")

def load_controller(filepath, device=None):
    """Load a trained controller"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(filepath, map_location=device)
    
    controller = Controller(
        input_dim=checkpoint['input_dim'],
        action_dim=checkpoint['action_dim']
    )
    
    controller.load_state_dict(checkpoint['model_state_dict'])
    controller.to(device)
    
    print(f"Controller loaded from {filepath}")
    return controller

if __name__ == "__main__":
    wandb.init(
        project="car-racing-controller",
        config={
            "latent_dim": 128,
            "action_dim": 5,
            "num_trajectories": 10000,
            "epochs": 100,
            "learning_rate": 1e-3,
            "discount_factor": 0.99,
            "max_trajectory_length": 200
        }
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load trained models
    print("Loading VAE model...")
    vae_model = load_vae_model('vae_model.pth', latent_dim=128, device=device)
    
    print("Loading world model...")
    world_model = load_world_model('world_model_final.pth', device)
    
    # Create controller
    print("Creating controller...")
    controller = Controller(input_dim=256, action_dim=5)  # 256 = hidden_dim from world model
    controller.to(device)
    
    total_params = sum(p.numel() for p in controller.parameters())
    print(f"Controller parameters: {total_params:,}")
    
    wandb.log({"controller_parameters": total_params})
    
    # Train controller
    print("Training controller...")
    controller = train_controller(
        controller, world_model, vae_model, 
        num_trajectories=10000, epochs=100, lr=1e-3, 
        trajectories_per_epoch=20, device=device
    )
    
    # Save final controller
    print("Saving final controller...")
    save_controller(controller, 'controller_final.pth')
    
    # Test controller in real environment
    print("Testing controller in real environment...")
    test_rewards = test_controller_in_real_env(controller, world_model, vae_model, num_episodes=5, render=False)
    
    # Create final test video
    print("Creating final test video...")
    final_video_result = create_validation_video(controller, world_model, vae_model, 
                                               'final', device=device)
    
    if final_video_result is not None:
        wandb.log({
            "final_test_video": wandb.Video(final_video_result['video_filename']),
            "final_test_reward": final_video_result['total_reward'],
            "final_test_steps": final_video_result['num_steps'],
            "final_test_completed": final_video_result['completed']
        })
        
        wandb.log_artifact(final_video_result['video_filename'], 
                         name='final_test_video', type='video')
        
        print(f"Final test video: {final_video_result['video_filename']}")
        print(f"Final test reward: {final_video_result['total_reward']:.2f}")
        print(f"Final test steps: {final_video_result['num_steps']}")
        print(f"Final test completed: {final_video_result['completed']}")
    
    wandb.log({
        "test_avg_reward": np.mean(test_rewards),
        "test_std_reward": np.std(test_rewards),
        "test_max_reward": max(test_rewards),
        "test_min_reward": min(test_rewards)
    })
    
    # Save artifacts
    wandb.log_artifact('controller_final.pth', name='trained_controller', type='model')
    wandb.log_artifact('controller_best.pth', name='best_controller', type='model')
    
    wandb.finish()
    print("Controller training complete!")