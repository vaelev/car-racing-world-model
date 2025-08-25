import torch
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import cv2
from transformer_world_model import TransformerWorldModel, sample_from_mog, create_prediction_video
from vae import load_model, encode_image, decode_latent, sample_action, preprocess_image
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygame")

def load_world_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
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
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"Loaded world model from {checkpoint_path}")
    print(f"Config: {config}")
    
    return model, device

def collect_context_sequence(vae_model, device, seq_len=128, warmup_steps=200):
    env = gym.make("CarRacing-v3", render_mode="rgb_array", continuous=False)
    
    obs, _ = env.reset()
    
    for _ in range(50):
        obs, _, terminated, truncated, _ = env.step(0)
        if terminated or truncated:
            break
    
    for _ in range(warmup_steps):
        action = sample_action(env)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
            for _ in range(50):
                obs, _, terminated, truncated, _ = env.step(0)
                if terminated or truncated:
                    break
    
    states = []
    actions = []
    real_images = []
    
    for _ in range(seq_len):
        action = sample_action(env)
        cropped_obs = preprocess_image(obs)
        latent = encode_image(vae_model, cropped_obs, device)
        
        states.append(latent.flatten())
        actions.append(action)
        real_images.append(cropped_obs)
        
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    
    env.close()
    
    if len(states) < seq_len:
        return None, None, None
    
    return np.array(states), np.array(actions), real_images

def predict_sequence(world_model, vae_model, device, context_states, context_actions, 
                    prediction_steps=64, temperature=1.0, action_strategy='random'):
    
    current_states = torch.FloatTensor(context_states).unsqueeze(0).to(device)
    current_actions = torch.LongTensor(context_actions).unsqueeze(0).to(device)
    
    predicted_latents = []
    predicted_actions = []
    
    with torch.no_grad():
        for step in range(prediction_steps):
            means, logvars, weights = world_model(current_states, current_actions)
            next_state = sample_from_mog(means[:, -1:], logvars[:, -1:], weights[:, -1:], temperature)
            
            predicted_latents.append(next_state.squeeze(1))
            
            if action_strategy == 'random':
                next_action = torch.randint(0, 5, (1, 1)).to(device)
            elif action_strategy == 'biased':
                action_probs = [0.1, 0.25, 0.25, 0.35, 0.05]
                next_action = torch.tensor([[np.random.choice(5, p=action_probs)]]).to(device)
            elif action_strategy == 'straight':
                next_action = torch.tensor([[3]]).to(device)  # gas
            
            predicted_actions.append(next_action.item())
            
            current_states = torch.cat([current_states[:, 1:], next_state], dim=1)
            current_actions = torch.cat([current_actions[:, 1:], next_action], dim=1)
    
    predicted_latents = torch.cat(predicted_latents, dim=0).cpu().numpy()
    decoded_images = decode_latent(vae_model, predicted_latents)
    
    return decoded_images, predicted_actions

def run_test_predictions():
    print("Loading models...")
    vae_model = load_model('vae_model.pth', latent_dim=128)
    world_model, device = load_world_model('world_model_final.pth')
    
    test_configs = [
        {"temp": 0.5, "action": "random", "name": "conservative_random"},
        {"temp": 1.0, "action": "random", "name": "normal_random"},  
        {"temp": 1.5, "action": "random", "name": "exploratory_random"},
        {"temp": 0.8, "action": "biased", "name": "conservative_biased"},
        {"temp": 1.2, "action": "biased", "name": "exploratory_biased"},
        {"temp": 1.0, "action": "straight", "name": "normal_straight"},
    ]
    
    for i in range(len(test_configs)):
        print(f"\nRunning test {i+1}/{len(test_configs)}: {test_configs[i]['name']}")
        
        print("Collecting context sequence...")
        context_states, context_actions, real_images = collect_context_sequence(
            vae_model, device, seq_len=128, warmup_steps=300
        )
        
        if context_states is None:
            print("Failed to collect context sequence, skipping...")
            continue
        
        print("Generating predictions...")
        predicted_images, predicted_actions = predict_sequence(
            world_model, vae_model, device,
            context_states, context_actions,
            prediction_steps=96,
            temperature=test_configs[i]["temp"],
            action_strategy=test_configs[i]["action"]
        )
        
        print("Creating videos...")
        real_decoded = decode_latent(vae_model, context_states[-32:])
        
        combined_images = np.concatenate([real_decoded, predicted_images], axis=0)
        
        video_filename = f"test_{test_configs[i]['name']}.mp4"
        create_prediction_video(combined_images, video_filename, fps=8)
        
        print(f"Saved: {video_filename}")
        print(f"Context: {len(real_decoded)} frames, Predictions: {len(predicted_images)} frames")
        print(f"Actions used: {predicted_actions[:10]}...")  # Show first 10 actions
    
    print("\nAll test predictions completed!")

def interactive_prediction():
    print("Loading models...")
    vae_model = load_model('vae_model.pth', latent_dim=128)
    world_model, device = load_world_model('world_model_final.pth')
    
    while True:
        print("\n=== Interactive World Model Testing ===")
        print("1. Generate prediction with custom settings")
        print("2. Quick test with default settings")
        print("3. Exit")
        
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == "3":
            break
        elif choice == "1":
            try:
                temp = float(input("Temperature (0.5-2.0): "))
                steps = int(input("Prediction steps (32-128): "))
                action_strategy = input("Action strategy (random/biased/straight): ")
                warmup = int(input("Warmup steps (100-500): "))
            except ValueError:
                print("Invalid input, using defaults...")
                temp, steps, action_strategy, warmup = 1.0, 64, "random", 200
        else:
            temp, steps, action_strategy, warmup = 1.0, 64, "random", 200
        
        print("Collecting context...")
        context_states, context_actions, _ = collect_context_sequence(
            vae_model, device, seq_len=96, warmup_steps=warmup
        )
        
        if context_states is None:
            print("Failed to collect context, trying again...")
            continue
        
        print("Predicting...")
        predicted_images, _ = predict_sequence(
            world_model, vae_model, device,
            context_states, context_actions,
            prediction_steps=steps,
            temperature=temp,
            action_strategy=action_strategy
        )
        
        video_name = f"interactive_temp{temp}_steps{steps}_{action_strategy}.mp4"
        create_prediction_video(predicted_images, video_name, fps=10)
        
        print(f"Created: {video_name}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_prediction()
    else:
        run_test_predictions()