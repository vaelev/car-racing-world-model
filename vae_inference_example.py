"""
Standalone script for using a trained VAE model for inference
Run this after training your VAE model
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from vae import VAE, load_model, encode_image, decode_latent, interpolate_in_latent_space

def load_and_test_model():
    """Load trained model and demonstrate inference capabilities"""
    
    # Load the trained model
    print("Loading trained VAE model...")
    model = load_model('vae_model.pth', latent_dim=128)
    device = next(model.parameters()).device
    
    # Example 1: Random generation
    print("\n1. Generating random images from latent space...")
    random_latent = np.random.normal(0, 1, (5, 128))  # 5 random latent vectors
    generated_images = decode_latent(model, random_latent, device)
    
    # Visualize generated images
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i, img in enumerate(generated_images):
        axes[i].imshow(img)
        axes[i].set_title(f'Generated {i+1}')
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig('generated_images.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Example 2: Encoding and reconstructing a custom image
    print("\n2. Encoding and reconstructing...")
    # Create a simple test image (you could load a real image here)
    test_image = np.random.rand(86, 96, 3)  # Random test image
    
    # Encode to latent space
    latent_vector = encode_image(model, test_image, device)
    print(f"Encoded image to latent vector of shape: {latent_vector.shape}")
    
    # Decode back to image
    reconstructed = decode_latent(model, latent_vector, device)
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(test_image)
    axes[0].set_title('Original')
    axes[0].axis('off')
    axes[1].imshow(reconstructed)
    axes[1].set_title('Reconstructed')
    axes[1].axis('off')
    plt.tight_layout()
    plt.savefig('reconstruction_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Example 3: Latent space interpolation
    print("\n3. Interpolating between two random images...")
    image1 = np.random.rand(86, 96, 3)
    image2 = np.random.rand(86, 96, 3)
    
    interpolations = interpolate_in_latent_space(model, image1, image2, steps=8)
    
    # Visualize interpolation
    fig, axes = plt.subplots(1, 8, figsize=(16, 2))
    for i, img in enumerate(interpolations):
        axes[i].imshow(img)
        axes[i].set_title(f'Step {i+1}')
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig('interpolation.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nInference examples completed!")
    return model

def encode_dataset_to_latents(model, images, batch_size=64):
    """Encode a dataset of images to latent vectors efficiently"""
    device = next(model.parameters()).device
    latents = []
    
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        batch_latents = encode_image(model, batch, device)
        latents.append(batch_latents)
    
    return np.concatenate(latents, axis=0)

if __name__ == "__main__":
    model = load_and_test_model()
    
    # Optional: If you have a dataset you want to encode
    # print("\nEncoding dataset to latent space...")
    # images = your_image_dataset  # Replace with your images
    # latent_dataset = encode_dataset_to_latents(model, images)
    # np.save('latent_dataset.npy', latent_dataset)
    # print(f"Saved {len(latent_dataset)} latent vectors to latent_dataset.npy")