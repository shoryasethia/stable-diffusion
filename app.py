import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import json
from tqdm import tqdm
import os

def get_models():
    # Load models from a JSON file
    with open('models.json', 'r') as ls:
        models = json.load(ls)
    return models

def generate_image(prompt: str, model_id: str, output_path: str):
    # Initialize the pipeline with the specified model
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    )
    
    # Move to GPU if available
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    
    # Disable safety checker 
    pipe.safety_checker = None
    
    # Set seed for reproducibility
    generator = torch.manual_seed(42) 
    
    # Generate image
    image = pipe(
        prompt=prompt,
        height=768,  
        width=768,
        num_inference_steps=60,  
        guidance_scale=9,  
        generator=generator
    ).images[0]
    
    # Save the image as JPG
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path, format="JPEG")
    print(f"Image saved to {output_path} using {model_id}")

if __name__ == "__main__":
    # Get prompt from user
    user_prompt = input("Enter your prompt: ")
    
    # Load model list from JSON
    models = get_models()
    
    # Generate and save image for each model
    for model in tqdm(models, desc="Generating images"):
        try:
            print(f"Generating image with {model}")
            output_file = f"output/{model.split('/')[1]}.jpg"  # Ensure file extension
            try:
                generate_image(user_prompt, model_id=model, output_path=f"output/{model.split('/')[1]}.jpg")
            except Exception as e:
                print(f"Error generating image with {model}: {e}")
        except Exception as e:
            print(f"Error generating image with {model}: {e}")
