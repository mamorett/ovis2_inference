import warnings
import torch
from PIL import Image
from pathlib import Path
import os
from tqdm import tqdm
import argparse
from dotenv import load_dotenv
from contextlib import contextmanager
import time
from transformers import GenerationConfig
from gptqmodel import GPTQModel

# Suppress all warnings
warnings.filterwarnings('ignore')


CONFIG = {
    'model_path': 'AIDC-AI/Ovis2-16B-GPTQ-Int4',
    'device': 'cuda:0',
    'supported_formats': ['.jpg', '.jpeg', '.png', '.bmp', '.webp'],
    'max_retries': 3,
    'max_image_size': 4096 * 4096,
    'max_file_size': 10 * 1024 * 1024  # 10MB
}

@contextmanager
def manage_model_resources():
    """Context manager for handling GPU memory resources."""
    try:
        torch.cuda.empty_cache()
        yield
    finally:
        torch.cuda.empty_cache()


def validate_image(image_path):
    try:
        with Image.open(image_path) as img:
            if img.size[0] * img.size[1] > 4096 * 4096:
                raise ValueError("Image dimensions too large")
            if os.path.getsize(image_path) > 10 * 1024 * 1024:  # 10MB
                raise ValueError("File size too large")
    except Exception as e:
        raise ValueError(f"Invalid image file: {str(e)}")

def process_batch(image_files, **kwargs):
    total_size = sum(os.path.getsize(f) for f in image_files)
    processed_size = 0
    with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
        for image in image_files:
            # Process image
            processed_size += os.path.getsize(image)
            pbar.update(os.path.getsize(image))


def load_model_and_tokenizer():
    with manage_model_resources():
        # Initialize tokenizer and model directly
        model = GPTQModel.load(CONFIG['model_path'], device=CONFIG['device'], trust_remote_code=True)
        model.model.generation_config = GenerationConfig.from_pretrained(CONFIG['model_path'])
        text_tokenizer = model.get_text_tokenizer()
        visual_tokenizer = model.get_visual_tokenizer()        
        model.eval()
        return model, text_tokenizer

def get_prompts_from_env():
    """Get all prompts from .env file that end with _PROMPT."""
    load_dotenv()
    prompts = {}
    for key, value in os.environ.items():
        if key.endswith('_PROMPT'):
            prompt_name = key.replace('_PROMPT', '').lower().replace('_', '-')
            prompts[prompt_name] = value
    return prompts

def process_single_image(image_path, prompt, model, tokenizer, force=False, no_save=False, quiet=False):
    try:
        validate_image(image_path)
        
        for attempt in range(CONFIG['max_retries']):
            try:
                with manage_model_resources():
                    output_path = image_path.with_suffix('.txt')
                    
                    if not no_save and output_path.exists() and not force:
                        if not quiet:
                            print(f"\nSkipping {image_path} - output file already exists")
                        return True
                        
                    image = Image.open(image_path).convert('RGB')
                    msgs = [{'role': 'user', 'content': [image, prompt]}]
                    answer = model.chat(msgs=msgs, tokenizer=tokenizer)
                    
                    if not no_save:
                        with open(output_path, 'w', encoding='utf-8') as f:
                            f.write(answer)
                        if not quiet:
                            print(f"\nCreated output file: {output_path}")
                    
                    # Print response only if not in quiet mode or if no_save is True
                    if not quiet or no_save:
                        print(f"\nResponse for {image_path}:\n{answer}\n")
                    return True
                    
            except Exception as e:
                if attempt == CONFIG['max_retries'] - 1:
                    raise
                if not quiet:
                    print(f"\nRetry {attempt + 1}/{CONFIG['max_retries']}: {str(e)}")
                time.sleep(1)
                
    except Exception as e:
        if not quiet:
            print(f"\nError processing {image_path}: {str(e)}")
        return False


def get_prompts():
    """Get prompts from .env file and command line arguments."""
    # Get prompts from .env
    load_dotenv()
    prompts = {}
    for key, value in os.environ.items():
        if key.endswith('_PROMPT'):
            prompt_name = key.replace('_PROMPT', '').lower().replace('_', '-')
            prompts[prompt_name] = value
    return prompts    

def main():
    # Get prompts from .env
    prompts = get_prompts_from_env()
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process images with GPT model')
    parser.add_argument('path', help='Path to image file or directory')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-t', '--prompt-type', choices=list(prompts.keys()),
                       help='Type of prompt to use (defined in .env file)')
    group.add_argument('-p', '--prompt', help='Custom prompt to use')
    parser.add_argument('-f', '--force', action='store_true',
                       help='Force processing even if output file exists')
    parser.add_argument('-n', '--no-save', action='store_true',
                       help='Do not save output to text files, print to terminal only')    
    parser.add_argument('-q', '--quiet', action='store_true',
                   help='Suppress response output (only valid when saving to files)')

    args = parser.parse_args()

        # Determine which prompt to use
    if args.prompt_type:
        selected_prompt = prompts[args.prompt_type]
    else:
        selected_prompt = args.prompt
    
    # Initialize model and tokenizer
    print("Initializing model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer()
    print("Initialization complete!")
    
    # Process path
    path = Path(args.path)
    if path.is_file():
        # Single file processing
        if path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
            print(f"Processing single image: {path}")
            process_single_image(path, selected_prompt, model, tokenizer, args.force, args.no_save, args.quiet)
    elif path.is_dir():
        image_files = []
        for ext in CONFIG['supported_formats']:
            image_files.extend(path.glob(f"*{ext}"))
            image_files.extend(path.glob(f"*{ext.upper()}"))
            
        if not image_files:
            print("No image files found in directory")
            return
            
        total_size = sum(os.path.getsize(f) for f in image_files)
        processed_size = 0
        successful = 0
        
        with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
            for image_path in image_files:
                if process_single_image(image_path, selected_prompt, model, tokenizer, args.force, args.no_save, args.quiet):
                    successful += 1
                processed_size += os.path.getsize(image_path)
                pbar.update(os.path.getsize(image_path))
                
        print(f"\nProcessing complete. Successfully processed {successful} out of {len(image_files)} images.")
    else:
        print("Invalid path. Please provide a valid image file or directory path.")

if __name__ == "__main__":
    main()
