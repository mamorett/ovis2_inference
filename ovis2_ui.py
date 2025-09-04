import json
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import pyperclip
from typing import Dict, Any, List, Optional
import threading
import glob
import warnings
import torch
from pathlib import Path
from transformers import GenerationConfig
from gptqmodel import GPTQModel
from dotenv import load_dotenv

# Suppress all warnings
warnings.filterwarnings('ignore')

# Try to import tkinterdnd2 for drag and drop
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    HAS_DND = True
except ImportError:
    HAS_DND = False

# Ovis2 Configuration
CONFIG = {
    'model_path': 'AIDC-AI/Ovis2-16B-GPTQ-Int4',
    'device': 'cuda:0',
    'supported_formats': ['.jpg', '.jpeg', '.png', '.bmp', '.webp'],
    'max_retries': 3,
    'max_image_size': 4096 * 4096,
    'max_file_size': 10 * 1024 * 1024,  # 10MB
    'max_partition': 9,
    'max_new_tokens': 1024
}

class Ovis2ImageAnalyzerUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Ovis2 Image Analyzer")
        self.root.geometry("1000x900")
        self.root.configure(bg='#f0f0f0')
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Initialize model variables
        self.model = None
        self.text_tokenizer = None
        self.visual_tokenizer = None
        self.model_loaded = False
        
        # Get prompts from .env
        self.prompts = self.get_prompts_from_env()
        
        self.setup_ui()
        
    def get_prompts_from_env(self):
        """Get all prompts from .env file that end with _PROMPT."""
        try:
            load_dotenv()
            prompts = {}
            for key, value in os.environ.items():
                if key.endswith('_PROMPT'):
                    prompt_name = key.replace('_PROMPT', '').lower().replace('_', '-')
                    prompts[prompt_name] = value
            return prompts
        except:
            return {}
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Ovis2 Image Analyzer", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, pady=(0, 20))
        
        # Model status and load button
        model_frame = ttk.LabelFrame(main_frame, text="Model Status", padding="10")
        model_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        model_frame.columnconfigure(1, weight=1)
        
        self.model_status_var = tk.StringVar()
        self.model_status_var.set("Model not loaded")
        
        ttk.Label(model_frame, text="Status:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.model_status_label = ttk.Label(model_frame, textvariable=self.model_status_var, 
                                           foreground='red', font=('Arial', 9))
        self.model_status_label.grid(row=0, column=1, sticky=tk.W, padx=(0, 10))
        
        self.load_model_btn = ttk.Button(model_frame, text="Load Model", command=self.load_model_thread)
        self.load_model_btn.grid(row=0, column=2)
        
        # Prompt configuration section
        prompt_frame = ttk.LabelFrame(main_frame, text="Prompt Configuration", padding="10")
        prompt_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        prompt_frame.columnconfigure(1, weight=1)
        
        # Preset prompts dropdown
        ttk.Label(prompt_frame, text="Preset:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        
        self.preset_var = tk.StringVar()
        preset_values = ["Custom"] + list(self.prompts.keys())
        self.preset_combo = ttk.Combobox(prompt_frame, textvariable=self.preset_var, 
                                        values=preset_values, state="readonly", width=20)
        self.preset_combo.set("Custom")
        self.preset_combo.grid(row=0, column=1, sticky=tk.W, padx=(0, 10))
        self.preset_combo.bind('<<ComboboxSelected>>', self.on_preset_change)
        
        # Custom prompt entry
        ttk.Label(prompt_frame, text="Prompt:").grid(row=1, column=0, sticky=(tk.W, tk.N), padx=(0, 5), pady=(5, 0))
        
        self.prompt_var = tk.StringVar()
        self.prompt_var.set("Describe this image")
        
        self.prompt_text = tk.Text(prompt_frame, height=3, wrap=tk.WORD, font=('Arial', 10))
        self.prompt_text.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=(5, 0))
        self.prompt_text.insert(1.0, "Describe this image")
        
        # File selection section
        file_frame = ttk.LabelFrame(main_frame, text="Select Image File(s)", padding="10")
        file_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        file_frame.columnconfigure(1, weight=1)
        
        # File path display
        self.file_path_var = tk.StringVar()
        self.file_path_var.set("No file selected")
        
        ttk.Label(file_frame, text="File(s):").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.file_label = ttk.Label(file_frame, textvariable=self.file_path_var, 
                                   foreground='gray', font=('Arial', 9))
        self.file_label.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        
        # Browse buttons frame
        browse_frame = ttk.Frame(file_frame)
        browse_frame.grid(row=0, column=2, padx=(5, 0))
        
        self.browse_file_btn = ttk.Button(browse_frame, text="Browse File", command=self.browse_file)
        self.browse_file_btn.grid(row=0, column=0, padx=(0, 2))
        
        self.browse_folder_btn = ttk.Button(browse_frame, text="Browse Folder", command=self.browse_folder)
        self.browse_folder_btn.grid(row=0, column=1)
        
        # Drop zone
        drop_text = "Drag & Drop image file(s) or folder here" if HAS_DND else "Click here to select image file(s)"
        if not HAS_DND:
            drop_text += "\n(Install tkinterdnd2 for drag & drop: pip install tkinterdnd2)"
        
        self.drop_frame = tk.Frame(file_frame, bg='#e8e8e8', relief='ridge', bd=2, height=100)
        self.drop_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        self.drop_frame.grid_propagate(False)
        
        self.drop_label = tk.Label(self.drop_frame, text=drop_text, 
                                  bg='#e8e8e8', fg='gray', font=('Arial', 10),
                                  justify='center')
        self.drop_label.place(relx=0.5, rely=0.5, anchor='center')
        
        # Setup drag and drop if available
        if HAS_DND:
            self.drop_frame.drop_target_register(DND_FILES)
            self.drop_frame.dnd_bind('<<Drop>>', self.on_drop)
            self.drop_frame.dnd_bind('<<DragEnter>>', self.on_drag_enter)
            self.drop_frame.dnd_bind('<<DragLeave>>', self.on_drag_leave)
        else:
            # Fallback to click
            self.drop_frame.bind('<Button-1>', self.browse_file)
            self.drop_label.bind('<Button-1>', self.browse_file)
        
        # Results section
        results_frame = ttk.LabelFrame(main_frame, text="Analysis Results", padding="10")
        results_frame.grid(row=4, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Text area for results
        self.results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD, 
                                                     height=15, font=('Arial', 10))
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Buttons frame
        buttons_frame = ttk.Frame(results_frame)
        buttons_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        buttons_frame.columnconfigure(0, weight=1)
        
        # Status and buttons
        self.status_var = tk.StringVar()
        status_text = "Ready - Load model and select image file(s)" if HAS_DND else "Ready - Load model and select image file(s)"
        self.status_var.set(status_text)
        
        self.status_label = ttk.Label(buttons_frame, textvariable=self.status_var, 
                                     foreground='gray', font=('Arial', 9))
        self.status_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        # Button frame
        btn_frame = ttk.Frame(buttons_frame)
        btn_frame.grid(row=1, column=0, sticky=tk.E)
        
        self.analyze_btn = ttk.Button(btn_frame, text="Analyze Images", 
                                     command=self.analyze_images, state='disabled')
        self.analyze_btn.grid(row=0, column=0, padx=(0, 5))
        
        self.copy_btn = ttk.Button(btn_frame, text="Copy Results", 
                                  command=self.copy_to_clipboard, state='disabled')
        self.copy_btn.grid(row=0, column=1, padx=(0, 5))
        
        self.save_btn = ttk.Button(btn_frame, text="Save to File", 
                                  command=self.save_to_file, state='disabled')
        self.save_btn.grid(row=0, column=2, padx=(0, 5))
        
        self.clear_btn = ttk.Button(btn_frame, text="Clear", command=self.clear_results)
        self.clear_btn.grid(row=0, column=3)
        
        # Progress bar (hidden by default)
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        self.progress.grid_remove()  # Hide initially
        
        # Store current files
        self.current_files = []
        self.current_results = []
    
    def on_preset_change(self, event=None):
        """Handle preset prompt selection"""
        selected = self.preset_var.get()
        if selected != "Custom" and selected in self.prompts:
            self.prompt_text.delete(1.0, tk.END)
            self.prompt_text.insert(1.0, self.prompts[selected])
    
    def validate_image(self, image_path):
        """Validate image file"""
        try:
            with Image.open(image_path) as img:
                if img.size[0] * img.size[1] > CONFIG['max_image_size']:
                    raise ValueError("Image dimensions too large")
                if os.path.getsize(image_path) > CONFIG['max_file_size']:
                    raise ValueError("File size too large")
            return True
        except Exception as e:
            raise ValueError(f"Invalid image file: {str(e)}")
    
    def load_model_and_tokenizer(self):
        """Load the Ovis2 model and tokenizers"""
        # Set device
        torch.cuda.set_device(CONFIG['device'])
        
        # Load model exactly as in the original example
        model = GPTQModel.load(CONFIG['model_path'], device=CONFIG['device'], trust_remote_code=True)
        
        # Set generation config
        try:
            generation_config = GenerationConfig.from_pretrained(CONFIG['model_path'])
            model.generation_config = generation_config
            if hasattr(model, 'model') and model.model is not None:
                model.model.generation_config = generation_config
        except:
            pass  # Continue without generation config if it fails
        
        text_tokenizer = model.get_text_tokenizer()
        visual_tokenizer = model.get_visual_tokenizer()
        
        return model, text_tokenizer, visual_tokenizer
    
    def load_model_thread(self):
        """Load model in separate thread"""
        if self.model_loaded:
            return
            
        self.model_status_var.set("Loading model...")
        self.model_status_label.configure(foreground='orange')
        self.load_model_btn.configure(state='disabled')
        
        def load_model():
            try:
                self.model, self.text_tokenizer, self.visual_tokenizer = self.load_model_and_tokenizer()
                self.root.after(0, self.on_model_loaded)
            except Exception as e:
                self.root.after(0, self.on_model_error, str(e))
        
        thread = threading.Thread(target=load_model)
        thread.daemon = True
        thread.start()
    
    def on_model_loaded(self):
        """Handle successful model loading"""
        self.model_loaded = True
        self.model_status_var.set("Model loaded successfully")
        self.model_status_label.configure(foreground='green')
        self.load_model_btn.configure(text="Model Loaded", state='disabled')
        self.analyze_btn.configure(state='normal')
        self.status_var.set("Ready - Select image file(s) to analyze")
    
    def on_model_error(self, error_message):
        """Handle model loading error"""
        self.model_status_var.set(f"Error loading model: {error_message}")
        self.model_status_label.configure(foreground='red')
        self.load_model_btn.configure(state='normal')
        messagebox.showerror("Model Error", f"Failed to load model:\n{error_message}")
    
    def on_drop(self, event):
        """Handle file drop event"""
        files = self.root.tk.splitlist(event.data)
        if files:
            self.load_files(files)
        
        # Reset drop zone appearance
        self.drop_frame.configure(bg='#e8e8e8')
        self.drop_label.configure(bg='#e8e8e8', fg='gray')
    
    def on_drag_enter(self, event):
        """Handle drag enter event"""
        self.drop_frame.configure(bg='#d0f0d0')  # Light green
        self.drop_label.configure(bg='#d0f0d0', fg='#006600', text="Drop image file(s) or folder here!")
    
    def on_drag_leave(self, event):
        """Handle drag leave event"""
        self.drop_frame.configure(bg='#e8e8e8')
        self.drop_label.configure(bg='#e8e8e8', fg='gray', text="Drag & Drop image file(s) or folder here")
        
    def browse_file(self, event=None):
        """Open file dialog to select image file(s)"""
        file_paths = filedialog.askopenfilenames(
            title="Select Image File(s)",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.webp"), ("All files", "*.*")]
        )
        
        if file_paths:
            self.load_files(list(file_paths))
    
    def browse_folder(self):
        """Open folder dialog to select directory with image files"""
        folder_path = filedialog.askdirectory(title="Select Folder with Image Files")
        
        if folder_path:
            # Find all image files in the folder
            image_files = []
            for ext in CONFIG['supported_formats']:
                image_files.extend(glob.glob(os.path.join(folder_path, f"**/*{ext}"), recursive=True))
                image_files.extend(glob.glob(os.path.join(folder_path, f"**/*{ext.upper()}"), recursive=True))
            
            if image_files:
                self.load_files(image_files)
            else:
                messagebox.showinfo("No Files", "No image files found in the selected folder.")
    
    def load_files(self, file_paths):
        """Load and validate the selected files"""
        # Filter for image files and existing files
        valid_files = []
        for file_path in file_paths:
            if os.path.isdir(file_path):
                # If it's a directory, find image files in it
                for ext in CONFIG['supported_formats']:
                    valid_files.extend(glob.glob(os.path.join(file_path, f"**/*{ext}"), recursive=True))
                    valid_files.extend(glob.glob(os.path.join(file_path, f"**/*{ext.upper()}"), recursive=True))
            elif os.path.exists(file_path):
                file_ext = os.path.splitext(file_path)[1].lower()
                if file_ext in CONFIG['supported_formats']:
                    valid_files.append(file_path)
        
        if not valid_files:
            messagebox.showwarning("Warning", "No valid image files found")
            return
        
        # Store files
        self.current_files = valid_files
        
        # Update UI
        if len(valid_files) == 1:
            self.file_path_var.set(os.path.basename(valid_files[0]))
        else:
            self.file_path_var.set(f"{len(valid_files)} image files selected")
        
        self.status_var.set(f"Ready - {len(valid_files)} image(s) selected")
    
    def analyze_images(self):
        """Analyze the selected images"""
        if not self.model_loaded:
            messagebox.showwarning("Warning", "Please load the model first")
            return
        
        if not self.current_files:
            messagebox.showwarning("Warning", "Please select image files first")
            return
        
        # Get current prompt
        current_prompt = self.prompt_text.get(1.0, tk.END).strip()
        if not current_prompt:
            messagebox.showwarning("Warning", "Please enter a prompt")
            return
        
        self.status_var.set("Analyzing images...")
        self.progress.grid()
        self.progress.start()
        
        # Disable buttons during processing
        self.analyze_btn.configure(state='disabled')
        self.browse_file_btn.configure(state='disabled')
        self.browse_folder_btn.configure(state='disabled')
        
        # Process files in separate thread
        thread = threading.Thread(target=self.analyze_images_thread, args=(self.current_files, current_prompt))
        thread.daemon = True
        thread.start()
    
    def analyze_images_thread(self, file_paths, prompt):
        """Analyze images in separate thread"""
        try:
            results = []
            for file_path in file_paths:
                try:
                    result = self.process_single_image(file_path, prompt)
                    results.append(result)
                except Exception as e:
                    results.append({
                        'file_path': file_path,
                        'success': False,
                        'error': str(e),
                        'response': None
                    })
            
            # Update UI in main thread
            self.root.after(0, self.update_results, results)
        except Exception as e:
            self.root.after(0, self.show_error, str(e))
    
    def process_single_image(self, image_path, prompt):
        """Process a single image with the model"""
        try:
            self.validate_image(image_path)
            
            for attempt in range(CONFIG['max_retries']):
                try:
                    # Load and process image
                    images = [Image.open(image_path)]
                    query = f'<image>\n{prompt}'
                    
                    # Preprocess inputs
                    prompt_formatted, input_ids, pixel_values = self.model.preprocess_inputs(
                        query, images, max_partition=CONFIG['max_partition'])
                    attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
                    input_ids = input_ids.unsqueeze(0).to(device=self.model.device)
                    attention_mask = attention_mask.unsqueeze(0).to(device=self.model.device)
                    if pixel_values is not None:
                        pixel_values = pixel_values.to(dtype=self.visual_tokenizer.dtype, 
                                                     device=self.visual_tokenizer.device)
                    pixel_values = [pixel_values]
                    
                    # Generate with simplified parameters
                    with torch.inference_mode():
                        gen_kwargs = {
                            'max_new_tokens': CONFIG['max_new_tokens'],
                            'do_sample': False,
                            'pad_token_id': self.text_tokenizer.pad_token_id,
                            'use_cache': True
                        }
                        
                        output_ids = self.model.generate(input_ids, pixel_values=pixel_values, 
                                                       attention_mask=attention_mask, **gen_kwargs)[0]
                        answer = self.text_tokenizer.decode(output_ids, skip_special_tokens=True)
                    
                    return {
                        'file_path': image_path,
                        'success': True,
                        'error': None,
                        'response': answer
                    }
                        
                except Exception as e:
                    if attempt == CONFIG['max_retries'] - 1:
                        raise
                    continue
                    
        except Exception as e:
            return {
                'file_path': image_path,
                'success': False,
                'error': str(e),
                'response': None
            }
    
    def update_results(self, results):
        """Update UI with analysis results"""
        # Stop progress bar
        self.progress.stop()
        self.progress.grid_remove()
        
        # Re-enable buttons
        self.analyze_btn.configure(state='normal')
        self.browse_file_btn.configure(state='normal')
        self.browse_folder_btn.configure(state='normal')
        
        # Store results
        self.current_results = results
        
        # Clear previous content
        self.results_text.delete(1.0, tk.END)
        
        successful = 0
        all_responses = []
        
        # Process results
        for i, result in enumerate(results):
            file_path = result['file_path']
            filename = os.path.basename(file_path)
            
            if len(results) > 1:
                self.results_text.insert(tk.END, f"=== {filename} ===\n")
            
            if result['success']:
                successful += 1
                response = result['response']
                self.results_text.insert(tk.END, f"{response}\n")
                all_responses.append(f"{response}")
            else:
                error_msg = result['error']
                self.results_text.insert(tk.END, f"Error: {error_msg}\n")
                all_responses.append(f"{filename}:\nError: {error_msg}")
            
            if i < len(results) - 1:
                self.results_text.insert(tk.END, "\n" + "="*60 + "\n\n")
        
        # Update status and enable buttons
        self.status_var.set(f"✓ Analyzed {successful}/{len(results)} images successfully")
        
        if successful > 0:
            self.copy_btn.configure(state='normal')
            self.save_btn.configure(state='normal')
            self.all_responses = all_responses
        else:
            self.all_responses = []
    
    def show_error(self, error_message):
        """Show error message"""
        self.progress.stop()
        self.progress.grid_remove()
        self.analyze_btn.configure(state='normal')
        self.browse_file_btn.configure(state='normal')
        self.browse_folder_btn.configure(state='normal')
        
        self.status_var.set(f"✗ Error: {error_message}")
        messagebox.showerror("Error", f"Failed to process images:\n{error_message}")
    
    def copy_to_clipboard(self):
        """Copy all results to clipboard"""
        if hasattr(self, 'all_responses') and self.all_responses:
            try:
                all_text = '\n\n'.join(self.all_responses)
                pyperclip.copy(all_text)
                self.status_var.set(f"✓ Results copied to clipboard!")
                
                # Reset status after 3 seconds
                self.root.after(3000, lambda: self.status_var.set("Ready"))
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to copy to clipboard:\n{e}")
    
    def save_to_file(self):
        """Save results to text file"""
        if hasattr(self, 'all_responses') and self.all_responses:
            # Default filename
            if len(self.current_files) == 1:
                base_name = os.path.splitext(os.path.basename(self.current_files[0]))[0]
                default_name = f"{base_name}_analysis.txt"
            else:
                default_name = "image_analysis_results.txt"
            
            file_path = filedialog.asksaveasfilename(
                title="Save Analysis Results",
                defaultextension=".txt",
                initialfile=default_name,  # Fixed: changed from initialfilename
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            
            if file_path:
                try:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write("Ovis2 Image Analysis Results\n")
                        f.write("=" * 30 + "\n\n")
                        f.write('\n\n'.join(self.all_responses))
                    
                    self.status_var.set(f"✓ Results saved to {os.path.basename(file_path)}")
                    
                    # Reset status after 3 seconds
                    self.root.after(3000, lambda: self.status_var.set("Ready"))
                    
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to save file:\n{e}")

    
    def clear_results(self):
        """Clear all results"""
        self.results_text.delete(1.0, tk.END)
        self.file_path_var.set("No file selected")
        self.status_var.set("Ready - Select image file(s) to analyze")
        self.copy_btn.configure(state='disabled')
        self.save_btn.configure(state='disabled')
        self.current_files = []
        self.current_results = []
        if hasattr(self, 'all_responses'):
            self.all_responses = []

def main():
    # Check if required packages are available
    missing_packages = []
    
    required_packages = ['pyperclip', 'Pillow', 'torch', 'transformers', 'gptqmodel']
    
    for package in required_packages:
        try:
            if package == 'Pillow':
                from PIL import Image
            elif package == 'pyperclip':
                import pyperclip
            elif package == 'torch':
                import torch
            elif package == 'transformers':
                import transformers
            elif package == 'gptqmodel':
                import gptqmodel
        except ImportError:
            missing_packages.append(package)
    
    # Check for drag and drop support
    if not HAS_DND:
        print("Note: For drag & drop functionality, install tkinterdnd2:")
        print("pip install tkinterdnd2")
    
    if missing_packages:
        print("Missing required packages:", ', '.join(missing_packages))
        print("Please install them using: pip install", ' '.join(missing_packages))
        return
    
    # Use TkinterDnD root if available, otherwise regular Tk
    if HAS_DND:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()
    
    app = Ovis2ImageAnalyzerUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
