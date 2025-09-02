# Ovis2 Inference

This project provides a Python script for running inference with the Ovis2 model (`AIDC-AI/Ovis2-16B-GPTQ-Int4`). It can process a single image or a directory of images, generating a text description for each based on a provided prompt.

## Features

*   Process a single image file or all images in a directory.
*   Supports various image formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`.
*   Saves the generated text to a `.txt` file with the same name as the image.
*   Allows using predefined prompts from a `.env` file or a custom prompt from the command line.
*   Includes options to force reprocessing of existing files and to print output to the terminal without saving.
*   Handles large images and files with configurable limits.
*   Includes retry logic for robustness.
*   **New:** A graphical user interface (GUI) for easier interaction.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/ovis2_inference.git
    cd ovis2_inference
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.

    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

    Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```
    For the UI, you might need to install `tkinterdnd2` for drag-and-drop support:
    ```bash
    pip install tkinterdnd2
    ```

## Configuration

You can define custom prompts in a `.env` file in the project's root directory. The script will automatically load any environment variables that end with `_PROMPT`.

1.  **Create a `.env` file:**
    ```bash
    touch .env
    ```

2.  **Add your prompts to the `.env` file.** The key should end with `_PROMPT`, and the value is the prompt text. For example:

    ```env
    # .env
    GENERAL_PROMPT="Describe the image in detail."
    OCR_PROMPT="Extract all text from the image."
    ```
    The script will make these prompts available as choices for the `-t` or `--prompt-type` argument (e.g., `general`, `ocr`).

## Usage

### Command-Line

The script can be run from the command line with various options.

#### Command-Line Arguments

*   `path`: (Required) The path to the image file or directory to process.
*   `-t`, `--prompt-type`: (Required, unless `-p` is used) The type of prompt to use, as defined in your `.env` file (e.g., `general`, `ocr`).
*   `-p`, `--prompt`: (Required, unless `-t` is used) A custom prompt to use directly.
*   `-f`, `--force`: (Optional) Force processing of an image, even if an output file already exists.
*   `-n`, `--no-save`: (Optional) Do not save the output to a text file; print it to the terminal instead.
*   `-q`, `--quiet`: (Optional) Suppress all output to the terminal, except for error messages.

#### Examples

**Process a single image with a predefined prompt:**
```bash
python ovis2.py /path/to/your/image.jpg -t general
```

**Process all images in a directory with a custom prompt:**
```bash
python ovis2.py /path/to/your/directory -p "What is the main subject of this image?"
```

**Force reprocessing of an image and print the output to the terminal:**
```bash
python ovis2.py /path/to/your/image.jpg -t ocr -f -n
```

### Graphical User Interface (GUI)

The project now includes a graphical user interface for a more interactive experience.

#### Running the GUI

To run the GUI, execute the following command:
```bash
python ovis2_ui.py
```

#### GUI Features

*   **Model Loading:** Load and monitor the status of the Ovis2 model.
*   **Prompt Configuration:** Choose from predefined prompts (from the `.env` file) or write a custom prompt.
*   **File Selection:**
    *   Browse for single or multiple image files.
    *   Browse for a folder of images.
    *   Drag and drop files and folders directly onto the application window (requires `tkinterdnd2`).
*   **Image Analysis:** Analyze the selected images with the chosen prompt.
*   **Results Display:** View the generated text for each image.
*   **Actions:**
    *   Copy the results to the clipboard.
    *   Save the results to a text file.
    *   Clear the results and file selection.

## License

This project is licensed under the terms of the LICENSE file.