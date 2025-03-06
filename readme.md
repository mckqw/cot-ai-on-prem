# Cost-Effective On-Premises AI Solution

This project implements a cost-effective, on-premises AI solution using Chain of Thought (CoT) Distilled Large Language Models (LLMs). It combines:

*   A **FastAPI server** that serves the DeepSeek-R1-Distill-Qwen-32B model with a Retrieval Augmented Generation (RAG) system using Disneyland Parks documentation.
*   A **Text-to-Speech (TTS) client** that queries the server and speaks the responses aloud.

Both functionalities are bundled into a single Python script, main.py, which can be run in either server or TTS mode based on command-line arguments.

## Requirements

*   **Python**: 3.8
*   **Operating System**: Ubuntu 20.04 LTS
*   **Hardware**: NVIDIA GPU with CUDA support (e.g., RTX 3090 with 24GB VRAM)
*   **Documentation File**: disneyland\_doc.txt containing Disneyland Parks documentation

## Installation

Follow these steps to set up the project:

1.  **Clone the Repository**
    
    `git clone https://github.com/your-repo-url.git cd your-repo-directory`
    
2.  **Install Python 3.8 and Dependencies**
    
    Ensure Python 3.8 is installed. Then, install the required libraries:
    
    `pip install -r requirements.txt`
    
3.  **Install PyTorch with CUDA Support**
    
    Install PyTorch compatible with your CUDA version. For example, for CUDA 11.1:
    
    `pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html`
    
4.  **Download and Install the DeepSeek-R1-Distill-Qwen-32B Model**
    
    *   Obtain the model from the [DeepSeek-R1 GitHub repository](https://github.com/deepseek-ai/deepseek-r1).
    *   Place it in a directory and update the model\_name variable in main.py (around line 70) to point to the correct path.
    *   `pip install "huggingface_hub[hf_transfer]"`
    *   `HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`
5.  **Prepare Documentation**
    
    Ensure disneyland\_doc.txt is in the project directory with the relevant Disneyland Parks documentation.
    

## Usage

The script can be run in two modes:

*   **Server Mode**: Starts the FastAPI server to handle queries.
*   **TTS Mode**: Runs the TTS client to interact with the server.

### Run the Server

`python main.py server`

This launches the FastAPI server on http://0.0.0.0:8000.

### Run the TTS Client

In a separate terminal (after starting the server):

`python main.py tts`

*   The client will prompt you for queries.
*   Enter a query, and the response will be printed and spoken aloud.
*   Type quit to exit.

## Notes

*   **Server Availability**: Ensure the server is running (python main.py server) before starting the TTS client.
*   **Server URL**: The TTS client assumes the server is at http://localhost:8000. Update the query\_server function in main.py if the server runs on a different address.
*   **Model Path**: Replace "path/to/deepseek-r1-distill-qwen-32b" in main.py with the actual model directory.
*   **Performance**: The Chroma database is recreated each time the server starts. For persistence, consider modifying the script to save and load the database.
  

## Installation Guide with iOS App Setup

This section help you set up a simple iOS app that requires the FastAPI server to be running. Follow the steps below to get everything running, including a new subsection for integrating the iOS Swift code.

### Requirements

*   **Python**: 3.8
*   **Operating System**: Ubuntu 20.04 LTS
*   **Hardware**: NVIDIA GPU with CUDA support (e.g., RTX 3090 with 24GB VRAM)
*   **Documentation File**: disneyland\_doc.txt containing Disneyland Parks documentation
*   **For iOS App**: Xcode 14+, iOS 16+, Swift 5

## iOS App Setup

This subsection guides you through setting up the iOS app using Swift code to connect to the FastAPI server.

1.  **Open Xcode**
    
    *   Launch Xcode and create a new project or open an existing one.
    *   Ensure the target is set to iOS 16 or later.
2.  **Add Alamofire Dependency**
    
    *   In Xcode, go to **File** > **Add Packages**.
    *   Enter the Alamofire URL: https://github.com/Alamofire/Alamofire.git.
    *   Choose a version (e.g., 5.6.0) and add it to your project.
3.  **Set Up the User Interface**
    
    *   In your storyboard, add:
        *   A UITextField for query input.
        *   A UIButton to submit the query.
        *   A UITextView to show the response.
        *   A UIActivityIndicatorView for loading feedback.
    *   Connect these to the outlets in the code (queryTextField, responseTextView, activityIndicator).
    *   Link the button’s action to submitQuery.
4.  **Update Server IP**
    
    *   In the sendQueryToServer function, replace "http://server-ip:8000/query" with your FastAPI server’s actual IP address.
5.  **Run the App**
    
    *   Build and run the app in a simulator or on a device.
    *   Ensure the server is running and accessible.


## Evaluation Script
This script provides a comprehensive, self-contained solution to evaluate the main script's performance.

1.  **Prepare Test Data**:
    *   Update the test\_data list with your own queries and ground truth answers relevant to the evaluation context (e.g., Disneyland-related queries).
    *   Ensure you have at least 10 queries if you want to match the TTS subset size, or adjust tts\_subset = results\[:10\] accordingly.
2.  **Run the Server**:
    *   Start the FastAPI server at http://localhost:8000/query before running the script.
    *   The server should accept POST requests with a JSON body (e.g., {"text": "query"}) and return a response like {"answer": "response text"}.
3.  **Execute the Script**:
    *   Run the script in a terminal:
        
        bash
        
        CollapseWrapCopy
        
        `python script_name.py`
        
    *   Ensure all required libraries are installed and nvidia-smi is available if GPU monitoring is needed.
4.  **Provide Ratings**:
    *   **Accuracy**: For each query-response pair, three assessors will be prompted to rate the response accuracy (1-5).
    *   **TTS Quality**: For each selected response, five participants will rate the TTS audio quality (1-5).
    *   Follow the prompts and enter valid integers between 1 and 5.
5.  **View Results**:
    *   The script outputs a summary of average metrics to the console.
    *   Detailed results, including individual query data and summary statistics, are saved to evaluation\_results.json.