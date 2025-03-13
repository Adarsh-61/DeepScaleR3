# DeepScale R3: Music Recommendation System

Welcome to **DeepScale R3**, the third iteration in our DeepScale AI series. This project showcases an advanced music recommendation system that leverages artificial intelligence to analyze your emotions from a photo and suggest Hindi songs tailored to your current mood. Additionally, it tracks your mood history over time, providing insights into your emotional trends through interactive visualizations. Built with Streamlit, this application offers a user-friendly interface to enhance your music listening experience.

## Features 🌟

- **Emotion Detection**: Uses state-of-the-art AI models to detect your emotions from a photo.
- **Music Recommendations**: Suggests Hindi songs based on your detected mood, with the ability to save your favorites.
- **Mood History Tracking**: Records your moods over time and displays them with interactive charts.
- **User-Friendly Interface**: Powered by Streamlit for an intuitive and interactive experience.
- **GPU Acceleration**: Optionally leverages GPU support for faster emotion detection.

## Prerequisites 📋

Before setting up the DeepScale R3 project, ensure you have the following installed on your system:

- **Python 3.8 or higher**: Required to run the application. Download it from [python.org](https://www.python.org/downloads/).
- **pip**: The Python package manager, typically included with Python. Verify it by running `pip --version` in your terminal.
- **Git**: Needed to clone the repository. Install it from [git-scm.com](https://git-scm.com/downloads) if not already present.

You will also need a terminal (e.g., Command Prompt on Windows or Terminal on macOS/Linux) to execute the commands.

## Installation 🛠️

Follow these detailed steps to set up DeepScale R3 on your local machine.

### 1. Clone the Repository

Download the project files from GitHub by running the following commands in your terminal:

```bash
git clone https://github.com/Adarsh3315/DeepScaleR3.git
cd DeepScaleR3
```

*Explanation*:  
- `git clone` downloads the repository.
- `cd DeepScaleR3` navigates you into the project directory where all project files are located.

### 2. Create a Virtual Environment

A virtual environment isolates the project’s dependencies from your system’s global Python installation, avoiding conflicts. Create and activate one as follows:

#### On Windows:
```bash
python -m venv myenv
myenv\Scripts\activate
```

#### On macOS/Linux:
```bash
python3 -m venv myenv
source myenv/bin/activate
```

*Explanation*:  
- This creates an isolated environment called `myenv` (you may choose any name).
- Activating the environment ensures that all subsequent Python and pip commands use this isolated setup.

### 3. Install Dependencies

Make sure your virtual environment is activated (you’ll see `(myenv)` in your prompt). Then, run these commands sequentially:

#### Update pip (Recommended)
```bash
python -m pip install --upgrade pip
```

#### Install Required Packages
```bash
pip install streamlit
pip install opencv-python
pip install deepface
pip install retina-face
pip install numpy
pip install Pillow
pip install pandas
pip install plotly
pip install mtcnn
pip install tensorflow
pip install tf-keras
```

*Explanation*:  
- **streamlit** powers the web-based user interface.
- **opencv-python** manages image processing and serves as a fallback for face detection.
- **deepface** provides robust emotion detection.
- **retina-face** increases face detection accuracy.
- **numpy**, **Pillow**, **pandas**, and **plotly** support various functionalities including image manipulation, data handling, and interactive visualizations.
- **mtcnn** offers an alternative face detection method.
- **tensorflow** and **tf-keras** support machine learning model operations.

#### Install PyTorch (with Optional GPU Support)

Choose one of the following based on your system:

- **With GPU Support** (requires NVIDIA GPU and compatible CUDA installation):
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
  ```
  *Note*: Adjust the CUDA version (e.g., `cu118` for CUDA 11.8) according to your setup.

- **Without GPU Support** (CPU-only):
  ```bash
  pip install torch torchvision torchaudio
  ```

*Verification*: After installation, you can use `pip show <package-name>` (e.g., `pip show streamlit`) to confirm each package is installed correctly.

### 4. Run the Application

With all dependencies installed, launch the application by running:
```bash
streamlit run app.py
```

*Explanation*:  
- The command starts a local Streamlit server.
- Your default web browser should automatically open at `http://localhost:8501` where you can interact with the Music Recommendation System.

## Usage Guide 📚

### Home Page
- **Overview**: Provides an introduction to the app and displays its main features.
- **Navigation**: Use the sidebar to switch between pages such as Music Recommendation and Mood History.

### Music Recommendation
- **Manual Mood Selection**: Click one of the mood buttons (e.g., Happy, Sad) to set your current mood.
- **Photo Input**: Alternatively, take or upload a photo so the AI can analyze your emotion.
- **Song Suggestions**: Based on your mood (either selected or AI-detected), the system recommends Hindi songs. You can click the heart icon (😍) to save your favorite songs.

### Mood History
- **Tracking Emotions**: View your mood history with interactive charts.
- **Visual Insights**: The system uses line and pie charts to display your mood trends over time.
- **Data Privacy**: All data is processed locally on your machine.

## Additional Notes 📝

- **GPU Support**: If an NVIDIA GPU is detected and configured, the app leverages it for faster processing. GPU details are shown in the sidebar.
- **Data Privacy**: No personal data or images are sent to external servers. All processing is local.
- **Troubleshooting**:
  - *Camera Issues*: Ensure your browser has permission to access your webcam.
  - *Face Detection Failures*: Use well-lit images, and try different angles if the face is not detected.
  - *Installation Errors*: Confirm that pip is up-to-date and your internet connection is stable.

## Contributing 🤝

Contributions, bug reports, and feature requests are welcome!  
Feel free to open an issue or submit a pull request on the [GitHub repository](https://github.com/Adarsh3315/DeepScaleR3).

## License 📝

DeepScale R3 is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments 🙏

- **DeepFace & RetinaFace**: For providing robust emotion and face detection capabilities.
- **Streamlit**: For powering the interactive and user-friendly interface.
- **Open Source Community**: Special thanks to all the contributors and open-source projects that made this system possible.

---

Thank you for exploring DeepScale R3! Enjoy discovering music that matches your mood, and feel free to provide feedback or contribute improvements.
