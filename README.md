# `ana.py` - Interview Analysis with Streamlit and Gemini API

`ana.py` is the core script of the **GSAS (Generalized Sentiment Analysis System)** repository. It provides a Streamlit-based web application for analyzing interview transcripts to predict career paths (Academia vs. Industry) and extract detailed insights. The script leverages the **Google Gemini API** for advanced natural language processing (NLP) and generates interactive visualizations using **Plotly**.

This document provides a detailed guide on how to use, customize, and extend the functionality of `ana.py`.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [How to Use the App](#how-to-use-the-app)
4. [Code Structure](#code-structure)
5. [Customizing the App](#customizing-the-app)
6. [Modifying the Sentiment Analysis Model](#modifying-the-sentiment-analysis-model)
7. [Rate Limiting](#rate-limiting)
8. [Visualizations](#visualizations)
9. [Chat Interface](#chat-interface)
10. [Contributing](#contributing)
11. [License](#license)

---

## Overview

The `ana.py` script is designed to:
- Analyze interview transcripts to predict whether a candidate is more suited for a career in **Academia** or **Industry**.
- Extract detailed insights such as **sentiment analysis**, **motivations**, **risk assessment**, and **long-term goals**.
- Provide interactive visualizations (e.g., radar charts, bar charts, pie charts) to explore the analysis results.
- Allow users to upload and analyze multiple interview transcripts in `.txt` or `.docx` formats.
- Include a **chat interface** for asking questions about the uploaded files.

---

## Key Features

1. **File Upload and Processing**:
   - Supports `.txt` and `.docx` file formats.
   - Extracts interviewee names and cleans text content for analysis.

2. **Sentiment Analysis**:
   - Analyzes sentiment for both Academia and Industry across multiple dimensions (e.g., research, teaching, product development, management).

3. **Career Path Prediction**:
   - Predicts whether the interviewee is more suited for Academia or Industry.
   - Provides a confidence score and rationale for the prediction.

4. **Interactive Visualizations**:
   - **Radar Chart**: Compares sentiment scores for Academia and Industry.
   - **Bar Chart**: Displays sentiment scores for different categories.
   - **Pie Chart**: Shows the distribution of motivations (primary, intrinsic, extrinsic).

5. **Chat Interface**:
   - Allows users to ask questions about the uploaded files and get responses based on the analysis.

6. **Rate Limiting**:
   - Implements rate limiting to comply with Gemini API constraints (2 requests per minute, 32,000 tokens per minute, 50 requests per day).

7. **Downloadable Results**:
   - Users can download analysis results in JSON or CSV format.

---

## How to Use the App

### 1. **Set Up the Environment**
   - Clone the repository:
     ```bash
     git clone https://github.com/pavlyhalim/GSAS.git
     cd GSAS
     ```
   - Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```

### 2. **Run the Streamlit App**
   - Start the app:
     ```bash
     streamlit run ana.py
     ```
   - The app will open in your browser at `http://localhost:8501`.

### 3. **Enter Gemini API Key**
   - In the sidebar, enter your **Google Gemini API Key**. This is required for the app to function.
   - Check this link to get Gemini API Key [Gemini API](https://ai.google.dev/gemini-api/docs)


### 4. **Upload Interview Transcripts**
   - Upload one or more `.txt` or `.docx` files containing interview transcripts.
   - The app will automatically process and analyze the files.

### 5. **View Analysis Results**
   - The app displays:
     - A summary table of all analyses.
     - Detailed sentiment analysis and career predictions for each interviewee.
     - Interactive visualizations (radar chart, bar chart, pie chart).

### 6. **Ask Questions Using the Chat Interface**
   - Use the chat interface to ask specific questions about the uploaded files.

### 7. **Download Results**
   - Download the analysis results in JSON or CSV format.

---

## Code Structure

The `ana.py` script is organized into the following components:

1. **Utility Functions**:
   - `read_file_content(file)`: Reads content from `.txt` or `.docx` files.

2. **Rate Limiter Class**:
   - `RateLimiter`: Implements rate limiting for API requests.

3. **Interview Analyzer Class**:
   - `InterviewAnalyzer`: Handles the core functionality, including:
     - Setting up the Gemini API.
     - Extracting and cleaning interview data.
     - Analyzing transcripts using the Gemini API.
     - Displaying results and visualizations.

4. **Main Application**:
   - Configures the Streamlit app layout and handles user interactions.

---

## Customizing the App

### 1. **Modify the User Interface**
   - The Streamlit app layout is defined in the `main()` function.
   - Add new components (e.g., sliders, checkboxes) to the sidebar or main area.

### 2. **Add New Features**
   - To add new features (e.g., emotion detection, topic modeling), create new functions and integrate them into the `InterviewAnalyzer` class.

### 3. **Change the Sentiment Analysis Model**
   - Replace the Gemini API with another NLP model (e.g., Hugging Face Transformers, VADER):
     ```python
     from transformers import pipeline

     sentiment_pipeline = pipeline("sentiment-analysis")

     def analyze_sentiment(text):
         return sentiment_pipeline(text)
     ```

---

## Modifying the Sentiment Analysis Model

The app uses the **Google Gemini API** for sentiment analysis. To use a different model:

1. **Install Required Libraries**:
   - For Hugging Face Transformers:
     ```bash
     pip install transformers
     ```
   - For VADER:
     ```bash
     pip install vaderSentiment
     ```

2. **Replace the Model**:
   - Update the `analyze_transcript` method in the `InterviewAnalyzer` class to use the new model.

---

## Rate Limiting

The `RateLimiter` class ensures compliance with Gemini API rate limits:
- **2 requests per minute**.
- **32,000 tokens per minute**.
- **50 requests per day**.

To adjust these limits, modify the `RateLimiter` initialization:
```python
self.rate_limiter = RateLimiter(rpm_limit=5, tpm_limit=50000, rpd_limit=100)
```

---

## Visualizations

The app uses **Plotly** to create interactive visualizations:
- **Radar Chart**: Compares sentiment scores for Academia and Industry.
- **Bar Chart**: Displays sentiment scores for different categories.
- **Pie Chart**: Shows the distribution of motivations.

To customize visualizations, modify the `create_radar_chart`, `create_bar_chart`, and `create_pie_chart` methods.

---

## Chat Interface

The chat interface allows users to ask questions about the uploaded files. Questions are sent to the Gemini API, and responses are displayed in the app.

To customize the chat interface:
- Modify the `chat_interface` and `handle_chat` methods in the `InterviewAnalyzer` class.

---

## Contributing

Contributions to the `ana.py` script are welcome! Follow these steps:
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Make your changes and commit them:
   ```bash
   git commit -m "Add your commit message"
   ```
4. Push your changes:
   ```bash
   git push origin feature/your-feature-name
   ```
5. Create a pull request.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or feedback, open an issue on GitHub

---

Thank you for using `ana.py`! We hope this tool helps you analyze interview transcripts effectively. Happy coding! 🚀
