# News Aggregator and Audio/Subtitle Generator

## 1. Project Description

This project is a Python application that allows users to:
*   Fetch news articles from specified URLs and RSS feeds.
*   Optionally summarize the fetched content using AI services (Ollama, Google Gemini, OpenRouter).
*   Generate audio (MP3) versions of the news content (original or summarized) using Text-to-Speech services (Microsoft Edge TTS, Azure Cognitive Services TTS, Google Cloud TTS, Minimax TTS - Minimax is placeholder).
*   Create subtitle files (SRT and LRC) synchronized with the generated audio.
*   Interact with these functionalities through a web-based user interface powered by Gradio.

## 2. Installation

To set up the project environment and install the necessary dependencies:

1.  **Clone the repository (if applicable) or ensure all project files (`.py`, `PDD.md`, etc.) are in a single directory.**
2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    # venv\Scripts\activate
    # On macOS/Linux
    # source venv/bin/activate
    ```
3.  **Install dependencies using pip:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Some TTS services like Azure and Google Cloud require their respective SDKs which are included in `requirements.txt`. Ensure you have any necessary system libraries or build tools if errors occur during installation of these packages (though typically they install smoothly via pip).*

## 3. How to Run the Application

Once the dependencies are installed:

1.  **Navigate to the project directory in your terminal.**
2.  **Run the Gradio application:**
    ```bash
    python app.py
    ```
3.  **Open your web browser and go to the URL displayed in the terminal (usually `http://127.0.0.1:7860` or similar).**

    *The application will create an `generated_files` directory in the same location as `app.py` to store the output MP3, SRT, and LRC files.*
    *The application also creates `tts_output` and `subtitle_output` from internal tests of `tts_generator.py` and `subtitle_generator.py` if they are run directly.*

## 4. API Key Configuration

To use AI summarization and cloud-based TTS services, you need to configure API keys:
*   Open the application in your browser.
*   Navigate to the **"Configuration"** tab.
*   Enter your API keys and service details (e.g., Azure region, Ollama URL) in the respective fields.
*   This configuration is used directly by the processing functions. There is no persistent save of keys in this version; they need to be entered per session if the app restarts.

## 5. Testing Checklist

This checklist is to help guide manual testing of the core features:

**A. News Fetching:**
*   [ ] Can you fetch news from a single valid RSS feed URL (e.g., `http://rss.cnn.com/rss/cnn_topstories.rss`)?
*   [ ] Can you fetch news from a single valid website article URL (e.g., a news article page)?
*   [ ] Can you fetch from multiple URLs/RSS feeds entered on separate lines?
*   [ ] Do fetched news items appear in the "Fetched News Items" table on the "News Fetching & Processing" tab?
*   [ ] Does the "Fetching Status/Log" show informative messages (e.g., number of items fetched, errors if any)?
*   [ ] Are items de-duplicated if the same URL is provided or fetched via different feeds? (Check source URLs in output)

**B. News Selection & Topic:**
*   [ ] Can you select a row in the "Fetched News Items" table? (The selection mechanism might be basic; check if the generation process uses the selection).
    *   *Developer Note: The current DataFrame selection logic is basic. Verify if the `selected_df_indices_state` correctly captures the intended row for processing. A simple test is to select one item and see if only that item is processed.*
*   [ ] Can you specify a "News Topic/Category" (e.g., "Technology") in the "Generation Options" tab?

**C. Summarization:**
*   (Requires API keys/local Ollama setup in "Configuration" tab)
*   [ ] If "Ollama" is selected as summarizer:
    *   [ ] Is the "Ollama Model Name" field visible?
    *   [ ] Does providing a valid Ollama API URL (e.g., `http://localhost:11434`) and model name (e.g., `llama3`, ensure it's pulled) result in summarization? (Check "Generation Status/Log").
*   [ ] If "Gemini" is selected and a valid Gemini API key is provided, does the log indicate successful summarization?
*   [ ] If "OpenRouter" is selected and a valid OpenRouter API key is provided, does the log indicate successful summarization?
*   [ ] If "None" is selected, is the summarization step skipped as expected?

**D. Text-to-Speech (TTS) & Output Files:**
*   (Cloud TTS options require API keys in "Configuration" tab)
*   **EdgeTTS (Default, no key needed):**
    *   [ ] Does selecting "edge_tts" and "female" voice generate an MP3 file?
    *   [ ] Does selecting "edge_tts" and "male" voice generate an MP3 file?
*   **Azure TTS:**
    *   [ ] If a valid Azure TTS API Key and Region are provided, does selecting "azure" for TTS generate an MP3?
*   **Google Cloud TTS:**
    *   [ ] If a valid Google Cloud TTS Credentials Path is provided (or ADC is set up), does selecting "google" for TTS generate an MP3?
*   **Minimax TTS:**
    *   [ ] (Currently a placeholder) Does selecting "minimax" result in a message indicating it's not yet implemented?

**E. File Naming and Content:**
*   [ ] Is the generated MP3 filename in the format `YYYYMMDDHHMMSS_Topic_News.mp3` (if topic specified) or `YYYYMMDDHHMMSS_News.mp3` (if no topic)? (Sanitized title part also included).
*   [ ] Are SRT (.srt) and LRC (.lrc) files generated alongside the MP3 when audio generation is successful?
*   [ ] Do the SRT and LRC files contain text content corresponding to the (potentially summarized) news item?
*   [ ] (Harder to verify precisely without listening) Do subtitles in SRT/LRC seem reasonably timed with the audio duration indicated in the log?

**F. UI & General:**
*   [ ] Does the "Generation Status/Log" provide clear feedback on the steps being performed and any errors?
*   [ ] Are download links (or paths) for generated files displayed in the "Results / Output" tab or indicated in the log?
    *   *Developer Note: Markdown links `./file=path/to/file.mp3` should work if `OUTPUT_DIR` is correctly configured in `app_ui.launch(allowed_paths=[OUTPUT_DIR])`.*
*   [ ] Does the UI generally respond as expected (e.g., Ollama model field visibility changing)?

This checklist provides a starting point for testing the application's functionality.
