# Product Design Document: News Aggregator and Audio/Subtitle Generator

## 1. Introduction

*   **Project Name:** News Aggregator and Audio/Subtitle Generator
*   **Purpose:** To collect news from various online sources, allow optional AI-powered summarization, and generate audio versions with subtitles.

## 2. Target Audience

*   Users who want to consume news content in audio format.
*   Users who need to create audio versions of news articles with accompanying subtitles.
*   Users interested in leveraging AI for news summarization.

## 3. Core Features

*   **News Fetching:**
    *   Sources: User-provided website URLs and RSS feed URLs.
    *   Data to be collected: Title, short summary/abstract, original source URL, publication date/time.
    *   Automatic de-duplication of news items will be performed based on URL or title to avoid redundancy when fetching from multiple sources.
    *   Users will be able to manually select/deselect news items from the fetched list before further processing (summarization, audio generation).
*   **News Topic/Category:**
    *   "Users can optionally specify a topic or category for a batch of news (e.g., 'Technology', 'AI', 'Economics'). This topic can be used for organizational purposes, including output filenames."
*   **Content Summarization (Optional):**
    *   User-selectable AI models.
    *   Supported Services:
        *   Local Ollama API (user needs to have it running).
        *   Google Gemini API (requires user API key).
        *   OpenRouter API (requires user API key).
*   **Text-to-Speech (TTS):**
    *   Language: Chinese (Mandarin).
    *   Format: MP3.
    *   Voice Gender: User-selectable (e.g., male, female, neutral if available).
    *   Supported Services:
        *   Microsoft Azure Cognitive Services TTS API (requires user API key/credentials).
        *   Google Cloud Text-to-Speech API (requires user API key/credentials).
        *   Minimax TTS API:
            *   Requires user API key and Group ID (configured via `.env`).
            *   Accessed via an HTTP streaming interface.
            *   Utilizes the `speech-02-turbo` model for generation.
            *   The default voice is "male-qn-qingse". *Note: When Minimax TTS is selected, the 'Select Voice (or Gender for non-Minimax TTS)' dropdown dynamically populates with a list of available Minimax-specific voices based on the chosen 'Target Language'. This allows users to select from a variety of voices (e.g., 'female_yujie', 'badao_shaoye' for Chinese) provided in the `MINIMAX_TTS_VOICE_MAPPING`. If a language is chosen for which no specific Minimax voices are mapped, or if a non-Minimax TTS service is selected, the dropdown defaults to generic 'female'/'male' options (where applicable to the TTS service).*
            *   Generates audio in MP3 format (32000 Hz sample rate, 128 kbps bitrate, mono channel, as per current implementation defaults).
    *   Output filenames will follow the convention: `YYYYMMDDHHMMSS_Topic_News.extension` (e.g., `20231027103000_AI_News.mp3`). If no topic is specified, it will be `YYYYMMDDHHMMSS_News.extension`.
*   **Subtitle Generation:**
    *   Formats: SRT (.srt) and LRC (.lrc).
    *   Synchronization: Based on generated audio.
*   **User Interface (Gradio):**
    *   Input fields for news sources (URLs, RSS feeds).
    *   Configuration options for summarization (model selection, API keys).
    *   Configuration options for TTS (service selection, voice gender, API keys).
    *   Display area for the list of fetched/processed news items.
    *   Controls to initiate news fetching, summarization (if desired), audio generation, and subtitle creation.
    *   Download links for generated MP3, SRT, and LRC files.
*   **Configuration Management:**
    *   Application settings, especially API keys and sensitive credentials, are managed via a `.env` file located in the project root.
    *   Users should create a `.env` file by copying the provided `.env.example` template and then fill in their specific configuration values (e.g., API keys, service URLs).
    *   The application uses the `python-dotenv` library to load these variables from the `.env` file into environment variables upon startup.
    *   The input fields in the 'Configuration' tab of the UI will display the values loaded from these environment variables or pre-defined defaults if a variable is not set.
    *   **Important:** Changes made to configuration values directly in the UI are for the current session only and are *not* saved back to any file. The `.env` file must be manually edited to persist configuration changes. The previous UI-based saving mechanism (to `app_config.json`) has been removed to encourage secure management of credentials.
    *   The `.env` file itself is ignored by Git (via `.gitignore`) to protect sensitive user data.

## 4. Technical Specifications

*   Programming Language: Python 3.x
*   User Interface Framework: Gradio
*   Key Python Libraries (preliminary list, may evolve):
    *   News Fetching: `requests`, `BeautifulSoup4`, `feedparser`
    *   Summarization APIs: `requests` (for REST APIs), potentially specific client libraries if available (e.g., `google-generativeai`).
    *   TTS APIs: `azure-cognitiveservices-speech`, `google-cloud-tts`, `requests` (for Minimax or other REST-based TTS). Consider `edge-tts` or `pyttsx3` as potential fallbacks or simpler options if direct API integration is complex or for local testing.
    *   Audio Processing: `pydub` (for MP3 handling if needed).
    *   GUI: `gradio`
    *   Environment Variable Management: `python-dotenv`
*   Data Handling: In-memory for processing, direct file downloads for outputs. Configuration is managed via environment variables (loaded from `.env`).
*   **Logging:**
    *   The application utilizes Python's built-in `logging` module for robust event tracking and diagnostics.
    *   Log messages are output to a file named `app.log` (configured with UTF-8 encoding) and simultaneously to the console.
    *   Log entries include timestamps, logger name, log level, and the message.
    *   This replaces previous `print()` statements, providing a more structured and manageable way to monitor application behavior and troubleshoot issues.
    *   The `app.log` file is included in `.gitignore`.

## 5. User Interface (UI) Design Sketch (Conceptual)

*   **Tab 1: Configuration**
    *   Section for API Keys:
        *   Google Gemini API Key (text input)
        *   OpenRouter API Key (text input)
        *   Azure TTS API Key & Region (text inputs)
        *   Google Cloud TTS Credentials Path (file path input or JSON text input)
        *   Minimax API Key (text input)
    *   Section for Ollama:
        *   Ollama API Base URL (text input, e.g., http://localhost:11434, value loaded from environment variable `OLLAMA_API_BASE_URL`).
    *   Note: UI fields display current session values loaded from environment variables. Changes in the UI do not persist unless the `.env` file or system environment variables are updated.
*   **Tab 2: News Fetching & Processing**
    *   Input area: Text box for multiple URLs/RSS feeds (one per line).
    *   Button: "Fetch News".
    *   News List Display: Table with columns (Select [Checkbox for manual filtering], Title, Summary, Source, Date).
*   **Tab 3: Generation Options (Applied to selected news from Tab 2)**
    *   Optional News Topic/Category: (text input, e.g., 'Technology') - to be used in filenames.
    *   Summarization:
        *   Dropdown: "Select Summarizer" (None, Ollama, Gemini, OpenRouter).
        *   Ollama Model Name (if Ollama selected, text input e.g., 'llama3').
    *   Text-to-Speech:
        *   Dropdown: "Select TTS Service" (Microsoft Azure, Google Cloud, Minimax, EdgeTTS [fallback]).
        *   Dropdown: "Select Voice Gender" (Male, Female).
    *   Button: "Generate Audio & Subtitles for Selected News".
*   **Tab 4: Results / Output**
    *   Display area showing progress/status of generation.
    *   List of generated content: News Title, Download MP3, Download SRT, Download LRC.
    *   **Note on Intermediate Files:** Additionally, for each processed news item, the system saves intermediate text files to the `generated_files/` output directory:
        *   `[filename_base]_original_content.txt`: Contains the full text content (title + summary) of the selected news item before any summarization is applied. This is the text used as input if summarization is chosen.
        *   `[filename_base]_summarized_content.txt`: Contains the text content after summarization has been applied. This file is created only if summarization is used for the item. This summarized text then becomes the input for the TTS stage.
        These files can be useful for reviewing the exact text input to the TTS stage or for debugging purposes. The `[filename_base]` corresponds to the one used for the MP3/SRT/LRC files.

## 6. Error Handling & User Feedback

*   Clear messages for API errors, network issues, invalid inputs.
*   Progress indicators for long-running tasks (fetching, generation).

## 7. Future Considerations (Optional)

*   Support for more news sources/APIs.
*   Batch processing controls.
*   More advanced subtitle customization.
*   Support for more TTS engines.
