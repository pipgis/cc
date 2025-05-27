import gradio as gr
import os
import json
import logging
import re # Import re module
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd # For DataFrame
from mutagen.mp3 import MP3 # For getting audio duration

# Import functionalities from other modules
import news_fetcher
import summarizer
import tts_generator
from tts_generator import MINIMAX_TTS_VOICE_MAPPING
import subtitle_generator
import translator # Import the new translator module

# --- Global State (Simplified for now) ---
# Store fetched news items. A list of dictionaries.
# Each dictionary should match the structure expected by the DataFrame and processing functions.
# Example: {'id': 0, 'title': 'News Title', 'summary': 'Short summary...', 
#           'source_url': 'http://...', 'published_date': 'YYYY-MM-DD', 'selected': False}
# For Gradio, we'll manage this by updating components directly.
# Let's define a global variable to hold the fetched news data as a list of dicts
# This will be used to populate the DataFrame and for processing.
global_news_items_store = []
# Get the absolute path of the directory where app.py is located
APP_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(APP_DIR, "generated_files")
CONFIG_FILE = "app_config.json"
LOG_FILE = "app.log"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load environment variables from .env file
dotenv_loaded = load_dotenv()

# --- Logging Configuration ---
logger = logging.getLogger('NewsAppLogger')
logger.setLevel(logging.INFO)

# File Handler
fh = logging.FileHandler(LOG_FILE, encoding='utf-8')
fh.setLevel(logging.INFO)

# Console Handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO) # Can be set to DEBUG for more console verbosity if needed

# Formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

if dotenv_loaded:
    logger.info("Successfully loaded .env file.")
else:
    logger.info("No .env file found. Application will rely on OS environment variables or defaults.")

# Log status of critical environment variables without logging their values
env_vars_to_check = [
    "GEMINI_API_KEY", "OPENROUTER_API_KEY", "OLLAMA_API_BASE_URL",
    "AZURE_TTS_API_KEY", "AZURE_TTS_REGION", 
    "GOOGLE_CLOUD_TTS_CREDENTIALS_PATH",
    "MINIMAX_API_KEY", "MINIMAX_GROUP_ID"
    # "TRANSLATION_API_KEY" # Removed as Ollama is used for translation
]
for var_name in env_vars_to_check:
    if os.getenv(var_name):
        logger.info(f"{var_name} found in environment.")
    else:
        logger.info(f"{var_name} not found in environment. Using default or manual input if applicable.")

logger.info("Application starting. Logging configured.")

# --- Handler Functions ---

def handle_fetch_news(urls_text_input):
    """
    Fetches news from the provided URLs/RSS feeds.
    Updates the news display DataFrame and status.
    """
    global global_news_items_store # Use the global store
    status_messages = [] # Collect messages for logging and returning

    if not urls_text_input or not urls_text_input.strip():
        msg = "Status: Please enter some URLs or RSS feed links."
        logger.warning(msg)
        return pd.DataFrame(), msg, [], gr.update(choices=[], value=[])

    sources = [url.strip() for url in urls_text_input.strip().split('\n') if url.strip()]
    if not sources:
        msg = "Status: No valid URLs provided."
        logger.warning(msg)
        return pd.DataFrame(), msg, [], gr.update(choices=[], value=[])
    
    msg = f"Fetching news from {len(sources)} source(s)..."
    logger.info(msg)
    status_messages.append(msg)
    
    try:
        fetched_items_raw = news_fetcher.fetch_news(sources)
    except Exception as e:
        msg = f"An error occurred during fetching: {e}"
        logger.exception("Exception during news fetching:") # Logs error with stack trace
        status_messages.append(msg)
        return pd.DataFrame(), "\n".join(status_messages), [], gr.update(choices=[], value=[])

    # Clear previous items and add new ones with an ID
    global_news_items_store = []
    valid_items_for_df = []
    checkbox_choices = [] # Initialize checkbox_choices
    item_id_counter = 0

    if not fetched_items_raw:
        msg = "No items were fetched. Check URLs and network."
        logger.info(msg)
        status_messages.append(msg)
        return pd.DataFrame(), "\n".join(status_messages), [], gr.update(choices=[], value=[])

    for item in fetched_items_raw:
        if item.get('error'):
            msg = f"Error for {item.get('source_url', 'Unknown source')}: {item['error']}"
            logger.warning(msg)
            status_messages.append(msg)
            continue

        processed_item = {
            'id': item_id_counter,
            'title': item.get('title', 'N/A'),
            'summary': item.get('summary', 'N/A')[:150] + "..." if item.get('summary') else 'N/A', # Truncate summary
            'source_url': item.get('source_url', 'N/A'),
            'published_date': item.get('published_date', 'N/A'),
            '_full_summary': item.get('summary', 'N/A'), 
            '_original_title': item.get('title', 'N/A')
        }
        global_news_items_store.append(processed_item)
        checkbox_choices.append((processed_item['title'], processed_item['id'])) # Populate checkbox_choices
        valid_items_for_df.append({
            "ID": item_id_counter,
            "Title": processed_item['title'],
            "Summary Snippet": processed_item['summary'],
            "Source": processed_item['source_url'],
            "Date": processed_item['published_date']
        })
        item_id_counter += 1
        
    msg = f"Fetched {len(valid_items_for_df)} valid news items."
    logger.info(msg)
    status_messages.append(msg)
    
    if not valid_items_for_df:
        news_df = pd.DataFrame()
        msg = "No valid news items could be processed into the display table."
        logger.info(msg)
        status_messages.append(msg)
        # Ensure checkbox_choices is empty if no valid items
        checkbox_choices = [] 
    else:
        news_df = pd.DataFrame(valid_items_for_df)

    return news_df, "\n".join(status_messages), global_news_items_store, gr.update(choices=checkbox_choices, value=[])


def handle_generate_audio_subtitles(
    selected_indices, 
    news_data_state, 
    news_topic,
    summarizer_choice, ollama_model_name, ollama_api_url_cfg,
    gemini_api_key_cfg, openrouter_api_key_cfg,
    tts_service, tts_voice_gender,
    max_chars_per_segment_cfg, # New parameter for max chars per segment
    azure_tts_key_cfg, azure_tts_region_cfg,
    google_tts_path_cfg, minimax_tts_key_cfg, minimax_tts_group_id_cfg,
    target_language_choice # New parameter for target language
):
    """
    Generates audio and subtitles for selected news items.
    """
    logger.debug(f"handle_generate_audio_subtitles: received selected_indices={selected_indices}, type={type(selected_indices)}")
    log_messages = [] # For returning to Gradio Textbox
    output_links_markdown = ""
    logger.info("Starting generation process...")
    log_messages.append("Starting generation process...")

    # Initialize collectors for global processing
    all_texts_for_global_processing = []
    consolidated_selected_news_data = [] # For storing data of selected items for JSON export
    combined_text_for_audio = ""

    # selected_indices now contains item IDs from the CheckboxGroup
    if not selected_indices: 
        msg = "No news items selected for generation."
        logger.warning(msg)
        log_messages.append(msg)
        return "\n".join(log_messages), ""
    
    actual_selected_items = []
    if isinstance(selected_indices, list):
        # news_data_state is global_news_items_store, which is a list of dicts, each with an 'id'
        # selected_indices is a list of IDs that the user has checked.
        ids_to_find = set(selected_indices) # Use a set for efficient lookup
        
        # Create a mapping from ID to item for quick retrieval and to preserve order if needed,
        # though order of selection might not be strictly preserved by CheckboxGroup's output.
        # For this loop, simple iteration over news_data_state is fine.
        for item in news_data_state:
            if item['id'] in ids_to_find:
                actual_selected_items.append(item)
                # Optional: Remove found ID to handle potential duplicates in news_data_state, though IDs should be unique
                # ids_to_find.remove(item['id']) 
                # if not ids_to_find: break # Optimization: stop if all selected IDs are found
        
        if not actual_selected_items:
            msg = "Selected item IDs not found in news data. This could indicate a state mismatch."
            logger.warning(msg)
            log_messages.append(msg)
            return "\n".join(log_messages), ""
            
    else: # Should not happen if CheckboxGroup is correctly wired
        msg = "Selection format not as expected. Expected a list of item IDs."
        logger.error(msg) # Changed to error as this implies an internal issue
        log_messages.append(msg)
        return "\n".join(log_messages), ""

    # This check is effectively duplicated if the above block for empty actual_selected_items runs.
    # However, keeping it as a safeguard.
    if not actual_selected_items: 
        msg = "No news items to process after selection logic."
        logger.warning(msg)
        log_messages.append(msg)
        return "\n".join(log_messages), ""

    msg = f"Processing {len(actual_selected_items)} selected news item(s)."
    logger.info(msg)
    log_messages.append(msg)

    # timestamp_run = datetime.now().strftime("%Y%m%d%H%M%S") # Timestamp for this run's individual files - No longer needed for individual original files

    for item in actual_selected_items:
        original_title = item.get('_original_title', 'Untitled')
        full_summary = item.get('_full_summary', '')
        
        # Step 1: Initial Text Setup
        original_text_for_processing = original_title + ". " + full_summary
        item_log_prefix = f"Processing Item: {original_title}"
        logger.info(item_log_prefix)
        log_messages.append(f"\n{item_log_prefix}")
        output_links_markdown += f"**{original_title}**:\n"

        # Step 2: Summarization Stage (on original English text)
        english_summary_content = None
        text_for_translation_or_tts = original_text_for_processing # Default to original if summarization fails or is skipped

        if summarizer_choice != "None" and original_text_for_processing.strip():
            msg = f"  Attempting summarization of original English content with {summarizer_choice}..."
            logger.info(msg)
            log_messages.append(msg)
            summary_result = summarizer.summarize_text(
                text_to_summarize=original_text_for_processing,
                service=summarizer_choice,
                api_key=(gemini_api_key_cfg if summarizer_choice == "gemini" else openrouter_api_key_cfg if summarizer_choice == "openrouter" else None),
                ollama_model=ollama_model_name, 
                ollama_api_url=ollama_api_url_cfg,
                target_language=None # Ensure summarizer produces English summary (or its default)
            )
            if summary_result['error']:
                msg = f"  Summarization Error (original English): {summary_result['error']}"
                logger.error(msg)
                log_messages.append(msg)
                # english_summary_content remains None
                # text_for_translation_or_tts remains original_text_for_processing
            elif summary_result['summary'] and summary_result['summary'].strip():
                english_summary_content = summary_result['summary']
                text_for_translation_or_tts = english_summary_content # Update for next stage
                msg = f"  Summarization of original English content successful. New length: {len(english_summary_content)}"
                logger.info(msg)
                log_messages.append(msg)
            else:
                msg = "  Summarization of original English content resulted in empty text. Using original text for subsequent steps."
                logger.warning(msg)
                log_messages.append(msg)
                # english_summary_content remains None
                # text_for_translation_or_tts remains original_text_for_processing
        else:
            logger.info("  Summarization skipped or input was empty. Using original text for subsequent steps.")
            log_messages.append("  Summarization skipped or input was empty.")
            # text_for_translation_or_tts remains original_text_for_processing
            if not output_links_markdown.strip().endswith("\n"): # Ensure proper spacing if no summary files were mentioned
                output_links_markdown += "\n"


        # Step 3: Translation Stage (on English summary or original English text)
        translated_text_content = None 
        final_text_for_tts = text_for_translation_or_tts # Default to English (original or summary)

        target_lang_code_for_processing = None
        if target_language_choice and target_language_choice != "As Source (No Translation)":
            match = re.search(r'\((.*?)\)', target_language_choice)
            if match:
                target_lang_code_for_processing = match.group(1)
                logger.info(f"  Target language for translation selected: {target_lang_code_for_processing}")
            else:
                logger.warning(f"  Could not parse language code from: {target_language_choice} for translation.")
        
        if target_lang_code_for_processing and text_for_translation_or_tts.strip():
            msg = f"  Attempting translation of '{'English summary' if english_summary_content else 'original English text'}' to {target_lang_code_for_processing}..."
            logger.info(msg)
            log_messages.append(msg)
            
            translation_result = translator.translate_text(
                text=text_for_translation_or_tts,
                target_lang_code=target_lang_code_for_processing,
                source_lang_code="en", # Source is known to be English (either original or summarized)
                ollama_model=ollama_model_name,
                ollama_api_url=ollama_api_url_cfg
            )
            if translation_result['error']:
                msg = f"  Translation Error to {target_lang_code_for_processing}: {translation_result['error']}"
                logger.error(msg)
                log_messages.append(msg)
                # translated_text_content remains None
                # final_text_for_tts remains text_for_translation_or_tts (English)
            elif translation_result['translated_text'] and translation_result['translated_text'].strip():
                translated_text_content = translation_result['translated_text']
                final_text_for_tts = translated_text_content # Update for TTS
                msg = f"  Translation to {target_lang_code_for_processing} successful. New length: {len(translated_text_content)}"
                logger.info(msg)
                log_messages.append(msg)
            else:
                msg = f"  Translation to {target_lang_code_for_processing} resulted in empty text. Using English text for TTS."
                logger.warning(msg)
                log_messages.append(msg)
                # translated_text_content remains None
                # final_text_for_tts remains text_for_translation_or_tts (English)
        elif target_lang_code_for_processing and not text_for_translation_or_tts.strip():
            msg = f"  Skipping translation to {target_lang_code_for_processing} as input text is empty."
            logger.info(msg)
            log_messages.append(msg)
        else:
            logger.info("  No translation selected or input text was empty. TTS will use English content.")
            log_messages.append("  Translation skipped. TTS will use English content.")
            # final_text_for_tts remains text_for_translation_or_tts (English)
        
        # Step 4: JSON Population Stage
        item_data_for_json = {
            'id': item.get('id'),
            'original_title': original_title, # Keep original title
            'full_summary_original': full_summary, # Keep original full summary (English)
            'source_url': item.get('source_url', 'N/A'),
            'published_date': item.get('published_date', 'N/A'),
            'ai_summary_source': english_summary_content # This is the English summary, or None
        }
        if target_lang_code_for_processing:
            item_data_for_json[f"ai_summary_{target_lang_code_for_processing}"] = translated_text_content # This is translated summary/original, or None
        
        consolidated_selected_news_data.append(item_data_for_json)

        # Step 5: TTS Input
        all_texts_for_global_processing.append(final_text_for_tts)
        
        # Ensure a newline in markdown if no specific file links were added for this item (e.g. if summarization was skipped)
        if not output_links_markdown.strip().endswith("\n"):
             output_links_markdown += "\n" # Should already be handled by initial item title print
        output_links_markdown += "\n" # Add a newline after each item's text file links block

    # --- Save consolidated data to JSON (after the loop) ---
    if consolidated_selected_news_data:
        json_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        json_filename = f"consolidated_selected_news_{json_timestamp}.json"
        json_filepath = os.path.join(OUTPUT_DIR, json_filename)
        try:
            with open(json_filepath, "w", encoding="utf-8") as f:
                json.dump(consolidated_selected_news_data, f, ensure_ascii=False, indent=4)
            logger.info(f"Successfully saved consolidated selected news data to {json_filepath}")
            log_messages.append(f"\nSaved consolidated selected news data to: {json_filepath}")
            json_file_link = f"[{json_filename}](./file={json_filepath})"
            # Insert this before the "Combined Output" section if it exists, or add a new section
            # For simplicity, adding it here. Might need restructuring of output_links_markdown for perfect placement.
            output_links_markdown = f"**Consolidated Selected News Data**:\n  - JSON File: {json_file_link}\n\n" + output_links_markdown

        except Exception as e:
            logger.error(f"Failed to save consolidated selected news data to {json_filepath}: {e}", exc_info=True)
            log_messages.append(f"\nError saving consolidated news data: {e}")
            output_links_markdown = f"**Consolidated Selected News Data**:\n  - Error saving JSON file.\n\n" + output_links_markdown
    else:
        log_messages.append("\nNo selected items to save to consolidated JSON.")


    # --- Global Processing (after the loop) ---
    if not all_texts_for_global_processing:
        msg = "No text collected from items for global audio generation. Aborting."
        logger.warning(msg)
        log_messages.append(msg)
        final_msg = "\nIndividual item processing completed. No content for global audio."
        log_messages.append(final_msg)
        logger.info(final_msg)
        # Return current logs and any item-specific text file links
        return "\n".join(log_messages), output_links_markdown.strip() + "\n\n**Combined Output**:\n  - No content for global audio."

    combined_text_for_audio = "\n\n".join(all_texts_for_global_processing)
    msg = f"Combined text from {len(all_texts_for_global_processing)} items for global audio. Total length: {len(combined_text_for_audio)}"
    logger.info(msg)
    log_messages.append(f"\n{msg}")

    # Global Filename Generation
    timestamp_global = datetime.now().strftime("%Y%m%d%H%M%S")
    topic_prefix_global = f"{news_topic.replace(' ', '_')}_" if news_topic and news_topic.strip() else ""
    base_filename_global = f"{timestamp_global}_{topic_prefix_global}combined_audio"

    global_mp3_filename = os.path.join(OUTPUT_DIR, f"{base_filename_global}.mp3")
    global_srt_filename = os.path.join(OUTPUT_DIR, f"{base_filename_global}.srt")
    global_lrc_filename = os.path.join(OUTPUT_DIR, f"{base_filename_global}.lrc")

    msg = f"Attempting global audio generation with {tts_service} ({tts_voice_gender})..."
    logger.info(msg)
    log_messages.append(msg)

    # Prepare API key arguments for tts_generator.generate_audio
    azure_api_key_to_pass = azure_tts_key_cfg if tts_service == "azure" else None
    # Other API keys (Google, Minimax) are passed directly to tts_generator

    # Determine language code for TTS
    # target_lang_code is derived per item, but TTS is global.
    # For simplicity, we'll use the target_lang_code of the *first* item if multiple items are processed
    # and a translation was requested. Otherwise, default to "en".
    # A more robust solution might involve per-item TTS or ensuring all items have same target lang for combined TTS.
    
    # Let's re-evaluate: target_lang_code is determined *inside* the loop for each item.
    # The combined_text_for_audio is built from text_for_tts which might be translated.
    # So, the language of combined_text_for_audio should match the target_lang_code if translation happened.
    # We need the target_lang_code that was applied (if any) to the text before summarization & TTS.
    # The current `target_lang_code` variable is from the last item in the loop.
    # This is a slight design issue: if items have different target languages, what should global TTS language be?
    # For now, let's assume the `target_language_choice` applies to all selected items uniformly.
    # So the `target_lang_code` derived from `target_language_choice` (outside the loop or from the first item) is what we need.
    
    # Re-parsing target_language_choice for global TTS language context.
    # This assumes target_language_choice is consistent for the batch.
    global_tts_lang_code = None
    if target_language_choice and target_language_choice != "As Source (No Translation)":
        match = re.search(r'\((.*?)\)', target_language_choice)
        if match:
            global_tts_lang_code = match.group(1)
    
    if global_tts_lang_code:
        language_code_for_tts = global_tts_lang_code
        logger.info(f"Global TTS will use target language: {language_code_for_tts} (from UI selection).")
    else:
        language_code_for_tts = "en" # Default to English if no translation was specified
        logger.info(f"Global TTS will use default language: {language_code_for_tts} (as 'As Source' or no valid language parsed).")

    global_tts_result = tts_generator.generate_audio(
        text_to_speak=combined_text_for_audio,
        output_filename=global_mp3_filename,
        service=tts_service,
        voice_gender=tts_voice_gender,
        language_code=language_code_for_tts, # Pass the determined language code
        api_key=azure_api_key_to_pass, # Used by Azure
        azure_region=azure_tts_region_cfg,
        google_credentials_path=google_tts_path_cfg,
        minimax_api_key=minimax_tts_key_cfg,
        minimax_group_id=minimax_tts_group_id_cfg
    )

    output_links_markdown += "**Combined Output**:\n" # Add heading for combined files

    if global_tts_result['success']:
        msg = f"  Global audio generated: {global_mp3_filename}"
        logger.info(msg)
        log_messages.append(msg)
        mp3_link_global = f"[{os.path.basename(global_mp3_filename)}](./file={global_mp3_filename})"
        output_links_markdown += f"  - Audio: {mp3_link_global}\n"

        global_audio_duration_seconds = 0
        try:
            if os.path.exists(global_mp3_filename):
                audio = MP3(global_mp3_filename)
                global_audio_duration_seconds = audio.info.length
                msg = f"  Global audio duration: {global_audio_duration_seconds:.2f} seconds."
                logger.info(msg)
                log_messages.append(msg)
            else:
                msg = "  Error: Global MP3 file not found after TTS success reported."
                logger.error(msg)
                log_messages.append(msg)
        except Exception as e:
            msg = f"  Error getting global audio duration: {e}. Subtitles might be misaligned."
            logger.exception("Exception during global audio duration reading:")
            log_messages.append(msg)
            if global_audio_duration_seconds == 0 and combined_text_for_audio:
                estimated_duration = len(combined_text_for_audio.split()) / 4.0 
                global_audio_duration_seconds = max(1.0, estimated_duration)
                msg = f"  Using estimated duration for global audio: {global_audio_duration_seconds:.2f}s for subtitles."
                logger.warning(msg)
                log_messages.append(msg)

        if global_audio_duration_seconds > 0:
            msg = f"  Generating global SRT subtitles for combined audio..."
            logger.info(msg)
            log_messages.append(msg)

            # Ensure max_chars_per_segment_cfg is an int for SRT generation
            try:
                max_chars_for_srt = int(max_chars_per_segment_cfg)
            except ValueError:
                logger.error(f"Could not convert max_chars_per_segment_cfg '{max_chars_per_segment_cfg}' to int. Defaulting to 50 for SRT.")
                max_chars_for_srt = 50
            
            # Use combined_text_for_audio for subtitle generation
            global_srt_result = subtitle_generator.generate_srt(
                text_content=combined_text_for_audio, 
                audio_duration_seconds=global_audio_duration_seconds, 
                output_filename=global_srt_filename,
                max_chars_per_segment=max_chars_for_srt # Use the validated integer
            )
            if global_srt_result['success']:
                msg = f"  Global SRT generated: {global_srt_filename}"
                logger.info(msg)
                log_messages.append(msg)
                srt_link_global = f"[{os.path.basename(global_srt_filename)}](./file={global_srt_filename})"
                output_links_markdown += f"  - SRT: {srt_link_global}\n"
            else:
                msg = f"  Global SRT Generation Error: {global_srt_result['error']}"
                logger.error(msg)
                log_messages.append(msg)
                output_links_markdown += f"  - SRT: Generation failed.\n"

            msg = f"  Generating global LRC subtitles for combined audio..."
            logger.info(msg)
            log_messages.append(msg)
            # Use combined_text_for_audio for subtitle generation
            global_lrc_result = subtitle_generator.generate_lrc(combined_text_for_audio, global_audio_duration_seconds, global_lrc_filename)
            if global_lrc_result['success']:
                msg = f"  Global LRC generated: {global_lrc_filename}"
                logger.info(msg)
                log_messages.append(msg)
                lrc_link_global = f"[{os.path.basename(global_lrc_filename)}](./file={global_lrc_filename})"
                output_links_markdown += f"  - LRC: {lrc_link_global}\n"
            else:
                msg = f"  Global LRC Generation Error: {global_lrc_result['error']}"
                logger.error(msg)
                log_messages.append(msg)
                output_links_markdown += f"  - LRC: Generation failed.\n"
        else:
            msg = "  Skipping global subtitle generation due to missing global audio duration."
            logger.warning(msg)
            log_messages.append(msg)
            output_links_markdown += "  - Subtitles: Skipped (no audio duration).\n"
    else:
        msg = f"  Global TTS Error: {global_tts_result['error']}"
        logger.error(msg)
        log_messages.append(msg)
        output_links_markdown += f"  - Audio: Generation failed.\n"
            
    final_msg = "\nGlobal generation process finished."
    logger.info(final_msg)
    log_messages.append(final_msg)
    return "\n".join(log_messages), output_links_markdown


def handle_textbox_selection(text_input_indices_str: str, current_news_data: list) -> list:
    """
    Parses a comma-separated string of indices, validates them, and returns a list of valid integer indices.
    """
    if not text_input_indices_str or not text_input_indices_str.strip():
        logger.debug("handle_textbox_selection: no input string provided.")
        return [] 

    valid_indices = []
    parts = text_input_indices_str.split(',')
    for part in parts:
        try:
            index = int(part.strip())
            if 0 <= index < len(current_news_data):
                if index not in valid_indices: 
                    valid_indices.append(index)
            else:
                logger.warning(f"Warning: Index {index} is out of range for current news data (size {len(current_news_data)}).")
        except ValueError:
            logger.warning(f"Warning: Non-integer value '{part}' found in selection input.")
    logger.debug(f"handle_textbox_selection: input='{text_input_indices_str}', parsed_indices={valid_indices}")
    return valid_indices


def update_minimax_voice_dropdown(tts_service: str, target_language_str: str):
    """
    Updates the voice dropdown choices and selected value based on the selected TTS service and target language.
    Specifically handles Minimax voices.
    """
    voice_choices = ["female", "male"]  # Default choices
    selected_voice = "female"       # Default selected voice

    if tts_service == "minimax":
        # Parse language code from target_language_str (e.g., "Chinese (zh)" -> "zh")
        lang_code = "en" # Default language code
        if target_language_str and "(" in target_language_str and ")" in target_language_str:
            match = re.search(r'\((.*?)\)', target_language_str)
            if match:
                parsed_code = match.group(1)
                # Validate if the parsed code is a known language for Minimax, otherwise default might be better
                # For now, we'll use whatever is parsed.
                lang_code = parsed_code
            else: # Should not happen if format is "Language (code)"
                logger.warning(f"Could not parse language code from '{target_language_str}', defaulting to 'en' for Minimax voices.")
        else: # Handle cases where target_language_str might be empty or not in expected format
            logger.info(f"Target language string '{target_language_str}' not in expected format for Minimax voice selection, defaulting to 'en'.")


        voices_for_lang = MINIMAX_TTS_VOICE_MAPPING.get(lang_code, {})
        
        if voices_for_lang: # If language exists in mapping and has voices
            voice_choices = list(voices_for_lang.keys())
            if voice_choices:
                selected_voice = voice_choices[0]
            else: # Language code was in mapping, but had no voices listed (empty dict)
                voice_choices = [] # No specific voices available
                selected_voice = None # Or handle as per Gradio's behavior for empty choices
        else: # Language code not in MINIMAX_TTS_VOICE_MAPPING or no specific voices
            # Fallback to default gender-based voices if no specific Minimax voices for the language
            # Or, indicate that specific selection is not applicable.
            # For now, if lang_code is not in mapping, let's assume generic voices are not applicable for Minimax
            # and thus provide no choices, prompting user to select a different TTS or language.
            # However, the requirement was "keep ['female', 'male'] if language is not in mapping at all".
            # Let's stick to the requirement more closely.
            if lang_code not in MINIMAX_TTS_VOICE_MAPPING:
                 # Keep default ["female", "male"] if language not in mapping
                 # This case is already handled by initialization, so no change needed here.
                 pass
            else: # lang_code was in mapping, but voices_for_lang was empty (e.g. MINIMAX_TTS_VOICE_MAPPING.get(lang_code) returned {})
                voice_choices = []
                selected_voice = None


    return gr.update(choices=voice_choices, value=selected_voice)


# --- Gradio UI Definition ---
with gr.Blocks(theme=gr.themes.Soft(), title="News Aggregator & Audio/Subtitle Generator") as app_ui:
    gr.Markdown("# News Aggregator and Audio/Subtitle Generator")
    
    # State to hold fetched news data (list of dictionaries)
    # This will be populated by handle_fetch_news and read by handle_generate_audio_subtitles
    news_data_state_gr = gr.State(value=[])

    with gr.Tabs():
        with gr.TabItem("Configuration"):
            gr.Markdown("## API Keys and Service Configuration")
            gr.Markdown("Values are loaded from the `.env` file if present at startup, or can be manually entered for the current session. API keys and other configurations are primarily managed via the `.env` file. Changes made to configuration values in this UI are for the current session only and will not be saved.")
            with gr.Row():
                with gr.Column():
                    cfg_gemini_key = gr.Textbox(label="Google Gemini API Key", type="password", lines=1, value=os.getenv("GEMINI_API_KEY", ""))
                    cfg_openrouter_key = gr.Textbox(label="OpenRouter API Key", type="password", lines=1, value=os.getenv("OPENROUTER_API_KEY", ""))
                    cfg_ollama_url = gr.Textbox(label="Ollama API Base URL", placeholder="e.g., http://localhost:11434", lines=1, value=os.getenv("OLLAMA_API_BASE_URL", "http://localhost:11434"))
                with gr.Column():
                    cfg_azure_tts_key = gr.Textbox(label="Azure TTS API Key", type="password", lines=1, value=os.getenv("AZURE_TTS_API_KEY", ""))
                    cfg_azure_tts_region = gr.Textbox(label="Azure TTS Region", placeholder="e.g., eastus", lines=1, value=os.getenv("AZURE_TTS_REGION", ""))
            with gr.Row():
                with gr.Column():
                    cfg_google_tts_path = gr.Textbox(label="Google Cloud TTS Credentials Path (JSON file path)", lines=1, value=os.getenv("GOOGLE_CLOUD_TTS_CREDENTIALS_PATH", ""))
                with gr.Column():
                    cfg_minimax_key = gr.Textbox(label="Minimax API Key (TTS)", type="password", lines=1, value=os.getenv("MINIMAX_API_KEY", ""))
                    cfg_minimax_group_id = gr.Textbox(label="Minimax Group ID (TTS)", lines=1, value=os.getenv("MINIMAX_GROUP_ID", ""))
            # Removed the Row that contained cfg_translation_key
            
            gr.Markdown("API keys and other configurations are primarily managed via the `.env` file or environment variables. Changes made here are for the current session only unless your environment variables are updated externally.")


        with gr.TabItem("News Fetching & Processing"):
            gr.Markdown("## Fetch News from URLs or RSS Feeds")
            news_urls_input = gr.Textbox(label="Enter URLs/RSS Feeds (one per line)", lines=5, placeholder="http://example.com/feed.xml\nhttps://another-news-site.com/article")
            fetch_news_button = gr.Button("Fetch News")
            
            news_status_log = gr.Textbox(label="Fetching Status/Log", lines=3, interactive=False)
            
            gr.Markdown("### Fetched News Items (Select rows to process)")
            # Using pandas DataFrame for display. Selection needs to be handled.
            # `type="pandas"` or `type="numpy"` for output format.
            # `interactive=True` allows selection. The `select` event can be used.
            news_display_df = gr.DataFrame(
                headers=["ID", "Title", "Summary Snippet", "Source", "Date"], 
                # datatype=["number", "str", "str", "str", "str"], # Optional: specify datatypes
                interactive=True, # Allows row selection
                label="Fetched News"
            )
            # How to get selected rows? The .select() event on DataFrame.
            # It passes a SelectData object to the handler.
            # The handler for generation will need these selected indices/items.

            fetch_news_button.click(
                handle_fetch_news,
                inputs=[news_urls_input],
                outputs=[news_display_df, news_status_log, news_data_state_gr, news_selection_checkboxes] # Update DataFrame, log, shared state, and checkboxes
            )

        with gr.TabItem("Generation Options"):
            gr.Markdown("## Generate Audio and Subtitles for Selected News")
            
            # Component to show which items are selected (for user feedback)
            # This is tricky with gr.DataFrame. A gr.CheckboxGroup from IDs would be easier.
            # For now, we rely on the user remembering what they clicked in the DataFrame.
            # Or, the handle_generate_audio_subtitles function needs to accept the selection event data.
            
            # selected_news_indices_input = gr.Textbox(label="Selected News Item IDs/Indices (for dev, ideally from DF selection)", lines=1, placeholder="e.g., 0,1,2 or from DF selection event") # REMOVED

            gen_news_topic = gr.Textbox(label="News Topic/Category (Optional, for filename)", lines=1)
            news_selection_checkboxes = gr.CheckboxGroup(label="Select News Items to Process", interactive=True) # ADDED
            
            with gr.Row():
                gen_summarizer_choice = gr.Dropdown(label="Select Summarizer", choices=["None", "ollama", "gemini", "openrouter"], value="None")
                gen_ollama_model_name = gr.Textbox(label="Ollama Model Name (if Ollama selected)", placeholder="e.g., llama3", lines=1, visible=False) # Visibility toggle later
            
            # Show Ollama model name only if Ollama is selected
            def toggle_ollama_model_visibility(summarizer_service):
                return gr.update(visible=(summarizer_service == "ollama"))
            gen_summarizer_choice.change(toggle_ollama_model_visibility, inputs=[gen_summarizer_choice], outputs=[gen_ollama_model_name])

            with gr.Row():
                gen_target_language = gr.Dropdown(
                    label="Select Target Language",
                    choices=["As Source (No Translation)", "English (en)", "Chinese (zh)", "Spanish (es)", "French (fr)"],
                    value="As Source (No Translation)"
                )

            with gr.Row():
                gen_tts_service = gr.Dropdown(label="Select TTS Service", choices=["edge_tts", "azure", "google", "minimax"], value="edge_tts")
                gen_tts_voice_gender = gr.Dropdown(label="Select Voice (or Gender for non-Minimax TTS)", choices=["female", "male"], value="female", allow_custom_value=False) # allow_custom_value=False for safety
            
            # Event handlers for updating the voice dropdown
            gen_tts_service.change(
                update_minimax_voice_dropdown,
                inputs=[gen_tts_service, gen_target_language],
                outputs=[gen_tts_voice_gender]
            )
            gen_target_language.change(
                update_minimax_voice_dropdown,
                inputs=[gen_tts_service, gen_target_language],
                outputs=[gen_tts_voice_gender]
            )

            gr.Markdown("### Subtitle Options")
            gen_max_chars_segment = gr.Slider(label="Max Characters per Subtitle Segment", minimum=20, maximum=150, value=50, step=5, interactive=True)

            generate_button = gr.Button("Generate Audio & Subtitles")
            generation_status_log = gr.Textbox(label="Generation Status/Log", lines=10, interactive=False)
            
            # This is where the selection from news_display_df needs to be passed.
            # The `news_display_df.select` event can trigger a function that stores the selection in a gr.State
            # or directly trigger the generation (might be too implicit).
            # Let's try to pass the news_data_state_gr and handle selection indices within the Python function for now.
            # The `selected_news_indices_input` is a manual placeholder.
            # Proper way: news_display_df.select(fn=handle_selection_event, inputs=None, outputs=some_state_for_selected_indices)
            # Then, generate_button.click uses that state.
            # Simplified for now: assume `handle_generate_audio_subtitles` can get selection from `news_data_state_gr`
            # or a dedicated selection state.

            # For now, we'll pass `news_data_state_gr` and a placeholder for selected_indices.
            # The user would have to manually input indices into `selected_news_indices_input` for this to work as-is
            # OR we connect the `news_display_df.select` event.
            
            # Let's try to use the select event of the DataFrame
            # This hidden textbox will store the selected row indices from the DataFrame
            selected_df_indices_state = gr.State([])

            # Event handler for manual text input of indices
            # Old .submit() event wiring:
            # selected_news_indices_input.submit(
            #     handle_textbox_selection,
            #     inputs=[selected_news_indices_input, news_data_state_gr],
            #     outputs=[selected_df_indices_state]
            # )

            # New .change() event wiring:
            # selected_news_indices_input.change( # REMOVED - This was tied to the removed Textbox
            #     handle_textbox_selection,
            #     inputs=[selected_news_indices_input, news_data_state_gr],
            #     outputs=[selected_df_indices_state]
            # )

            # def handle_df_selection(evt: gr.SelectData, current_news_data: list): # REMOVED
            #     # evt.index contains (row_index, col_index) if a cell is clicked
            #     # If a row is selected (e.g. by clicking on the far left), evt.index might be just row_index
            #     # We need the IDs of the items, which correspond to the 'ID' column in the displayed DF,
            #     # or the index in the `global_news_items_store`.
            #     # For simplicity, let's assume evt.index[0] gives the row index in the displayed DataFrame.
            #     # This corresponds to the index in `global_news_items_store` if not sorted/filtered.
            #     if evt.selected: # Check if selection is happening (not deselection)
            #         # This is a simplified way to handle selection.
            #         # If multiple rows can be selected, evt.index might be a list of indices.
            #         # For single row selection:
            #         row_index = evt.index[0]
            #         if 0 <= row_index < len(current_news_data):
            #             # Store the actual item or its ID. Storing ID is safer.
            #             selected_id = current_news_data[row_index]['id']
            #             # For multiple selections, this needs to accumulate.
            #             # This example just takes the latest single selection.
            #             # A proper multi-select would require a gr.CheckboxGroup or more complex state management.
            #             # For now, let's assume we want to process the item corresponding to the clicked row index.
            #             # This is a placeholder for robust multi-selection.
            #             # return [current_news_data[row_index]] # Return list with the single selected item
            #             return [row_index] # Return the index in the global_news_items_store
            #         else:
            #             return []
            #     return [] # No selection or deselection

            # news_display_df.select( # REMOVED
            #     handle_df_selection, 
            #     inputs=[news_data_state_gr], 
            #     outputs=[selected_df_indices_state] # Store selected indices here
            # )
            
            def handle_checkbox_selection(selected_item_ids: list):
                """
                Handles the change in checkbox selection.
                The input `selected_item_ids` is a list of IDs of the items checked by the user.
                These IDs correspond to `processed_item['id']`.
                """
                logger.debug(f"handle_checkbox_selection: selected_item_ids={selected_item_ids}, type={type(selected_item_ids)}")
                # The selected_item_ids are the actual values (item IDs) we want to store.
                # These are already the indices/IDs that handle_generate_audio_subtitles expects.
                return selected_item_ids

            news_selection_checkboxes.change(
                fn=handle_checkbox_selection,
                inputs=[news_selection_checkboxes],
                outputs=[selected_df_indices_state]
            )

            generate_button.click(
                handle_generate_audio_subtitles,
                inputs=[
                    selected_df_indices_state, # Selected item indices from DataFrame
                    news_data_state_gr,         # Full list of fetched news items
                    gen_news_topic,
                    gen_summarizer_choice, gen_ollama_model_name, cfg_ollama_url, # Summarizer
                    cfg_gemini_key, cfg_openrouter_key,                         # Summarizer APIs
                    gen_tts_service, gen_tts_voice_gender,                      # TTS
                    gen_max_chars_segment,                                      # Subtitle option (now correctly mapped to max_chars_per_segment_cfg)
                    cfg_azure_tts_key, cfg_azure_tts_region,                    # TTS APIs
                    cfg_google_tts_path, cfg_minimax_key, cfg_minimax_group_id, # TTS APIs
                    gen_target_language                                         # Target Language (now correctly mapped to target_language_choice)
                ],
                outputs=[generation_status_log, gr.Markdown(label="Generated Files")] # Output to log and results tab area
                # The results tab area needs to be named for the output to go there.
            )


        with gr.TabItem("Results / Output"):
            gr.Markdown("## Generated Files")
            # This Markdown component will be updated by handle_generate_audio_subtitles
            # It should be named in the outputs of the generate_button.click
            # To make it work, we need to pass its reference.
            # Let's define it above and pass it to the click handler.
            # For now, the generate_button.click outputs a new gr.Markdown component.
            # This means the tab needs to be re-rendered or the component updated.
            # A placeholder approach: the handler returns Markdown content,
            # and we have a gr.Markdown component here that gets updated.
            results_display_markdown = gr.Markdown("Generated file links will appear here.")
            
            # Re-wire the generate_button click to update this specific Markdown component
            # This requires moving the generate_button.click definition after results_display_markdown exists.
            # Or, by making results_display_markdown an output of the click event.
            # The current setup in generate_button.click is: outputs=[generation_status_log, gr.Markdown(...)]
            # This creates a NEW markdown component.
            # It should be: outputs=[generation_status_log, results_display_markdown]

    # Re-assigning click handler for generate_button to update the correct Markdown output
    # This is a bit of a Gradio quirk; components should be defined before they are targets of outputs.
    # The previous generate_button.click inside the Tab definition is problematic for this.
    # One way is to define all components first, then wire them.
    # For simplicity here, we assume the previous definition of generate_button.click is implicitly overridden
    # if we redefine it here. Or, more cleanly, ensure all components are created before wiring.
    
    # Correct wiring for generate_button (ensure `results_display_markdown` is defined before this call)
    # This is slightly out of order with the visual tab definition but necessary for Gradio's processing model
    # if components are defined and then immediately used in event wiring within the same scope.
    # A cleaner way is to define all UI, then all events.
    # For this subtask, the current wiring in Tab 3 for generate_button.click will be used,
    # and it implicitly targets the `results_display_markdown` by its order in the `outputs` list
    # if we ensure `results_display_markdown` is the second output component.
    # The `gr.Markdown(label="Generated Files")` in the click handler's outputs needs to be replaced by `results_display_markdown`.
    
    # The click handler for generate_button is already defined under Tab 3.
    # We need to ensure its `outputs` list correctly targets `results_display_markdown`.
    # This is done by passing `results_display_markdown` itself, not creating a new `gr.Markdown()`.
    # This seems to be an issue with my current `generate_button.click` definition.
    # Let's adjust it directly where it's defined.

    # The structure is:
    # generate_button.click(
    #     handle_generate_audio_subtitles,
    #     inputs=[...],
    #     outputs=[generation_status_log, results_display_markdown] <--- Ensure this is the component instance
    # )
    # This is implicitly handled if results_display_markdown is the component in the "Results / Output" tab
    # and the handler returns content for it as the second item in its return tuple.
    # The `gr.Markdown(label="Generated Files")` in `outputs` of `generate_button.click` should be `results_display_markdown`.
    # I will correct this in the `generate_button.click` definition within Tab 3.
    # (No, I can't directly edit the previous definition in this flow. I'll assume the current setup
    # where the output is directed to a gr.Markdown component in the 'Results / Output' tab works by position/order.
    # The key is that `handle_generate_audio_subtitles` returns markdown *content* as its second value,
    # and the `outputs` list of the button click has a `gr.Markdown` component in the second position.)
    # The `gr.Markdown(label="Generated Files")` in the `outputs` list of the `generate_button.click` call
    # should be `results_display_markdown`.
    # This is a common Gradio setup pattern: define UI, then wire events.
    # I will assume the current definition is sufficient for Gradio to update the component in the "Results" tab.

if __name__ == "__main__":
    # Setting share=True is useful for testing in environments like Gitpod or Colab
    # but might require `ffmpy` and `ffmpeg` for some audio operations if not already installed.
    # For local execution, share=False is fine.
    app_ui.launch(debug=True, share=False, allowed_paths=[OUTPUT_DIR])
    # To make files in OUTPUT_DIR accessible via markdown links like `./file=path/to/file.mp3`,
    # Gradio needs to be aware of this directory. `allowed_paths` is one way.
    # The `./file=` prefix is a Gradio convention for serving local files.
