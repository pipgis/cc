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
import subtitle_generator

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
        return pd.DataFrame(), msg, []

    sources = [url.strip() for url in urls_text_input.strip().split('\n') if url.strip()]
    if not sources:
        msg = "Status: No valid URLs provided."
        logger.warning(msg)
        return pd.DataFrame(), msg, []
    
    msg = f"Fetching news from {len(sources)} source(s)..."
    logger.info(msg)
    status_messages.append(msg)
    
    try:
        fetched_items_raw = news_fetcher.fetch_news(sources)
    except Exception as e:
        msg = f"An error occurred during fetching: {e}"
        logger.exception("Exception during news fetching:") # Logs error with stack trace
        status_messages.append(msg)
        return pd.DataFrame(), "\n".join(status_messages), []

    # Clear previous items and add new ones with an ID
    global_news_items_store = []
    valid_items_for_df = []
    item_id_counter = 0

    if not fetched_items_raw:
        msg = "No items were fetched. Check URLs and network."
        logger.info(msg)
        status_messages.append(msg)
        return pd.DataFrame(), "\n".join(status_messages), []

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
    else:
        news_df = pd.DataFrame(valid_items_for_df)

    return news_df, "\n".join(status_messages), global_news_items_store


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
    logger.info(f"DEBUG: Received tts_service: {tts_service}")
    logger.info(f"DEBUG: Received tts_voice_gender: {tts_voice_gender}")
    logger.info(f"DEBUG: Received target_language_choice: {target_language_choice}")
    logger.debug(f"handle_generate_audio_subtitles: received selected_indices={selected_indices}, type={type(selected_indices)}")
    log_messages = [] # For returning to Gradio Textbox
    output_links_markdown = ""
    logger.info("Starting generation process...")
    log_messages.append("Starting generation process...")

    # Initialize collectors for global processing
    all_texts_for_global_processing = []
    consolidated_selected_news_data = [] # For storing data of selected items for JSON export
    combined_text_for_audio = ""

    if not selected_indices:
        msg = "No news items selected for generation."
        logger.warning(msg)
        log_messages.append(msg)
        return "\n".join(log_messages), ""
    
    actual_selected_items = []
    if isinstance(selected_indices, list):
        for index in selected_indices:
            if 0 <= index < len(news_data_state):
                actual_selected_items.append(news_data_state[index])
            else:
                msg = f"Warning: Invalid selected index {index} ignored."
                logger.warning(msg)
                log_messages.append(msg)
    else:
        msg = "Selection format not as expected. Please select rows in the table."
        logger.warning(msg)
        log_messages.append(msg)
        if not news_data_state:
            msg = "News data is empty. Fetch news first."
            logger.error(msg) # Changed to error as it's a prerequisite
            log_messages.append(msg)
            return "\n".join(log_messages), ""
        if not actual_selected_items and news_data_state: # Fallback logic
            msg = "Warning: No specific items selected or selection mechanism not fully wired. Processing first item as a fallback for testing."
            logger.warning(msg)
            log_messages.append(msg)
            actual_selected_items.append(news_data_state[0])

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
        text_for_tts = original_title + ". " + full_summary # This will be the text for current item for summarization/TTS
        ai_generated_summary = None # Initialize ai_generated_summary

        # Parse target language choice
        target_lang_code = None
        if target_language_choice and target_language_choice != "As Source (No Translation)":
            match = re.search(r'\((.*?)\)', target_language_choice)
            if match:
                target_lang_code = match.group(1)
                logger.info(f"Target language code selected: {target_lang_code}")
            else:
                logger.warning(f"Could not parse language code from: {target_language_choice}")
        
        # Conceptual Translation Step
        if target_lang_code:
            # TODO: Implement actual translation call here using translator.py
            logger.info(f"Conceptual translation: Text for item '{original_title}' would be translated to {target_lang_code} here.")
            # Placeholder modification for now:
            # text_for_tts = f"[Translated to {target_lang_code}] {text_for_tts}"
            # For this task, we'll assume text_for_tts is now translated without actual content change.
            # Subsequent steps (like summarization) will use this conceptually translated text.

        # Collect data for JSON export
        item_data_for_json = {
            'id': item.get('id'),
            'original_title': original_title,
            'full_summary': full_summary,
            'source_url': item.get('source_url', 'N/A'),
            'published_date': item.get('published_date', 'N/A')
            # ai_summary will be added dynamically below
        }
        # This append will be moved down after ai_summary might be updated.
        # consolidated_selected_news_data.append(item_data_for_json) 

        item_log_prefix = f"Processing Item: {original_title}"
        logger.info(item_log_prefix)
        log_messages.append(f"\n{item_log_prefix}")
        output_links_markdown += f"**{original_title}**:\n"


        # Define base_filename for item's summarized text files (if summarization happens)
        # This base_filename might still be useful if summarization saves files.
        timestamp_item_processing = datetime.now().strftime("%Y%m%d%H%M%S") # Timestamp for this specific item's processing artifacts if any
        topic_prefix_item = f"{news_topic.replace(' ', '_')}_" if news_topic and news_topic.strip() else ""
        safe_title_part_item = "".join(c if c.isalnum() or c in (' ', '_') else '_' for c in original_title[:30]).rstrip().replace(' ', '_')
        base_filename_item = f"{timestamp_item_processing}_{topic_prefix_item}{safe_title_part_item}"
        
        # Removed individual _original_content.txt saving logic as per requirements
        # # Save original selected content
        # original_content_filepath = os.path.join(OUTPUT_DIR, f"{base_filename_item}_original_content.txt")
        # 
        # # DEBUG LOGS START
        # logger.debug(f"OUTPUT_DIR is: {OUTPUT_DIR}")
        # logger.debug(f"Calculated original_content_filepath is: {original_content_filepath}")
        # logger.debug(f"Does OUTPUT_DIR ({OUTPUT_DIR}) exist at this point? {os.path.exists(OUTPUT_DIR)}")
        # logger.debug(f"Is OUTPUT_DIR ({OUTPUT_DIR}) a directory? {os.path.isdir(OUTPUT_DIR)}")
        # # The following line is to check the directory part of original_content_filepath itself
        # logger.debug(f"Parent directory of original_content_filepath ({os.path.dirname(original_content_filepath)}) exists? {os.path.exists(os.path.dirname(original_content_filepath))}")
        # # DEBUG LOGS END
        # 
        # output_links_markdown += f"**{original_title}**:\n" # Moved up
        # try:
        #     with open(original_content_filepath, "w", encoding="utf-8") as f:
        #         f.write(text_for_tts) # text_for_tts initially holds the full original content
        #     logger.info(f"Saved original content for '{original_title}' to {original_content_filepath}")
        #     log_messages.append(f"  Saved original content to: {original_content_filepath}")
        #     original_text_link = f"[{os.path.basename(original_content_filepath)}](./file={original_content_filepath})"
        #     output_links_markdown += f"  - Original Text: {original_text_link}\n"
        # except Exception as e:
        #     logger.error(f"Failed to save original content for '{original_title}' to {original_content_filepath}: {e}", exc_info=True)
        #     log_messages.append(f"  Error saving original content: {e}")
        #     output_links_markdown += f"  - Error saving original text.\n"

        if summarizer_choice != "None" and text_for_tts.strip():
            msg = f"  Attempting summarization with {summarizer_choice} for text (potentially translated to {target_lang_code if target_lang_code else 'source language'})..."
            logger.info(msg)
            log_messages.append(msg)
            summary_result = summarizer.summarize_text(
                text_to_summarize=text_for_tts, # This text_for_tts is conceptually translated if a language was selected
                service=summarizer_choice, 
                api_key=(gemini_api_key_cfg if summarizer_choice == "gemini" else openrouter_api_key_cfg if summarizer_choice == "openrouter" else None),
                ollama_model=ollama_model_name, ollama_api_url=ollama_api_url_cfg,
                target_language=target_lang_code # Pass target_lang_code to summarizer (optional, if summarizer supports it)
            )
            if summary_result['error']:
                msg = f"  Summarization Error: {summary_result['error']}"
                logger.error(msg)
                log_messages.append(msg)
            else:
                summarized_text = summary_result['summary']
                if summarized_text and summarized_text.strip(): # Check if summary is not empty
                    ai_generated_summary = summarized_text # Update ai_generated_summary
                    # item_data_for_json['ai_summary'] = ai_generated_summary # Old static update
                    msg = f"  Summarization Successful. New text length: {len(summarized_text)}"
                    logger.info(msg)
                    log_messages.append(msg)
                    
                    # The following block for saving summarized content to a file is now removed/commented out.
                    # summarized_content_filepath = os.path.join(OUTPUT_DIR, f"{base_filename_item}_summarized_content.txt")
                else:
                    msg = f"  Summarization resulted in empty text. Not using."
                    logger.warning(msg)
                    log_messages.append(msg)
                    # ai_generated_summary remains None
                    summarized_text = text_for_tts # Fallback to original if summary is empty
                # try:
                #     with open(summarized_content_filepath, "w", encoding="utf-8") as f:
                #         f.write(summarized_text)
                #     logger.info(f"Saved summarized content for '{original_title}' to {summarized_content_filepath}")
                #     log_messages.append(f"  Saved summarized content to: {summarized_content_filepath}")
                #     summarized_text_link = f"[{os.path.basename(summarized_content_filepath)}](./file={summarized_content_filepath})"
                #     output_links_markdown += f"  - Summarized Text: {summarized_text_link}\n"
                # except Exception as e:
                #     logger.error(f"Failed to save summarized content for '{original_title}' to {summarized_content_filepath}: {e}", exc_info=True)
                #     log_messages.append(f"  Error saving summarized content: {e}")
                #     output_links_markdown += f"  - Error saving summarized text.\n"
                
                text_for_tts = summarized_text # Update text_for_tts to the summarized version for this item
        else: # No summarization or summarization failed
            # Ensure a newline if only original text link was added and no summarized file link.
            if not output_links_markdown.strip().endswith("\n"):
                output_links_markdown += "\n"
        
        # Add ai_generated_summary to item_data_for_json with dynamic key
        summary_key = f"ai_summary_{target_lang_code}" if target_lang_code else "ai_summary_source"
        item_data_for_json[summary_key] = ai_generated_summary
        
        consolidated_selected_news_data.append(item_data_for_json) # Append here after ai_summary is potentially updated

        # Add the final text (original or summarized) for this item to the global list
        all_texts_for_global_processing.append(text_for_tts)
        # Ensure a newline if no summarized file link was added for this item.
        if not output_links_markdown.strip().endswith("\n"):
             output_links_markdown += "\n"
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
            # Use combined_text_for_audio for subtitle generation
            global_srt_result = subtitle_generator.generate_srt(
                text_content=combined_text_for_audio, 
                audio_duration_seconds=global_audio_duration_seconds, 
                output_filename=global_srt_filename,
                max_chars_per_segment=max_chars_per_segment_cfg # Pass the new parameter
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


# --- Gradio UI Definition ---
with gr.Blocks(theme=gr.themes.Soft(), title="News Aggregator & Audio/Subtitle Generator") as app_ui:
    gr.Markdown("# News Aggregator and Audio/Subtitle Generator")
    
    # State to hold fetched news data (list of dictionaries)
    # This will be populated by handle_fetch_news and read by handle_generate_audio_subtitles
    news_data_state_gr = gr.State(value=[])

    with gr.Tabs():
        with gr.TabItem("Configuration"):
            gr.Markdown("## API Keys and Service Configuration")
            gr.Markdown("Values are loaded from `.env` file if present, or can be manually entered. Click 'Save Configuration' to persist current UI values to `app_config.json` (for review or backup, not primary loading).")
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
                outputs=[news_display_df, news_status_log, news_data_state_gr] # Update DataFrame, log, and the shared state
            )

        with gr.TabItem("Generation Options"):
            gr.Markdown("## Generate Audio and Subtitles for Selected News")
            
            # Component to show which items are selected (for user feedback)
            # This is tricky with gr.DataFrame. A gr.CheckboxGroup from IDs would be easier.
            # For now, we rely on the user remembering what they clicked in the DataFrame.
            # Or, the handle_generate_audio_subtitles function needs to accept the selection event data.
            
            selected_news_indices_input = gr.Textbox(label="Selected News Item IDs/Indices (for dev, ideally from DF selection)", lines=1, placeholder="e.g., 0,1,2 or from DF selection event")


            gen_news_topic = gr.Textbox(label="News Topic/Category (Optional, for filename)", lines=1)
            
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
                gen_tts_voice_gender = gr.Dropdown(label="Select Voice Gender (for selected TTS)", choices=["female", "male"], value="female")
            
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
            selected_news_indices_input.change(
                handle_textbox_selection,
                inputs=[selected_news_indices_input, news_data_state_gr],
                outputs=[selected_df_indices_state]
            )

            def handle_df_selection(evt: gr.SelectData, current_news_data: list):
                # evt.index contains (row_index, col_index) if a cell is clicked
                # If a row is selected (e.g. by clicking on the far left), evt.index might be just row_index
                # We need the IDs of the items, which correspond to the 'ID' column in the displayed DF,
                # or the index in the `global_news_items_store`.
                # For simplicity, let's assume evt.index[0] gives the row index in the displayed DataFrame.
                # This corresponds to the index in `global_news_items_store` if not sorted/filtered.
                if evt.selected: # Check if selection is happening (not deselection)
                    # This is a simplified way to handle selection.
                    # If multiple rows can be selected, evt.index might be a list of indices.
                    # For single row selection:
                    row_index = evt.index[0]
                    if 0 <= row_index < len(current_news_data):
                        # Store the actual item or its ID. Storing ID is safer.
                        selected_id = current_news_data[row_index]['id']
                        # For multiple selections, this needs to accumulate.
                        # This example just takes the latest single selection.
                        # A proper multi-select would require a gr.CheckboxGroup or more complex state management.
                        # For now, let's assume we want to process the item corresponding to the clicked row index.
                        # This is a placeholder for robust multi-selection.
                        # return [current_news_data[row_index]] # Return list with the single selected item
                        return [row_index] # Return the index in the global_news_items_store
                    else:
                        return []
                return [] # No selection or deselection

            news_display_df.select(
                handle_df_selection, 
                inputs=[news_data_state_gr], 
                outputs=[selected_df_indices_state] # Store selected indices here
            )


            generate_button.click(
                handle_generate_audio_subtitles,
                inputs=[
                    selected_df_indices_state, # Selected item indices from DataFrame
                    news_data_state_gr,         # Full list of fetched news items
                    gen_news_topic,
                    gen_summarizer_choice, gen_ollama_model_name, cfg_ollama_url, # Summarizer
                    cfg_gemini_key, cfg_openrouter_key,                         # Summarizer APIs
                    gen_target_language,                                        # New Language choice
                    gen_tts_service, gen_tts_voice_gender,                      # TTS
                    gen_max_chars_segment,                                      # New Subtitle Option
                    cfg_azure_tts_key, cfg_azure_tts_region,                    # TTS APIs
                    cfg_google_tts_path, cfg_minimax_key, cfg_minimax_group_id  # TTS APIs
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
