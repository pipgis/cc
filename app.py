import gradio as gr
import os
import json
import logging
import re # Import re module
import copy # ADDED: For deepcopy
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


def handle_stage_selected_news(selected_indices, all_news_items):
    """
    Handles staging selected news items for generation.
    Creates deep copies of selected items and prepares them for display in the staging DataFrame.
    """
    logger.debug(f"handle_stage_selected_news: received selected_indices={selected_indices}")
    
    if not selected_indices:
        logger.info("No news items selected to stage.")
        return [], pd.DataFrame(columns=["ID", "Title", "Summary"]) 

    staged_items_copies = []
    for index in selected_indices:
        if isinstance(index, int) and 0 <= index < len(all_news_items):
            original_item = all_news_items[index]
            copied_item = copy.deepcopy(original_item)
            staged_items_copies.append(copied_item)
            logger.debug(f"Staging item (ID: {copied_item['id']}, Title: {copied_item['title']})")
        else:
            logger.warning(f"Invalid index {index} encountered while staging news items.")
            
    if not staged_items_copies:
        logger.info("No valid news items were staged.")
        return [], pd.DataFrame(columns=["ID", "Title", "Summary"])

    display_data_for_df = [
        [item['id'], item['title'], item['_full_summary']] for item in staged_items_copies
    ]
    staged_df = pd.DataFrame(display_data_for_df, columns=["ID", "Title", "Summary"])

    logger.info(f"Staged {len(staged_items_copies)} news items for generation.")
    return staged_items_copies, staged_df


def handle_edit_staged_news(edited_staged_df: pd.DataFrame, current_staged_state_list: list):
    """
    Handles edits made to the staged_news_df.
    Updates the staged_news_data_state with the changes.
    """
    logger.debug(f"handle_edit_staged_news: received edited DataFrame with {edited_staged_df.shape[0]} rows.")
    
    if current_staged_state_list is None:
        current_staged_state_list = []

    edited_map = {}
    for _index, row in edited_staged_df.iterrows():
        try:
            item_id = int(row['ID']) 
            edited_map[item_id] = {'title': row['Title'], 'summary': row['Summary']}
        except ValueError:
            logger.error(f"Could not parse ID {row['ID']} as integer during edit handling. Skipping row.")
            continue
        except KeyError as e:
            logger.error(f"Missing expected column {e} in edited DataFrame row. Skipping row: {row}")
            continue

    updated_staged_items_state = []
    items_to_process = copy.deepcopy(current_staged_state_list)

    for item_dict in items_to_process:
        item_id = item_dict.get('id')
        
        if item_id in edited_map:
            edited_content = edited_map[item_id]
            
            if item_dict.get('title') != edited_content['title']:
                logger.info(f"Updating title for staged item ID {item_id}: '{item_dict.get('title')}' -> '{edited_content['title']}'")
                item_dict['title'] = edited_content['title']
                if '_original_title' in item_dict:
                     item_dict['_original_title'] = edited_content['title']

            if item_dict.get('_full_summary') != edited_content['summary']:
                logger.info(f"Updating summary for staged item ID {item_id}.")
                item_dict['_full_summary'] = edited_content['summary']
        
        updated_staged_items_state.append(item_dict)

    logger.debug(f"handle_edit_staged_news: Updated staged_news_data_state: {updated_staged_items_state}")
    return updated_staged_items_state

def handle_staged_df_selection(evt: gr.SelectData):
    logger.debug(f"handle_staged_df_selection: evt.index={evt.index}")
    if isinstance(evt.index, list):
        return evt.index
    elif isinstance(evt.index, tuple) and len(evt.index) == 2: # cell click
        return [evt.index[0]] # return row index as a list
    logger.debug("handle_staged_df_selection: No valid indices returned from event.")
    return []

def handle_remove_staged_items(indices_to_remove, current_staged_items_data):
    """
    Removes selected items from the staged_news_data_state and updates the staged_news_df.
    """
    if not indices_to_remove or current_staged_items_data is None: 
        logger.info("No indices to remove or no data in staging area.")
        current_display_data = [[item['id'], item['title'], item['_full_summary']] for item in current_staged_items_data if item] if current_staged_items_data else []
        current_df = pd.DataFrame(current_display_data, columns=["ID", "Title", "Summary"])
        return current_staged_items_data if current_staged_items_data is not None else [], current_df

    logger.info(f"Attempting to remove items at visual indices: {indices_to_remove} from staged data (current size: {len(current_staged_items_data)}).")

    valid_indices_to_remove = sorted([idx for idx in indices_to_remove if isinstance(idx, int)], reverse=True)
    
    if not valid_indices_to_remove:
        logger.info("No valid integer indices provided for removal.")
        current_display_data = [[item['id'], item['title'], item['_full_summary']] for item in current_staged_items_data if item]
        current_df = pd.DataFrame(current_display_data, columns=["ID", "Title", "Summary"])
        return current_staged_items_data, current_df

    modified_staged_items_data = copy.deepcopy(current_staged_items_data)
    
    removed_count = 0
    for index in valid_indices_to_remove:
        if 0 <= index < len(modified_staged_items_data):
            removed_item = modified_staged_items_data.pop(index)
            logger.info(f"Removed item: ID {removed_item.get('id')}, Title '{removed_item.get('title')}' at visual index {index}.")
            removed_count += 1
        else:
            logger.warning(f"Index {index} is out of bounds for removal. List size: {len(modified_staged_items_data)}")

    if removed_count == 0:
        logger.info("No items were actually removed based on provided indices.")
        current_display_data = [[item['id'], item['title'], item['_full_summary']] for item in current_staged_items_data if item]
        current_df = pd.DataFrame(current_display_data, columns=["ID", "Title", "Summary"])
        return current_staged_items_data, current_df

    if not modified_staged_items_data:
        new_staged_df = pd.DataFrame(columns=["ID", "Title", "Summary"])
    else:
        new_display_data = [[item['id'], item['title'], item['_full_summary']] for item in modified_staged_items_data if item]
        new_staged_df = pd.DataFrame(new_display_data, columns=["ID", "Title", "Summary"])
        
    logger.info(f"Successfully removed {removed_count} items. Staged data updated. New size: {len(modified_staged_items_data)}")
    return modified_staged_items_data, new_staged_df


def handle_generate_audio_subtitles(
    news_data_state,  # This is `staged_news_data_state`
    news_topic,
    summarizer_choice, ollama_model_name, ollama_api_url_cfg,
    gemini_api_key_cfg, openrouter_api_key_cfg,
    tts_service, tts_voice_gender,
    max_chars_per_segment_cfg, 
    azure_tts_key_cfg, azure_tts_region_cfg,
    google_tts_path_cfg, minimax_tts_key_cfg, minimax_tts_group_id_cfg,
    target_language_choice 
):
    """
    Generates audio and subtitles for selected news items from the STAGING AREA.
    """
    # The 'selected_indices' parameter was removed as we process all items in news_data_state (staged_news_data_state)
    logger.debug(f"handle_generate_audio_subtitles: using news_data_state (staged items) directly. Number of items: {len(news_data_state) if news_data_state else 0}")
    log_messages = [] 
    output_links_markdown = ""
    logger.info("Starting generation process for staged items...")
    log_messages.append("Starting generation process for staged items...")

    all_texts_for_global_processing = []
    consolidated_selected_news_data = [] 
    combined_text_for_audio = ""

    actual_selected_items = news_data_state # Use all items from the staged state
    
    if not actual_selected_items:
        msg = "No news items in the staging area for generation."
        logger.warning(msg)
        log_messages.append(msg)
        return "\n".join(log_messages), ""

    msg = f"Processing {len(actual_selected_items)} staged news item(s)."
    logger.info(msg)
    log_messages.append(msg)

    for item in actual_selected_items:
        original_title = item.get('title', item.get('_original_title', 'Untitled')) # Use 'title' first, then '_original_title'
        full_summary = item.get('_full_summary', '')
        
        original_text_for_processing = original_title + ". " + full_summary
        item_log_prefix = f"Processing Item: {original_title}"
        logger.info(item_log_prefix)
        log_messages.append(f"\n{item_log_prefix}")
        output_links_markdown += f"**{original_title}**:\n"

        english_summary_content = None
        text_for_translation_or_tts = original_text_for_processing

        if summarizer_choice != "None" and original_text_for_processing.strip():
            msg = f"  Attempting summarization of content with {summarizer_choice}..."
            logger.info(msg)
            log_messages.append(msg)
            summary_result = summarizer.summarize_text(
                text_to_summarize=original_text_for_processing,
                service=summarizer_choice,
                api_key=(gemini_api_key_cfg if summarizer_choice == "gemini" else openrouter_api_key_cfg if summarizer_choice == "openrouter" else None),
                ollama_model=ollama_model_name, 
                ollama_api_url=ollama_api_url_cfg,
                target_language=None 
            )
            if summary_result['error']:
                msg = f"  Summarization Error: {summary_result['error']}"
                logger.error(msg)
                log_messages.append(msg)
            elif summary_result['summary'] and summary_result['summary'].strip():
                english_summary_content = summary_result['summary']
                text_for_translation_or_tts = english_summary_content
                msg = f"  Summarization successful. New length: {len(english_summary_content)}"
                logger.info(msg)
                log_messages.append(msg)
            else:
                msg = "  Summarization resulted in empty text. Using original text."
                logger.warning(msg)
                log_messages.append(msg)
        else:
            logger.info("  Summarization skipped or input was empty.")
            log_messages.append("  Summarization skipped or input was empty.")
            if not output_links_markdown.strip().endswith("\n"): 
                output_links_markdown += "\n"

        translated_text_content = None 
        final_text_for_tts = text_for_translation_or_tts

        target_lang_code_for_processing = None
        if target_language_choice and target_language_choice != "As Source (No Translation)":
            match = re.search(r'\((.*?)\)', target_language_choice)
            if match:
                target_lang_code_for_processing = match.group(1)
                logger.info(f"  Target language for translation selected: {target_lang_code_for_processing}")
            else:
                logger.warning(f"  Could not parse language code from: {target_language_choice} for translation.")
        
        if target_lang_code_for_processing and text_for_translation_or_tts.strip():
            msg = f"  Attempting translation to {target_lang_code_for_processing}..."
            logger.info(msg)
            log_messages.append(msg)
            
            translation_result = translator.translate_text(
                text=text_for_translation_or_tts,
                target_lang_code=target_lang_code_for_processing,
                source_lang_code="en", 
                ollama_model=ollama_model_name,
                ollama_api_url=ollama_api_url_cfg
            )
            if translation_result['error']:
                msg = f"  Translation Error to {target_lang_code_for_processing}: {translation_result['error']}"
                logger.error(msg)
                log_messages.append(msg)
            elif translation_result['translated_text'] and translation_result['translated_text'].strip():
                translated_text_content = translation_result['translated_text']
                final_text_for_tts = translated_text_content
                msg = f"  Translation to {target_lang_code_for_processing} successful. New length: {len(translated_text_content)}"
                logger.info(msg)
                log_messages.append(msg)
            else:
                msg = f"  Translation to {target_lang_code_for_processing} resulted in empty text. Using previous text for TTS."
                logger.warning(msg)
                log_messages.append(msg)
        elif target_lang_code_for_processing and not text_for_translation_or_tts.strip():
            msg = f"  Skipping translation to {target_lang_code_for_processing} as input text is empty."
            logger.info(msg)
            log_messages.append(msg)
        else:
            logger.info("  No translation selected or input text was empty. TTS will use previous content.")
            log_messages.append("  Translation skipped.")
        
        item_data_for_json = {
            'id': item.get('id'),
            'original_title': item.get('_original_title', 'Untitled'), 
            'editable_title': item.get('title', 'Untitled'), # The title that might have been edited
            'full_summary_original_language': item.get('_full_summary', ''), # The summary that might have been edited
            'source_url': item.get('source_url', 'N/A'),
            'published_date': item.get('published_date', 'N/A'),
            'ai_summary_english': english_summary_content 
        }
        if target_lang_code_for_processing:
            item_data_for_json[f"content_for_tts_{target_lang_code_for_processing}"] = translated_text_content
        else:
             item_data_for_json[f"content_for_tts_english"] = final_text_for_tts

        consolidated_selected_news_data.append(item_data_for_json)
        all_texts_for_global_processing.append(final_text_for_tts)
        
        if not output_links_markdown.strip().endswith("\n"):
             output_links_markdown += "\n"
        output_links_markdown += "\n"

    if consolidated_selected_news_data:
        json_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        json_filename = f"consolidated_staged_news_{json_timestamp}.json"
        json_filepath = os.path.join(OUTPUT_DIR, json_filename)
        try:
            with open(json_filepath, "w", encoding="utf-8") as f:
                json.dump(consolidated_selected_news_data, f, ensure_ascii=False, indent=4)
            logger.info(f"Successfully saved consolidated staged news data to {json_filepath}")
            log_messages.append(f"\nSaved consolidated staged news data to: {json_filepath}")
            json_file_link = f"[{json_filename}](./file={json_filepath})"
            output_links_markdown = f"**Consolidated Staged News Data**:\n  - JSON File: {json_file_link}\n\n" + output_links_markdown
        except Exception as e:
            logger.error(f"Failed to save consolidated staged news data to {json_filepath}: {e}", exc_info=True)
            log_messages.append(f"\nError saving consolidated staged news data: {e}")
            output_links_markdown = f"**Consolidated Staged News Data**:\n  - Error saving JSON file.\n\n" + output_links_markdown
    else:
        log_messages.append("\nNo staged items to save to consolidated JSON.")

    if not all_texts_for_global_processing:
        msg = "No text collected from staged items for global audio generation. Aborting."
        logger.warning(msg)
        log_messages.append(msg)
        final_msg = "\nStaged item processing completed. No content for global audio."
        log_messages.append(final_msg)
        logger.info(final_msg)
        return "\n".join(log_messages), output_links_markdown.strip() + "\n\n**Combined Output**:\n  - No content for global audio."

    combined_text_for_audio = "\n\n".join(filter(None, all_texts_for_global_processing)) # Filter out None or empty strings
    if not combined_text_for_audio.strip():
        msg = "Combined text for TTS is empty after processing all staged items. Aborting audio generation."
        logger.warning(msg)
        log_messages.append(msg)
        return "\n".join(log_messages), output_links_markdown

    msg = f"Combined text from {len(all_texts_for_global_processing)} staged items for global audio. Total length: {len(combined_text_for_audio)}"
    logger.info(msg)
    log_messages.append(f"\n{msg}")

    timestamp_global = datetime.now().strftime("%Y%m%d%H%M%S")
    topic_prefix_global = f"{news_topic.replace(' ', '_')}_" if news_topic and news_topic.strip() else ""
    base_filename_global = f"{timestamp_global}_{topic_prefix_global}staged_combined_audio"

    global_mp3_filename = os.path.join(OUTPUT_DIR, f"{base_filename_global}.mp3")
    global_srt_filename = os.path.join(OUTPUT_DIR, f"{base_filename_global}.srt")
    global_lrc_filename = os.path.join(OUTPUT_DIR, f"{base_filename_global}.lrc")

    msg = f"Attempting global audio generation with {tts_service} ({tts_voice_gender})..."
    logger.info(msg)
    log_messages.append(msg)

    azure_api_key_to_pass = azure_tts_key_cfg if tts_service == "azure" else None
    
    global_tts_lang_code = None
    if target_language_choice and target_language_choice != "As Source (No Translation)":
        match = re.search(r'\((.*?)\)', target_language_choice)
        if match:
            global_tts_lang_code = match.group(1)
    
    language_code_for_tts = global_tts_lang_code if global_tts_lang_code else "en"
    logger.info(f"Global TTS will use language: {language_code_for_tts}.")

    global_tts_result = tts_generator.generate_audio(
        text_to_speak=combined_text_for_audio,
        output_filename=global_mp3_filename,
        service=tts_service,
        voice_gender=tts_voice_gender,
        language_code=language_code_for_tts,
        api_key=azure_api_key_to_pass, 
        azure_region=azure_tts_region_cfg,
        google_credentials_path=google_tts_path_cfg,
        minimax_api_key=minimax_tts_key_cfg,
        minimax_group_id=minimax_tts_group_id_cfg
    )

    output_links_markdown += "**Combined Output (from Staged Items)**:\n"

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

            try:
                max_chars_for_srt = int(max_chars_per_segment_cfg)
            except ValueError:
                logger.error(f"Could not convert max_chars_per_segment_cfg '{max_chars_per_segment_cfg}' to int. Defaulting to 50 for SRT.")
                max_chars_for_srt = 50
            
            global_srt_result = subtitle_generator.generate_srt(
                text_content=combined_text_for_audio, 
                audio_duration_seconds=global_audio_duration_seconds, 
                output_filename=global_srt_filename,
                max_chars_per_segment=max_chars_for_srt 
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
    voice_choices = ["female", "male"] 
    selected_voice = "female"      

    if tts_service == "minimax":
        lang_code = "en" 
        if target_language_str and "(" in target_language_str and ")" in target_language_str:
            match = re.search(r'\((.*?)\)', target_language_str)
            if match:
                parsed_code = match.group(1)
                lang_code = parsed_code
            else: 
                logger.warning(f"Could not parse language code from '{target_language_str}', defaulting to 'en' for Minimax voices.")
        else: 
            logger.info(f"Target language string '{target_language_str}' not in expected format for Minimax voice selection, defaulting to 'en'.")

        voices_for_lang = MINIMAX_TTS_VOICE_MAPPING.get(lang_code, {})
        
        if voices_for_lang: 
            voice_choices = list(voices_for_lang.keys())
            if voice_choices:
                selected_voice = voice_choices[0]
            else: 
                voice_choices = [] 
                selected_voice = None 
        else: 
            if lang_code not in MINIMAX_TTS_VOICE_MAPPING:
                 pass
            else: 
                voice_choices = []
                selected_voice = None
    return gr.update(choices=voice_choices, value=selected_voice)


def handle_df_selection(evt: gr.SelectData, current_news_data: list):
    logger.debug(f"handle_df_selection: evt.index={evt.index}, evt.selected={evt.selected}, evt.value={evt.value}")
    
    selected_row_indices = []
    if evt.index is None:
        return []

    if isinstance(evt.index, list): 
        selected_row_indices = evt.index
    elif isinstance(evt.index, tuple) and len(evt.index) == 2: 
        selected_row_indices = [evt.index[0]] 
    else:
        logger.warning(f"handle_df_selection: Unexpected evt.index format: {evt.index}")
        return []

    valid_indices = []
    for idx in selected_row_indices:
        if isinstance(idx, int) and 0 <= idx < len(current_news_data):
            valid_indices.append(idx)
        else:
            logger.warning(f"Warning: Invalid or out-of-range index {idx} for current_news_data size {len(current_news_data)}.")
            
    logger.debug(f"handle_df_selection: final_selected_indices={valid_indices}")
    return valid_indices


# --- Gradio UI Definition ---
with gr.Blocks(theme=gr.themes.Soft(), title="News Aggregator & Audio/Subtitle Generator") as app_ui:
    gr.Markdown("# News Aggregator and Audio/Subtitle Generator")
    
    news_data_state_gr = gr.State(value=[])
    staged_news_data_state = gr.State([]) 
    staged_df_selected_indices_state = gr.State([])


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
            
            gr.Markdown("API keys and other configurations are primarily managed via the `.env` file or environment variables. Changes made here are for the current session only unless your environment variables are updated externally.")


        with gr.TabItem("News Fetching & Processing"):
            gr.Markdown("## Fetch News from URLs or RSS Feeds")
            news_urls_input = gr.Textbox(label="Enter URLs/RSS Feeds (one per line)", lines=5, placeholder="http://example.com/feed.xml\nhttps://another-news-site.com/article")
            fetch_news_button = gr.Button("Fetch News")
            
            news_status_log = gr.Textbox(label="Fetching Status/Log", lines=3, interactive=False)
            
            gr.Markdown("### Fetched News Items (Select rows to process)")
            news_display_df = gr.DataFrame(
                headers=["ID", "Title", "Summary Snippet", "Source", "Date"], 
                interactive=True, 
                label="Fetched News"
            )
            stage_button = gr.Button("Stage Selected News for Generation") 


        with gr.TabItem("Generation Options"):
            gr.Markdown("## Generate Audio and Subtitles for Selected News")
            
            gr.Markdown("### News Items Staged for Generation") 
            staged_news_df = gr.DataFrame( 
                headers=["ID", "Title", "Summary"],
                label="Staged News Items",
                interactive=True 
            )
            remove_staged_item_button = gr.Button("Remove Selected Rows from Staging Area")

            gen_news_topic = gr.Textbox(label="News Topic/Category (Optional, for filename)", lines=1)
            
            with gr.Row():
                gen_summarizer_choice = gr.Dropdown(label="Select Summarizer", choices=["None", "ollama", "gemini", "openrouter"], value="None")
                gen_ollama_model_name = gr.Textbox(label="Ollama Model Name (if Ollama selected)", placeholder="e.g., llama3", lines=1, visible=False) 
            
            def toggle_ollama_model_visibility(summarizer_service):
                return gr.update(visible=(summarizer_service == "ollama"))

            with gr.Row():
                gen_target_language = gr.Dropdown(
                    label="Select Target Language",
                    choices=["As Source (No Translation)", "English (en)", "Chinese (zh)", "Spanish (es)", "French (fr)"],
                    value="As Source (No Translation)"
                )

            with gr.Row():
                gen_tts_service = gr.Dropdown(label="Select TTS Service", choices=["edge_tts", "azure", "google", "minimax"], value="edge_tts")
                gen_tts_voice_gender = gr.Dropdown(label="Select Voice (or Gender for non-Minimax TTS)", choices=["female", "male"], value="female", allow_custom_value=False) 
            
            gr.Markdown("### Subtitle Options")
            gen_max_chars_segment = gr.Slider(label="Max Characters per Subtitle Segment", minimum=20, maximum=150, value=50, step=5, interactive=True)

            generate_button = gr.Button("Generate Audio & Subtitles")
            generation_status_log = gr.Textbox(label="Generation Status/Log", lines=10, interactive=False)
            # selected_df_indices_state = gr.State([]) # REMOVED: This was redundant. The one at the app_ui scope is used.

        with gr.TabItem("Results / Output"):
            gr.Markdown("## Generated Files")
            results_display_markdown = gr.Markdown("Generated file links will appear here.")
            
    # --- Event Wiring ---
    fetch_news_button.click(
        handle_fetch_news,
        inputs=[news_urls_input],
        outputs=[news_display_df, news_status_log, news_data_state_gr] 
    )
    
    gen_summarizer_choice.change(
        toggle_ollama_model_visibility, 
        inputs=[gen_summarizer_choice], 
        outputs=[gen_ollama_model_name]
    )
    
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

    stage_button.click(
        fn=handle_stage_selected_news,
        inputs=[selected_df_indices_state, news_data_state_gr], # Uses selection from main DF
        outputs=[staged_news_data_state, staged_news_df]
    )

    staged_news_df.select( # Selection event for the STAGED DataFrame
        fn=handle_staged_df_selection,
        inputs=None, 
        outputs=[staged_df_selected_indices_state] # Outputs to its own state
    )

    remove_staged_item_button.click(
        fn=handle_remove_staged_items,
        inputs=[staged_df_selected_indices_state, staged_news_data_state],
        outputs=[staged_news_data_state, staged_news_df]
    )

    staged_news_df.input(
        fn=handle_edit_staged_news,
        inputs=[staged_news_df, staged_news_data_state],
        outputs=[staged_news_data_state]
    )
    
    news_display_df.select( # Selection event for the MAIN DataFrame
        fn=handle_df_selection,
        inputs=[news_data_state_gr], 
        outputs=[selected_df_indices_state] # Outputs to the main selection state
    )
    
    generate_button.click(
        handle_generate_audio_subtitles,
        inputs=[
            staged_news_data_state, # This is passed to the 'news_data_state' parameter of the function
            gen_news_topic,
            gen_summarizer_choice, gen_ollama_model_name, cfg_ollama_url, 
            cfg_gemini_key, cfg_openrouter_key,                         
            gen_tts_service, gen_tts_voice_gender,                      
            gen_max_chars_segment,                                      
            cfg_azure_tts_key, cfg_azure_tts_region,                    
            cfg_google_tts_path, cfg_minimax_key, cfg_minimax_group_id, 
            gen_target_language                                         
        ],
        outputs=[generation_status_log, results_display_markdown] 
    )

if __name__ == "__main__":
    app_ui.launch(debug=True, share=False, allowed_paths=[OUTPUT_DIR])
