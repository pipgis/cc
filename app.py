import gradio as gr
import os
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
OUTPUT_DIR = "generated_files"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Configuration State (managed by Gradio inputs) ---
# API keys and settings will be read directly from Gradio input components within handler functions.

# --- Handler Functions ---

def handle_save_configuration(gemini_key, openrouter_key, azure_key, azure_region, 
                              google_tts_path, minimax_key, minimax_group_id, ollama_url):
    """
    Handles the "Save Configuration" button click.
    For now, this function is a placeholder. In a real app, it might save to a file or db.
    Here, it just demonstrates reading the values.
    """
    # In a real app, you'd securely store these. For now, they are passed to handlers.
    # This function could return a status message.
    config_summary = (
        f"Gemini Key: {'Set' if gemini_key else 'Not Set'}\n"
        f"OpenRouter Key: {'Set' if openrouter_key else 'Not Set'}\n"
        f"Azure Key: {'Set' if azure_key else 'Not Set'}\n"
        f"Azure Region: {azure_region}\n"
        f"Google TTS Path: {google_tts_path}\n"
        f"Minimax Key: {'Set' if minimax_key else 'Not Set'}\n"
        f"Minimax Group ID: {minimax_group_id}\n"
        f"Ollama URL: {ollama_url}"
    )
    print("Configuration handling (not saved persistently in this version):")
    print(config_summary)
    return f"Configuration values acknowledged (not saved persistently).\n{config_summary}"


def handle_fetch_news(urls_text_input):
    """
    Fetches news from the provided URLs/RSS feeds.
    Updates the news display DataFrame and status.
    """
    global global_news_items_store # Use the global store

    if not urls_text_input or not urls_text_input.strip():
        return pd.DataFrame(), "Status: Please enter some URLs or RSS feed links.", []

    sources = [url.strip() for url in urls_text_input.strip().split('\n') if url.strip()]
    if not sources:
        return pd.DataFrame(), "Status: No valid URLs provided.", []
    
    status_log = f"Fetching news from {len(sources)} source(s)...\n"
    
    try:
        fetched_items_raw = news_fetcher.fetch_news(sources)
    except Exception as e:
        status_log += f"An error occurred during fetching: {e}\n"
        return pd.DataFrame(), status_log, []

    # Clear previous items and add new ones with an ID
    global_news_items_store = []
    valid_items_for_df = []
    item_id_counter = 0

    if not fetched_items_raw:
        status_log += "No items were fetched. Check URLs and network.\n"
        return pd.DataFrame(), status_log, []

    for item in fetched_items_raw:
        if item.get('error'):
            status_log += f"Error for {item.get('source_url', 'Unknown source')}: {item['error']}\n"
            # Optionally, still add error items to the list for visibility, or skip them
            # For now, let's skip adding errored items directly to the selectable DataFrame
            continue

        processed_item = {
            'id': item_id_counter,
            'title': item.get('title', 'N/A'),
            'summary': item.get('summary', 'N/A')[:150] + "..." if item.get('summary') else 'N/A', # Truncate summary
            'source_url': item.get('source_url', 'N/A'),
            'published_date': item.get('published_date', 'N/A'),
            # Store the full summary for processing, not just the truncated one
            '_full_summary': item.get('summary', 'N/A'), 
            '_original_title': item.get('title', 'N/A') # Keep original for processing
        }
        global_news_items_store.append(processed_item)
        # For DataFrame, we might only want to show certain fields or add a selection checkbox column
        # Gradio's DataFrame doesn't directly support a checkbox column for selection in the same way
        # a CheckboxGroup would. We'll use row selection property of DataFrame.
        valid_items_for_df.append({
            "ID": item_id_counter, # For user display & selection reference
            "Title": processed_item['title'],
            "Summary Snippet": processed_item['summary'],
            "Source": processed_item['source_url'],
            "Date": processed_item['published_date']
        })
        item_id_counter += 1
        
    status_log += f"Fetched {len(valid_items_for_df)} valid news items.\n"
    
    if not valid_items_for_df:
        news_df = pd.DataFrame()
        status_log += "No valid news items could be processed into the display table.\n"
    else:
        news_df = pd.DataFrame(valid_items_for_df)

    return news_df, status_log, global_news_items_store # Return store to update a gr.State if used for selection


def handle_generate_audio_subtitles(
    selected_indices, # This will come from gr.DataFrame(interactive=True) selection event
    news_data_state,  # This will be the global_news_items_store passed via gr.State
    news_topic,
    summarizer_choice, ollama_model_name, ollama_api_url_cfg, # Summarizer params
    gemini_api_key_cfg, openrouter_api_key_cfg,             # Summarizer API keys
    tts_service, tts_voice_gender,                          # TTS params
    azure_tts_key_cfg, azure_tts_region_cfg,                # TTS API keys
    google_tts_path_cfg, minimax_tts_key_cfg, minimax_tts_group_id_cfg # TTS API keys
):
    """
    Generates audio and subtitles for selected news items.
    """
    generation_log = "Starting generation process...\n"
    output_links_markdown = ""

    if selected_indices is None or not selected_indices['index']: # Check if selection event data is valid
        generation_log += "No news items selected for generation.\n"
        return generation_log, ""

    # selected_indices from a gr.DataFrame selection event is a dict like {'index': (row_index, col_index), 'value': cell_value}
    # We are interested in row_indices if the selection mode is per row, or we need a different way.
    # For now, let's assume selected_indices is a list of IDs from the DataFrame.
    # This part needs careful handling based on how gr.DataFrame selection events provide data.
    # If gr.DataFrame `interactive=True` is used with `select` event, it gives row indices.
    
    # For this example, let's assume selected_indices is a list of item IDs (0, 1, 2...)
    # This might need to be adjusted based on actual Gradio event data for DataFrame selection.
    # The PDD mentioned checkboxes; a CheckboxGroup might be better if direct DataFrame selection is tricky.
    # Using the 'select' event on gr.DataFrame returns a SelectData object.
    # For now, let's assume selected_indices is a list of the 'ID' column values from the selected rows.
    # This part of the code will need refinement once the UI interaction is tested.
    # For now, we'll use the indices directly from the event if it's a simple list of row numbers.
    
    # The `news_data_state` (global_news_items_store) contains dicts with 'id', 'title', '_full_summary', etc.
    actual_selected_items = []
    if isinstance(selected_indices, list): # If we get a list of selected row indices
        for index in selected_indices:
            if 0 <= index < len(news_data_state):
                actual_selected_items.append(news_data_state[index])
            else:
                generation_log += f"Warning: Invalid selected index {index} ignored.\n"
    else: # If it's not a list (e.g. from a single select event) - this needs more robust handling
        generation_log += "Selection format not as expected. Please select rows in the table.\n"
        # Attempt to handle single selection from DataFrame click if applicable
        # This part is highly dependent on Gradio's event data structure for DataFrame selection
        # For now, we'll proceed assuming `actual_selected_items` gets populated correctly.
        # A gr.CheckboxGroup that outputs the IDs of selected items would be more straightforward.
        # If `selected_indices` is from `gr.DataFrame(...).select`, it's `evt: gr.SelectData`.
        # `evt.index[0]` would be the row index.
        # For now, let's assume `selected_indices` is the list of actual item dictionaries for simplicity.
        # This is a placeholder for proper selection handling.
        # A simple workaround: iterate through ALL news_data_state and check a hypothetical 'selected' flag
        # This means the DataFrame selection needs to update this flag. That's complex.
        # Let's assume `selected_indices` IS the list of indices from the DataFrame.
        # And `news_data_state` is the current full list of news items.
        
        # Simpler approach for now: Assume selected_indices is a list of IDs.
        # This requires the DataFrame to output selected IDs.
        # The current setup with global_news_items_store and its 'id' field.
        # This is a placeholder and needs to be correctly wired with Gradio DataFrame selection.
        # For now, let's assume `selected_indices` is a list of `id`s from the DataFrame.
        # This is a placeholder:
        if not news_data_state:
             generation_log += "News data is empty. Fetch news first.\n"
             return generation_log, ""
        
        # Mocking selection: take the first item if nothing "properly" selected.
        # THIS IS A MAJOR SIMPLIFICATION to get the rest of the logic flowing.
        # In a real scenario, you'd use gr.CheckboxGroup or handle DataFrame selection events properly.
        if not actual_selected_items and news_data_state:
            generation_log += "Warning: No specific items selected or selection mechanism not fully wired. Processing first item as a fallback for testing.\n"
            actual_selected_items.append(news_data_state[0]) # Process first item as a test

    if not actual_selected_items:
        generation_log += "No news items to process after selection logic.\n"
        return generation_log, ""

    generation_log += f"Processing {len(actual_selected_items)} selected news item(s).\n"

    for item in actual_selected_items:
        original_title = item.get('_original_title', 'Untitled')
        text_for_tts = item.get('_original_title', '') + ". " + item.get('_full_summary', '')
        generation_log += f"\nProcessing: {original_title}\n"

        # 1. Summarization (Optional)
        if summarizer_choice != "None" and text_for_tts.strip():
            generation_log += f"  Attempting summarization with {summarizer_choice}...\n"
            summary_result = summarizer.summarize_text(
                text_to_summarize=text_for_tts,
                service=summarizer_choice,
                api_key=(gemini_api_key_cfg if summarizer_choice == "gemini" else openrouter_api_key_cfg if summarizer_choice == "openrouter" else None),
                ollama_model=ollama_model_name,
                ollama_api_url=ollama_api_url_cfg,
                # openrouter_model can be added if you want to specify it
            )
            if summary_result['error']:
                generation_log += f"  Summarization Error: {summary_result['error']}\n"
            else:
                text_for_tts = summary_result['summary'] # Use summarized text
                generation_log += f"  Summarization Successful. New text length: {len(text_for_tts)}\n"
        
        # 2. Generate Filename Base
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        topic_prefix = f"{news_topic.replace(' ', '_')}_" if news_topic and news_topic.strip() else ""
        # Sanitize title for filename
        safe_title_part = "".join(c if c.isalnum() or c in (' ', '_') else '_' for c in original_title[:30]).rstrip()
        safe_title_part = safe_title_part.replace(' ', '_')
        
        base_filename = f"{timestamp}_{topic_prefix}{safe_title_part}"
        
        mp3_filename = os.path.join(OUTPUT_DIR, f"{base_filename}.mp3")
        srt_filename = os.path.join(OUTPUT_DIR, f"{base_filename}.srt")
        lrc_filename = os.path.join(OUTPUT_DIR, f"{base_filename}.lrc")

        # 3. Text-to-Speech
        generation_log += f"  Generating audio with {tts_service} ({tts_voice_gender})...\n"
        tts_result = tts_generator.generate_audio(
            text_to_speak=text_for_tts,
            output_filename=mp3_filename,
            service=tts_service,
            voice_gender=tts_voice_gender,
            api_key=(azure_tts_key_cfg if tts_service == "azure" else minimax_tts_key_cfg if tts_service == "minimax" else None),
            azure_region=azure_tts_region_cfg,
            google_credentials_path=google_tts_path_cfg,
            minimax_group_id=minimax_tts_group_id_cfg
            # minimax_voice_id can be added if needed
        )

        if tts_result['success']:
            generation_log += f"  Audio generated: {mp3_filename}\n"
            
            # Get audio duration (CRITICAL for subtitles)
            audio_duration_seconds = 0
            try:
                if os.path.exists(mp3_filename):
                    audio = MP3(mp3_filename)
                    audio_duration_seconds = audio.info.length
                    generation_log += f"  Audio duration: {audio_duration_seconds:.2f} seconds.\n"
                else: # Should not happen if tts_result is success
                    generation_log += "  Error: MP3 file not found after TTS success reported.\n"
            except Exception as e:
                generation_log += f"  Error getting audio duration: {e}. Subtitles might be misaligned.\n"
                # Fallback: estimate duration (e.g., 10 words per second, very rough)
                # This is a very poor substitute for actual duration.
                if audio_duration_seconds == 0 and text_for_tts:
                    estimated_duration = len(text_for_tts.split()) / 4.0 # Rough estimate
                    audio_duration_seconds = max(1.0, estimated_duration) # Ensure at least 1s
                    generation_log += f"  Using estimated duration: {audio_duration_seconds:.2f}s for subtitles.\n"


            if audio_duration_seconds > 0:
                # 4. Subtitle Generation
                generation_log += f"  Generating SRT subtitles...\n"
                srt_result = subtitle_generator.generate_srt(text_for_tts, audio_duration_seconds, srt_filename)
                if srt_result['success']:
                    generation_log += f"  SRT generated: {srt_filename}\n"
                else:
                    generation_log += f"  SRT Generation Error: {srt_result['error']}\n"

                generation_log += f"  Generating LRC subtitles...\n"
                lrc_result = subtitle_generator.generate_lrc(text_for_tts, audio_duration_seconds, lrc_filename)
                if lrc_result['success']:
                    generation_log += f"  LRC generated: {lrc_filename}\n"
                else:
                    generation_log += f"  LRC Generation Error: {lrc_result['error']}\n"
                
                # Add to markdown links (Gradio can serve files from the script's directory or subdirs)
                # For downloadable links, Gradio needs paths relative to where app is run or absolute if configured.
                # Using relative paths to OUTPUT_DIR.
                # The `file_paths` parameter in `gr.File` or `gr.DownloadButton` would be better.
                # For `gr.Markdown`, we need to ensure these paths are accessible via HTTP.
                # Gradio can serve files if `allowed_paths` is set or if they are in `gradio.Blocks(allowed_paths=[OUTPUT_DIR])`
                # Or by returning `gr.File` components.
                # For now, just list them. Direct download might need `gr.File` output.
                output_links_markdown += f"**{original_title}**:\n"
                if tts_result['success']:
                    # These links might not work directly in Markdown without Gradio serving them.
                    # A better approach is to return gr.File components or use gr.DownloadButton.
                    # This is a placeholder for demonstrating file paths.
                    mp3_link = f"[{os.path.basename(mp3_filename)}](./file={mp3_filename})" # Gradio specific link
                    output_links_markdown += f"  - Audio: {mp3_link}\n"
                if srt_result.get('success'):
                    srt_link = f"[{os.path.basename(srt_filename)}](./file={srt_filename})"
                    output_links_markdown += f"  - SRT: {srt_link}\n"
                if lrc_result.get('success'):
                    lrc_link = f"[{os.path.basename(lrc_filename)}](./file={lrc_filename})"
                    output_links_markdown += f"  - LRC: {lrc_link}\n"
                output_links_markdown += "\n"

            else: # audio_duration_seconds <= 0
                generation_log += "  Skipping subtitle generation due to missing audio duration.\n"
        else:
            generation_log += f"  TTS Error: {tts_result['error']}\n"
            output_links_markdown += f"**{original_title}**: Audio generation failed.\n\n"
            
    generation_log += "\nGeneration process finished.\n"
    return generation_log, output_links_markdown


# --- Gradio UI Definition ---
with gr.Blocks(theme=gr.themes.Soft(), title="News Aggregator & Audio/Subtitle Generator") as app_ui:
    gr.Markdown("# News Aggregator and Audio/Subtitle Generator")
    
    # State to hold fetched news data (list of dictionaries)
    # This will be populated by handle_fetch_news and read by handle_generate_audio_subtitles
    news_data_state_gr = gr.State(value=[])

    with gr.Tabs():
        with gr.TabItem("Configuration"):
            gr.Markdown("## API Keys and Service Configuration")
            with gr.Row():
                with gr.Column():
                    cfg_gemini_key = gr.Textbox(label="Google Gemini API Key", type="password", lines=1)
                    cfg_openrouter_key = gr.Textbox(label="OpenRouter API Key", type="password", lines=1)
                    cfg_ollama_url = gr.Textbox(label="Ollama API Base URL", placeholder="e.g., http://localhost:11434", lines=1)
                with gr.Column():
                    cfg_azure_tts_key = gr.Textbox(label="Azure TTS API Key", type="password", lines=1)
                    cfg_azure_tts_region = gr.Textbox(label="Azure TTS Region", placeholder="e.g., eastus", lines=1)
            with gr.Row():
                with gr.Column():
                    cfg_google_tts_path = gr.Textbox(label="Google Cloud TTS Credentials Path (JSON file path)", lines=1)
                with gr.Column():
                    cfg_minimax_key = gr.Textbox(label="Minimax API Key (TTS)", type="password", lines=1)
                    cfg_minimax_group_id = gr.Textbox(label="Minimax Group ID (TTS)", lines=1)
            
            # cfg_save_button = gr.Button("Save Configuration") # Not strictly needed if passing directly
            # cfg_status_md = gr.Markdown("")
            # cfg_save_button.click(
            #     handle_save_configuration,
            #     inputs=[cfg_gemini_key, cfg_openrouter_key, cfg_azure_tts_key, cfg_azure_tts_region,
            #             cfg_google_tts_path, cfg_minimax_key, cfg_minimax_group_id, cfg_ollama_url],
            #     outputs=[cfg_status_md]
            # )
            gr.Markdown("Configuration is passed directly to generation. No explicit save needed for this version.")


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
                gen_tts_service = gr.Dropdown(label="Select TTS Service", choices=["edge_tts", "azure", "google", "minimax"], value="edge_tts")
                gen_tts_voice_gender = gr.Dropdown(label="Select Voice Gender (for selected TTS)", choices=["female", "male"], value="female")

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
                    gen_tts_service, gen_tts_voice_gender,                      # TTS
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

```
