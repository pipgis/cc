import logging
import requests
import json

logger = logging.getLogger(__name__)

def translate_text(text: str, target_lang_code: str, source_lang_code: str = "auto", 
                   ollama_model: str = "llama3", 
                   ollama_api_url: str = "http://localhost:11434") -> dict:
    """
    Translates text using a local Ollama API.

    Args:
        text (str): The text to translate.
        target_lang_code (str): The target language code (e.g., "en", "zh", "es", "fr").
        source_lang_code (str, optional): The source language code. Defaults to "auto".
                                         If "auto", the prompt will reflect this.
        ollama_model (str, optional): The Ollama model to use for translation.
        ollama_api_url (str, optional): The base URL for the Ollama API.

    Returns:
        dict: A dictionary containing 'translated_text' and 'error'.
              e.g., {'translated_text': '...', 'error': None} or
              {'translated_text': None, 'error': 'Translation failed'}
    """
    logger.info(f"Attempting to translate text to target language: {target_lang_code} using Ollama model: {ollama_model}.")
    logger.debug(f"Received text for translation (first 100 chars): {text[:100]}")
    logger.debug(f"Target language: {target_lang_code}, Source language: {source_lang_code}, Model: {ollama_model}, URL: {ollama_api_url}")

    if not text:
        return {'translated_text': "", 'error': None}

    if not ollama_api_url:
        return {'translated_text': None, 'error': "Ollama API URL not provided."}
    if not ollama_model:
        return {'translated_text': None, 'error': "Ollama model not specified."}

    # Ensure the URL has a scheme
    if not ollama_api_url.startswith(('http://', 'https://')):
        ollama_api_url = 'http://' + ollama_api_url

    # Construct the prompt
    source_lang_display = "the source language (auto-detected if not specified)"
    if source_lang_code and source_lang_code.lower() != "auto":
        source_lang_display = f"'{source_lang_code}'"
    
    # Basic mapping for target language display (can be expanded)
    lang_map = {"en": "English", "zh": "Chinese", "es": "Spanish", "fr": "French"}
    target_lang_display = lang_map.get(target_lang_code.lower(), f"'{target_lang_code}'")

    prompt = f"Translate the following text from {source_lang_display} to {target_lang_display}. Output ONLY the translated text and nothing else:\n\n{text}\n\nTranslated text:"

    payload = {
        "model": ollama_model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": { # Adding some options to potentially improve consistency for translation
            "temperature": 0.2, # Lower temperature for more deterministic output
        }
    }
    
    chat_api_url = f"{ollama_api_url.rstrip('/')}/api/chat"
    generate_api_url = f"{ollama_api_url.rstrip('/')}/api/generate" # Fallback
    
    logger.debug(f"Attempting Ollama translation with model '{ollama_model}' via /api/chat at {chat_api_url}")

    try:
        response = requests.post(chat_api_url, json=payload, timeout=60)
        response.raise_for_status()
        response_data = response.json()
        
        if 'message' in response_data and 'content' in response_data['message']:
            translated_text = response_data['message']['content'].strip()
            logger.info(f"Ollama /api/chat translation successful for model '{ollama_model}'.")
            return {'translated_text': translated_text, 'error': None}
        elif 'response' in response_data: # Fallback for generate-like response from chat endpoint
            translated_text = response_data['response'].strip()
            logger.info(f"Ollama /api/chat (generate-like response) translation successful for model '{ollama_model}'.")
            return {'translated_text': translated_text, 'error': None}
        else:
            logger.error(f"Unexpected response structure from Ollama /api/chat for model '{ollama_model}': {response_data}")
            return {'translated_text': None, 'error': f"Unexpected response structure from Ollama /api/chat: {response_data}"}

    except requests.exceptions.RequestException as e_chat:
        logger.warning(f"Ollama /api/chat failed for model '{ollama_model}': {e_chat}. Trying /api/generate...")
        payload_generate = {
            "model": ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": payload["options"] # Carry over options
        }
        logger.debug(f"Attempting Ollama /api/generate for model '{ollama_model}' at {generate_api_url}")
        try:
            response = requests.post(generate_api_url, json=payload_generate, timeout=60)
            response.raise_for_status()
            response_data = response.json()
            if 'response' in response_data:
                translated_text = response_data['response'].strip()
                logger.info(f"Ollama /api/generate translation successful for model '{ollama_model}'.")
                return {'translated_text': translated_text, 'error': None}
            else:
                logger.error(f"Unexpected response structure from Ollama /api/generate for model '{ollama_model}': {response_data}")
                return {'translated_text': None, 'error': f"Unexpected response structure from Ollama /api/generate: {response_data}"}
        except requests.exceptions.RequestException as e_generate:
            logger.error(f"Ollama API request failed for both /api/chat and /api/generate for model '{ollama_model}'. Chat error: {e_chat}. Generate error: {e_generate}", exc_info=True)
            return {'translated_text': None, 'error': f"Ollama API request failed for both /api/chat and /api/generate. Chat error: {e_chat}. Generate error: {e_generate}"}
    except json.JSONDecodeError as e_json:
        logger.error(f"Failed to decode Ollama API response as JSON from {chat_api_url if 'e_chat' in locals() else generate_api_url}.", exc_info=True)
        return {'translated_text': None, 'error': "Failed to decode Ollama API response as JSON."}
    except Exception as e: # Catch-all for other unexpected errors
        logger.error(f"An unexpected error occurred during Ollama translation: {e}", exc_info=True)
        return {'translated_text': None, 'error': f"An unexpected error occurred: {e}"}


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logger.info("NOTE: To run these tests effectively, ensure an Ollama instance is running with the specified model (e.g., 'llama3').")
    logger.info("Default Ollama API URL: http://localhost:11434")

    sample_text_en = "Hello, world! This is a test of the translation module."
    sample_text_zh_src = "你好，世界！这是一个翻译模块的测试。"
    ollama_test_model = "llama3" # Or your preferred model for testing
    ollama_test_url = "http://localhost:11434"

    # Test 1: English to Chinese
    logger.info("\n--- Test 1: English to Chinese (Ollama) ---")
    result1 = translate_text(sample_text_en, target_lang_code="zh", source_lang_code="en", 
                             ollama_model=ollama_test_model, ollama_api_url=ollama_test_url)
    if result1['error']:
        logger.error(f"Error: {result1['error']}")
    else:
        logger.info(f"Original (EN): {sample_text_en}")
        logger.info(f"Translated (ZH): {result1['translated_text']}")

    # Test 2: Chinese to English (auto-detect source)
    logger.info("\n--- Test 2: Chinese to English (Ollama, source 'auto') ---")
    result2 = translate_text(sample_text_zh_src, target_lang_code="en", source_lang_code="auto",
                             ollama_model=ollama_test_model, ollama_api_url=ollama_test_url)
    if result2['error']:
        logger.error(f"Error: {result2['error']}")
    else:
        logger.info(f"Original (ZH): {sample_text_zh_src}")
        logger.info(f"Translated (EN): {result2['translated_text']}")

    # Test 3: Empty text
    logger.info("\n--- Test 3: Empty text (Ollama) ---")
    result3 = translate_text("", target_lang_code="fr", ollama_model=ollama_test_model, ollama_api_url=ollama_test_url)
    if result3['error']:
        logger.error(f"Error: {result3['error']}")
    else:
        logger.info(f"Original: ''")
        logger.info(f"Translated: '{result3['translated_text']}'")

    # Test 4: English to Spanish
    logger.info("\n--- Test 4: English to Spanish (Ollama) ---")
    result4 = translate_text(sample_text_en, target_lang_code="es", source_lang_code="en",
                             ollama_model=ollama_test_model, ollama_api_url=ollama_test_url)
    if result4['error']:
        logger.error(f"Error: {result4['error']}")
    else:
        logger.info(f"Original (EN): {sample_text_en}")
        logger.info(f"Translated (ES): {result4['translated_text']}")
