import logging
from google.cloud import translate_v2 as translate # Import Google Cloud Translate library

logger = logging.getLogger(__name__)

# Placeholder for actual API client or request logic - Now implementing Google Cloud Translate
# For now, this module will simulate translation. - No longer simulating

def translate_text(text: str, target_lang_code: str, source_lang_code: str = "auto", api_key: str = None) -> dict:
    """
    Translates text using the Google Cloud Translation API.

    Args:
        text (str): The text to translate.
        target_lang_code (str): The target language code (e.g., "en", "zh").
        source_lang_code (str, optional): The source language code. Defaults to "auto".
                                         If "auto", Google will attempt to detect the source language.
        api_key (str, optional): API key. Note: Google Cloud client libraries typically use
                                 Application Default Credentials (ADC) or a service account JSON
                                 file specified by GOOGLE_APPLICATION_CREDENTIALS. This parameter
                                 is logged but not directly used for client instantiation if ADC is active.

    Returns:
        dict: A dictionary containing 'translated_text', 'error', and 'detected_source_language'.
              e.g., {'translated_text': '...', 'error': None, 'detected_source_language': 'en'} or
              {'translated_text': None, 'error': 'Translation failed', 'detected_source_language': None}
    """
    logger.info(f"Attempting to translate text to target language: {target_lang_code} using Google Cloud Translation.")
    logger.debug(f"Received text for translation (first 100 chars): {text[:100]}")
    logger.debug(f"Target language: {target_lang_code}, Source language: {source_lang_code}")

    if api_key:
        logger.info("API key provided (note: Google Cloud client library typically uses ADC or GOOGLE_APPLICATION_CREDENTIALS env var).")

    try:
        translate_client = translate.Client()
    except Exception as e:
        logger.error(f"Failed to initialize Google Translate client: {e}", exc_info=True)
        return {'translated_text': None, 'error': f"Failed to initialize Google Translate client: {e}", 'detected_source_language': None}

    if not text:
        return {'translated_text': "", 'error': None, 'detected_source_language': None}

    try:
        source_for_api = source_lang_code if source_lang_code != "auto" else None
            
        result = translate_client.translate(
            text,
            target_language=target_lang_code,
            source_language=source_for_api
        )
        translated_text = result['translatedText']
        detected_source_lang = result.get('detectedSourceLanguage', source_for_api)
            
        logger.info(f"Successfully translated text to {target_lang_code}. Detected source: {detected_source_lang or 'N/A'}.")
        return {'translated_text': translated_text, 'error': None, 'detected_source_language': detected_source_lang}

    except Exception as e:
        logger.error(f"Google Cloud Translation API error: {e}", exc_info=True)
        return {'translated_text': None, 'error': f"Google Cloud Translation API error: {e}", 'detected_source_language': None}


if __name__ == '__main__':
    # Basic setup for testing this module directly
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logger.info("NOTE: To run these tests effectively, ensure Google Cloud authentication is configured (e.g., GOOGLE_APPLICATION_CREDENTIALS environment variable).")

    sample_text_en = "Hello, world! This is a test of the translation module."
    sample_text_zh_src = "你好，世界！这是一个翻译模块的测试。"

    # Test 1: English to Chinese (Real API call if auth is configured)
    logger.info("\n--- Test 1: English to Chinese ---")
    result1 = translate_text(sample_text_en, target_lang_code="zh", source_lang_code="en")
    if result1['error']:
        logger.error(f"Error: {result1['error']}")
    else:
        logger.info(f"Original: {sample_text_en}")
        logger.info(f"Translated: {result1['translated_text']}")
        logger.info(f"Detected Source Language: {result1['detected_source_language']}")

    # Test 2: Chinese to English (Real API call, auto-detect source)
    logger.info("\n--- Test 2: Chinese to English (auto-detect source) ---")
    result2 = translate_text(sample_text_zh_src, target_lang_code="en", source_lang_code="auto")
    if result2['error']:
        logger.error(f"Error: {result2['error']}")
    else:
        logger.info(f"Original: {sample_text_zh_src}")
        logger.info(f"Translated: {result2['translated_text']}")
        logger.info(f"Detected Source Language: {result2['detected_source_language']}")

    # Test 3: Empty text
    logger.info("\n--- Test 3: Empty text ---")
    result3 = translate_text("", target_lang_code="fr")
    if result3['error']:
        logger.error(f"Error: {result3['error']}")
    else:
        logger.info(f"Original: ''")
        logger.info(f"Translated: '{result3['translated_text']}'")
        logger.info(f"Detected Source Language: {result3['detected_source_language']}")

    # Test 4: Translation to Spanish (auto-detect source)
    logger.info("\n--- Test 4: English to Spanish (auto-detect source) ---")
    result4 = translate_text(sample_text_en, target_lang_code="es") # Default source_lang_code="auto"
    if result4['error']:
        logger.error(f"Error: {result4['error']}")
    else:
        logger.info(f"Original: {sample_text_en}")
        logger.info(f"Translated: {result4['translated_text']}")
        logger.info(f"Detected Source Language: {result4['detected_source_language']}")
