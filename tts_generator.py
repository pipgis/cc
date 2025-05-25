import requests
import json
import os
import logging
import asyncio

# Attempt to import cloud SDKs, but don't fail if not installed yet.
# The actual functions will handle cases where the module is not available.
try:
    import edge_tts
except ImportError:
    edge_tts = None # Placeholder if not installed

try:
    from azure.cognitiveservices.speech import SpeechConfig, SpeechSynthesizer, ResultReason, CancellationReason
    from azure.cognitiveservices.speech.audio import AudioOutputConfig
except ImportError:
    SpeechConfig = None # Placeholder

try:
    from google.cloud import texttospeech
except ImportError:
    texttospeech = None # Placeholder

logger = logging.getLogger(__name__)

# --- Voice Mappings ---
EDGE_TTS_VOICE_MAPPING = {
    "en": {"female": "en-US-AriaNeural", "male": "en-US-GuyNeural"},
    "zh": {"female": "zh-CN-XiaoxiaoNeural", "male": "zh-CN-YunxiNeural"},
    "es": {"female": "es-ES-ElviraNeural", "male": "es-ES-AlvaroNeural"},
    "fr": {"female": "fr-FR-DeniseNeural", "male": "fr-FR-HenriNeural"},
}

AZURE_TTS_VOICE_MAPPING = {
    "en": {"female": "en-US-AriaNeural", "male": "en-US-GuyNeural"},
    "zh": {"female": "zh-CN-XiaoxiaoNeural", "male": "zh-CN-YunxiNeural"},
    "es": {"female": "es-ES-ElviraNeural", "male": "es-ES-AlvaroNeural"},
    "fr": {"female": "fr-FR-DeniseNeural", "male": "fr-FR-HenriNeural"},
}

GOOGLE_TTS_VOICE_MAPPING = {
    # Google uses language_code and gender directly, but specific voice names can be mapped if needed.
    # Using standard voices for broader compatibility and potentially lower cost.
    # Format: {lang_code: {gender: voice_name}}
    "en": {"female": "en-US-Standard-C", "male": "en-US-Standard-D"}, # Example voices
    "zh": {"female": "cmn-CN-Standard-A", "male": "cmn-CN-Standard-B"}, # For Mandarin Chinese
    "es": {"female": "es-ES-Standard-A", "male": "es-ES-Standard-B"},
    "fr": {"female": "fr-FR-Standard-A", "male": "fr-FR-Standard-B"},
}

MINIMAX_TTS_VOICE_MAPPING = {
    "zh": {
        "male_qn_qingse": "male-qn-qingse",
        "male_qn_jingying": "male-qn-jingying",
        "male_qn_badao": "male-qn-badao",
        "male_qn_daxuesheng": "male-qn-daxuesheng",
        "female_shaonv": "female-shaonv",
        "female_yujie": "female-yujie",
        "female_chengshu": "female-chengshu",
        "female_tianmei": "female-tianmei",
        "presenter_male": "presenter_male",
        "presenter_female": "presenter_female",
        "audiobook_male_1": "audiobook_male_1",
        "audiobook_male_2": "audiobook_male_2",
        "audiobook_female_1": "audiobook_female_1",
        "audiobook_female_2": "audiobook_female_2",
        "male_qn_qingse_jingpin": "male-qn-qingse-jingpin",
        "male_qn_jingying_jingpin": "male-qn-jingying-jingpin",
        "male_qn_badao_jingpin": "male-qn-badao-jingpin",
        "male_qn_daxuesheng_jingpin": "male-qn-daxuesheng-jingpin",
        "female_shaonv_jingpin": "female-shaonv-jingpin",
        "female_yujie_jingpin": "female-yujie-jingpin",
        "female_chengshu_jingpin": "female-chengshu-jingpin",
        "female_tianmei_jingpin": "female-tianmei-jingpin",
        "clever_boy": "clever_boy",
        "cute_boy": "cute_boy",
        "lovely_girl": "lovely_girl",
        "cartoon_pig": "cartoon_pig",
        "bingjiao_didi": "bingjiao_didi",
        "junlang_nanyou": "junlang_nanyou",
        "chunzhen_xuedi": "chunzhen_xuedi",
        "lengdan_xiongzhang": "lengdan_xiongzhang",
        "badao_shaoye": "badao_shaoye",
        "tianxin_xiaoling": "tianxin_xiaoling",
        "qiaopi_mengmei": "qiaopi_mengmei",
        "wumei_yujie": "wumei_yujie",
        "diadia_xuemei": "diadia_xuemei",
        "danya_xuejie": "danya_xuejie",
    },
    "en": {
        "santa_claus": "Santa_Claus",
        "grinch": "Grinch",
        "rudolph": "Rudolph",
        "arnold": "Arnold",
        "charming_santa": "Charming_Santa",
        "charming_lady": "Charming_Lady",
        "sweet_girl": "Sweet_Girl",
        "cute_elf": "Cute_Elf",
        "attractive_girl": "Attractive_Girl",
        "serene_woman": "Serene_Woman",
    }
}


# --- Helper Functions for Each Service ---

async def _generate_edge_tts_async(text_to_speak: str, output_filename: str, language_code: str, voice_gender: str = 'female') -> dict:
    """
    Asynchronous helper for EdgeTTS generation.
    """
    if not edge_tts:
        return {'success': False, 'error': "edge_tts library is not installed. Please install it using: pip install edge-tts"}

    voice = EDGE_TTS_VOICE_MAPPING.get(language_code, {}).get(voice_gender.lower())
    if not voice:
        logger.error(f"EdgeTTS: Unsupported language_code '{language_code}' or voice_gender '{voice_gender}'.")
        return {'success': False, 'error': f"EdgeTTS: Unsupported language_code '{language_code}' or voice_gender '{voice_gender}'."}
    
    logger.info(f"EdgeTTS: Using voice '{voice}' for language '{language_code}' and gender '{voice_gender}'.")
    try:
        communicate = edge_tts.Communicate(text_to_speak, voice)
        await communicate.save(output_filename)
        logger.info(f"EdgeTTS successfully generated audio to {output_filename}")
        return {'success': True, 'error': None}
    except Exception as e:
        logger.error(f"EdgeTTS generation failed for {output_filename}", exc_info=True)
        return {'success': False, 'error': f"EdgeTTS generation failed: {e}"}

def _generate_edge_tts(text_to_speak: str, output_filename: str, voice_gender: str = 'female') -> dict:
    """
    Synchronous wrapper for EdgeTTS generation.
    """
    try:
        asyncio.run(_generate_edge_tts_async(text_to_speak, output_filename, voice_gender))
        # Check if file was created and has size, as edge_tts.save might not raise error for all failures
        if os.path.exists(output_filename) and os.path.getsize(output_filename) > 0:
            # Already logged success in async version
            return {'success': True, 'error': None}
        else:
            logger.error(f"EdgeTTS file not created or empty for {output_filename}, check text or voice.")
            return {'success': False, 'error': "EdgeTTS file not created or empty, check text or voice."}
    except Exception as e:
        logger.error(f"EdgeTTS asyncio execution failed for {output_filename}", exc_info=True)
        return {'success': False, 'error': f"EdgeTTS asyncio execution failed: {e}"}


def _generate_edge_tts(text_to_speak: str, output_filename: str, language_code: str, voice_gender: str = 'female') -> dict:
    """
    Synchronous wrapper for EdgeTTS generation.
    """
    try:
        # Pass language_code and voice_gender to the async helper
        result = asyncio.run(_generate_edge_tts_async(text_to_speak, output_filename, language_code, voice_gender))
        if not result['success']: # If async helper already reported failure, propagate it
            return result
            
        # Check if file was created and has size, as edge_tts.save might not raise error for all failures
        if os.path.exists(output_filename) and os.path.getsize(output_filename) > 0:
            # Already logged success in async version
            return {'success': True, 'error': None}
        else:
            logger.error(f"EdgeTTS file not created or empty for {output_filename}, check text or voice.")
            # If the async call reported success but file is missing/empty, this is a new error condition.
            return {'success': False, 'error': "EdgeTTS file not created or empty, check text or voice."}
    except Exception as e:
        logger.error(f"EdgeTTS asyncio execution failed for {output_filename}", exc_info=True)
        return {'success': False, 'error': f"EdgeTTS asyncio execution failed: {e}"}


def _generate_azure_tts(text_to_speak: str, output_filename: str, language_code: str, voice_gender: str = 'female', 
                        api_key: str = None, azure_region: str = None) -> dict:
    """
    Generates speech using Microsoft Azure Cognitive Services TTS.
    """
    if not SpeechConfig:
        return {'success': False, 'error': "azure-cognitiveservices-speech library not installed. Please install it."}
    if not api_key or not azure_region:
        return {'success': False, 'error': "Azure API Key and Region are required."}

    voice_name = AZURE_TTS_VOICE_MAPPING.get(language_code, {}).get(voice_gender.lower())
    if not voice_name:
        logger.error(f"AzureTTS: Unsupported language_code '{language_code}' or voice_gender '{voice_gender}'.")
        return {'success': False, 'error': f"AzureTTS: Unsupported language_code '{language_code}' or voice_gender '{voice_gender}'."}

    try:
        speech_config = SpeechConfig(subscription=api_key, region=azure_region)
        speech_config.speech_synthesis_language = language_code # Set language
        speech_config.speech_synthesis_voice_name = voice_name  # Set voice
        logger.info(f"AzureTTS: Using voice '{voice_name}' for language '{language_code}' and gender '{voice_gender}'.")

        # Set audio output format to MP3
        speech_config.set_speech_synthesis_output_format(speech_config.speech_synthesis_output_format.Audio16Khz32KBitRateMonoMp3)

        audio_config = AudioOutputConfig(filename=output_filename)
        synthesizer = SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

        result = synthesizer.speak_text_async(text_to_speak).get()

        if result.reason == ResultReason.SynthesizingAudioCompleted:
            logger.info(f"Azure TTS successfully generated audio to {output_filename}")
            return {'success': True, 'error': None}
        elif result.reason == ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            error_message = f"Azure TTS synthesis canceled: {cancellation_details.reason}"
            if cancellation_details.reason == CancellationReason.Error:
                error_message += f" - Error details: {cancellation_details.error_details}"
            logger.error(f"Azure TTS synthesis canceled for {output_filename}. Details: {error_message}")
            return {'success': False, 'error': error_message}
        else:
            logger.error(f"Azure TTS synthesis failed for {output_filename} with reason: {result.reason}")
            return {'success': False, 'error': f"Azure TTS synthesis failed with reason: {result.reason}"}

    except Exception as e:
        logger.error(f"Azure TTS generation failed for {output_filename}", exc_info=True)
        return {'success': False, 'error': f"Azure TTS generation failed: {e}"}


def _generate_google_tts(text_to_speak: str, output_filename: str, language_code: str, voice_gender: str = 'female', 
                         google_credentials_path: str = None) -> dict:
    """
    Generates speech using Google Cloud Text-to-Speech.
    """
    if not texttospeech:
        return {'success': False, 'error': "google-cloud-texttospeech library not installed. Please install it."}

    voice_name = GOOGLE_TTS_VOICE_MAPPING.get(language_code, {}).get(voice_gender.lower())
    if not voice_name:
        # Google can often select a voice if only language_code and gender are provided,
        # but for consistency and control, we use a mapping.
        logger.error(f"GoogleTTS: Unsupported language_code '{language_code}' or voice_gender '{voice_gender}' in mapping.")
        return {'success': False, 'error': f"GoogleTTS: Unsupported language_code '{language_code}' or voice_gender '{voice_gender}' in mapping."}

    try:
        if google_credentials_path:
            client = texttospeech.TextToSpeechClient(credentials_path=google_credentials_path)
        else:
            client = texttospeech.TextToSpeechClient()
            logger.info("Google TTS: Attempting to use Application Default Credentials.")

        input_text = texttospeech.SynthesisInput(text=text_to_speak)
        
        ssml_gender_value = texttospeech.SsmlVoiceGender.MALE if voice_gender.lower() == 'male' else texttospeech.SsmlVoiceGender.FEMALE

        voice_params = texttospeech.VoiceSelectionParams(
            language_code=language_code,  # Set language
            name=voice_name,              # Set specific voice name from mapping
            ssml_gender=ssml_gender_value # Set gender
        )
        logger.info(f"GoogleTTS: Using voice '{voice_name}' for language '{language_code}' and gender '{voice_gender}'.")

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )

        response = client.synthesize_speech(
            request={"input": input_text, "voice": voice, "audio_config": audio_config}
        )

        with open(output_filename, "wb") as out:
            out.write(response.audio_content)
        logger.info(f"Google TTS successfully generated audio to {output_filename}")
        return {'success': True, 'error': None}

    except Exception as e:
        logger.error(f"Google TTS generation failed for {output_filename}", exc_info=True)
        return {'success': False, 'error': f"Google TTS generation failed: {e}"}


def _generate_minimax_tts(text_to_speak: str, output_filename: str, language_code: str, voice_name_key: str = None,
                          minimax_api_key: str = None, minimax_group_id: str = None, 
                          direct_voice_id_override: str = None, additional_settings: dict = None) -> dict:
    """
    Generates speech using Minimax TTS.
    'voice_name_key' is used to look up the voice_id from MINIMAX_TTS_VOICE_MAPPING.
    'direct_voice_id_override' allows specifying a voice_id directly, bypassing the mapping.
    'additional_settings' allows specifying other parameters like speed, vol, pitch, emotion, etc.
    """
    if not minimax_api_key or not minimax_group_id:
        return {'success': False, 'error': "Minimax API Key and Group ID are required."}

    selected_voice_id = None
    if direct_voice_id_override:
        selected_voice_id = direct_voice_id_override
        logger.info(f"MinimaxTTS: Using direct_voice_id_override: '{selected_voice_id}'.")
    elif voice_name_key:
        # Backward compatibility for generic 'male'/'female' keys
        # This check is implicitly covered by being inside 'elif voice_name_key:'
        # and the 'if direct_voice_id_override:' check has already passed.
        original_voice_name_key = voice_name_key # Keep original for logging if needed
        if language_code == "zh":
            if voice_name_key == "male":
                voice_name_key = "male_qn_qingse" # Default Chinese male
                logger.info(f"MinimaxTTS: Remapped generic key '{original_voice_name_key}' to '{voice_name_key}' for lang 'zh'.")
            elif voice_name_key == "female":
                voice_name_key = "female_shaonv" # Default Chinese female
                logger.info(f"MinimaxTTS: Remapped generic key '{original_voice_name_key}' to '{voice_name_key}' for lang 'zh'.")
        elif language_code == "en":
            if voice_name_key == "male":
                voice_name_key = "arnold" # Default English male (maps to 'Arnold')
                logger.info(f"MinimaxTTS: Remapped generic key '{original_voice_name_key}' to '{voice_name_key}' for lang 'en'.")
            elif voice_name_key == "female":
                voice_name_key = "charming_lady" # Default English female (maps to 'Charming_Lady')
                logger.info(f"MinimaxTTS: Remapped generic key '{original_voice_name_key}' to '{voice_name_key}' for lang 'en'.")
        # Add more remappings if other generic keys were previously used and need defaults.
        
        selected_voice_id = MINIMAX_TTS_VOICE_MAPPING.get(language_code, {}).get(voice_name_key)
        if selected_voice_id:
            logger.info(f"MinimaxTTS: Found voice_id '{selected_voice_id}' for lang '{language_code}', key '{voice_name_key}' (original key if remapped: '{original_voice_name_key}').")
        else:
            # If voice_name_key was remapped, original_voice_name_key holds the original.
            # If not remapped, original_voice_name_key == voice_name_key.
            logger.warning(f"MinimaxTTS: Voice key '{voice_name_key}' (original key if remapped: '{original_voice_name_key}') not found in mapping for language '{language_code}'.")
            selected_voice_id = None 
    
    if not selected_voice_id:
        # If voice_name_key was provided (even if it was a generic one that got remapped and still failed, or was specific and failed)
        if voice_name_key: 
             # Use original_voice_name_key in error if it exists (i.e., if we entered the voice_name_key block)
             key_in_error = original_voice_name_key if 'original_voice_name_key' in locals() else voice_name_key
             logger.error(f"MinimaxTTS: Specified voice_name_key '{key_in_error}' for language '{language_code}' is not valid (or remapped key '{voice_name_key}' is not valid) and no direct_voice_id_override was given.")
             return {'success': False, 'error': f"MinimaxTTS: Invalid voice_name_key '{key_in_error}' for language '{language_code}'."}
        else: # No voice_name_key provided at all (and no direct_voice_id_override). Attempt to use a default.
            if language_code == "zh":
                default_key = "male_qn_qingse" 
                selected_voice_id = MINIMAX_TTS_VOICE_MAPPING.get(language_code, {}).get(default_key)
                logger.warning(f"MinimaxTTS: No voice_name_key or direct_voice_id_override specified for lang '{language_code}'. Using default key '{default_key}' -> voice_id '{selected_voice_id}'.")
            elif language_code == "en":
                default_key = "arnold" 
                selected_voice_id = MINIMAX_TTS_VOICE_MAPPING.get(language_code, {}).get(default_key)
                logger.warning(f"MinimaxTTS: No voice_name_key or direct_voice_id_override specified for lang '{language_code}'. Using default key '{default_key}' -> voice_id '{selected_voice_id}'.")
            else:
                logger.error(f"MinimaxTTS: No voice_name_key or direct_voice_id_override specified, and no default voice configured for language_code '{language_code}'.")
                return {'success': False, 'error': f"MinimaxTTS: No voice specified and no default for lang '{language_code}'."}

    if not selected_voice_id: 
        logger.error(f"MinimaxTTS: Could not determine a voice_id to use for language_code '{language_code}'. Default key might be missing from mapping.")
        return {'success': False, 'error': f"MinimaxTTS: Could not determine voice_id for lang '{language_code}'. Default key configuration issue."}

    logger.info(f"MinimaxTTS: Using voice_id '{selected_voice_id}' for lang '{language_code}'. Text: \"{text_to_speak[:30]}...\"")
    
    url = f"https://api.minimax.chat/v1/t2a_v2?GroupId={minimax_group_id}"
    headers = {
        'accept': 'application/json, text/plain, */*',
        'Content-Type': 'application/json',
        'Authorization': f"Bearer {minimax_api_key}"
    }

    # Initialize voice_setting payload with defaults
    voice_setting_payload = {
        "speed": 1.0,
        "vol": 1.0,
        "pitch": 0
    }
    # Ensure selected_voice_id is primary
    voice_setting_payload['voice_id'] = selected_voice_id

    if additional_settings and isinstance(additional_settings, dict):
        if 'speed' in additional_settings:
            voice_setting_payload['speed'] = additional_settings['speed']
        if 'vol' in additional_settings:
            voice_setting_payload['vol'] = additional_settings['vol']
        if 'pitch' in additional_settings:
            voice_setting_payload['pitch'] = additional_settings['pitch']
        # If additional_settings contains 'voice_id', it's ignored here to prioritize
        # selected_voice_id determined by earlier logic (direct_override or mapping).
        # If override from additional_settings was desired for voice_id itself,
        # selected_voice_id logic would need adjustment or this part would change.
        # Current task implies selected_voice_id is from prior logic.
    
    body_payload = {
        "model": "speech-02-turbo", 
        "text": text_to_speak,
        "stream": True,
        "voice_setting": voice_setting_payload,
        "audio_setting": {
            "sample_rate": 32000,
            "bitrate": 128000,
            "format": "mp3",
            "channel": 1
        }
    }

    if additional_settings and isinstance(additional_settings, dict):
        if 'emotion' in additional_settings:
            body_payload['emotion'] = additional_settings['emotion']
        if 'latex_read' in additional_settings:
            body_payload['latex_read'] = additional_settings['latex_read']
        if 'english_normalization' in additional_settings:
            body_payload['english_normalization'] = additional_settings['english_normalization']
            
    try:
        response = requests.post(url, headers=headers, json=body_payload, stream=True, timeout=60)
        response.raise_for_status() # Check for initial HTTP errors (4xx or 5xx)

        audio_bytes_list = []
        for line_bytes in response.iter_lines():
            if line_bytes: # Filter out keep-alive new lines
                line_bytes = line_bytes.strip()
                if line_bytes.startswith(b'data:'):
                    json_str_part = line_bytes[5:].strip()
                    if not json_str_part:
                        continue
                    try:
                        data_obj = json.loads(json_str_part.decode('utf-8'))
                        # Check for actual audio data vs. other messages like task_id
                        if "data" in data_obj and "audio" in data_obj["data"] and "extra_info" not in data_obj:
                            hex_audio_chunk = data_obj["data"]["audio"]
                            try:
                                audio_bytes_list.append(bytes.fromhex(hex_audio_chunk))
                            except ValueError as e_hex:
                                logger.warning(f"Failed to decode hex audio chunk: {e_hex}. Chunk: {hex_audio_chunk[:30]}...")
                        # Check for error messages within the stream
                        elif data_obj.get("base_resp", {}).get("status_code", 0) != 0:
                            error_msg = data_obj["base_resp"].get("status_msg", "Unknown error in stream")
                            logger.error(f"Minimax API error in stream: {error_msg}")
                            # If an error is reported in the stream, we should probably stop and return failure.
                            return {'success': False, 'error': f"Minimax API error in stream: {error_msg}"}
                        # Handle other potential data messages if necessary (e.g., task_id, extra_info if it becomes relevant)
                        elif "extra_info" in data_obj:
                             logger.debug(f"Received Minimax stream message with extra_info: {data_obj}")


                    except (json.JSONDecodeError, ValueError) as e_parse:
                        logger.warning(f"Failed to parse data from Minimax stream: {e_parse}. Line: '{json_str_part.decode('utf-8', errors='ignore')[:100]}'")
        
        if not audio_bytes_list:
            logger.error(f"No audio data extracted from Minimax stream for {output_filename}.")
            # Check if the response content type was something other than expected, e.g. an error page not caught by raise_for_status
            # This is less likely if raise_for_status passed and iter_lines began.
            # But if the stream was empty but valid, this is the correct error.
            return {'success': False, 'error': "No audio data received from Minimax stream."}

        final_audio_bytes = b"".join(audio_bytes_list)
        with open(output_filename, 'wb') as f:
            f.write(final_audio_bytes)
        logger.info(f"Minimax TTS successfully generated audio to {output_filename}")
        return {'success': True, 'error': None}

    except requests.exceptions.RequestException as e:
        error_message = f"Minimax API request failed for {output_filename}: {e}"
        if e.response is not None:
            error_message += f" - Response: {e.response.text[:200]}"
        logger.error(error_message, exc_info=True)
        return {'success': False, 'error': error_message}
    except Exception as e: # Catch any other unexpected errors
        logger.error(f"An unexpected error occurred during Minimax TTS processing for {output_filename}.", exc_info=True)
        return {'success': False, 'error': f"Minimax TTS processing failed: {e}"}

# --- Main Dispatch Function ---

def generate_audio(text_to_speak: str, output_filename: str, service: str, 
                   voice_gender: str = 'female', language_code: str = "en", # Added language_code with default "en"
                   api_key: str = None, azure_region: str = None, 
                   google_credentials_path: str = None,
                   minimax_api_key: str = None, minimax_group_id: str = None, 
                   minimax_voice_id: str = None,
                   minimax_additional_settings: dict = None) -> dict: # New parameter for Minimax
    """
    Generates audio using the specified TTS service.
    Ensures the output directory exists.
    The `language_code` parameter determines the language of the speech.
    """
    if not text_to_speak or not text_to_speak.strip():
        return {'success': False, 'error': "Input text is empty or whitespace."}
    if not output_filename:
        return {'success': False, 'error': "Output filename not provided."}

    try:
        output_dir = os.path.dirname(output_filename)
        if output_dir and not os.path.exists(output_dir):
            logger.info(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create output directory for {output_filename}", exc_info=True)
        return {'success': False, 'error': f"Failed to create output directory for {output_filename}: {e}"}

    logger.info(f"Generating audio for text: \"{text_to_speak[:30]}...\" using service: {service}, lang: {language_code} to file: {output_filename}")
    service_lower = service.lower()

    # Validate language_code (basic check, specific checks within helpers)
    if not language_code or not isinstance(language_code, str) or len(language_code) < 2:
        logger.error(f"Invalid language_code provided: {language_code}. Must be a string like 'en', 'zh'.")
        return {'success': False, 'error': f"Invalid language_code: {language_code}"}
        
    if service_lower == "edge_tts":
        return _generate_edge_tts(text_to_speak, output_filename, language_code, voice_gender)
    elif service_lower == "azure":
        return _generate_azure_tts(text_to_speak, output_filename, language_code, voice_gender, api_key, azure_region)
    elif service_lower == "google":
        return _generate_google_tts(text_to_speak, output_filename, language_code, voice_gender, google_credentials_path)
    elif service_lower == "minimax":
        # Pass parameters including the new minimax_additional_settings
        return _generate_minimax_tts(text_to_speak, output_filename, language_code, voice_gender,
                                     minimax_api_key, minimax_group_id, minimax_voice_id,
                                     additional_settings=minimax_additional_settings)
    else:
        logger.error(f"Unsupported TTS service requested: {service}")
        return {'success': False, 'error': f"Unsupported TTS service: {service}"}


# --- Test Block ---
if __name__ == '__main__':
    # Basic setup for testing this module directly
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    sample_chinese_text = "你好，这是一个测试语音。今天天气怎么样？希望你能喜欢这个声音。"
    output_dir = "tts_output" 
    os.makedirs(output_dir, exist_ok=True) 
    logger.info(f"Created output directory: {os.path.abspath(output_dir)}")

    logger.info("\n--- Testing EdgeTTS ---")
    test_langs = {"en": "Hello, this is a test.", "zh": "你好，这是一个测试语音。", "es": "Hola, esta es una prueba de voz.", "fr": "Bonjour, ceci est un test vocal."}

    if edge_tts:
        for lang, text in test_langs.items():
            edge_output_file_female = os.path.join(output_dir, f"edge_tts_female_{lang}_test.mp3")
            edge_result_female = generate_audio(text, edge_output_file_female, "edge_tts", voice_gender="female", language_code=lang)
            logger.info(f"EdgeTTS (Female, {lang}) Result: {edge_result_female}")
            if edge_result_female['success']:
                logger.info(f"  Output: {os.path.abspath(edge_output_file_female)}")
    else:
        logger.warning("EdgeTTS library not found. Skipping EdgeTTS tests. Install with: pip install edge-tts")
    
    logger.info("\n--- Testing Azure TTS ---")
    azure_api_key_test = os.environ.get("AZURE_SPEECH_KEY") 
    azure_region_test = os.environ.get("AZURE_SPEECH_REGION")

    if not SpeechConfig:
        logger.warning("azure-cognitiveservices-speech library not found. Skipping Azure TTS tests.")
    elif not azure_api_key_test or not azure_region_test:
        logger.warning(f"Azure credentials not set (AZURE_SPEECH_KEY, AZURE_SPEECH_REGION). Skipping.")
    else:
        for lang, text in test_langs.items():
            azure_output_file = os.path.join(output_dir, f"azure_tts_female_{lang}_test.mp3")
            azure_result = generate_audio(text, azure_output_file, "azure", 
                                          voice_gender="female", language_code=lang, 
                                          api_key=azure_api_key_test, azure_region=azure_region_test)
            logger.info(f"AzureTTS (Female, {lang}) Result: {azure_result}")
            if azure_result['success']:
                logger.info(f"  Output: {os.path.abspath(azure_output_file)}")

    logger.info("\n--- Testing Google Cloud TTS ---")
    google_creds_path_test = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

    if not texttospeech:
        logger.warning("google-cloud-texttospeech library not found. Skipping Google TTS tests.")
    elif not google_creds_path_test: # Assuming ADC might not be set up for direct test runs easily
        logger.warning("GOOGLE_APPLICATION_CREDENTIALS not set. Skipping Google TTS test.")
    else:
        for lang, text in test_langs.items():
            google_output_file = os.path.join(output_dir, f"google_tts_female_{lang}_test.mp3")
            google_result = generate_audio(text, google_output_file, "google", 
                                           voice_gender="female", language_code=lang, 
                                           google_credentials_path=google_creds_path_test)
            logger.info(f"GoogleTTS (Female, {lang}) Result: {google_result}")
            if google_result['success']:
                logger.info(f"  Output: {os.path.abspath(google_output_file)}")

    logger.info("\n--- Testing Minimax TTS ---")
    minimax_api_key_test = os.environ.get("MINIMAX_API_KEY")
    minimax_group_id_test = os.environ.get("MINIMAX_GROUP_ID")

    if not minimax_api_key_test or not minimax_group_id_test:
        logger.warning("Minimax credentials not set (MINIMAX_API_KEY, MINIMAX_GROUP_ID). Skipping Minimax TTS tests.")
    else:
        logger.info("--- Starting Minimax TTS Tests ---")
        text_zh = test_langs["zh"]
        text_en = test_langs["en"]

        # 1. Test specific Chinese voice via mapping
        logger.info("Testing MinimaxTTS: Chinese voice 'female_yujie' via mapping...")
        minimax_output_file_zh_yujie = os.path.join(output_dir, "minimax_tts_zh_yujie_test.mp3")
        result_zh_yujie = generate_audio(
            text_to_speak=text_zh,
            output_filename=minimax_output_file_zh_yujie,
            service="minimax",
            language_code="zh",
            voice_gender="female_yujie", # This is the voice_name_key
            minimax_api_key=minimax_api_key_test,
            minimax_group_id=minimax_group_id_test
        )
        logger.info(f"MinimaxTTS (Zh, female_yujie) Result: {result_zh_yujie}")
        if result_zh_yujie.get('success'):
            logger.info(f"  Output: {os.path.abspath(minimax_output_file_zh_yujie)}")

        # 2. Test specific English voice via mapping
        logger.info("Testing MinimaxTTS: English voice 'santa_claus' via mapping...")
        minimax_output_file_en_santa = os.path.join(output_dir, "minimax_tts_en_santa_test.mp3")
        result_en_santa = generate_audio(
            text_to_speak=text_en,
            output_filename=minimax_output_file_en_santa,
            service="minimax",
            language_code="en",
            voice_gender="santa_claus", # This is the voice_name_key
            minimax_api_key=minimax_api_key_test,
            minimax_group_id=minimax_group_id_test
        )
        logger.info(f"MinimaxTTS (En, santa_claus) Result: {result_en_santa}")
        if result_en_santa.get('success'):
            logger.info(f"  Output: {os.path.abspath(minimax_output_file_en_santa)}")

        # 3. Test direct voice ID override
        logger.info("Testing MinimaxTTS: Direct voice ID override 'male-qn-jingying' for Chinese...")
        minimax_output_file_direct_override = os.path.join(output_dir, "minimax_tts_zh_direct_override_test.mp3")
        result_direct_override = generate_audio(
            text_to_speak=text_zh,
            output_filename=minimax_output_file_direct_override,
            service="minimax",
            language_code="zh",
            minimax_voice_id="male-qn-jingying", # Direct override
            minimax_api_key=minimax_api_key_test,
            minimax_group_id=minimax_group_id_test
        )
        logger.info(f"MinimaxTTS (Zh, Direct Override 'male-qn-jingying') Result: {result_direct_override}")
        if result_direct_override.get('success'):
            logger.info(f"  Output: {os.path.abspath(minimax_output_file_direct_override)}")
            
        # 4. Test with minimax_additional_settings
        logger.info("Testing MinimaxTTS: Chinese voice 'female_tianmei' with additional settings...")
        minimax_output_file_custom_settings = os.path.join(output_dir, "minimax_tts_zh_custom_settings_test.mp3")
        custom_settings = {
            "speed": 1.2,
            "vol": 0.8,
            "pitch": 2,
            # "emotion": "happy" # Emotion might not be supported by all voices or models, test carefully based on docs.
                               # For now, testing without emotion to ensure broader compatibility.
        }
        result_custom_settings = generate_audio(
            text_to_speak=text_zh,
            output_filename=minimax_output_file_custom_settings,
            service="minimax",
            language_code="zh",
            voice_gender="female_tianmei",
            minimax_additional_settings=custom_settings,
            minimax_api_key=minimax_api_key_test,
            minimax_group_id=minimax_group_id_test
        )
        logger.info(f"MinimaxTTS (Zh, female_tianmei, Custom Settings) Result: {result_custom_settings}")
        if result_custom_settings.get('success'):
            logger.info(f"  Output: {os.path.abspath(minimax_output_file_custom_settings)}")

        # 5. Test default voice (no specific voice key or override)
        logger.info("Testing MinimaxTTS: Default Chinese voice (no voice_gender or minimax_voice_id)...")
        minimax_output_file_zh_default = os.path.join(output_dir, "minimax_tts_zh_default_test.mp3")
        # Note: generate_audio's voice_gender defaults to 'female'. 
        # To truly test the default logic in _generate_minimax_tts (where voice_name_key is None),
        # we might need to pass voice_gender=None if the default 'female' isn't a valid key.
        # However, the current _generate_minimax_tts logic handles a non-None voice_name_key ('female')
        # that's not in the mapping by erroring out, which is fine.
        # To test the "no voice_name_key provided" path in _generate_minimax_tts,
        # we ensure voice_gender is not a valid key and minimax_voice_id is not set.
        # The default 'female' for voice_gender in generate_audio will cause an error as 'female' is not a valid key.
        # To test the intended default fallback in _generate_minimax_tts, we should call it such that voice_name_key becomes None.
        # This means passing voice_gender=None to generate_audio.
        result_zh_default = generate_audio(
            text_to_speak=text_zh,
            output_filename=minimax_output_file_zh_default,
            service="minimax",
            language_code="zh",
            voice_gender=None, # Explicitly None to test the default logic in _generate_minimax_tts
            minimax_api_key=minimax_api_key_test,
            minimax_group_id=minimax_group_id_test
        )
        logger.info(f"MinimaxTTS (Zh, Default) Result: {result_zh_default}")
        if result_zh_default.get('success'):
            logger.info(f"  Output: {os.path.abspath(minimax_output_file_zh_default)}")

        logger.info("Testing MinimaxTTS: Default English voice (no voice_gender or minimax_voice_id)...")
        minimax_output_file_en_default = os.path.join(output_dir, "minimax_tts_en_default_test.mp3")
        result_en_default = generate_audio(
            text_to_speak=text_en,
            output_filename=minimax_output_file_en_default,
            service="minimax",
            language_code="en",
            voice_gender=None, # Explicitly None to test the default logic in _generate_minimax_tts
            minimax_api_key=minimax_api_key_test,
            minimax_group_id=minimax_group_id_test
        )
        logger.info(f"MinimaxTTS (En, Default) Result: {result_en_default}")
        if result_en_default.get('success'):
            logger.info(f"  Output: {os.path.abspath(minimax_output_file_en_default)}")

        # 6. Test backward compatibility for generic voice keys
        logger.info("--- Testing MinimaxTTS: Backward Compatibility for Generic Keys ---")
        # Chinese 'male'
        logger.info("Testing MinimaxTTS: Chinese 'male' (backward compatibility)...")
        minimax_output_file_zh_compat_male = os.path.join(output_dir, "minimax_tts_zh_compat_male_test.mp3")
        result_zh_compat_male = generate_audio(
            text_to_speak=text_zh,
            output_filename=minimax_output_file_zh_compat_male,
            service="minimax",
            language_code="zh",
            voice_gender="male", # Generic key
            minimax_api_key=minimax_api_key_test,
            minimax_group_id=minimax_group_id_test
        )
        logger.info(f"MinimaxTTS (Zh, Compat 'male') Result: {result_zh_compat_male}")
        if result_zh_compat_male.get('success'):
            logger.info(f"  Output: {os.path.abspath(minimax_output_file_zh_compat_male)}")

        # Chinese 'female'
        logger.info("Testing MinimaxTTS: Chinese 'female' (backward compatibility)...")
        minimax_output_file_zh_compat_female = os.path.join(output_dir, "minimax_tts_zh_compat_female_test.mp3")
        result_zh_compat_female = generate_audio(
            text_to_speak=text_zh,
            output_filename=minimax_output_file_zh_compat_female,
            service="minimax",
            language_code="zh",
            voice_gender="female", # Generic key
            minimax_api_key=minimax_api_key_test,
            minimax_group_id=minimax_group_id_test
        )
        logger.info(f"MinimaxTTS (Zh, Compat 'female') Result: {result_zh_compat_female}")
        if result_zh_compat_female.get('success'):
            logger.info(f"  Output: {os.path.abspath(minimax_output_file_zh_compat_female)}")

        # English 'male'
        logger.info("Testing MinimaxTTS: English 'male' (backward compatibility)...")
        minimax_output_file_en_compat_male = os.path.join(output_dir, "minimax_tts_en_compat_male_test.mp3")
        result_en_compat_male = generate_audio(
            text_to_speak=text_en,
            output_filename=minimax_output_file_en_compat_male,
            service="minimax",
            language_code="en",
            voice_gender="male", # Generic key
            minimax_api_key=minimax_api_key_test,
            minimax_group_id=minimax_group_id_test
        )
        logger.info(f"MinimaxTTS (En, Compat 'male') Result: {result_en_compat_male}")
        if result_en_compat_male.get('success'):
            logger.info(f"  Output: {os.path.abspath(minimax_output_file_en_compat_male)}")

        # English 'female'
        logger.info("Testing MinimaxTTS: English 'female' (backward compatibility)...")
        minimax_output_file_en_compat_female = os.path.join(output_dir, "minimax_tts_en_compat_female_test.mp3")
        result_en_compat_female = generate_audio(
            text_to_speak=text_en,
            output_filename=minimax_output_file_en_compat_female,
            service="minimax",
            language_code="en",
            voice_gender="female", # Generic key
            minimax_api_key=minimax_api_key_test,
            minimax_group_id=minimax_group_id_test
        )
        logger.info(f"MinimaxTTS (En, Compat 'female') Result: {result_en_compat_female}")
        if result_en_compat_female.get('success'):
            logger.info(f"  Output: {os.path.abspath(minimax_output_file_en_compat_female)}")
        
        logger.info("--- Finished Minimax TTS Tests ---")

    logger.info("\n--- Finished ALL TTS Tests ---")
    logger.info(f"Please check the '{output_dir}' directory for any generated MP3 files.")
    logger.info("For cloud services, ensure you have installed their respective SDKs (e.g., pip install azure-cognitiveservices-speech google-cloud-texttospeech) and configured credentials.")
