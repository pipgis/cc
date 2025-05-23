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

# --- Helper Functions for Each Service ---

async def _generate_edge_tts_async(text_to_speak: str, output_filename: str, voice_gender: str = 'female') -> dict:
    """
    Asynchronous helper for EdgeTTS generation.
    """
    if not edge_tts:
        return {'success': False, 'error': "edge_tts library is not installed. Please install it using: pip install edge-tts"}

    # Voice selection for Chinese
    # Full list: https://docs.microsoft.com/en-us/azure/cognitive-services/speech-service/language-support?tabs=stt-tts#text-to-speech
    if voice_gender.lower() == 'male':
        voice = "zh-CN-YunxiNeural"  # Male
    else:
        voice = "zh-CN-XiaoxiaoNeural" # Female (default)
    
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


def _generate_azure_tts(text_to_speak: str, output_filename: str, voice_gender: str = 'female', 
                        api_key: str = None, azure_region: str = None) -> dict:
    """
    Generates speech using Microsoft Azure Cognitive Services TTS.
    """
    if not SpeechConfig:
        return {'success': False, 'error': "azure-cognitiveservices-speech library not installed. Please install it."}
    if not api_key or not azure_region:
        return {'success': False, 'error': "Azure API Key and Region are required."}

    try:
        speech_config = SpeechConfig(subscription=api_key, region=azure_region)
        
        # Voice selection for Chinese
        if voice_gender.lower() == 'male':
            speech_config.speech_synthesis_voice_name = "zh-CN-YunxiNeural"
        else:
            speech_config.speech_synthesis_voice_name = "zh-CN-XiaoxiaoNeural" # Female (default)

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


def _generate_google_tts(text_to_speak: str, output_filename: str, voice_gender: str = 'female', 
                         google_credentials_path: str = None) -> dict:
    """
    Generates speech using Google Cloud Text-to-Speech.
    """
    if not texttospeech:
        return {'success': False, 'error': "google-cloud-texttospeech library not installed. Please install it."}

    try:
        # If credentials path is provided, use it. Otherwise, client tries ADC.
        if google_credentials_path:
            client = texttospeech.TextToSpeechClient(credentials_path=google_credentials_path)
        else:
            # This will work if GOOGLE_APPLICATION_CREDENTIALS env var is set,
            # or if running on GCP with a service account.
            client = texttospeech.TextToSpeechClient() 
            logger.info("Google TTS: Attempting to use Application Default Credentials.")


        input_text = texttospeech.SynthesisInput(text=text_to_speak)

        # Voice selection for Chinese (Mandarin)
        # Full list: https://cloud.google.com/text-to-speech/docs/voices
        if voice_gender.lower() == 'male':
            # Using Standard voices as Wavenet might be more expensive / require more setup
            # cmn-CN-Wavenet-B / cmn-CN-Wavenet-D are male Wavenet
            # cmn-CN-Standard-B / cmn-CN-Standard-D are male Standard
            voice_name = "cmn-CN-Standard-B" 
        else:
            # cmn-CN-Wavenet-A / cmn-CN-Wavenet-C are female Wavenet
            # cmn-CN-Standard-A / cmn-CN-Standard-C are female Standard
            voice_name = "cmn-CN-Standard-A" # Female (default)

        voice = texttospeech.VoiceSelectionParams(
            language_code="cmn-CN", # Mandarin Chinese
            name=voice_name
            # ssml_gender=texttospeech.SsmlVoiceGender.MALE if voice_gender.lower() == 'male' else texttospeech.SsmlVoiceGender.FEMALE
        )

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


def _generate_minimax_tts(text_to_speak: str, output_filename: str, voice_gender: str = 'female',
                          minimax_api_key: str = None, minimax_group_id: str = None, minimax_voice_id: str = None) -> dict:
    """
    Placeholder for Minimax TTS generation.
    Actual implementation depends on Minimax API details.
    """
    if not minimax_api_key or not minimax_group_id: # Assuming these are essential
        return {'success': False, 'error': "Minimax API Key and Group ID are required."}

    # This is a hypothetical structure. Replace with actual API endpoint and payload.
    logger.info(f"Attempting Minimax TTS (streaming) for text: \"{text_to_speak[:30]}...\" to file: {output_filename}")
    
    if not minimax_api_key or not minimax_group_id:
        logger.error("Minimax API Key or Group ID not provided.")
        return {'success': False, 'error': "Minimax API Key and Group ID are required."}

    url = f"https://api.minimax.chat/v1/t2a_v2?GroupId={minimax_group_id}"
    headers = {
        'accept': 'application/json, text/plain, */*',
        'Content-Type': 'application/json',
        'Authorization': f"Bearer {minimax_api_key}"
    }
    
    body_payload = {
        "model": "speech-02-turbo",
        "text": text_to_speak,
        "stream": True,
        "voice_setting": {
            "voice_id": "male-qn-qingse", # Default from new example, ignoring voice_gender and minimax_voice_id params
            "speed": 1.0,
            "vol": 1.0,
            "pitch": 0
        },
        "audio_setting": {
            "sample_rate": 32000,
            "bitrate": 128000,
            "format": "mp3",
            "channel": 1
        }
    }

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
                   voice_gender: str = 'female', api_key: str = None, 
                   azure_region: str = None, google_credentials_path: str = None,
                   minimax_api_key: str = None, minimax_group_id: str = None, 
                   minimax_voice_id: str = None) -> dict:
    """
    Generates audio using the specified TTS service.
    Ensures the output directory exists.
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

    logger.info(f"Generating audio for text: \"{text_to_speak[:30]}...\" using service: {service} to file: {output_filename}")
    service = service.lower()
    if service == "edge_tts":
        return _generate_edge_tts(text_to_speak, output_filename, voice_gender)
    elif service == "azure":
        return _generate_azure_tts(text_to_speak, output_filename, voice_gender, api_key, azure_region)
    elif service == "google":
        return _generate_google_tts(text_to_speak, output_filename, voice_gender, google_credentials_path)
    elif service == "minimax":
        return _generate_minimax_tts(text_to_speak, output_filename, voice_gender, 
                                     minimax_api_key, minimax_group_id, minimax_voice_id)
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
    if edge_tts:
        edge_output_file_female = os.path.join(output_dir, "edge_tts_female_test.mp3")
        edge_result_female = generate_audio(sample_chinese_text, edge_output_file_female, "edge_tts", voice_gender="female")
        logger.info(f"EdgeTTS (Female) Result: {edge_result_female}")
        if edge_result_female['success']:
            logger.info(f"  Output: {os.path.abspath(edge_output_file_female)}")

        edge_output_file_male = os.path.join(output_dir, "edge_tts_male_test.mp3")
        edge_result_male = generate_audio(sample_chinese_text, edge_output_file_male, "edge_tts", voice_gender="male")
        logger.info(f"EdgeTTS (Male) Result: {edge_result_male}")
        if edge_result_male['success']:
            logger.info(f"  Output: {os.path.abspath(edge_output_file_male)}")
    else:
        logger.warning("EdgeTTS library not found. Skipping EdgeTTS tests. Install with: pip install edge-tts")
    
    logger.info("\n--- Testing Azure TTS (Placeholder) ---")
    azure_api_key_test = os.environ.get("AZURE_SPEECH_KEY") or "YOUR_AZURE_SPEECH_KEY" 
    azure_region_test = os.environ.get("AZURE_SPEECH_REGION") or "YOUR_AZURE_REGION"

    if not SpeechConfig:
        logger.warning("azure-cognitiveservices-speech library not found. Skipping Azure TTS tests. Install with: pip install azure-cognitiveservices-speech")
    elif azure_api_key_test == "YOUR_AZURE_SPEECH_KEY" or azure_region_test == "YOUR_AZURE_REGION":
        logger.warning(f"Azure credentials not set (use env vars AZURE_SPEECH_KEY, AZURE_SPEECH_REGION or edit script). Skipping.")
    else:
        logger.info(f"Attempting Azure TTS with key: '...{azure_api_key_test[-4:]}' and region: '{azure_region_test}'")
        azure_output_file = os.path.join(output_dir, "azure_tts_female_test.mp3")
        azure_result = generate_audio(sample_chinese_text, azure_output_file, "azure", 
                                      voice_gender="female", api_key=azure_api_key_test, azure_region=azure_region_test)
        logger.info(f"AzureTTS (Female) Result: {azure_result}")
        if azure_result['success']:
            logger.info(f"  Output: {os.path.abspath(azure_output_file)}")

    logger.info("\n--- Testing Google Cloud TTS (Placeholder) ---")
    google_creds_path_test = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") 

    if not texttospeech:
        logger.warning("google-cloud-texttospeech library not found. Skipping Google TTS tests. Install with: pip install google-cloud-texttospeech")
    elif not google_creds_path_test:
        logger.warning("GOOGLE_APPLICATION_CREDENTIALS environment variable not set. Skipping Google TTS test.")
        logger.warning("Alternatively, ensure you are authenticated for ADC if running on GCP or with gcloud CLI.")
    else:
        logger.info(f"Attempting Google TTS with credentials from: '{google_creds_path_test}'")
        google_output_file = os.path.join(output_dir, "google_tts_female_test.mp3")
        google_result = generate_audio(sample_chinese_text, google_output_file, "google", 
                                       voice_gender="female", google_credentials_path=google_creds_path_test)
        logger.info(f"GoogleTTS (Female) Result: {google_result}")
        if google_result['success']:
            logger.info(f"  Output: {os.path.abspath(google_output_file)}")

    logger.info("\n--- Testing Minimax TTS (Placeholder) ---")
    minimax_api_key_test = os.environ.get("MINIMAX_API_KEY") or "YOUR_MINIMAX_API_KEY"
    minimax_group_id_test = os.environ.get("MINIMAX_GROUP_ID") or "YOUR_MINIMAX_GROUP_ID"

    if minimax_api_key_test == "YOUR_MINIMAX_API_KEY" or minimax_group_id_test == "YOUR_MINIMAX_GROUP_ID":
        logger.warning("Minimax credentials not set (use env vars MINIMAX_API_KEY, MINIMAX_GROUP_ID or edit script). Skipping.")
    else:
        logger.info(f"Attempting Minimax TTS with API key '...{minimax_api_key_test[-4:]}' and Group ID '{minimax_group_id_test}'")
        minimax_output_file = os.path.join(output_dir, "minimax_tts_female_test.mp3")
        minimax_result = generate_audio(sample_chinese_text, minimax_output_file, "minimax",
                                        voice_gender="female", 
                                        minimax_api_key=minimax_api_key_test, 
                                        minimax_group_id=minimax_group_id_test)
        logger.info(f"MinimaxTTS (Female) Result: {minimax_result}")

    logger.info("\n--- Finished TTS Tests ---")
    logger.info(f"Please check the '{output_dir}' directory for any generated MP3 files.")
    logger.info("For cloud services, ensure you have installed their respective SDKs (e.g., pip install azure-cognitiveservices-speech google-cloud-texttospeech) and configured credentials.")
