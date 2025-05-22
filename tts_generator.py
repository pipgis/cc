import requests
import json
import os
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
        return {'success': True, 'error': None}
    except Exception as e:
        return {'success': False, 'error': f"EdgeTTS generation failed: {e}"}

def _generate_edge_tts(text_to_speak: str, output_filename: str, voice_gender: str = 'female') -> dict:
    """
    Synchronous wrapper for EdgeTTS generation.
    """
    try:
        asyncio.run(_generate_edge_tts_async(text_to_speak, output_filename, voice_gender))
        # Check if file was created and has size, as edge_tts.save might not raise error for all failures
        if os.path.exists(output_filename) and os.path.getsize(output_filename) > 0:
            return {'success': True, 'error': None}
        else:
            return {'success': False, 'error': "EdgeTTS file not created or empty, check text or voice."}
    except Exception as e:
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
            return {'success': True, 'error': None}
        elif result.reason == ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            error_message = f"Azure TTS synthesis canceled: {cancellation_details.reason}"
            if cancellation_details.reason == CancellationReason.Error:
                error_message += f" - Error details: {cancellation_details.error_details}"
            return {'success': False, 'error': error_message}
        else:
            return {'success': False, 'error': f"Azure TTS synthesis failed with reason: {result.reason}"}

    except Exception as e:
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
            print("Google TTS: Attempting to use Application Default Credentials.")


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
        return {'success': True, 'error': None}

    except Exception as e:
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
    # MINIMAX_TTS_API_URL = "https://api.minimax.ai/v1/text_to_speech" # Example URL
    
    # print(f"Attempting Minimax TTS for: {text_to_speak[:30]}...")
    # print(f"Output to: {output_filename}")
    # print(f"Voice Gender: {voice_gender}, Voice ID: {minimax_voice_id}")
    # print("Note: Minimax TTS is currently a placeholder and will not produce audio.")

    # Example payload structure (needs to be verified with Minimax docs)
    # payload = {
    #     "text": text_to_speak,
    #     "voice_id": minimax_voice_id or ("male_voice_default" if voice_gender == 'male' else "female_voice_default"),
    #     "output_format": "mp3"
    # }
    # headers = {
    #     "Authorization": f"Bearer {minimax_api_key}",
    #     "X-Group-ID": minimax_group_id, # Or however group ID is passed
    #     "Content-Type": "application/json"
    # }

    # try:
    #     response = requests.post(MINIMAX_TTS_API_URL, json=payload, headers=headers, timeout=30)
    #     response.raise_for_status()
    #     with open(output_filename, 'wb') as f:
    #         f.write(response.content)
    #     return {'success': True, 'error': None}
    # except requests.exceptions.RequestException as e:
    #     return {'success': False, 'error': f"Minimax API request failed: {e}"}
    # except Exception as e:
    #     return {'success': False, 'error': f"Minimax TTS processing failed: {e}"}
    
    return {'success': False, 'error': 'Minimax TTS not yet implemented or API details needed.'}


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
            os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        return {'success': False, 'error': f"Failed to create output directory for {output_filename}: {e}"}


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
        return {'success': False, 'error': f"Unsupported TTS service: {service}"}


# --- Test Block ---
if __name__ == '__main__':
    sample_chinese_text = "你好，这是一个测试语音。今天天气怎么样？希望你能喜欢这个声音。"
    output_dir = "tts_output" # Relative to where the script is run
    # Ensure the output directory exists for the test
    # The generate_audio function also does this, but good for clarity here.
    os.makedirs(output_dir, exist_ok=True) 
    print(f"Created output directory: {os.path.abspath(output_dir)}")

    print("\n--- Testing EdgeTTS ---")
    if edge_tts:
        edge_output_file_female = os.path.join(output_dir, "edge_tts_female_test.mp3")
        edge_result_female = generate_audio(sample_chinese_text, edge_output_file_female, "edge_tts", voice_gender="female")
        print(f"EdgeTTS (Female) Result: {edge_result_female}")
        if edge_result_female['success']:
            print(f"  Output: {os.path.abspath(edge_output_file_female)}")

        edge_output_file_male = os.path.join(output_dir, "edge_tts_male_test.mp3")
        edge_result_male = generate_audio(sample_chinese_text, edge_output_file_male, "edge_tts", voice_gender="male")
        print(f"EdgeTTS (Male) Result: {edge_result_male}")
        if edge_result_male['success']:
            print(f"  Output: {os.path.abspath(edge_output_file_male)}")
    else:
        print("EdgeTTS library not found. Skipping EdgeTTS tests. Install with: pip install edge-tts")
    
    # --- Placeholder for Azure ---
    print("\n--- Testing Azure TTS (Placeholder) ---")
    # Replace with your actual key and region to test
    azure_api_key_test = os.environ.get("AZURE_SPEECH_KEY") or "YOUR_AZURE_SPEECH_KEY" 
    azure_region_test = os.environ.get("AZURE_SPEECH_REGION") or "YOUR_AZURE_REGION"

    if not SpeechConfig:
        print("azure-cognitiveservices-speech library not found. Skipping Azure TTS tests. Install with: pip install azure-cognitiveservices-speech")
    elif azure_api_key_test == "YOUR_AZURE_SPEECH_KEY" or azure_region_test == "YOUR_AZURE_REGION":
        print(f"Azure credentials not set (use env vars AZURE_SPEECH_KEY, AZURE_SPEECH_REGION or edit script). Skipping.")
    else:
        print(f"Attempting Azure TTS with key: '...{azure_api_key_test[-4:]}' and region: '{azure_region_test}'")
        azure_output_file = os.path.join(output_dir, "azure_tts_female_test.mp3")
        azure_result = generate_audio(sample_chinese_text, azure_output_file, "azure", 
                                      voice_gender="female", api_key=azure_api_key_test, azure_region=azure_region_test)
        print(f"AzureTTS (Female) Result: {azure_result}")
        if azure_result['success']:
            print(f"  Output: {os.path.abspath(azure_output_file)}")

    # --- Placeholder for Google Cloud TTS ---
    print("\n--- Testing Google Cloud TTS (Placeholder) ---")
    # Set GOOGLE_APPLICATION_CREDENTIALS environment variable to the path of your JSON credentials file
    google_creds_path_test = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") # or "path/to/your/credentials.json"

    if not texttospeech:
        print("google-cloud-texttospeech library not found. Skipping Google TTS tests. Install with: pip install google-cloud-texttospeech")
    elif not google_creds_path_test:
        print("GOOGLE_APPLICATION_CREDENTIALS environment variable not set. Skipping Google TTS test.")
        print("Alternatively, ensure you are authenticated for ADC if running on GCP or with gcloud CLI.")
    else:
        print(f"Attempting Google TTS with credentials from: '{google_creds_path_test}'")
        google_output_file = os.path.join(output_dir, "google_tts_female_test.mp3")
        google_result = generate_audio(sample_chinese_text, google_output_file, "google", 
                                       voice_gender="female", google_credentials_path=google_creds_path_test)
        print(f"GoogleTTS (Female) Result: {google_result}")
        if google_result['success']:
            print(f"  Output: {os.path.abspath(google_output_file)}")

    # --- Placeholder for Minimax TTS ---
    print("\n--- Testing Minimax TTS (Placeholder) ---")
    # Replace with your actual keys/IDs to test
    minimax_api_key_test = os.environ.get("MINIMAX_API_KEY") or "YOUR_MINIMAX_API_KEY"
    minimax_group_id_test = os.environ.get("MINIMAX_GROUP_ID") or "YOUR_MINIMAX_GROUP_ID"
    # minimax_voice_id_test = "some_voice_id_for_chinese_female" # Example

    if minimax_api_key_test == "YOUR_MINIMAX_API_KEY" or minimax_group_id_test == "YOUR_MINIMAX_GROUP_ID":
        print("Minimax credentials not set (use env vars MINIMAX_API_KEY, MINIMAX_GROUP_ID or edit script). Skipping.")
    else:
        print(f"Attempting Minimax TTS with API key '...{minimax_api_key_test[-4:]}' and Group ID '{minimax_group_id_test}'")
        minimax_output_file = os.path.join(output_dir, "minimax_tts_female_test.mp3")
        # Note: _generate_minimax_tts is currently a placeholder and will return an error.
        minimax_result = generate_audio(sample_chinese_text, minimax_output_file, "minimax",
                                        voice_gender="female", 
                                        minimax_api_key=minimax_api_key_test, 
                                        minimax_group_id=minimax_group_id_test)
                                        # minimax_voice_id=minimax_voice_id_test)
        print(f"MinimaxTTS (Female) Result: {minimax_result}")
        # No output path check here as it's expected to fail for now.

    print("\n--- Finished TTS Tests ---")
    print(f"Please check the '{output_dir}' directory for any generated MP3 files.")
    print("For cloud services, ensure you have installed their respective SDKs (e.g., pip install azure-cognitiveservices-speech google-cloud-texttospeech) and configured credentials.")
