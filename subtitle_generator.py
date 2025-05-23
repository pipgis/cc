import os
from datetime import timedelta
import math
import logging

logger = logging.getLogger(__name__)

def format_srt_time(seconds_float: float) -> str:
    """
    Formats total seconds into SRT timecode HH:MM:SS,mmm.
    """
    if seconds_float < 0:
        seconds_float = 0
    # Using timedelta for robust time calculation
    td = timedelta(seconds=seconds_float)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int(td.microseconds / 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def format_lrc_time(seconds_float: float) -> str:
    """
    Formats total seconds into LRC timecode [mm:ss.xx].
    """
    if seconds_float < 0:
        seconds_float = 0
    minutes = int(seconds_float // 60)
    seconds = seconds_float % 60
    # xx represents hundredths of a second
    hundredths = int((seconds_float - (minutes * 60) - math.floor(seconds)) * 100)
    return f"[{minutes:02}:{math.floor(seconds):02}.{hundredths:02}]"

def generate_srt(text_content: str, audio_duration_seconds: float, output_filename: str, 
                 lines_per_segment: int = 2, segment_duration_seconds: float = 5.0) -> dict:
    """
    Generates an SRT subtitle file.
    """
    if not text_content or not text_content.strip():
        logger.error("SRT generation error: Input text is empty.")
        return {'success': False, 'error': "Input text is empty."}
    if audio_duration_seconds <= 0:
        logger.error("SRT generation error: Audio duration must be positive.")
        return {'success': False, 'error': "Audio duration must be positive."}
    
    logger.info(f"Generating SRT file for '{output_filename}' with audio duration {audio_duration_seconds}s.")
    try:
        output_dir = os.path.dirname(output_filename)
        if output_dir and not os.path.exists(output_dir):
            logger.info(f"Creating directory for SRT file: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)

        lines = [line.strip() for line in text_content.strip().split('\n') if line.strip()]
        if not lines:
            logger.warning("SRT generation: No valid lines in text content after stripping.")
            return {'success': False, 'error': "No valid lines in text content after stripping."}

        with open(output_filename, 'w', encoding='utf-8') as f:
            segment_index = 1
            current_time_seconds = 0.0
            
            for i in range(0, len(lines), lines_per_segment):
                if current_time_seconds >= audio_duration_seconds:
                    logger.info("SRT generation: Reached end of audio duration.")
                    break 

                segment_lines = lines[i : i + lines_per_segment]
                segment_text = "\n".join(segment_lines)

                start_time_str = format_srt_time(current_time_seconds)
                
                end_time_seconds = min(current_time_seconds + segment_duration_seconds, audio_duration_seconds)
                
                if (end_time_seconds - current_time_seconds) < 0.1 and segment_index > 1:
                    logger.debug(f"SRT segment {segment_index} too short, stopping generation.")
                    break

                end_time_str = format_srt_time(end_time_seconds)

                f.write(f"{segment_index}\n")
                f.write(f"{start_time_str} --> {end_time_str}\n")
                f.write(f"{segment_text}\n\n")

                segment_index += 1
                current_time_seconds = end_time_seconds 

                if current_time_seconds < audio_duration_seconds:
                    current_time_seconds += 0.001 
            
            logger.info(f"SRT file generated successfully: {output_filename} with {segment_index-1} segments.")
        return {'success': True, 'error': None}
    except IOError as e:
        logger.error(f"File I/O error during SRT generation for {output_filename}: {e}", exc_info=True)
        return {'success': False, 'error': f"File I/O error: {e}"}
    except Exception as e:
        logger.error(f"An unexpected error occurred during SRT generation for {output_filename}: {e}", exc_info=True)
        return {'success': False, 'error': f"An unexpected error occurred: {e}"}


def generate_lrc(text_content: str, audio_duration_seconds: float, output_filename: str) -> dict:
    """
    Generates an LRC subtitle file.
    Lines are distributed evenly across the audio duration.
    """
    if not text_content or not text_content.strip():
        logger.error("LRC generation error: Input text is empty.")
        return {'success': False, 'error': "Input text is empty."}
    if audio_duration_seconds <= 0:
        logger.error("LRC generation error: Audio duration must be positive.")
        return {'success': False, 'error': "Audio duration must be positive."}

    logger.info(f"Generating LRC file for '{output_filename}' with audio duration {audio_duration_seconds}s.")
    try:
        output_dir = os.path.dirname(output_filename)
        if output_dir and not os.path.exists(output_dir):
            logger.info(f"Creating directory for LRC file: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)

        lines = [line.strip() for line in text_content.strip().split('\n') if line.strip()]
        if not lines:
            logger.warning("LRC generation: No valid lines in text content after stripping.")
            return {'success': False, 'error': "No valid lines in text content after stripping."}

        num_lines = len(lines)
        if num_lines == 0: # Should be caught by `if not lines` already, but as a safeguard.
            logger.warning("LRC generation: No lines to process.")
            return {'success': False, 'error': "No lines to process for LRC."}
            
        time_per_line = audio_duration_seconds / num_lines
        logger.debug(f"LRC generation: {num_lines} lines, {time_per_line:.3f}s per line.")

        with open(output_filename, 'w', encoding='utf-8') as f:
            current_time_seconds = 0.0
            for line_index, line_text in enumerate(lines):
                lrc_time_str = format_lrc_time(current_time_seconds)
                f.write(f"{lrc_time_str}{line_text}\n")
                current_time_seconds += time_per_line
                if line_index == num_lines -1 : # If it's the last line
                     # Ensure the last timestamp doesn't exceed audio_duration by a tiny fraction
                     # or is exactly at the start of the last segment if that's preferred.
                     # For LRC, each line has its own timestamp, so this adjustment is mainly for theoretical precision.
                     current_time_seconds = min(current_time_seconds, audio_duration_seconds)


        logger.info(f"LRC file generated successfully: {output_filename}")
        return {'success': True, 'error': None}
    except IOError as e:
        logger.error(f"File I/O error during LRC generation for {output_filename}: {e}", exc_info=True)
        return {'success': False, 'error': f"File I/O error: {e}"}
    except Exception as e:
        logger.error(f"An unexpected error occurred during LRC generation for {output_filename}: {e}", exc_info=True)
        return {'success': False, 'error': f"An unexpected error occurred: {e}"}


if __name__ == '__main__':
    # Basic setup for testing this module directly
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    sample_text_raw = "这是第一行字幕。\n这是第二行，稍微长一点。\n然后是第三行。\n最后是第四行，作为结束。"
    processed_sample_text = sample_text_raw 

    sample_audio_duration = 15.0  # seconds
    output_dir = "subtitle_output"
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {os.path.abspath(output_dir)}")

    logger.info(f"\n--- Testing Time Formatters ---")
    logger.info(f"SRT format for 65.123 seconds: {format_srt_time(65.123)}") 
    logger.info(f"SRT format for 3600 seconds: {format_srt_time(3600.0)}")   
    logger.info(f"SRT format for 0.5 seconds: {format_srt_time(0.5)}")       
    logger.info(f"LRC format for 65.12 seconds: {format_lrc_time(65.12)}")   
    logger.info(f"LRC format for 5.789 seconds: {format_lrc_time(5.789)}")   
    logger.info(f"LRC format for 123.456 seconds: {format_lrc_time(123.456)}")


    logger.info(f"\n--- Generating SRT subtitle ---")
    srt_output_file = os.path.join(output_dir, "test_subtitle.srt")
    srt_result = generate_srt(processed_sample_text, sample_audio_duration, srt_output_file, 
                              lines_per_segment=1, segment_duration_seconds=3.5)
    logger.info(f"SRT Generation Result: {srt_result}")
    if srt_result['success']:
        logger.info(f"  SRT file: {os.path.abspath(srt_output_file)}")
        with open(srt_output_file, 'r', encoding='utf-8') as f_srt:
            logger.debug(f"  SRT Content:\n{f_srt.read()}")

    logger.info(f"\n--- Generating SRT subtitle (2 lines per segment) ---")
    srt_output_file_2l = os.path.join(output_dir, "test_subtitle_2lines.srt")
    srt_result_2l = generate_srt(processed_sample_text, sample_audio_duration, srt_output_file_2l,
                                 lines_per_segment=2, segment_duration_seconds=6)
    logger.info(f"SRT Generation Result (2 lines): {srt_result_2l}")
    if srt_result_2l['success']:
        logger.info(f"  SRT file: {os.path.abspath(srt_output_file_2l)}")
        with open(srt_output_file_2l, 'r', encoding='utf-8') as f_srt_2l:
            logger.debug(f"  SRT Content (2 lines):\n{f_srt_2l.read()}")


    logger.info(f"\n--- Generating LRC subtitle ---")
    lrc_output_file = os.path.join(output_dir, "test_subtitle.lrc")
    lrc_result = generate_lrc(processed_sample_text, sample_audio_duration, lrc_output_file)
    logger.info(f"LRC Generation Result: {lrc_result}")
    if lrc_result['success']:
        logger.info(f"  LRC file: {os.path.abspath(lrc_output_file)}")
        with open(lrc_output_file, 'r', encoding='utf-8') as f_lrc:
            logger.debug(f"  LRC Content:\n{f_lrc.read()}")

    logger.info(f"\n--- Test with short audio duration for SRT ---")
    short_audio_duration = 3.0 
    srt_short_output_file = os.path.join(output_dir, "test_subtitle_short.srt")
    srt_short_result = generate_srt(processed_sample_text, short_audio_duration, srt_short_output_file,
                                    lines_per_segment=1, segment_duration_seconds=5.0)
    logger.info(f"SRT Short Audio Result: {srt_short_result}")
    if srt_short_result['success']:
        logger.info(f"  SRT file (short): {os.path.abspath(srt_short_output_file)}")
        with open(srt_short_output_file, 'r', encoding='utf-8') as f_srt_short:
            logger.debug(f"  SRT Content (short):\n{f_srt_short.read()}")
