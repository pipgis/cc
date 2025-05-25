import os
from datetime import timedelta
import math
import logging
import re

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
                 lines_per_segment: int = 2, max_chars_per_segment: int = 50) -> dict:
    """
    Generates an SRT subtitle file with adaptive segmentation based on character limits.
    """
    if not text_content or not text_content.strip():
        logger.error("SRT generation error: Input text is empty.")
        return {'success': False, 'error': "Input text is empty."}
    if audio_duration_seconds <= 0:
        logger.error("SRT generation error: Audio duration must be positive.")
        return {'success': False, 'error': "Audio duration must be positive."}
    
    logger.info(f"Generating SRT file for '{output_filename}' with audio duration {audio_duration_seconds}s, "
                f"max_chars_per_segment={max_chars_per_segment}, lines_per_segment={lines_per_segment}.")
    try:
        output_dir = os.path.dirname(output_filename)
        if output_dir and not os.path.exists(output_dir):
            logger.info(f"Creating directory for SRT file: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)

        # Normalize text: replace newlines with spaces to treat content as a single stream for segmentation
        # Preserve original newlines if they are intended as hard breaks by choice, but problem implies continuous text flow.
        # For this implementation, let's assume text_content is a single block of text or pre-formatted with meaningful newlines.
        # If text_content has many \n, it might interfere with max_chars logic or require pre-processing.
        # Let's initially process the text as a whole, then handle potential pre-existing newlines if they complicate things.
        
        # For duration calculation based on characters
        total_chars = len(text_content.replace('\n', '')) # Count characters, excluding newlines for density calculation
        if total_chars == 0:
            logger.warning("SRT generation: Text content is empty after stripping newlines.")
            return {'success': False, 'error': "Text content is empty."}

        # Regex for finding natural split points (includes the punctuation in the split)
        # Prioritizes major punctuation, then minor, then space.
        # The regex captures the character *before* the punctuation and the punctuation itself.
        # This helps in including the punctuation in the current segment.
        split_pattern = re.compile(r'([^\s.。？！，；：]*[.。？！])|([^\s.。？！，；：]*[，；：])|(\S+\s)')
        # More refined regex: match up to max_chars, then backtrack to nearest punctuation or space.
        # This is complex; let's try a simpler iterative approach first.

        segments = []
        remaining_text = text_content.strip()
        current_pos = 0

        while current_pos < len(remaining_text):
            # Determine the end of the current segment
            # Ideal end is current_pos + max_chars_per_segment
            potential_end = min(current_pos + max_chars_per_segment, len(remaining_text))
            segment_text_chunk = remaining_text[current_pos:potential_end]
            
            actual_end = potential_end
            
            # If the chunk is not the end of the text, try to find a natural break
            if potential_end < len(remaining_text):
                # Search for major punctuation from right to left
                # Regex for major/minor punctuation and spaces for splitting
                # Punctuation: . ! ? 。 ！ ？ , ; : ， ； ：
                # We want to split *after* the punctuation.
                # Let's search for the last occurrence of these within the segment_text_chunk
                
                best_split_point = -1

                # Try to find major punctuation first
                for punct_match in re.finditer(r'[.。？！!\?]', segment_text_chunk):
                    best_split_point = punct_match.end() # split after this punctuation
                
                # If no major, try minor punctuation
                if best_split_point == -1:
                    for punct_match in re.finditer(r'[，；：,;:]', segment_text_chunk):
                        best_split_point = punct_match.end()

                # If no punctuation, try space (for non-Chinese text)
                # This needs to be language aware, but for now, we assume space is a valid split point.
                if best_split_point == -1:
                    space_pos = segment_text_chunk.rfind(' ')
                    if space_pos != -1:
                        # Check if the space is not too early (e.g. less than half of max_chars)
                        # This avoids very short segments if a space is found early.
                        # However, for now, any space is a candidate.
                        best_split_point = space_pos + 1 # split after space

                if best_split_point != -1 and best_split_point > 0: # Ensure it's a valid point
                    actual_end = current_pos + best_split_point
                else:
                    # If no natural break is found, and we are not at the end of the text,
                    # we might have to do a hard split.
                    # Or, if the remaining text is short, just take it all.
                    if len(remaining_text) - current_pos <= max_chars_per_segment:
                         actual_end = len(remaining_text)
                    # else, actual_end remains potential_end (hard split at max_chars)

            segment_text = remaining_text[current_pos:actual_end].strip()
            
            if not segment_text: # Should not happen if logic is correct
                current_pos = actual_end
                continue

            # Distribute segment_text into lines_per_segment
            # This part takes the segment_text (which respects max_chars_per_segment)
            # and tries to split it into `lines_per_segment` lines.

            processed_segment_lines = []
            original_lines_in_segment = [line.strip() for line in segment_text.split('\n') if line.strip()]

            if lines_per_segment == 1:
                # Join any original lines with spaces, as we only want one line for display
                processed_segment_lines.append(" ".join(original_lines_in_segment))
            elif original_lines_in_segment and len(original_lines_in_segment) >= lines_per_segment:
                # If the segment text already has enough (or more) newlines, respect them.
                # Take the first `lines_per_segment`. If there are more, join them to the last line.
                processed_segment_lines.extend(original_lines_in_segment[:lines_per_segment-1])
                processed_segment_lines.append(" ".join(original_lines_in_segment[lines_per_segment-1:]))
            elif lines_per_segment == 2: # And original lines are 0 or 1
                # Join original lines first (if any, e.g. if segment_text was "Line1\nTooMuch")
                # then re-split.
                single_line_text = " ".join(original_lines_in_segment) if original_lines_in_segment else segment_text
                
                if not single_line_text.strip(): # If segment is effectively empty
                    processed_segment_lines.append("") # Avoid issues with empty strings later
                elif len(single_line_text) < max_chars_per_segment / lines_per_segment * 1.25 and '\n' not in single_line_text : # If short enough, keep as one line
                    # Heuristic: if it's reasonably short for one line even if two are allowed.
                    # This prevents overly aggressive splitting of short phrases.
                    # max_chars_per_segment / lines_per_segment gives ideal length per line.
                    # 1.25 is a tolerance factor.
                    processed_segment_lines.append(single_line_text)
                else:
                    # Try to split into two lines, near the middle or at a natural break.
                    mid_point = len(single_line_text) // 2
                    best_split_idx = -1
                    
                    # Prefer splitting at punctuation near middle (search window around midpoint)
                    # Search window could be e.g., +/- 25% of midpoint
                    search_radius = mid_point // 4
                    
                    # Look for sentence-ending punctuation first, then commas/colons, then spaces
                    split_chars_priority = [r'[.。？！!\?]', r'[，；：,;:]', r'\s']
                    
                    found_split = False
                    for p_idx, pattern in enumerate(split_chars_priority):
                        # Search backwards from mid + radius
                        for i in range(min(mid_point + search_radius, len(single_line_text) -1), max(mid_point - search_radius -1, 0), -1):
                            if re.match(pattern, single_line_text[i]):
                                # For punctuation, we want to include it in the first line.
                                # For space, we split after the space (so space is not included in second line start)
                                best_split_idx = i + 1 if p_idx < 2 else i # Punctuation included, space excluded
                                found_split = True
                                break
                        if found_split: break
                        # Search forwards from mid - radius (if not found backwards)
                        for i in range(max(mid_point - search_radius, 0), min(mid_point + search_radius + 1, len(single_line_text))):
                             if re.match(pattern, single_line_text[i]):
                                best_split_idx = i + 1 if p_idx < 2 else i
                                found_split = True
                                break
                        if found_split: break

                    if best_split_idx != -1:
                        line1 = single_line_text[:best_split_idx].strip()
                        line2 = single_line_text[best_split_idx:].strip()
                        if line1: processed_segment_lines.append(line1)
                        if line2: processed_segment_lines.append(line2)
                    else:
                        # No ideal natural break found, hard split near middle
                        # Ensure first part isn't empty if mid_point is 0
                        split_p = mid_point if mid_point > 0 else 1 
                        line1 = single_line_text[:split_p].strip()
                        line2 = single_line_text[split_p:].strip()
                        if line1: processed_segment_lines.append(line1)
                        if line2: processed_segment_lines.append(line2)
            else: # Fallback or lines_per_segment > 2 (not explicitly handled for smart splitting here)
                 processed_segment_lines.append(segment_text.replace('\n', ' ').strip())


            # Filter out any empty lines that might have been created
            processed_segment_lines = [line for line in processed_segment_lines if line]
            if not processed_segment_lines and segment_text.strip(): # If all lines became empty but original segment was not
                processed_segment_lines.append(segment_text.strip()) # Put the stripped original back
            elif not processed_segment_lines and not segment_text.strip(): # If original segment was also empty/whitespace
                 # This segment will have 0 char_count and 0 duration.
                 # Let it pass through, it will be handled by the duration logic (skipped or min duration)
                 pass


            final_segment_text_to_display = "\n".join(processed_segment_lines)
            # char_count for duration should be based on the original segment_text that respected max_chars_per_segment
            # not final_segment_text_to_display, as line breaks don't add to reading time in the same way.
            segments.append({'text': final_segment_text_to_display, 'char_count': len(segment_text.replace('\n',''))})
            current_pos = actual_end
            # Skip any leading spaces for the next segment
            while current_pos < len(remaining_text) and remaining_text[current_pos].isspace():
                current_pos += 1
        
        if not segments:
            logger.warning("SRT generation: No segments created.")
            # This might happen if text_content was only whitespace
            return {'success': False, 'error': "No segments could be created from the text."}

        with open(output_filename, 'w', encoding='utf-8') as f:
            segment_index = 1
            current_time_seconds = 0.0
            
            for seg_info in segments:
                segment_text_for_srt = seg_info['text']
                segment_char_count = seg_info['char_count']

                if current_time_seconds >= audio_duration_seconds:
                    logger.info("SRT generation: Reached end of audio duration before processing all segments.")
                    break

                start_time_str = format_srt_time(current_time_seconds)
                
                # Calculate duration for this segment
                # Proportional to character count
                segment_duration = (segment_char_count / total_chars) * audio_duration_seconds if total_chars > 0 else 0
                # Ensure a minimum duration for very short segments if audio_duration is long enough
                # This avoids super fast flashing subtitles. Let's say min 0.5s if possible.
                # segment_duration = max(segment_duration, 0.5) # This could make total duration exceed audio_duration

                end_time_seconds = current_time_seconds + segment_duration
                end_time_seconds = min(end_time_seconds, audio_duration_seconds) # Don't exceed total audio duration
                
                # Sanity check for duration (e.g., minimum 100ms)
                if (end_time_seconds - current_time_seconds) < 0.1:
                    if segment_index > 1 or len(segments) == 1 : # if it's not the first segment or it's the only segment
                         # if it's too short, try to extend it slightly, or merge, or drop.
                         # For now, if it's not the very last possible moment, let it be at least 0.1s
                         end_time_seconds = min(current_time_seconds + 0.1, audio_duration_seconds)
                    else: # if it is the first and there are more, this should not happen with proportional calc unless total_chars is huge
                          pass


                # If this is the last segment, make sure it ends exactly at audio_duration_seconds
                if segment_index == len(segments):
                    end_time_seconds = audio_duration_seconds

                end_time_str = format_srt_time(end_time_seconds)
                
                # Avoid overlapping or negative duration subtitles
                if end_time_seconds <= current_time_seconds and current_time_seconds > 0:
                    # This might happen if a segment has 0 chars (e.g. only newlines) and total_chars is positive
                    # Or if audio_duration_seconds is extremely small
                    # Skip this segment or assign a minimal duration
                    if audio_duration_seconds > current_time_seconds: # if there's still time left
                        end_time_seconds = min(current_time_seconds + 0.1, audio_duration_seconds) # min 0.1s duration
                        end_time_str = format_srt_time(end_time_seconds)
                    else: # no time left, skip
                        logger.debug(f"SRT segment {segment_index} has no duration or negative, skipping.")
                        current_time_seconds = end_time_seconds # which is likely audio_duration_seconds
                        continue


                f.write(f"{segment_index}\n")
                f.write(f"{start_time_str} --> {end_time_str}\n")
                f.write(f"{segment_text_for_srt}\n\n")

                segment_index += 1
                current_time_seconds = end_time_seconds 

                # Small gap between subtitles, but SRT standard implies continuous is fine.
                # A small increment was in the old code (current_time_seconds += 0.001)
                # If using proportional duration, this might not be needed or could cause drift.
                # Let's remove it for now, as time is continuous based on segment content.
                # If a tiny gap is desired, it should be factored into duration calculation or added explicitly.
                # For now, let's make it strictly continuous.
                # If current_time_seconds >= audio_duration_seconds and segment_index <= len(segments):
                #    break # stop if we have already reached the end of audio

            logger.info(f"SRT file generated successfully: {output_filename} with {segment_index - 1} segments.")
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

    logger.info(f"\n--- Generating SRT subtitle (New Logic) ---")
    srt_output_file_new = os.path.join(output_dir, "test_subtitle_new.srt")
    # Using max_chars_per_segment = 30 as per example, lines_per_segment = 2
    srt_result_new = generate_srt(processed_sample_text, sample_audio_duration, srt_output_file_new,
                                  lines_per_segment=2, max_chars_per_segment=30)
    logger.info(f"SRT Generation Result (New Logic): {srt_result_new}")
    if srt_result_new['success']:
        logger.info(f"  SRT file (New Logic): {os.path.abspath(srt_output_file_new)}")
        with open(srt_output_file_new, 'r', encoding='utf-8') as f_srt_new:
            logger.debug(f"  SRT Content (New Logic):\n{f_srt_new.read()}")

    logger.info(f"\n--- Generating SRT subtitle (New Logic, 1 line per segment) ---")
    srt_output_file_new_1l = os.path.join(output_dir, "test_subtitle_new_1line.srt")
    srt_result_new_1l = generate_srt(processed_sample_text, sample_audio_duration, srt_output_file_new_1l,
                                     lines_per_segment=1, max_chars_per_segment=40)
    logger.info(f"SRT Generation Result (New Logic, 1 line): {srt_result_new_1l}")
    if srt_result_new_1l['success']:
        logger.info(f"  SRT file (New Logic, 1 line): {os.path.abspath(srt_output_file_new_1l)}")
        with open(srt_output_file_new_1l, 'r', encoding='utf-8') as f_srt_new_1l:
            logger.debug(f"  SRT Content (New Logic, 1 line):\n{f_srt_new_1l.read()}")

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
    srt_short_output_file = os.path.join(output_dir, "test_subtitle_short_new.srt")
    srt_short_result = generate_srt(processed_sample_text, short_audio_duration, srt_short_output_file,
                                    lines_per_segment=1, max_chars_per_segment=20) # Shorter segments for short audio
    logger.info(f"SRT Short Audio Result (New Logic): {srt_short_result}")
    if srt_short_result['success']:
        logger.info(f"  SRT file (short, New Logic): {os.path.abspath(srt_short_output_file)}")
        with open(srt_short_output_file, 'r', encoding='utf-8') as f_srt_short:
            logger.debug(f"  SRT Content (short, New Logic):\n{f_srt_short.read()}")
