"""
render_event.py

Description:
    Renders a score event JSON file into a MIDI file and optionally into audio (e.g., WAV).

Usage: 
    python3 render_event.py --event path/to/event.json --output_dir path/to/output 
                            [--mode MODE] [--fs SAMPLING_RATE] [--to_audio]

Arguments:
    --event_file         Path to the input event file (JSON format). (required)
    --output_dir    Directory to save the rendered MIDI and audio files. (required)
    --mode          Score expansion mode: 'volta_only', 'no_repeat', or 'full'. Defaults to 'no_repeat'. (optional)
                    Given the pattern ["A", "A1", "A", "A2", "B", "B"], the unfolded score will be:
                        - ["A", "A1", "A", "A2", "B"]      if mode = "volta_only"
                        - ["A", "A1", "A", "A2", "B", "B"]  if mode = "full"
                        - ["A", "A2", "B"]                  if mode = "no_repeat"
    --fs            Sampling frequency for audio rendering. 
                    Defaults to 44100.0 Hz. (optional)
    --to_audio      If set, also renders audio from the MIDI file. (optional; boolean flag)

Examples:
    # Render an event file to MIDI only
    python3 render_event.py --event ../event/mozart/sample1.json --output_dir ../midi/mozart/

    # Render an event file to MIDI and audio
    python3 render_event.py --event ../event/mozart/sample1.json --output_dir ../midi/mozart/ --to_audio
"""
import os
import json
import scipy.io.wavfile

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.common import load_event
from utils.event import expand_score, event_to_pm


def main(event_file, output_dir, repeat_mode="no_repeat", to_audio=True, fs=44100.0):

    composer = os.path.basename(os.path.dirname(event_file))
    prefix = os.path.basename(event_file).split(".")[0]

    os.makedirs(os.path.join(output_dir, composer), exist_ok=True)

    midi_file = os.path.join(output_dir, composer, f"{prefix}.mid")
    mapping_file = os.path.join(output_dir, composer, f"{prefix}.json")
    audio_file = os.path.join(output_dir, composer, f"{prefix}.wav")

    # Render event to midi
    score_event, mark = load_event(event_file)
    event, idx_mapping = expand_score(score_event, mark, repeat_mode)

    # render to midi with quantized tempo
    pm, cpt = event_to_pm(event, quantize_tp=True)

    pm.write(midi_file)

    with open(mapping_file, "w") as f:
        json.dump({"idx_mapping": idx_mapping, "cpt": cpt}, f)

    if to_audio:
        # Render midi to audio
        audio = pm.fluidsynth(fs=float(fs))
        scipy.io.wavfile.write(audio_file, int(fs), audio)

    return


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--event_file", type=str, required=True,
                        help="Path to the input event JSON file ")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the rendered MIDI and audio files")
    parser.add_argument("--fs", type=float, default=44100.0,
                        help="Rendered audio sampling frequency.")
    parser.add_argument("--mode", type=str, default="no_repeat",
                        help="'volta_only', 'no_repeat' or 'full'. Default to 'no_repeat'.")
    parser.add_argument("--to_audio", action="store_true",
                        help="Render to audio. Defaults to false.")

    args = parser.parse_args()
    main(args.event_file, args.output_dir, repeat_mode=args.mode, fs=args.fs, to_audio=args.to_audio)
