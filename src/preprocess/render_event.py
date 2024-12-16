"""Render event tokens to midi and audio.

Usage: python3 render_event.py [--event path/to/event] [--output_dir output_dir]

"""
import os
import json
import scipy.io.wavfile

import sys
sys.path.append("..")
from utils.common import load_event
from utils.event import expand_score, event_to_pm

# Constant
from utils.constants import DATA_DIR


def main(event_file, output_dir, repeat_mode="no_repeat", to_audio=True, fs=44100.0):

    composer = os.path.basename(os.path.dirname(event_file))
    prefix = os.path.basename(event_file).split(".")[0]

    os.makedirs(os.path.join(output_dir, composer), exist_ok=True)

    midi_file = os.path.join(output_dir, composer, f"{prefix}.mid")
    mapping_file = os.path.join(output_dir, composer, f"{prefix}.json")
    audio_file = os.path.join(output_dir, composer, f"{prefix}.wav")

    # Render event to midi
    score_event, struct = load_event(event_file)
    event, idx_mapping = expand_score(score_event, struct, repeat_mode)

    # render to midi with quantized tempo
    pm, cpt = event_to_pm(event, quantize_tp=True)

    pm.write(midi_file)

    with open(mapping_file, "w") as f:
        json.dump({"idx_mapping": idx_mapping, "onset": cpt}, f)

    if to_audio:
        # Render midi to audio
        audio = pm.fluidsynth(fs=float(fs))
        scipy.io.wavfile.write(audio_file, int(fs), audio)

    return


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--event", dest="event_file", type=str,
                        help="Input event file name.")
    parser.add_argument("--output_dir", dest="output_dir", type=str,
                        default=f"{DATA_DIR}/midi",
                        help="Output path. Defaults to DATA_DIR/midi.")
    parser.add_argument("--fs", dest="fs", type=float,
                        default=44100.0, help="Rendered audio sampling frequency.")
    parser.add_argument(
        "--unroll_mode", dest="unroll_mode", type=str, default="volta_only",
        help="Unroll score mode. `volta_only`, `no_repeat` or `full`. Default to `volta_only`.")
    parser.add_argument(
        "--to_audio", dest="to_audio", action="store_true",
        help="Render to audio. Defaults to false.")

    args = parser.parse_args()

    main(args.event_file, args.output_dir,
         repeat_mode=args.unroll_mode, fs=args.fs, to_audio=args.to_audio)
