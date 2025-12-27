# Codes by Vision
import argparse
import sys
import os
from detector import DeepTraceDetector
from cli_utils import display_banner, print_status, print_success, print_error, print_warning

def main():
    display_banner()
    parser = argparse.ArgumentParser(description="DeepTrace: Deepfake Detection System")
    parser.add_argument("input", help="Path to the video or audio file to analyze")
    args = parser.parse_args()
    if not os.path.exists(args.input):
        print_error(f"File not found: {args.input}")
        sys.exit(1)
    print_status(f"Starting analysis for: {os.path.basename(args.input)}")
    try:
        detector = DeepTraceDetector()
        results = detector.analyze(args.input)
        print("\n" + "="*30)
        print("      DETECTION RESULTS      ")
        print("="*30)
        overall_fake = False
        v_res = results.get("video")
        a_res = results.get("audio")
        i_res = results.get("image")
        if v_res:
            label = "\033[1;31mFAKE\033[0m" if v_res['prediction'] == "FAKE" else "\033[1;32mREAL\033[0m"
            print(f"Video Stream: {label} (Conf: {v_res['confidence']:.2%})")
            if v_res['prediction'] == "FAKE": overall_fake = True
        if a_res:
            label = "\033[1;31mFAKE\033[0m" if a_res['prediction'] == "FAKE" else "\033[1;32mREAL\033[0m"
            print(f"Audio Stream: {label} (Conf: {a_res['confidence']:.2%})")
            if a_res['prediction'] == "FAKE": overall_fake = True
        if i_res:
            label = "\033[1;31mFAKE\033[0m" if i_res['prediction'] == "FAKE" else "\033[1;32mREAL\033[0m"
            print(f"Image Analysis: {label} (Conf: {i_res['confidence']:.2%})")
            if i_res['prediction'] == "FAKE": overall_fake = True
        if not v_res and not a_res and not i_res:
            print_warning("No analysis results generated. Models may be missing or file type unsupported.")
        else:
            final_verdict = "\033[1;31mFAKE\033[0m" if overall_fake else "\033[1;32mREAL\033[0m"
            print("-" * 30)
            print(f"FINAL VERDICT: {final_verdict}")
            print("=" * 30)
    except Exception as e:
        print_error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()
