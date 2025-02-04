import os
import argparse
import time
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
# python anonymise_transcript.py --folder transcripts --out anon

# initialize parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "--file", help="name of the text file to be anonymised", default=None
)
parser.add_argument(
    "--folder", help="name of the directory of text files to be anonymised", default=None
)
parser.add_argument(
    "--out", help="name of the output directory to save the anonymised text files", default=None
)
args = parser.parse_args()

# anonymise a folder of text files
def anonymise_from_dir(folder, outdir):
    text_files = [f for f in os.listdir(folder) if f.lower().endswith('.txt')]
    print(f"Found {len(text_files)} text files to anonymise in {folder}...")

    for file in text_files:
        filepath = os.path.join(folder, file)
        anonymise_file(filepath, outdir)

# anonymise a single text file
def anonymise_file(filepath, outdir):
    dirname = os.path.dirname(filepath) 
    filename = os.path.basename(filepath)

    if not outdir:
        outdir = dirname

    # initialize the Presidio Analyzer and Anonymizer
    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()

    with open(filepath, "r") as f:
        text = f.read()

    start = time.time()
    print(f"Anonymising {filename}...")

    # analyze the text to detect PII entities
    results = analyzer.analyze(text=text, entities=None, language="en") # entities=["PERSON", "LOCATION", "PHONE_NUMBER"]

    # anonymize detected PII entities
    anonymized_text = anonymizer.anonymize(text=text, analyzer_results=results)

    outpath = os.path.join(outdir, f"ANONYMISED_{filename}")
    with open(outpath, "w") as f:
        f.write(anonymized_text.text)
    print(f"Anonymised text file saved to {outpath}")
    print(f"Operation complete in {time.time() - start:.2f} seconds")


if args.folder:
    anonymise_from_dir(args.folder, args.out)
elif args.file:
    anonymise_file(args.file, args.out)
else:
    print("Error: Invalid input. Please provide a valid file path with --file or a directory with --folder.")


