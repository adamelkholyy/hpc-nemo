import jiwer
import os 
import re
import argparse

# initialize parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "-r", "--reference", help="path of the reference (ground truth) transcript", default=None
)
parser.add_argument(
    "-h", "--hypothesis", help="path of the hypothesis transcript", default=None
)

def write_to_file(filepath, text):
    dirname = os.path.dirname(filepath) 
    filename = os.path.basename(filepath)
    out_folder = os.path.join(dirname, "processed")
    os.makedirs(out_folder, exist_ok=True)
    filepath = os.path.join(out_folder, "processed_" + filename[:-4])

    with open(filepath, "w") as file:
        file.write(text)

    return filepath

def preprocess_verity_hypothesis(hypothesis_filepath):
    with open(hypothesis_filepath, "r") as file:
        hypothesis = file.read()

    # remove the first line from the hypothesis content.
    hypothesis = "\n".join(hypothesis.splitlines()[1:])

    # remove speaker labels from the hypothesis content
    hypothesis = hypothesis.replace("Speaker 1: ", "").replace("Speaker 0: ", "")

    filepath = write_to_file(hypothesis_filepath, hypothesis)
    return filepath

def preprocess_verity_reference(reference_filepath):

    with open(reference_filepath, "r") as file:
        reference = file.read()

    # remove the first line from the reference content.
    reference = "\n".join(reference.splitlines()[1:])

    # remove any text within square brackets and the brackets themselves.
    reference = re.sub(r"\[.*?\]", "", reference)

    # replace times in the form dd:dd or hh:mm:ss (including 1:00:00 format) that are between two newlines with a space
    reference = re.sub(r"\n\d{1,2}(:\d{2}){1,2}\n", " ", reference)

    # replace two or more consecutive newlines with optional whitespace in between with a single newline
    reference = re.sub(r"\n[\s]*\n+", "\n", reference)

    # replace single newlines with double to match hypothesis
    reference = re.sub(r"\n", "\n\n", reference)

    # remove all leading whitespace from the beginning of the text
    reference = reference.lstrip()

    # remove all trailing whitespace from the end of the text
    reference = reference.rstrip()

    filepath = write_to_file(reference_filepath, reference)
    return filepath


def calculate_wer(processed_reference_filepath, processed_hypothesis_filepath):

    with open(processed_reference_filepath, "r") as file:
        reference = file.read()

    with open(processed_hypothesis_filepath, "r") as file:
        hypothesis = file.read()

    # https://medium.com/@johnidouglasmarangon/how-to-calculate-the-word-error-rate-in-python-ce0751a46052#:~:text=In%20short%2C%20WER%20is%20calculated,words%20in%20the%20reference%20transcription.
    transforms = jiwer.Compose(
        [
            jiwer.ExpandCommonEnglishContractions(),
            jiwer.RemoveEmptyStrings(),
            jiwer.ToLowerCase(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.RemovePunctuation(),
            jiwer.ReduceToListOfListOfWords(),
        ]
    )

    wer = jiwer.wer(
        reference,
        hypothesis,
        truth_transform=transforms,
        hypothesis_transform=transforms,
    )

    dirname = os.path.dirname(processed_hypothesis_filepath) 
    filename = os.path.basename(processed_hypothesis_filepath)
    filepath = os.path.join(dirname, filename[10:][:-4] + "_results.txt")

    with open(filepath, "w") as file:
        file.write(f"Reference: {os.path.basename(processed_reference_filepath)}\n")
        file.write(f"Hypothesis: {os.path.basename(processed_hypothesis_filepath)}\n")
        file.write(f"Word Error Rate (WER) : {wer:.2f}\n")
        file.write(f"Accuracy              : {((1 - wer) * 100):.2f}\n")
    print(f"WER successfully written to {filepath}")


args = parser.parse_args()
print("Preprocessing reference and hypothesis files...")
processed_reference_filepath = preprocess_verity_reference(args.reference)
processed_hypothesis_filepath = preprocess_verity_hypothesis(args.hypothesis)
print("Calculating Word Error Rate (WER)...")
calculate_wer(processed_reference_filepath, processed_hypothesis_filepath)
print("Word Error Rate (WER) calculation completed successfully.")