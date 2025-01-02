import re

# TODO
# change to work for any OS
# command line input

# read both files
dirname = "wer_calculations/"
reference_filename = "reference_AutSWYS01.txt"
hypothesis_filename = "hypothesis_AutSWYS01.txt"

with open(dirname + reference_filename, "r") as file:
    reference = file.read()

with open(dirname + hypothesis_filename, "r") as file:
    hypothesis = file.read()

"""
   hypothesis preprocessing
"""
# remove the first line from the hypothesis content.
hypothesis = "\n".join(hypothesis.splitlines()[1:])

# remove speaker labels from the hypothesis content
hypothesis = hypothesis.replace("Speaker 1: ", "").replace("Speaker 0: ", "")

"""
   verity transcript reference preprocessing
"""
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

with open(dirname + "processed/processed_"  + reference_filename, "w") as file:
    file.write(reference)

with open(dirname + "processed/processed_" + hypothesis_filename, "w") as file:
    file.write(hypothesis)

print("Preprocessing completed succesfully.")