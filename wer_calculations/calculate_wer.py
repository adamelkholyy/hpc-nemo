import jiwer

# TODO
# change to work for any OS
# command line input calculate_wer -r reference_filename -t hypothesis_filename
# append results to txt file - rows: ref_filename, hyp_filename, WER, ACC

reference_filename = "wer_calculations\\processed\\processed_reference_AutHERTS01.txt"
hypothesis_filename = "wer_calculations\\processed\\processed_hypothesis_AutHERTS01.txt"

with open(reference_filename, "r") as file:
    reference = file.read()

with open(hypothesis_filename, "r") as file:
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
print(f"Reference: {reference_filename}")
print(f"Hypothesis: {hypothesis_filename}")
print(f"Word Error Rate (WER) : {wer}")
print(f"Accuracy              : {(1 - wer) * 100}")

