import os
import shutil 
DIR = "/lustre/projects/Research_Project-T116269/cobalt-text-srt"
mp3DIR = "/lustre/projects/Research_Project-T116269/cobalt-audio-mp3"

check_audio_dir = False


if __name__ == "__main__":

    """
    DIR = "/lustre/projects/Research_Project-T116269/cobalt-text-srt"

    text_dir_files = [f for f in os.listdir(DIR) if f.endswith(".srt")]
    audio_dir_files = [f for f in os.listdir(mp3DIR) if f.endswith(".srt")]

    
    target = "/lustre/projects/Research_Project-T116269/cobalt-srt-final"
    
    i = 0
    for file in text_dir_files:
        if file in audio_dir_files:
            shutil.copy(os.path.join(mp3DIR, file), target)
        else:
            shutil.copy(os.path.join(DIR, file), target)
        i += 1
        if i % 50 == 0:
            print(f"{i}/{len(text_dir_files)}")

    """
    # exit()



if check_audio_dir:
    files = [f for f in os.listdir(mp3DIR) if f.endswith(".srt")]
else:
    files = [f for f in os.listdir(DIR) if f.endswith(".srt")]
    files = [f for f in files if f not in os.listdir(mp3DIR)]


diar_errs = []
for i, file in enumerate(files):

    if check_audio_dir:
        dir = mp3DIR
    else:
        dir = DIR

    with open(f"{dir}/{file}", "r", encoding="utf-8") as f:
        content = f.read()
    
    lines = len(content.split("\n"))

    if lines < 100:
        # diar_errs.append(file)
        print(lines, file)
        pass
    elif "Speaker 2" in content:
        diar_errs.append(file)
    elif "Speaker 1" not in content:
        diar_errs.append(file)
        
    if i % 50 == 0:
        print(f"{i}/{len(files)}")


print(diar_errs)
print(len(diar_errs))



# with open("diar_errs.txt", "w", encoding="utf-8") as f:
#    f.write("\n".join(diar_errs))