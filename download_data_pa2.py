import torchaudio
import os
import shutil

torchaudio.datasets.SPEECHCOMMANDS(root='./', download=True)

src = "SpeechCommands/speech_commands_v0.02"
dst = "SpeechCommands"

# Make sure destination exists
os.makedirs(dst, exist_ok=True)

for item in os.listdir(src):
    s = os.path.join(src, item)
    d = os.path.join(dst, item)
    shutil.move(s, d)

os.rmdir(src)

old_name = "SpeechCommands/_background_noise_"
new_name = "SpeechCommands/noise"

os.rename(old_name, new_name)

base_dir = "SpeechCommands"
destination = os.path.join(base_dir, "speech")
skip_folder = "noise"   # Name only, not full path

os.makedirs(destination, exist_ok=True)

for item in os.listdir(base_dir):
    if item == skip_folder or item == os.path.basename(destination):
        continue  # Skip the folder we want to keep (and avoid moving the destination into itself)

    src = os.path.join(base_dir, item)
    dst = os.path.join(destination, item)

    shutil.move(src, dst)

dst = "SpeechCommands/speech/noise"
os.makedirs(dst, exist_ok=True)
src = "SpeechCommands/noise/doing_the_dishes.wav"
shutil.copy(src, dst)