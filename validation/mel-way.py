import librosa
import numpy as np
import tensorflow as tf
import os

def extract_spectrogram(raw_audio):

    feature_1 = librosa.power_to_db(librosa.feature.melspectrogram(y=raw_audio, sr=44100, n_mels=64, n_fft=2048, hop_length=512))
    feature_2 = librosa.power_to_db(librosa.feature.melspectrogram(y=raw_audio, sr=44100, n_mels=64, n_fft=1024, hop_length=512))
    feature_3 = librosa.power_to_db(librosa.feature.melspectrogram(y=raw_audio, sr=44100, n_mels=64, n_fft=512, hop_length=512))

    three_chanel = np.stack((feature_1, feature_2, feature_3), axis=2)

    return(three_chanel)

dir_list = os.listdir("validation/val-data/")

class_names = ["Fire","Rain","Thunderstorm","Wind","Tree Falling","Engine","Axe","Chainsaw","Handsaw","Gun shot","Speaking","Footsteps","Insect","Frog","Bird Chirp", "Wing Flap", "Lion", "Wolf", "Squirrel","Ambient"]

full_class_names = ["Fire","Rain","Thunderstorm","WaterDrops","Wind","Silence","Tree Falling","Helicopter","Engine","Axe","Chainsaw","Generator","Handsaw","Firework","Gun shot","WoodChop","Whistling","Speaking","Footsteps","Clapping","Insect","Frog","Bird Chirp","Wing Flap","Lion","Wolf","Squirrel","Ambient"]

count = 0 

for audio_name in dir_list:

    actual_index = int(list(audio_name.split("_"))[0]) - 1

    if (full_class_names[actual_index] in class_names):

        path = "validation/val-data/" + audio_name

        raw_audio, sr = librosa.load(path, sr=44100, mono=True, duration=5)

        three_chanel = extract_spectrogram(raw_audio)

        feature = np.expand_dims(three_chanel, axis=0)

        inp = tf.lite.Interpreter('mel-model-1.tflite')
        inp.allocate_tensors()

        input_details = inp.get_input_details()
        output_details = inp.get_output_details()

        inp.set_tensor(input_details[0]['index'], feature)
        inp.invoke()
        label = inp.get_tensor(output_details[0]['index'])[0]

        max_index = np.argmax(label)
        max_prob = max(label)

        print(class_names[max_index],full_class_names[actual_index])

        if (class_names[max_index]==full_class_names[actual_index]):
            count = count + 1

print("Correctly Identified",count)
print("Accuracy",count/40)

