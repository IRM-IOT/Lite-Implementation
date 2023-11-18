import numpy as np
import tensorflow as tf
import time
import soundfile
import librosa

# Class Names

class_names = ["Fire","Rain","Thunderstorm","Wind","Tree Falling","Engine","Axe","Chainsaw","Handsaw","Gun shot","Speaking","Footsteps","Insect","Frog","Bird Chirp", "Wing Flap", "Lion", "Wolf", "Squirrel","Ambient"]
threats = ["Fire","Tree Falling","Engine","Axe","Chainsaw","Handsaw","Gun shot","Speaking","Footsteps"]

# tflite structure to load the fsc22 classifier

inp = tf.lite.Interpreter('mel-model-full-lite')
inp.allocate_tensors()

input_details = inp.get_input_details()
output_details = inp.get_output_details()

# Function to load audios | using this instead of librosa.load

def load_wav_audio(filename):
    audio_data, sample_rate = soundfile.read(filename, dtype='float32', always_2d=True)
    audio_data = audio_data[:, 0]  # take only the first channel
    duration = len(audio_data) / sample_rate
    if duration > 5.0:
        audio_data = audio_data[:int(5.0 * sample_rate)]
    return audio_data, sample_rate

# # Function to classify a given audio and write it to the predictions.txt

def classify(audio_name, feature):

    f = open("predictions.txt", "a")

    inp.set_tensor(input_details[0]['index'], feature)
    inp.invoke()
    label = inp.get_tensor(output_details[0]['index'])[0]

    max_index = np.argmax(label)
    max_prob = max(label)
    
    f.write("{0} -- {1}\n".format(audio_name, class_names[max_index], max_prob))

    print(audio_name,":",class_names[np.argmax(label)],max_prob)

    if (class_names[max_index] in threats):
        print("Threat detected" + class_names[max_index])

# # Function to extract embeddings from yamnet

def extract_spectrogram(raw_audio):

    feature_1 = librosa.power_to_db(librosa.feature.melspectrogram(y=raw_audio, sr=22050, n_mels=128, n_fft=2048, hop_length=512))
    feature_2 = librosa.power_to_db(librosa.feature.melspectrogram(y=raw_audio, sr=22050, n_mels=128, n_fft=1024, hop_length=512))
    feature_3 = librosa.power_to_db(librosa.feature.melspectrogram(y=raw_audio, sr=22050, n_mels=128, n_fft=512, hop_length=512))

    three_chanel = np.stack((feature_1, feature_2, feature_3), axis=2)

    feature = np.expand_dims(three_chanel, axis=0)

    return(feature)

# # Main handler code with sleeping

def classify_audio(index):

    audio_name = "audio_" + str(index)

    audio_path = "audios/audio_"+str(index)+".wav"

    try:

        raw_audio, sr = load_wav_audio(audio_path)

        print("audio_"+str(index)+" : Loaded with SR :", sr)

        embedding = extract_spectrogram(raw_audio)

        print("audio_"+str(index)+" : embeddings extracted", np.shape(embedding))

        classify(audio_name, embedding)

        return(True)
    
    except Exception as e:

        print("All audios classified")
        return(False)
    
# # Realtime classifier execution code

index = 1

while True:
    for i in range(0, 10):
        status = classify_audio(index)
        if status==True:
            index = index + 1
        else:
            break
    time.sleep(10)