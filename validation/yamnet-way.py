import librosa
import numpy as np
import tensorflow as tf
import os

# Download the model to yamnet.tflite
interpreter = tf.lite.Interpreter('yamnet.tflite')

input_details = interpreter.get_input_details()
waveform_input_index = input_details[0]['index']
output_details = interpreter.get_output_details()
scores_output_index = output_details[0]['index']
embeddings_output_index = output_details[1]['index']
spectrogram_output_index = output_details[2]['index']

def extract_embedding(waveform):

    interpreter.resize_tensor_input(waveform_input_index, [len(waveform)], strict=True)
    interpreter.allocate_tensors()
    interpreter.set_tensor(waveform_input_index, waveform)
    interpreter.invoke()

    spectrogram = interpreter.get_tensor(embeddings_output_index)

    return(spectrogram)


dir_list = os.listdir("validation/val-data/")

class_names = ["Fire","Rain","Thunderstorm","Wind","Tree Falling","Engine","Axe","Chainsaw","Handsaw","Gun shot","Speaking","Footsteps","Insect","Frog","Bird Chirp", "Wing Flap", "Lion", "Wolf", "Squirrel","Ambient"]

full_class_names = ["Fire","Rain","Thunderstorm","WaterDrops","Wind","Silence","Tree Falling","Helicopter","Engine","Axe","Chainsaw","Generator","Handsaw","Firework","Gun shot","WoodChop","Whistling","Speaking","Footsteps","Clapping","Insect","Frog","Bird Chirp","Wing Flap","Lion","Wolf","Squirrel","Ambient"]

count = 0 

for audio_name in dir_list:

    actual_index = int(list(audio_name.split("_"))[0]) - 1

    if (full_class_names[actual_index] in class_names):

        path = "validation/val-data/" + audio_name

        raw_audio, sr = librosa.load(path, sr=44100, mono=True, duration=5)

        embedding = extract_embedding(raw_audio)

        three_chanel = np.stack((embedding, embedding, embedding), axis=2)

        feature = np.expand_dims(three_chanel, axis=0)

        inp = tf.lite.Interpreter('yam-model-1.tflite')
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



# raw_audio, sr = librosa.load("validation/val-data/1_10105.wav", sr=44100, mono=True, duration=5)

# embedding = extract_embedding(raw_audio)

# three_chanel = np.stack((embedding, embedding, embedding), axis=2)

# feature = np.expand_dims(three_chanel, axis=0)

# inp = tf.lite.Interpreter('classifier.tflite')
# inp.allocate_tensors()

# input_details = inp.get_input_details()
# output_details = inp.get_output_details()

# def Spect(spectrogram):
    
#     inp.set_tensor(input_details[0]['index'], spectrogram)
#     inp.invoke()
#     predictions = inp.get_tensor(output_details[0]['index'])[0]
#     return(predictions)

# class_names = ["Fire","Rain","Thunderstorm","Wind","Tree Falling","Engine","Axe","Chainsaw","Handsaw","Gun shot","Speaking","Footsteps","Insect","Frog","Bird Chirp", "Wing Flap", "Lion", "Wolf", "Squirrel","Ambient"]

# label = Spect(feature)

# max_index = np.argmax(label)
# max_prob = max(label)

# print(class_names[max_index])