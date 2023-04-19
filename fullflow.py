import librosa
import numpy as np
import tensorflow as tf

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

    spectrogram = interpreter.get_tensor(spectrogram_output_index)

    return(spectrogram)

raw_audio, sr = librosa.load("test.wav", sr=44100, mono=True, duration=5)

embedding = extract_embedding(raw_audio)

three_chanel = np.stack((embedding, embedding, embedding), axis=2)

feature = np.expand_dims(three_chanel, axis=0)

inp = tf.lite.Interpreter('classifier.tflite')
inp.allocate_tensors()

input_details = inp.get_input_details()
output_details = inp.get_output_details()

def Spect(spectrogram):
    
    inp.set_tensor(input_details[0]['index'], spectrogram)
    inp.invoke()
    predictions = inp.get_tensor(output_details[0]['index'])[0]
    return(predictions)

class_names = ["Fire","Rain","Thunderstorm","Wind","Tree Falling","Engine","Axe","Chainsaw","Handsaw","Gun shot","Speaking","Footsteps","Insect","Frog","Bird Chirp", "Wing Flap", "Lion", "Wolf", "Squirrel","Ambient"]

label = Spect(feature)

max_index = np.argmax(label)
max_prob = max(label)

print(class_names[max_index])