import numpy as np
import tensorflow as tf
import time
import soundfile

# Class Names

class_names = ["Fire","Rain","Thunderstorm","Wind","Tree Falling","Engine","Axe","Chainsaw","Handsaw","Gun shot","Speaking","Footsteps","Insect","Frog","Bird Chirp", "Wing Flap", "Lion", "Wolf", "Squirrel","Ambient"]
threats = ["Fire","Tree Falling","Engine","Axe","Chainsaw","Handsaw","Gun shot","Speaking","Footsteps"]

# tflite structure to load yamnet model

interpreter = tf.lite.Interpreter('yamnet.tflite')

input_details = interpreter.get_input_details()
waveform_input_index = input_details[0]['index']
output_details = interpreter.get_output_details()
scores_output_index = output_details[0]['index']
embeddings_output_index = output_details[1]['index']
spectrogram_output_index = output_details[2]['index']

# tflite structure to load the fsc22 classifier

inp = tf.lite.Interpreter('yam-model-1.tflite')
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

# Function to classify a given audio and write it to the predictions.txt

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

# Function to extract embeddings from yamnet

def extract_spectrogram(waveform):

    interpreter.resize_tensor_input(waveform_input_index, [len(waveform)], strict=True)
    interpreter.allocate_tensors()
    interpreter.set_tensor(waveform_input_index, waveform)
    interpreter.invoke()

    embeddings = interpreter.get_tensor(embeddings_output_index)

    # scores  = interpreter.get_tensor(scores_output_index)
    # spectrogram = interpreter.get_tensor(spectrogram_output_index)

    return(embeddings)

# Main handler code with sleeping

def classify_audio(index):

    audio_name = "audio_" + str(index)

    audio_path = "audios/audio_"+str(index)+".wav"

    try:

        raw_audio, sr = load_wav_audio(audio_path)

        print("audio_"+str(index)+" : Loaded")

        embedding = extract_spectrogram(raw_audio)

        three_chanel = np.stack((embedding, embedding, embedding), axis=2)

        feature = np.expand_dims(three_chanel, axis=0)

        print("audio_"+str(index)+" : embeddings extracted")

        classify(audio_name, feature)

        return(True)
    
    except:

        print("All audios classified")
        return(False)
    
# Realtime classifier execution code

index = 1

while True:
    for i in range(0, 10):
        status = classify_audio(index)
        if status==True:
            index = index + 1
        else:
            break
    time.sleep(10)