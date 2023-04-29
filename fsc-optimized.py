import librosa
import numpy as np
import tensorflow as tf
import time

# Download the model to yamnet.tflite
interpreter = tf.lite.Interpreter('yamnet.tflite')

input_details = interpreter.get_input_details()
waveform_input_index = input_details[0]['index']
output_details = interpreter.get_output_details()
scores_output_index = output_details[0]['index']
embeddings_output_index = output_details[1]['index']
spectrogram_output_index = output_details[2]['index']

inp = tf.lite.Interpreter('classifier.tflite')
inp.allocate_tensors()

input_details = inp.get_input_details()
output_details = inp.get_output_details()

class_names = ["Fire","Rain","Thunderstorm","Wind","Tree Falling","Engine","Axe","Chainsaw","Handsaw","Gun shot","Speaking","Footsteps","Insect","Frog","Bird Chirp", "Wing Flap", "Lion", "Wolf", "Squirrel","Ambient"]
threats = ["Fire","Tree Falling","Engine","Axe","Chainsaw","Handsaw","Gun shot","Speaking","Footsteps"]
def Spect(spectrogram):
    
    inp.set_tensor(input_details[0]['index'], spectrogram)
    inp.invoke()
    predictions = inp.get_tensor(output_details[0]['index'])[0]
    return(predictions)

def classify(audio_name, feature):
    print("Classifying...")
    f = open("predictions.txt", "a")
    label = Spect(feature)

    max_index = np.argmax(label)
    max_prob = max(label)
    
    f.write("{0} -- {1}\n".format(audio_name, class_names[max_index], max_prob))
    print(audio_name,class_names[np.argmax(label)],max_prob)
    if (class_names[max_index] in threats):
        print("Threat detected" + class_names[max_index])


def extract_embedding(waveform):

    interpreter.resize_tensor_input(waveform_input_index, [len(waveform)], strict=True)
    interpreter.allocate_tensors()
    interpreter.set_tensor(waveform_input_index, waveform)
    interpreter.invoke()

    # embeddings = interpreter.get_tensor(embeddings_output_index)
    # scores  = interpreter.get_tensor(scores_output_index)

    spectrogram = interpreter.get_tensor(spectrogram_output_index)

    return(spectrogram)

def feature_extractor(index):
    print("extracting")
    print(index)
    audio_name = "audios/audio_"+str(index)+".wav"
    print(audio_name)
    raw_audio, sr = librosa.load(audio_name, sr=44100, mono=True, duration=5)

    embedding = extract_embedding(raw_audio)

    three_chanel = np.stack((embedding, embedding, embedding), axis=2)

    feature = np.expand_dims(three_chanel, axis=0)

    classify(audio_name, feature)
    return(index+1)
    # print("embedding for : " + str(audio_name) + " done")

index = 1
def write_clip(index):  
    while True:
        for i in range(0, 10):
            index = feature_extractor(index)
        time.sleep(10)

write_clip(index)
