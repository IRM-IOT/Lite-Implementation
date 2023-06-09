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

    # embeddings = interpreter.get_tensor(embeddings_output_index)
    # scores  = interpreter.get_tensor(scores_output_index)

    spectrogram = interpreter.get_tensor(spectrogram_output_index)

    return(spectrogram)

def feature_extractor(audioIndex,features):
    while True:
        if (len(audioIndex)>0):
            audio_name = audioIndex.pop(0)
            raw_audio, sr = librosa.load(audio_name, sr=44100, mono=True, duration=5)

            embedding = extract_embedding(raw_audio)

            three_chanel = np.stack((embedding, embedding, embedding), axis=2)

            feature = np.expand_dims(three_chanel, axis=0)

            features.append([audio_name,feature])
            print("embedding for : " + str(audio_name) + " done")
        else:
            continue


# raw_audio, sr = librosa.load("audios/audio_1.wav", sr=44100, mono=True, duration=5)

# embedding = extract_embedding(raw_audio)

# three_chanel = np.stack((embedding, embedding, embedding), axis=2)

# print(np.shape(three_chanel))

# feature = np.expand_dims(three_chanel, axis=0)

# print(np.shape(feature))