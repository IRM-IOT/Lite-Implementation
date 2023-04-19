import numpy as np
import tensorflow as tf

inp = tf.lite.Interpreter('classifier.tflite')
inp.allocate_tensors()

input_details = inp.get_input_details()
output_details = inp.get_output_details()

class_names = ["Fire","Rain","Thunderstorm","Wind","Tree Falling","Engine","Axe","Chainsaw","Handsaw","Gun shot","Speaking","Footsteps","Insect","Frog","Bird Chirp", "Wing Flap", "Lion", "Wolf", "Squirrel","Ambient"]

def Spect(spectrogram):
    
    inp.set_tensor(input_details[0]['index'], spectrogram)
    inp.invoke()
    predictions = inp.get_tensor(output_details[0]['index'])[0]
    return(predictions)

def classify(features):
    f = open("predictions.txt", "a")
    while True:
        if (len(features)>0):
            feature = features.pop(0)
            label = Spect(feature[1])

            max_index = np.argmax(label)
            max_prob = max(label)
            
            f.write("{0} -- {1}\n".format(feature[0], class_names[max_index], max_prob))
            print(feature[0],class_names[np.argmax(label)],max_prob)
        else:
            continue



