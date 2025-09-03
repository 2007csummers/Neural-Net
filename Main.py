# Creates and Maintains training dataset and Machine Learning Model 
# used to read handwritten digits.

#TO IMPLEMENT
# 1) Add bmp file functionality and update visualizer for bmps.
# 2) Add a user interface for drawing their own digits.
# 3) Combine both applications.

import os
import numpy as np
import tensorflow as tf
import re

def main():
    #Define the number of images to be loaded and announce to user
    num_images = 100

    current_dir = os.getcwd()

    # loads dataset into memory
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    

    #loads previously trained model into memory and adds a softmax() layer to the model for prediction probabilities
    model =tf.keras.models.load_model("mnist_model.keras")
    prob_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

    #Evaluate Accuracy
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print(f"Accuracy {test_acc}")


    #produces the probability array for each of the images included in the test batch.
    probabilities = prob_model.predict(test_images)

    #prints the actual value of the digit followed by the predicted one.
    for i in range(num_images):
        print(test_labels[i])
        print(f"Probability: {probabilities[i]} \nPrediction: {np.argmax(probabilities[i])}")
    
    visualizer(0, "combined.txt")
    

    
    

#Creates a new model, trains it on the training data passed in through the "images" and "labels" variables for an "epo" number of epochs, and the stores the entire model in a .keras file locateed at "end_loc"
def train_and_save(images, labels, end_loc, epo):
    #create the Learning Model
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)), #This layer is the input layer. Simply flattens the 28x28 input array into a 1D array of length 28^2
        tf.keras.layers.Dense(128, activation="relu"), #This is the hidden layer. performs an operation (relu) to determine the level of 'activation' of each neuron before passing it on to the next layer
        tf.keras.layers.Dense(10) #output layer. 10 neurons for the 10 numerical digits
    ])

    #compile the model (essentially just tweak some settings so that the model knows how to behave and train)
    model.compile(  optimizer="adam", # chooses the algorithm used to modify the weights and biases in each layer
                    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), #chooses the loss function that the model uses. different loss functions are better for different types of problems
                    metrics=["accuracy"] #gives one or more metrics by which the model should focus on/think about improving
                  )
    
    
    
    #train the model
    model.fit(images, labels, epochs = epo)

    #save the model
    model.save(end_loc)
    return


#load the txt file's data into memory in the form of a three dimensional array where each image gets a 28x28 2D list and an element in a 1D labels list.  also normalizes each value into a 0-1 value
def load_list(txt_file_loc, num_images):
    txt_f = open(txt_file_loc, "r")
    output_list = []
    label_list = []
    
    #loop through each row of the text file and convert the row into a two dimensional list usable by the model and a 1 dimensional list containing the labels
    for i in range(num_images):
        index = 0
        temp_list = re.split("\t", txt_f.readline())
        image_list = []
        
        #loop through each pixel value and record it in a two dimensional list (28x28)
        for j in range(28):
            row_list = []
            for k in range(28):
                row_list.append(float(temp_list[index]) / 255)
                index += 1
            image_list.append(row_list)
        #Assign final values to output lists
        label_list.append(int(temp_list[index]))
        output_list.append(image_list)

    numpy_images = np.array(output_list)
    numpy_labels = np.array(label_list)
    return (numpy_images, numpy_labels)


#produces a text file that combines both images and labels in a tsv style format with each line being a different character.
def convert(images_file, labels_file, text_file,  num_convert):
    #Open each file
    img_f = open(images_file, "rb")
    lbl_f = open(labels_file, "rb")
    txt_f = open(text_file, "w")

    #Skip over file headers
    img_f.read(16)
    lbl_f.read(8)

    print("FORMATED: 784 pixels followed by 1 label. Separated by tabs")
    
    #loop through num_convert many images
    for i in range(num_convert):
        #Reads one byte and converts to the integer equivalent using ord()
        label = ord(lbl_f.read(1)) 

        #Loop over each pixel value and write it to the file
        for j in range(784):
            pixel = ord(img_f.read(1))
            txt_f.write(f"{pixel}\t")
        txt_f.write(f"{label}\t")
        txt_f.write("\n")

    img_f.close()
    lbl_f.close()
    txt_f.close()

    print("Binary to Text Conversion Completed")


#produces a readable visual of a given letter from the combined txt file
def visualizer(index, text_file):
    txt_f = open(text_file, "r")

    for i in range(index):
        txt_f.readline()

    #create variables to read out the line containing the number's pixels  
    line = txt_f.readline()
    pixels = []
    
    #read out the file into a list
    line_list = re.split("\t", line)
    
    #Loop over list and print out the values in a grid
    ind = 0
    for i in range(28):
        for j in range(28):
            str_dig = str(line_list[ind])
            if len(str_dig) == 1:
                str_dig = " " + str_dig + " "
            elif len(str_dig) == 2:
                str_dig = " " + str_dig
            print(f"{str_dig}", end="")
            ind += 1
        print()

    print(f"The number is {line_list[ind]}")


def predict(image_arr, model):
    return np.argmax(model.predict(image_arr))

#Uncomment when attempting to run Main as a standalone program. Comment when Main is a dependency
#main()