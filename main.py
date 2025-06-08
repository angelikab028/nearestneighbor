from termcolor import colored
from pyfiglet import Figlet

import numpy as np 
import pandas as pd

def nearestneighborAccuracy(data, features):
    #https://www.geeksforgeeks.org/calculate-the-euclidean-distance-using-numpy/
    features = data[features].to_numpy()
    labels = data[0].to_numpy()
    rows = len(data)

    # print(features[0])
    correctClassified = 0
    for i in range(rows):
        # print("Classifying row", i)
        objToClass = features[i]
        objLabels = labels[i]
        
        #https://numpy.org/doc/2.1/reference/generated/numpy.linalg.norm.html
        distances = np.linalg.norm(features - objToClass, axis = 1)
        distances[i] = np.inf

        #https://stackoverflow.com/questions/2474015/getting-the-index-of-the-returned-max-or-min-item-using-max-min-on-a-list
        nearestNeighborIndex = np.argmin(distances)
        # print("   Nearest Neighbor is row", nearestNeighborIndex)
        if labels[i] == labels[nearestNeighborIndex]:
            correctClassified += 1
    
    # print("Correctly Classified", correctClassified)
    # print("Accuracy =", correctClassified/rows)
    return correctClassified/rows

def forwardSelection(data, features):
    # Start with an empty set of features
    currentSelection = []
    accuracies = []
    # unselectedFeatures = features
    unselectedFeatures = list(features)
    
    # Add one feature at a time to current set of features
    # All features will be added once
    for i in range(len(features)):
        iterAccuracies = []
        iterFeatures = []

        # Try adding one feature at a time to current sent of features
        for feature in unselectedFeatures:
            featuresToTest = currentSelection + [feature]
            accuracy = nearestneighborAccuracy(data, featuresToTest)
            iterAccuracies.append(accuracy)
            iterFeatures.append(feature)

        # Pick the added feature that has the highest accuracy
        maxAccIndex = np.argmax(iterAccuracies)
        maxFeature = iterFeatures[maxAccIndex]
        accuracies.append(iterAccuracies[maxAccIndex])

        # Add highest accuracy feature of current iteration to current set of features
        currentSelection.append(maxFeature)
        # unselectedFeatures.delete(maxFeature)
        unselectedFeatures.remove(maxFeature)

        print("Highest accuracy is", iterAccuracies[maxAccIndex])
        print("   Obtained by adding", maxFeature)
        print("   Our current selection is now", currentSelection)
    
    return currentSelection, accuracies

def backwardSelection(data, features):
    # Start with all of the features
    currentSelection = list(features)
    accuracies = []
    # unselectedFeatures = features
    unselectedFeatures = list(features)
    
    accuracies.append(nearestneighborAccuracy(data, currentSelection))
    # Renive one feature at a time to current set of features
    # All features will be removed once
    for i in range(len(features)):
        iterAccuracies = []
        iterFeatures = []

        # Try removing one feature at a time to current sent of features
        for feature in unselectedFeatures:
            # print(feature)
            featuresToTest = currentSelection.copy()
            featuresToTest.remove(feature)
            # print(featuresToTest)
            accuracy = nearestneighborAccuracy(data, featuresToTest)
            iterAccuracies.append(accuracy)
            iterFeatures.append(feature)

        # Pick the added feature that has the highest accuracy
        maxAccIndex = np.argmax(iterAccuracies)
        maxFeature = iterFeatures[maxAccIndex]
        accuracies.append(iterAccuracies[maxAccIndex])
        
        # Add highest accuracy feature of current iteration to current set of features
        currentSelection.remove(maxFeature)
        # unselectedFeatures.delete(maxFeature)
        unselectedFeatures.remove(maxFeature)

        print("Highest accuracy is", iterAccuracies[maxAccIndex])
        print("   Obtained by removing", maxFeature)
        print("   Our current selection is now", currentSelection)
    
    return currentSelection, accuracies

def main():
    f = Figlet(font="small")
    print(colored(f.renderText("Feature Selection with Nearest Neighbor"), "blue"))
    print("Welcome to my feature selection project for CS205: AI!")
    print("Note that this code only works for any small and large dataset provided by the project description.")
    print("Extra data cleaning is necessary for other datasets")

    fileName = "hello"
    while fileName != 'q':
        print("To get started, please input the data file name: ")
        fileName = input()

        if fileName == 'q': continue

        df = pd.read_fwf(fileName, header=None)

        print("Type the number for the selection algorithm you want to run:")
        print("  1.) Forwards")
        print("  2.) Backwards")
        selection = int(input())

        if selection == 1:
            forwardSelection(df, df.columns[1:])
        elif selection == 2:
            backwardSelection(df, df.columns[1:])

if __name__ == "__main__":
    main()
    #https://www.geeksforgeeks.org/how-to-read-text-files-with-pandas/
    #The code for reading in the txt as a dataframe was obtained from the above link
    # df = pd.read_csv("CS205_small_Data__22.txt", sep=" ", header=None)
    # df = pd.read_table("CS205_small_Data__22.txt", delimiter=" ")
    # df = pd.read_fwf("CS205_small_Data__22.txt", header=None)

    # nearestneighborAccuracy(df, df.columns[1:])
    # forwardSelection(df, df.columns[1:])
    # backwardSelection(df, df.columns[1:])
