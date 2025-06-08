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
        print("Classifying row", i)
        objToClass = features[i]
        objLabels = labels[i]
        
        #https://numpy.org/doc/2.1/reference/generated/numpy.linalg.norm.html
        distances = np.linalg.norm(features - objToClass, axis = 1)
        distances[i] = np.inf
        nearestNeighborIndex = np.argmin(distances)
        print("   Nearest Neighbor is row", nearestNeighborIndex)
        if labels[i] == labels[nearestNeighborIndex]:
            correctClassified += 1
    
    print("Correctly Classified", correctClassified)
    print("Accuracy =", correctClassified/rows)
    return correctClassified/rows

def main():
    f = Figlet(font="small")
    print(colored(f.renderText("Feature Selection with Nearest Neighbor"), "blue"))
    print("Welcome to my feature selection project for CS205: AI!")
    print("To get started, please input the data file name: ")

if __name__ == "__main__":
    main()
    #https://www.geeksforgeeks.org/how-to-read-text-files-with-pandas/
    #The code for reading in the txt as a dataframe was obtained from the above link
    # df = pd.read_csv("CS205_small_Data__22.txt", sep=" ", header=None)
    # df = pd.read_table("CS205_small_Data__22.txt", delimiter=" ")
    df = pd.read_fwf("CS205_small_Data__22.txt", header=None)
    print(df)

    nearestneighborAccuracy(df, df.columns[1:])
