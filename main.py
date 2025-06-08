from termcolor import colored
from pyfiglet import Figlet

import numpy as np 
import pandas as pd

def nearestneighbor(data, features):
    features = data[features]
    labels = data[0]

    print(features)
    print(labels)

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

    nearestneighbor(df, df.columns[1:])
