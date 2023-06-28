#Project 1 Affective Computing 
#Zaber Raiyan Choudhury

#RUN to save npy files.
#then proceed to proj1.py

import os
import numpy as np

# function to read bnd files
def read_text_file(file_path):
    
    #initializing array for storing lines
    lines =[]
    data =[]

    #read file
    file1 = open(file_path, 'r')
    lines = file1.readlines()

    #format line to get raw data
    for line in lines:
        line = line[:len(line)-1]
        line = line.split(' ')
        line = line[1:]
        
        data.append(line)
    
    #return array for each file read    
    return np.array(data, dtype=float)

#function to read folder of each subject
def read_expression(abs_path):
    
    rel_path = "" #initializtion

    #Y_train classification refers to this map
    hash = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise" ]

    #To store all expressions of each subject
    dx = None
    dy = None
    
    #iterating through expressions to make x and y train
    for expression in os.listdir(abs_path):
        if os.path.isdir(expression):
            
            ##print(expression)
            rel_path = expression
            full_path = os.path.join(abs_path, rel_path)

            result_arr = []
            for file in os.listdir(full_path):
                os.chdir(full_path)
                if file.endswith(".bnd"):
                
                    #call read text file function
                    x = read_text_file(file)
                    result_arr.append(x)
            x_train = np.array(result_arr) #call at end
            
            i = hash.index(expression)
            y_train = np.full((len(x_train)), i)

            ##print(x_train.shape)
            ##print(y_train.shape)

            os.chdir(abs_path)
            
            #compile all expression data
            if dx is None:
                dx = x_train
            else:
                dx = np.row_stack((dx, x_train))
            
            if dy is None:
                dy = y_train
            else:
                dy = np.concatenate((dy, y_train), axis=0)

    return dx,dy


#initialize
x_train = None
y_train = None

#iterate through subjects
main_path = os.path.dirname(__file__)
for subjects in os.listdir(main_path):
    if os.path.isdir(subjects):
        print(subjects)
        
        relpath = subjects
        fullpath = os.path.join(main_path, relpath)
        
        os.chdir(fullpath)
        
        #read files of each subjects
        a, b = read_expression(fullpath)
        if x_train is None:
            x_train = a
        else:
            x_train = np.row_stack((x_train, a))
        
        if y_train is None:
            y_train = b
        else:
            y_train = np.concatenate((y_train, b), axis=0)
        os.chdir(main_path)
    
        print(x_train.shape, y_train.shape)

#create .npy files
np.save('x_train.npy', x_train)
np.save('y_train.npy', y_train)