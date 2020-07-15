# simulates colour blindness from webcam

from PIL import Image
import numpy as np
import numpy
import cv2

#Restructuring laterally inverted image
def normalise(editablePhoto,sizeX,sizeY):
    NormalPhoto =  np.zeros((sizeX,sizeY,3),'float')
    x=sizeX-1
    for i in range(0,sizeX):
        for j in range(0,sizeY):
            for k in range(0,3):
                NormalPhoto[x,j,k]=editablePhoto[i,j,k]
        x=x-1
    return NormalPhoto
    
#Matrix Multiplication Block (Common for all operations, just varying matrix)
def getImageArray(respectiveArray, editablePhoto, sizeX, sizeY):
    for i in range(0,sizeX):
        for j in range(0,sizeY):
            currMatrix = np.array((0,0,0),dtype=float)
            for k in range(0,3):
                currMatrix[k]=editablePhoto[i,j,k]
            lmsImage = np.dot(respectiveArray,currMatrix)
            for k in range(0,3):
                editablePhoto[i,j,k]=lmsImage[k]
    return editablePhoto

#Converting RGB to LMS
def convertToLMS(im,sizeX,sizeY):
    photo = im
    editablePhoto = np.zeros((sizeX,sizeY,3),'float')
    for i in range(0,sizeX):
        for j in range(0,sizeY):
            for k in range(0,3):
                editablePhoto[i,j,k] = photo[i,j][k]
                editablePhoto[i,j,k] = ((editablePhoto[i,j,k])/255)

    lmsConvert = numpy.array([[17.8824,43.5161,4.11935],[3.45565,27.1554,3.86714],[0.0299566,0.184309,1.46709]])
    editablePhoto = getImageArray(lmsConvert, editablePhoto, sizeX, sizeY)

    NormalPhoto =  normalise(editablePhoto,sizeX,sizeY)
    return NormalPhoto

#Simulating Deutranopia
def ConvertToDeuteranopes(editablePhoto,sizeX,sizeY):
    DeuteranopesConvert = numpy.array([[1,0,0],[0.494207,0,1.24827],[0,0,1]])
    editablePhoto = getImageArray(DeuteranopesConvert, editablePhoto, sizeX, sizeY)
    NormalPhoto = normalise(editablePhoto, sizeX, sizeY)
    return NormalPhoto

#Simulating Protanopia
def ConvertToProtanopes(editablePhoto,sizeX,sizeY):
    protanopeConvert = numpy.array([[0,2.02344,-2.52581],[0,1,0],[0,0,1]])
    editablePhoto = getImageArray(protanopeConvert, editablePhoto, sizeX, sizeY)
    NormalPhoto = normalise(editablePhoto, sizeX, sizeY)
    return NormalPhoto

#Simulating Tritanopia
def ConvertToTritanopes(editablePhoto,sizeX,sizeY):
    TritanopeConvert = numpy.array([[1,0,0],[0,1,0],[-0.395913,0.801109,0]])
    editablePhoto = getImageArray(TritanopeConvert, editablePhoto, sizeX, sizeY)
    NormalPhoto = normalise(editablePhoto, sizeX, sizeY)
    return NormalPhoto

#Converting LMS to RGB
def convertToRGB(editablePhoto,sizeX,sizeY):
    rgb2lms = numpy.array([[17.8824,43.5161,4.11935],[3.45565,27.1554,3.86714],[0.0299566,0.184309,1.46709]])
    RGBConvert = numpy.linalg.inv(rgb2lms)
    editablePhoto = getImageArray(RGBConvert, editablePhoto, sizeX, sizeY)
    for i in range(0,sizeX):
        for j in range(0,sizeY):
            for k in range(0,3):
                editablePhoto[i,j,k]=((editablePhoto[i,j,k]))*255

    NormalPhoto = normalise(editablePhoto, sizeX, sizeY)
    return NormalPhoto

#Converting Processed Array to Image
def arrayToImage(editablePhoto,sizeX,sizeY):
    #return rgbArray
    rgbArray = np.zeros((sizeX,sizeY,3),'uint8')
    
    for i in range(0,sizeX):
        for j in range(0,sizeY):
            for k in range(0,3):
                rgbArray[i,j,k] = editablePhoto[i,j,k]
    img = Image.fromarray(rgbArray)
    return img
    

# accesses webcam
cap = cv2.VideoCapture(0)                           

while(True):                                        
    ret, frame = cap.read()
    
    scale_percent = 40 # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA) # resize image
    
    cv2.imshow('what normal people see:', resized) # displys the video feed (un-processed)

    sizeX = resized.shape[0]
    sizeY = resized.shape[1]
    
    lmsPhoto = convertToLMS(resized,sizeX,sizeY)
    simPhoto = ConvertToDeuteranopes(lmsPhoto,sizeX,sizeY) 
    #simPhoto = ConvertToProtanopes(lmsPhoto,sizeX,sizeY)  # uncomment depending on colourblindness you wish to simulate
    #simPhoto = ConvertToTritanopes(lmsPhoto,sizeX,sizeY)
    
    rgbPhoto = convertToRGB(simPhoto,sizeX,sizeY)
    #colourPhoto = arrayToImage(rgbPhoto,sizeX,sizeY)
    
    rgbPhoto = rgbPhoto.astype(np.uint8)
    cv2.imshow('what I see:', rgbPhoto) # displays the processed frames

    if cv2.waitKey(1) & 0xFF == ord('q'):           # checks for quit (q) every frame
        break
    

cap.release()                                       # turns off webcam
cv2.destroyAllWindows()                             # clean up