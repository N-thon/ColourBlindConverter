# using multithreading on video capture and processing to improve performance 

# import the necessary packages
from __future__ import print_function
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import numpy
import imutils
import cv2
    
#Matrix Multiplication Block (Common for all operations, just varying matrix)
def getImageArray(respectiveArray, editablePhoto, sizeX, sizeY):
    for i in range(0,sizeX):
        for j in range(0,sizeY):
            currMatrix = numpy.array((0,0,0),dtype=float)
            for k in range(0,3):
                currMatrix[k]=editablePhoto[i,j,k]
            lmsImage = numpy.dot(respectiveArray,currMatrix)
            for k in range(0,3):
                editablePhoto[i,j,k]=lmsImage[k]
    return editablePhoto

#Converting RGB to LMS
def convertToLMS(im,sizeX,sizeY):
    photo = im
    editablePhoto = numpy.zeros((sizeX,sizeY,3),'float')
    for i in range(0,sizeX):
        for j in range(0,sizeY):
            for k in range(0,3):
                editablePhoto[i,j,k] = photo[i,j][k]
                editablePhoto[i,j,k] = ((editablePhoto[i,j,k])/255)
    lmsConvert = numpy.array([[17.8824,43.5161,4.11935],[3.45565,27.1554,3.86714],[0.0299566,0.184309,1.46709]])
    editablePhoto = getImageArray(lmsConvert, editablePhoto, sizeX, sizeY)
    return editablePhoto

#Simulating Deutranopia
def ConvertToDeuteranopes(editablePhoto,sizeX,sizeY):
    DeuteranopesConvert = numpy.array([[1,0,0],[0.494207,0,1.24827],[0,0,1]])
    editablePhoto = getImageArray(DeuteranopesConvert, editablePhoto, sizeX, sizeY)
    return editablePhoto

#Simulating Protanopia
def ConvertToProtanopes(editablePhoto,sizeX,sizeY):
    protanopeConvert = numpy.array([[0,2.02344,-2.52581],[0,1,0],[0,0,1]])
    editablePhoto = getImageArray(protanopeConvert, editablePhoto, sizeX, sizeY)
    return editablePhoto

#Simulating Tritanopia
def ConvertToTritanopes(editablePhoto,sizeX,sizeY):
    TritanopeConvert = numpy.array([[1,0,0],[0,1,0],[-0.395913,0.801109,0]])
    editablePhoto = getImageArray(TritanopeConvert, editablePhoto, sizeX, sizeY)
    return editablePhoto

#Converting LMS to RGB
def convertToRGB(editablePhoto,sizeX,sizeY):
    rgb2lms = numpy.array([[17.8824,43.5161,4.11935],[3.45565,27.1554,3.86714],[0.0299566,0.184309,1.46709]])
    RGBConvert = numpy.linalg.inv(rgb2lms)
    editablePhoto = getImageArray(RGBConvert, editablePhoto, sizeX, sizeY)
    for i in range(0,sizeX):
        for j in range(0,sizeY):
            for k in range(0,3):
                editablePhoto[i,j,k]=((editablePhoto[i,j,k]))*255
    return editablePhoto


# created a *threaded* video stream, allow the camera sensor to warmup,
# and start the FPS counter
vs = WebcamVideoStream(src=0).start() 
fps = FPS().start()

# loop over frames
while (True):
    # grab the frame from the threaded video stream and resize it
    # adjust width value to increace processing speeds (recommended for slow devices)
    frame = vs.read()
    frame = imutils.resize(frame, width=80)
    sizeX = frame.shape[0]
    sizeY = frame.shape[1]
    
    lmsPhoto = convertToLMS(frame,sizeX,sizeY)
    
    # un-comment depending on colourblindness you wish to simulate
    simPhoto = ConvertToDeuteranopes(lmsPhoto,sizeX,sizeY) 
    #simPhoto = ConvertToProtanopes(lmsPhoto,sizeX,sizeY)  
    #simPhoto = ConvertToTritanopes(lmsPhoto,sizeX,sizeY)

    rgbPhoto = convertToRGB(simPhoto,sizeX,sizeY) 
    # convert the frame to uint8 before imshow()
    rgbPhoto = rgbPhoto.astype(numpy.uint8)

    
    # un-comment to save most recent frame
    #cv2.imwrite('C:/Path/to/Folder/converted_image_name.png',rgbPhoto) 
    #cv2.imwrite('C:/Path/to/Folder/normal_image_name.png',frame) 

    cv2.imshow("What I See", rgbPhoto)
    cv2.imshow("Normal Vision", frame)
    
    # checks for quit (q) every frame
    if cv2.waitKey(1) & 0xFF == ord('q'):           
        break
    # update the FPS counter
    fps.update()
# stop the timer and display FPS information
fps.stop()

print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()