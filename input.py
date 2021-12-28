import os
from PIL import Image, ImageOps
import numpy as np

def getPixels( imageName ):
    image = Image.open( imageName )
    grayImage = ImageOps.grayscale( image )
    imageArray = np.asarray( grayImage )

    return imageArray

def getData( dataSets, dir = "/Users/noaheverett/Documents/Codes/Vision/Neural Network/Data Sets/" ):
    nDataSets = len( dataSets )

    data = []
    classifications = []
    oneHot = []
    
    nFileCurr = 0
    folders = os.listdir( dir ) 
    for nFolder in range( len( folders ) ):
        if folders[ nFolder ] != ".DS_Store":
            files = os.listdir( dir + "/" + folders[ nFolder ] )
            for nFile in range( len( files ) ):
                if files[ nFile ] != ".DS_Store":
                    data.append( [] )
                    classifications.append( files[ nFile ][ :len( folders[ nFolder ] ) ] )
                    current_oneHot = [ 0 for n in range( nDataSets ) ]
                    for nSet in range( nDataSets ):
                        if files[ nFile ][ :len( folders[ nFolder ] ) ] == dataSets[ nSet ]: current_oneHot[ nSet ] = 1
                    oneHot.append( current_oneHot )
                    for row in list( getPixels( dir + folders[ nFolder ] + "/" + files[ nFile ] ) ):
                        for pixel in row:
                            data[ nFileCurr ].append( pixel / 255 )
                    nFileCurr += 1
                        
    return data, classifications, oneHot