# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 16:14:15 2022

@author: ypatia
"""

#!/usr/bin/env python
import itk

if len(sys.argv) != 4:
    print('Usage: ' + sys.argv[0] +
          ' input3DImageFile  output3DImageFile  sliceNumber')
    sys.exit(1)

Dimension = 3
PixelType = itk.ctype('short')
ImageType = itk.Image[PixelType, Dimension]

# Here we recover the file names from the command line arguments
inputFilename = sys.argv[1]
outputFilename = sys.argv[2]
reader = itk.ImageFileReader[ImageType].New()
reader.SetFileName(inputFilename)
reader.Update()

extractFilter = itk.ExtractImageFilter[ImageType, ImageType].New()
extractFilter.SetInput(reader.GetOutput())
extractFilter.SetDirectionCollapseToSubmatrix()

# set up the extraction region [one slice]
inputImage = reader.GetOutput()
inputRegion = inputImage.GetBufferedRegion()
size = inputRegion.GetSize()
size[2] = 1  # we extract along z direction
start = inputRegion.GetIndex()
sliceNumber = int(sys.argv[3])
start[2] = sliceNumber
desiredRegion = inputRegion
desiredRegion.SetSize(size)
desiredRegion.SetIndex(start)

extractFilter.SetExtractionRegion(desiredRegion)
pasteFilter = itk.PasteImageFilter[ImageType].New()
medianFilter = itk.MedianImageFilter[ImageType, ImageType].New()
extractFilter.SetInput(inputImage)
medianFilter.SetInput(extractFilter.GetOutput())
pasteFilter.SetSourceImage(medianFilter.GetOutput())
pasteFilter.SetDestinationImage(inputImage)
pasteFilter.SetDestinationIndex(start)

indexRadius = size
indexRadius[0] = 1  # radius along x
indexRadius[1] = 1  # radius along y
indexRadius[2] = 0  # radius along z
medianFilter.SetRadius(indexRadius)
medianFilter.UpdateLargestPossibleRegion()
medianImage = medianFilter.GetOutput()
pasteFilter.SetSourceRegion(medianImage.GetBufferedRegion())

writer = itk.ImageFileWriter[ImageType].New()
writer.SetFileName(outputFilename)
writer.SetInput(pasteFilter.GetOutput())
writer.Update()