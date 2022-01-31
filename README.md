# Object Detection (faces) using the Viola-Jones Algorithm

This project is a simple example of using the Viola-Jones algorithm to detect objects in an image, we built the algorithm based on faces but it can be used for any object.

The pipeline was built from scratch using only numpy. opencv was used for reading images and simple preprocessing before entering **our** pipeline.

The pipeline doesn't involve training. It takes an xml file (not just facedetction ones) from opencv and uses the already trained weight for object detection.

More details can be found in the paper
    [Viola-Jones Paper](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf)

# Usage

To use this code, you have to give the path to a xml file and the path to an image.

The xml file which can be obtained from opencv can be downloaded from [here](https://github.com/opencv/opencv/tree/master/data/haarcascades) contain the trained weights for the object detection.

You can detect multiple types objects in an image by giving multiple xml files to different instiations of the function.

The function will return a list containing a bounding box for each detected object.

# Performance

The detection process is quite fast as cython was used to optimize that performance.

Using a webcam, the algorithm can detect faces in slightly less than 0.2s. which means that it can be roughly considered real time.

# Preprocessing

The image is preprocessed by converting it to grayscale and resizing it to a fixed size.

There is another preprocessing step that happens to remove the effect of lighting from each image.

Before computing the features, each window is normalized by dividing by the variance in that window.

# Optimization Process

Several tricks where made to optimize the performance

- Integral Image

    The integral image was used quite a bit in the algorithm which hugely sped up the performance. For details about how the concept works, see the paper mentioned abobe

- Canny (Edge Detection)

    For each sliding window, we would sum the edge response in that window utilizing the integral image.

    If the sum was below a certain threshold, this meant that there were no edges in that window. which consequently means that the window was not a face.

    This allowed us to avoid a lot of unnecessary computation.

- Cython
    
    While python is very easy to write and experiment with, the performance drops quite a bit especially when dealing with loops which are very common in the algorithm.

    Cython was used to speed up the algorithm making the loops much more efficient. 

- Strides

    Initially we used a stride of 1 for the sliding window in both the x and y direction. This was not optimal as this caused the algorithm to check certain parts of the image multiple times even though it "knows" there are no faces there

# Using conda to install or update the required packages

use `conda env create -f env.yml` to create a new environment with all the needed packages.

use `conda env update -f env.yml` to update the environment if you have already created it.

use `conda activate fer` to activate the environment.

use `conda env export --no-build > env.yml` to update the env file in case you installed any new packages.