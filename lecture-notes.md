Nov 6
--------
Notes from class:

1. Average height in the third column is the average height of all connections in the whole video. (so we need to identify the same crop?)
2. Dr. Mohler suggests to model the average height of each frame using a linear regression (with pixels(i) of each detection as input). Pixel j could also be used, say, if j is changing from frame to frame, it indicates how the camera moves.
3. linear regression model
```
Frame
    |                            |
    | +(i1)                      |
    |          +(i3)             |
    |               +(i2)        |
    |                            |
    |                            |
 -----------------------------------  bottom line
 ```
 Linear Regression: Label = A*(a1+a2+a3)/3 + B
 
 Label is data from the csv file Dr. Mohler gave us.
 i1,i2,i3 is the corn-ear-connection our model detected.
 a1,a2,a3 is the height of i1,i2,i3 in the frame.(#pixels from bottom line to the detected connection)
 A, B is trained by data.
 
 The same connection may generate multiple datas at different time point.

Oct 30
--------
Some notes from today's class:

1. Two cameras(at 4 feet and 2 feet hight respectively) are used in pair in the measuring device(something like a cart), so videos come in pairs.
```

    |
    +-   <- camera 2(4 feet in hight)
    |
    +-   <- camera 1(2 feet high, facing up)
    |
  [_+_]  <- cart
```
2. The lower camera has a angle facing up. Both camera is tightly fastened in the rod, but the rod can vibrate when the cart is moving forward.
3. They traditional way of  corn ear height measurement has a tolerance of around 3 inches.
4. People measure "height" differently, and they are using the connection of the corn ear. But if we use the whole corn ear, there can be a variation of around 1 feet, because of the orientation of the corn ear.

10-27
------

1. model should be easy run (no external packages)
2. good documentation
3. give some example (images with detection)
4. he will run on test images and see what can be detected

10-16
------
1. Tutorial: https://towardsdatascience.com/creating-your-own-object-detector-ad69dda69c85
2. other link: https://www.geeksforgeeks.org/ml-training-image-classifier-using-tensorflow-object-detection-api/

Stage 1 description
-------------------

Beck's Hybrids has provided videos of experimental corn plots.  Normally they measure plants by hand (from ground to where corn ear meets the stalk) and then get an average of each plot (to assess performance of different strains). 

The goal of the competition is to estimate the average height from these videos using deep learning.  

We will split this into stages.  The goal of stage 1 is to detect in an image any place where the corn ear meets the stalk.  

Here are the rules:

    You can work in teams of up to 3 people
    You can use transfer learning or other data sets
    You can (and I would recommend) doing some labeling for training

What to turn in:

A trained model along with executable code that takes as input an image and outputs a new image with regions of the image highlighted (either shading or bounding box) where your algorithm has detected places where corn ear meets stalk.   Include a readme file with instructions on how to run the code.  You will be evaluated qualitatively on how well your algorithm does with 5 or so test images.

Getting started:

You are free to use any deep learning approaches you want.  My suggestion would be to try to get a dataset of at least 50-100 small sub-images 20x20 or 30x30 that contain the corn-stalk connection (these are positive class images).  Then use data augmentation combined with negatively sampled sub-images to train a classifier.  Finally, you can drag that classifier across a larger image to detect places in the image where corn ear meets stalk.  This is just one idea.   For labeling, there are some software tools for doing this like http://www.robots.ox.ac.uk/~vgg/software/via/

Datahttps://iu.box.com/s/3kotou1u2v7fuqk7gwhbew480hrwk1e3 (Links to an external site.)

example_labels.zip

