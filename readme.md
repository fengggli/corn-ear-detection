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
