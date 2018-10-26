# Assignment 4: Histogram Equalization

Assignment No 4 for the multi-core programming course. Implement histogram equalization for a gray scale image in CPU and GPU. The result of applying the algorithm to an image with low contrast can be seen in Figure 1:

![Figure 1](Images/histogram_equalization.png)
<br/>Figure 1: Expected Result.

The programs have to do the following:

1. Using Opencv, load and image and convert it to grayscale.
2. Calculate de histogram of the image.
3. Calculate the normalized sum of the histogram.
4. Create an output image based on the normalized histogram.
5. Display both the input and output images.

Test your code with the different images that are included in the *Images* folder. Include the average calculation time for both the CPU and GPU versions, as well as the speedup obtained, in the Readme.

Rubric:

1. Image is loaded correctly.
2. The histogram is calculated correctly using atomic operations.
3. The normalized histogram is correctly calculated.
4. The output image is correctly calculated.
5. For the GPU version, used shared memory where necessary.
6. Both images are displayed at the end.
7. Calculation times and speedup obtained are incuded in the Readme.

## Results

* GPU: Nvidia GTX 980 Ti
* CPU: Intel Core i7 4790k (4 physical cores, 8 virtual)

| Image       | CPU     | GPU     | Speed Up    |
| :---------: |:-------:|:-------:|:-----------:|
| dog1.jpeg   | 39.0436 | 0.0325  |   1201.34   |
| dog2.jpeg   | 51.9778 | 0.0335  |   1551.57   |
| dog3.jpeg   | 52.2572 | 0.0315  |   1658.95   |
| scenery.jpg |  2.2863 | 0.0217  |   105.35    |
| woman.jpg   | 37.1473 | 0.0317  |   1171.83   |
| woman2.jpg  | 30.1481 | 0.0332  |   908.07    |
| woman3.jpg  | 39.2198 | 0.0312  |   1257.04   |
