# Image Processing Utilities in MATLAB

# 📄 Abstract

This repository provides a set of MATLAB functions for various image processing tasks. These functions cover a range of techniques from noise addition, gamma correction, and filtering to histogram matching and bounding box overlay. The implementations are crafted for educational purposes, incorporating fundamental image processing concepts taught in the **Image Processing** course at MSA University, Egypt, by **Dr. Tarek Ghoniemy**. You can find more about Dr. Ghoniemy's work [here](https://www.researchgate.net/profile/Tarek-Ghoniemy-2).


# 📖 Table of Contents

- [📜 Overview]
  - Brief introduction to the functionalities and applications of the image processing functions.

- ### 🛠️ Functions Overview()
  - #### 🔧 Utility Functions
    - [📊 Histogram Calculation]

  - #### 🎛️ Image Adjustment
    - [🌞 Gamma Correction]
    - [⚖️ Contrast Stretching]

  - #### 🔍 Segmentation Techniques
    - [📐 Local Adaptive Segmentation]
    - [🎨 HSV Color Segmentation]

  - #### 📈 Histogram Operations
    - [🔄 Histogram Equalization]
    - [🎯 Histogram Matching]

  - #### 📊 Image Analysis and Comparison
    - [📷 Compare Outputs for Grayscale]
    - [🎨 Compare Outputs for Colored Images])

  - #### 📉 Quality and Distortion Metrics
    - [📏 Image Distortion Measures]

  - #### 🔊 Noise Operations
    - [⚙️ Salt Noise Addition]

  - #### 🧹 Filtering Techniques
    - [3️⃣ 3x3 Average Filter]
    - [3️⃣ 3x3 Median Filter]
    - [5️⃣ 5x5 Median Filter]

  - #### 📐 Bounding Box Annotation
    - [✏️ Show Drawn Bounding Boxes]




# 🔍 Overview

This collection of functions serves as a comprehensive toolkit for essential image processing operations, facilitating various tasks such as enhancing image quality, performing effective noise reduction, and enabling precise segmentation. The provided functions are designed to support educational and practical applications in fields like computer vision, remote sensing, and medical imaging. Users can easily implement these functions to manipulate images, assess their quality, and visualize outcomes, thus fostering a deeper understanding of image processing principles and techniques.


# 🛠️ Functions Overview()

## 📊 Histogram Calculation

```matlab
% Utility Functions: Histogram Calculation
% Calculate the histogram of the input grayscale image.
% @param IM: Input image
% @return outt: Histogram of the input image
% @brief Computes the frequency of each grayscale intensity level (0-255) in the input image.
function outt = My_Hist(IM)

% Calculate the histogram of a multi-channel image.
% @param Image: Input RGB image
% @return outt: Histogram of the image
function outt = my_histogram(Image)
```
## 🎛️ Image Adjustment
### ⚖️ Contrast Stretching

```matlab
% Perform Contrast Stretching on the input image.
% @param IM: Input image
% @return outt: Processed image after contrast stretching
% @brief Expands the intensity range of the image for better contrast by stretching from minimum to maximum values.
function outt = My_Contrast_Stretching(IM)
```
### 🌞 Gamma Correction

```matlab
% Gamma Correction for brightness adjustment.
% @param IM: Input image
% @param Gamma_Level: Gamma level (between 0.8 and 1.2 recommended)
% @return outt: Image after gamma correction
% @brief Useful for brightness control; gamma > 1 brightens, < 1 darkens.
function outt = My_Gamma_Correction(IM,Gamma_Level)
```
## 🔍 Segmentation Techniques
### 📐 Local Adaptive Segmentation

```matlab
% Perform Local Adaptive Segmentation.
% @param IM: Input image
% @param Length: Window length for segmentation
% @param Width: Window width for segmentation
% @return outt: Segmented image with local adaptive thresholding
% @brief Local adaptive segmentation improves thresholding where shadows are present.
function outt = My_Local_Adaptive_Segmentation(IM, Length, Width)
```
### 🎨 HSV Color Segmentation

```matlab
% Perform HSV-based color segmentation.
% @param IM: Input image
% @param Hue_Magrin: Range for hue threshold (min, max in [0, 360])
% @param Sat_Magrin: Range for saturation threshold (min, max in [0, 1])
% @param Val_Magrin: Range for value threshold (min, max in [0, 1])
% @return outt: Binary image after HSV segmentation
% @brief HSV segmentation isolates colors in the specified hue range.
function outt = My_HSV_Segmentation(IM, Hue_Magrin, Sat_Magrin, Val_Magrin)
```
## 📈 Histogram Operations
### 🔄 Histogram Equalization

```matlab
% Histogram Equalization for contrast enhancement.
% @param IM: Input grayscale image
% @return outt: Image with equalized histogram
% @brief Spreads pixel intensity distribution across the range, enhancing contrast.
function outt = My_Histogram_Equalization(IM)
```
### 🎯 Histogram Matching

```matlab
% Perform histogram matching on the input image based on a reference image.
% Matches the histogram of the input image to that of the reference image, creating a similar brightness distribution.
% @param Input_Image: Input image whose histogram needs adjustment
% @param ref: Reference image with the desired histogram distribution
% @return outt: Image after histogram matching
% @brief Useful for standardizing brightness and contrast across different images.
function outt = my_histogram_maching(Input_Image, ref)
```
## 📊 Image Analysis and Comparison
### 📷 Compare Outputs for Grayscale

```matlab
% Compare the output of a given function applied to a grayscale image with its original version.
% This function plots the original and manipulated images along with their histograms.
% @param func: Function to be applied to the image
% @param image_name_string: Name of the input image file
function My_Compare_Outputs_Gray(func, image_name_string)
```
### 🎨 Compare Outputs for Colored Images

```matlab
% Compare the output of a given function applied to a colored image with its original version.
% This function plots the original and manipulated images along with their histograms.
% @param func: Function to be applied to the image
% @param image_name_string: Name of the input image file
function My_Compare_Outputs_Colored(func, image_name_string)
```
## 📉 Quality and Distortion Metrics
### 📏 Image Distortion Measures

```matlab
% Calculate distortion measures between an original image and a distorted image.
% This function computes Mean Absolute Error, Mean Squared Error, and Peak Signal-to-Noise Ratio (PSNR).
% @param original_image: Original image (reference)
% @param distorted_image: Distorted version of the original image
% @return outt: Array containing Mean Absolute Error, Mean Squared Error, and PSNR
function outt = Image_Dostortion_Mesures(original_image, distorted_image)
```
## 🔊 Noise Operations
### ⚙️ Salt Noise Addition

```matlab
% Add noise to the input image.
% This function introduces random noise (salt) to the input image based on the specified percentage.
% @param IM: Input image
% @param percent: Percentage of pixels to be affected by noise (in percentage, e.g., 10 for 10%)
% @return outt: Image with added noise
% @brief Adds random salt noise to enhance testing robustness for noise reduction filters or detection algorithms.
function outt = Add_salt_Noise(IM, percent)
```
## 🧹 Filtering Techniques
### 3️⃣ 3x3 Average Filter

```matlab
% Apply a 3x3 average filter to the input image.
% This function performs spatial filtering using a 3x3 average filter.
% @param input_image: Input grayscale or RGB image
% @return outt: Image after 3x3 average filtering
% @brief Reduces noise by averaging pixel values in a 3x3 neighborhood, effective for low-frequency noise.
function outt = average_filter3x3(input_image)
```
### 3️⃣ 3x3 Median Filter

```matlab
% Apply a 3x3 median filter to the input image.
% This function performs spatial filtering using a 3x3 median filter.
% @param input_image: Input grayscale or RGB image
% @return outt: Image after 3x3 median filtering
% @brief Reduces salt-and-pepper noise by taking the median value of pixels in a 3x3 neighborhood.
function outt = median_filter3x3(input_image)
```
### 5️⃣ 5x5 Median Filter

```matlab
% Apply a 5x5 median filter to the input image.
% @param input_image: Input grayscale or RGB image
% @return outt: Image after 5x5 median filtering
% @brief Reduces noise while preserving edges by applying a median filter in a larger neighborhood.
function outt = median_filter5x5(input_image)
```
## 📐 Bounding Box Annotation
### ✏️ Show Drawn Bounding Boxes

```matlab
% Draw bounding boxes on a colored image based on regions in a segmented binary image (segmentation) like hsv segmentation, local adaptive segmentation, threshold segmentation.
% This function highlights connected regions above a specified area in the binary segmented image.
% @param segmented_image: Segmented binary image with identified regions
% @param colored_image: Colored image on which bounding boxes will be drawn
% @param minAreaThreshold: Minimum area threshold to display bounding boxes
% @return plots the original image with bounding boxes
% @brief Useful for visualizing object detection by drawing bounding boxes around detected objects.
function Show_Draw_Bounding_Boxes(segmented_image, colored_image, minAreaThreshold)
```





