clc
clear
close all

Image1 = imread('./example_images/vqDittm.jpg');
Image2 = imread('./example_images/maxresdefault.jpg');
Image3 = My_Gamma_Correction(Image1,1.5);
Image4 = My_Gamma_Correction(Image2,1.5);

subplot(2,2,1),imshow(Image1),title('original image 1')
subplot(2,2,2),imshow(Image2),title('original image 2')
subplot(2,2,3),imshow(Image3),title('manipulated image 1')
subplot(2,2,4),imshow(Image4),title('manipulated image 1')
%%
% Utility Functions: Histogram Calculation
% Calculate the histogram of the input grayscale image.
% @param IM: Input image
% @return outt: Histogram of the input image
% @brief Computes the frequency of each grayscale intensity level (0-255) in the input image.
function outt = My_Hist(IM)
    if size(IM, 3) > 1
       IM = rgb2gray(IM); 
    end
    
    histogram_value = zeros(256, 1);
    [rows, cols] = size(IM);
    for i = 1:rows
        for j = 1:cols
            histogram_value(IM(i, j) + 1) = histogram_value(IM(i, j) + 1) + 1;
        end
    end
    outt = histogram_value;
end

%%
% Calculate the histogram of a multi-channel image.
% @param Image: Input RGB image
% @return outt: Histogram of the image
function outt = my_histogram(Image)
    histogram = zeros(256, 1);
    [rows, cols] = size(Image); 
    
    for i = 1:rows
        for j = 1:cols
            value = Image(i, j);
            histogram(value + 1) = histogram(value + 1) + 1;     
        end
    end
    outt = histogram;
end

%%
% Perform Contrast Stretching on the input image.
% @param IM: Input image
% @return outt: Processed image after contrast stretching
% @brief Expands the intensity range of the image for better contrast by stretching from minimum to maximum values.
function outt = My_Contrast_Stretching(IM)
    if size(IM, 3) > 1
       IM = rgb2gray(IM); 
    end
    IM = double(IM);
    my_max = double(max(max(IM)));
    my_min = double(min(min(IM)));
    IM_new = ((IM - my_min) / (my_max - my_min)) * 255;
    outt = uint8(IM_new);
end

% Gamma Correction for brightness adjustment.
% @param IM: Input image
% @param Gamma_Level: Gamma level (between 0.8 and 1.2 recommended)
% @return outt: Image after gamma correction
% @brief Useful for brightness control; gamma > 1 brightens, < 1 darkens.
function outt = My_Gamma_Correction(IM,Gamma_Level)
    IM = double(IM) .^ Gamma_Level;
    outt = uint8(IM);
end

% Perform Local Adaptive Segmentation.
% @param IM: Input image
% @param Length: Window length for segmentation
% @param Width: Window width for segmentation
% @return outt: Segmented image with local adaptive thresholding
% @brief Local adaptive segmentation improves thresholding where shadows are present.
function outt = My_Local_Adaptive_Segmentation(IM, Length, Width)
    if size(IM, 3) > 1
       IM = rgb2gray(IM); 
    end
    
    [rows, cols] = size(IM);
    Length = 9;
    Width = 8;
    new_rows = round(rows / Length);
    new_cols = round(cols / Width);
    
    for i = 1:new_rows:rows-new_rows+1  
        for j = 1:new_cols:cols-new_cols+1  
            section_image = IM(i:i+new_rows-1, j:j+new_cols-1);
            the_new_mean = mean(mean(section_image));

            for ii = i:i+new_rows-1
                for jj = j:j+new_cols-1
                    if IM(ii, jj) >= the_new_mean
                        IM(ii, jj) = 255;
                    else
                        IM(ii, jj) = 0;
                    end
                end
            end
        end
    end   
    outt = IM;
end

% Perform HSV-based color segmentation.
% @param IM: Input image
% @param Hue_Magrin: Range for hue threshold (min, max in [0, 360])
% @param Sat_Magrin: Range for saturation threshold (min, max in [0, 1])
% @param Val_Magrin: Range for value threshold (min, max in [0, 1])
% @return outt: Binary image after HSV segmentation
% @brief HSV segmentation isolates colors in the specified hue range.
function outt = My_HSV_Segmentation(IM, Hue_Magrin, Sat_Magrin, Val_Magrin)
    [IM_H, IM_S, IM_V] = rgb2hsv(IM);
    [rows, cols] = size(IM_H);
    output_image = zeros(rows);

    for i = 1:rows
        for j = 1:cols
            if (IM_H(i, j) > Hue_Magrin(1)/360 && IM_H(i, j) < Hue_Magrin(2)/360 && ...
                IM_S(i, j) > Sat_Magrin(1) && IM_S(i, j) < Sat_Magrin(2) && ...
                IM_V(i, j) > Val_Magrin(1) && IM_V(i, j) < Val_Magrin(2))
                output_image(i, j) = 255;
            else
                output_image(i, j) = 0;
            end
        end
    end
    outt = uint8(output_image);
end

% Histogram Equalization for contrast enhancement.
% @param IM: Input grayscale image
% @return outt: Image with equalized histogram
% @brief Spreads pixel intensity distribution across the range, enhancing contrast.
function outt = My_Histogram_Equalization(IM)
    if size(IM, 3) > 1
       IM = rgb2gray(IM); 
    end
    
    [rows, cols] = size(IM);
    my_histogramm = My_Hist(IM); 
    
    probability = my_histogramm / (rows * cols);
    
    cdf = zeros(256, 1);
    cdf(1) = probability(1);
    
    for i = 2:256
        cdf(i) = cdf(i - 1) + probability(i);
    end
    
    equalized_img = zeros(rows, cols);
    
    for i = 1:rows
        for j = 1:cols
            intensity = IM(i, j) + 1;
            equalized_img(i, j) = cdf(intensity) * 255;
        end
    end
    
    outt = uint8(equalized_img);  
end

%%
% Perform histogram matching on the input image based on a reference image.
% Matches the histogram of the input image to that of the reference image, creating a similar brightness distribution.
% @param Input_Image: Input image whose histogram needs adjustment
% @param ref: Reference image with the desired histogram distribution
% @return outt: Image after histogram matching
% @brief Useful for standardizing brightness and contrast across different images.
function outt = my_histogram_maching(Input_Image, ref)
    if size(ref, 3) > 1
       ref = rgb2gray(ref); 
    end
    if size(Input_Image, 3) > 1
       Input_Image = rgb2gray(Input_Image); 
    end
    
    histo1 = my_histogram(ref);
    histo2 = my_histogram(Input_Image);
    histo_match = from_histo_to_match(histo1, histo2);
    histo_match_CDF = get_cdf(histo_match);
    
    [rows, cols]  = size(Input_Image);
     
    for i = 1:rows
        for j = 1:cols
            value  = Input_Image(i, j) + 1;
            mask = histo_match_CDF(value) * 255;
            Input_Image(i, j) = round(mask);
        end
    end
    
    outt = Input_Image;
end

%%
% Map two histograms to match the target distribution.
% @param Original_Image_histo: Histogram of the original image
% @param Desired_Image_histo: Histogram of the target image
% @return outt: Adjusted histogram to match target distribution
function outt = from_histo_to_match(Original_Image_histo, Desired_Image_histo)
    val1 = get_cdf(Original_Image_histo);
    val2 = get_cdf(Desired_Image_histo); 
    outt = cdfhistomatch(val1, val2, Desired_Image_histo);
end

%%
% Generate the cumulative distribution function (CDF) from a histogram.
% @param histo: Histogram data
% @return outt: CDF of the input histogram
% @brief Used in histogram matching to calculate cumulative distribution for image intensity transformation.
function outt = get_cdf(histo)
    histo = double(histo);
    summ = sum(histo);
    sa = length(histo);
    
    for i = 2:sa
        histo(i) = histo(i-1) + histo(i);
    end

    outt = (histo ./ summ);
end

%%
% Match two cumulative distribution functions (CDFs) to perform histogram matching.
% @param cdf1: CDF of the input image
% @param cdf2: CDF of the reference image
% @param original_histo: Histogram of the reference image
% @return outt: Adjusted histogram for matched distribution
function outt = cdfhistomatch(cdf1, cdf2, original_histo)
    cdf1 = round(cdf1 * 7);
    cdf2 = round(cdf2 * 7);
    
    lop = length(cdf1);
    map = zeros(lop, 1);
    outt = zeros(lop, 1);
    
    for i = 1:lop
        for j = 1:lop
            if cdf1(i) <= cdf2(j)
                map(i) = j - 1;
                break;
            end
        end
    end
    
    for i = 1:lop
        val = map(i);
        outt(val + 1) = outt(val + 1) + original_histo(i);
    end
end


%%
% Compare the output of a given function applied to a grayscale image with its original version.
% This function plots the original and manipulated images along with their histograms.
% @param func: Function to be applied to the image
% @param image_name_string: Name of the input image file
function My_Compare_Outputs_Gray(func, image_name_string)
    % Read the input image
    image = imread(image_name_string);
    % Convert the image to grayscale if it's not already
    if size(image, 3) > 1
       image = rgb2gray(image); 
    end
    % Apply the function to the image
    result = func(image);
    
    % Plotting
    subplot(2, 2, 1), imshow(image), title('Original Image')
    subplot(2, 2, 2), imshow(result), title('Manipulated Image')
    
    subplot(2, 2, 3), plot(My_Hist(image)), title('Histogram of Original Image')
    subplot(2, 2, 4), plot(My_Hist(result)), title('Histogram of Manipulated Image')
end


% Compare the output of a given function applied to a colored image with its original version.
% This function plots the original and manipulated images along with their histograms.
% @param func: Function to be applied to the image
% @param image_name_string: Name of the input image file
function My_Compare_Outputs_Colored(func, image_name_string)
    % Read the input image
    image = imread(image_name_string);
    % Apply the function to the image
    result = func(image);
    
    % Plotting
    subplot(2, 2, 1), imshow(image), title('Original Image')
    subplot(2, 2, 2), imshow(result), title('Manipulated Image')
    subplot(2, 2, 3), hold on;
    plot(My_Hist(image(:,:,1)), 'r');
    plot(My_Hist(image(:,:,2)), 'g');
    plot(My_Hist(image(:,:,3)), 'b');
    title('Histogram of Original Image');
    legend('Red', 'Green', 'Blue');
    hold off;
    subplot(2, 2, 4), plot(My_Hist(result)), title('Histogram of Manipulated Image')
end


%%
% Calculate distortion measures between an original image and a distorted image.
% This function computes Mean Absolute Error, Mean Squared Error, and Peak Signal-to-Noise Ratio (PSNR).
% @param original_image: Original image (reference)
% @param distorted_image: Distorted version of the original image
% @return outt: Array containing Mean Absolute Error, Mean Squared Error, and PSNR
function outt = Image_Dostortion_Mesures(original_image, distorted_image)

    Z_Image = abs(original_image - distorted_image);
    
    [rows, cols] = size(Z_Image);
    num = size(rows) * size(cols);
    
    Mean_Abs_error = 0;
    Mean_Squ_err0r = 0;
    
    for i = 1:rows
        for j = 1:cols
            Mean_Abs_error = Mean_Abs_error + Z_Image(i, j);
            Mean_Squ_error = Mean_Squ_error + Z_Image(i, j);
        end
    end
    Mean_Abs_error = (1/num)*(Mean_Abs_error);
    Mean_Squ_error = (1/num)*(Mean_Squ_error);
    Peak_Signal_to_Noise_Ratio = log10(255^2 / Mean_Squ_error);
    
    outt = [Mean_Abs_error, Mean_Squ_error, Peak_Signal_to_Noise_Ratio];
end 

%%
% Add noise to the input image.
% This function introduces random noise (salt) to the input image based on the specified percentage.
% @param IM: Input image
% @param percent: Percentage of pixels to be affected by noise (in percentage, e.g., 10 for 10%)
% @return outt: Image with added noise
% @brief Adds random salt noise to enhance testing robustness for noise reduction filters or detection algorithms.
function outt = Add_salt_Noise(IM, percent)
    input_image = IM;
    
    [rows, cols] = size(input_image);
    noise = rand(rows, cols, 1);
        
    for i = 1:rows
        for j = 1:cols
            if noise(i, j) < percent/100
                input_image(i, j) = 255;
            end
        end
    end
    
    outt = input_image;
end

%%
% Apply a 3x3 average filter to the input image.
% This function performs spatial filtering using a 3x3 average filter.
% @param input_image: Input grayscale or RGB image
% @return outt: Image after 3x3 average filtering
% @brief Reduces noise by averaging pixel values in a 3x3 neighborhood, effective for low-frequency noise.
function outt = average_filter3x3(input_image)
    if size(input_image, 3) > 1
       input_image = rgb2gray(input_image); 
    end
    
    [rows, cols] = size(input_image);
    output_image = zeros(rows, cols);
    
    for i = 2:rows-1
        for j = 2:cols-1
            neighborhood = input_image(i-1:i+1, j-1:j+1);
            average_value = round(mean(neighborhood(:)));
            output_image(i, j) = average_value;
        end
    end
    
    outt = uint8(output_image);
end

%%
% Apply a 3x3 median filter to the input image.
% This function performs spatial filtering using a 3x3 median filter.
% @param input_image: Input grayscale or RGB image
% @return outt: Image after 3x3 median filtering
% @brief Reduces salt-and-pepper noise by taking the median value of pixels in a 3x3 neighborhood.
function outt = median_filter3x3(input_image)
    if size(input_image, 3) > 1
       input_image = rgb2gray(input_image); 
    end
    
    [rows, cols] = size(input_image);
    output_image = zeros(rows, cols);
    
    for i = 2:rows-1
        for j = 2:cols-1
            neighborhood = input_image(i-1:i+1, j-1:j+1);
            temp = neighborhood(:);
            sorted_neighborhood = sort(temp);
            median_value = sorted_neighborhood(5);
            output_image(i, j) = median_value;
        end
    end
    
    outt = uint8(output_image);
end

%%
% Apply a 5x5 median filter to the input image.
% @param input_image: Input grayscale or RGB image
% @return outt: Image after 5x5 median filtering
% @brief Reduces noise while preserving edges by applying a median filter in a larger neighborhood.
function outt = median_filter5x5(input_image)
    if size(input_image, 3) > 1
       input_image = rgb2gray(input_image); 
    end
    
    [rows, cols] = size(input_image);
    output_image = zeros(rows, cols);
    
    for i = 3:rows-2
        for j = 3:cols-2
            neighborhood = input_image(i-2:i+2, j-2:j+2);
            temp = neighborhood(:);
            sorted_neighborhood = sort(temp);
            median_value = sorted_neighborhood(13);
            output_image(i, j) = median_value;
        end
    end
    
    outt = uint8(output_image);
end

%%
% Draw bounding boxes on a colored image based on regions in a segmented binary image (segmentation) like hsv segmentation, local adaptive segmentation, threshold segmentation.
% This function highlights connected regions above a specified area in the binary segmented image.
% @param segmented_image: Segmented binary image with identified regions
% @param colored_image: Colored image on which bounding boxes will be drawn
% @param minAreaThreshold: Minimum area threshold to display bounding boxes
% @return plots the original image with bounding boxes
% @brief Useful for visualizing object detection by drawing bounding boxes around detected objects.
function Show_Draw_Bounding_Boxes(segmented_image, colored_image, minAreaThreshold)

    % Label connected components in the binary image
    [L, num] = bwlabel(segmented_image);
    % L: Labeled image where connected components are assigned unique integer labels.
    % num: Number of connected components (objects) found in the image.
    
    % Compute bounding boxes and areas of the regions
    bboxes = regionprops(L, 'BoundingBox', 'Area');
    % regionprops: Measures properties of labeled image regions.
    % Inputs:
    % L: Labeled matrix where each pixel in the original image is assigned a label indicating its connected component.
    % 'BoundingBox': Specifies to compute the bounding box of each labeled region.
    % 'Area': Specifies to compute the area of each labeled region.
    % Outputs:
    % bboxes: Structure array containing information about each labeled region, including bounding box coordinates and area.

    % Display the original colored image
    imshow(colored_image);
    hold on; % Enable hold to overlay bounding boxes
    
    % Loop over each bounding box
    for k = 1:length(bboxes)
        CurrBB = bboxes(k).BoundingBox; % Extract current bounding box
        
        % Only draw the bounding box if its area is above the threshold
        if bboxes(k).Area > minAreaThreshold
            % Draw a rectangle representing the bounding box
            rectangle('Position', [CurrBB(1), CurrBB(2), CurrBB(3), CurrBB(4)], ...
                'EdgeColor', 'r', 'LineWidth', 2);
        end
    end
    
    hold off; % Disable hold to allow subsequent plot commands
end

