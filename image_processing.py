# Importing necessary libraries:
from skimage import color, filters, io, util, segmentation, feature, transform
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import time

start_time = time.time()

print('--------------------------------------------------------------------------------------------------')
#--------------------------------------------------------------------------------------------------
# Question 2.1:
print('Question 2.1')

# Reading Avengers image:
avengers_image = io.imread('data/avengers_imdb.jpg')

# Determining size of Avengers image:
avengers_size = avengers_image.shape

# Printing out size of the Avengers image:
print('The size of the Avengers image is: {}'.format(avengers_size))

# Converting the image to grayscale:
avengers_gray = color.rgb2gray(avengers_image)
plt.imsave('outputs/avengers_grayscale.jpg', avengers_gray, cmap = 'gray')

# Finding the threshold to convert Avengers image from grayscale to black and white:
threshold_for_bw = filters.threshold_isodata(avengers_gray)

# Creating black and white representation of Avengers image based on threshold found:
avengers_black_white = avengers_gray > threshold_for_bw
plt.imsave('outputs/avengers_black_white.jpg', avengers_black_white, cmap = 'gray')

print('--------------------------------------------------------------------------------------------------')
#--------------------------------------------------------------------------------------------------
# Question 2.2:
print('Question 2.2')
print('Please see report for Bush House image with noise and its filtered representations.')

# Importing bush_house image:
bush_house_image = io.imread('data/bush_house_wikipedia.jpg')

# Adding Gaussian random noise to image, with variance specified as 0.1:
bush_house_image = util.random_noise(bush_house_image, mode = 'gaussian', var = 0.1)
plt.imsave('outputs/bush_house_with_noise.jpg', bush_house_image)

# Filtering the bush_house image with noise with a Gaussian mask with sigma specified as 1:
bush_house_with_gaussian = filters.gaussian(bush_house_image, sigma = 1, multichannel = True)
# Saving bush_house image after applying Gaussian mask to reduce noise:
plt.imsave('outputs/bush_house_with_gaussian.jpg', bush_house_with_gaussian)

# Filtering the bush_house image with noise with a uniform smoothing mask. The size is specified to be 9 x 9:
bush_house_with_uniform = ndimage.uniform_filter(bush_house_image, size = 9)
# saving bush_house image after applying Uniform smoothing mask to reduce noise:
plt.imsave('outputs/bush_house_with_uniform.jpg', bush_house_with_uniform)

print('--------------------------------------------------------------------------------------------------')
#--------------------------------------------------------------------------------------------------
# Question 2.3:
print('Question 2.3')
print('Please see report for k-means segmentation of Forestry Commission image.')

# Importing forestry commission image:
forestry_commission_image = io.imread('data/forestry_commission_gov_uk.jpg')

# Using k-means segmentation to divide forestry_commission_image into 5 components:
segmented_forest = segmentation.slic(forestry_commission_image, n_segments = 5, multichannel=True,
                                     compactness= 25)

# Saving forestry commission image with k-means segmentation applied to it:
plt.imsave('outputs/kmeans_segmented_forestry_commission.jpg', segmented_forest)

print('--------------------------------------------------------------------------------------------------')
#--------------------------------------------------------------------------------------------------
# Question 2.4:
print('Question 2.4')
print('Please see report for Rolland Garros images. Includes Canny Edge Detection followed by Hough Transform.')

# Importing rolland garros image:
rolland_garros_image = io.imread('data/rolland_garros_tv5monde.jpg')

# Converting rolland garros image to grayscale:
rolland_garros_grayscale = color.rgb2gray(rolland_garros_image)

# Applying Canny Edge Detection to rolland garros image:
canny_rolland_garros = feature.canny(rolland_garros_grayscale, sigma = 1)

# Saving rolland garros image with Canny Edge Detection applied to it:
plt.imsave('outputs/canny_edge_detection_rolland_garros.jpg', canny_rolland_garros, cmap = 'gray')

# Applying Hough Transform to rolland garros image with Canny Edge Detection in place:
hough_lines = transform.probabilistic_hough_line(canny_rolland_garros, threshold=10, line_length=5, line_gap=3)

plt.imshow(canny_rolland_garros * 0)
for line in hough_lines:
    p0, p1 = line
    plt.plot((p0[0], p1[0]), (p0[1], p1[1]))
plt.xlim((0, rolland_garros_image.shape[1]))
plt.ylim((rolland_garros_image.shape[0], 0))
plt.axis('off')
plt.title('Probabilistic Hough Transform of Rolland Garros Image')

plt.tight_layout()
plt.savefig('outputs/hough_transform_rolland_garros.jpg')

#--------------------------------------------------------------------------------------------------

end_time = time.time()
print('The script took {} seconds to run.'.format(end_time - start_time))

# Citations:
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html
# https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.slic
# https://scikit-image.org/docs/dev/auto_examples/edges/plot_line_hough_transform.html