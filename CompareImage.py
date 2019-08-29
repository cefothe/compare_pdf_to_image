import cv2
import matplotlib.pyplot as plt
from skimage.measure import compare_mse as mse
# from pdf2image import convert_from_path, convert_from_bytes
from skimage.measure import compare_ssim as ssim

# images = convert_from_path("pdf/final.pdf",output_folder="output/",fmt="jpeg")

'''
Compare images used MSE and SSIM algorithm
'''


def compare_images(imageA, imageB, title):
    # compute the mean squared error and structural similarity
    # index for the images
    m = mse(imageA, imageB)
    s = ssim(imageA, imageB, multichannel=True)

    # setup the figure
    fig = plt.figure(title)
    plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))

    # show first image
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(imageA)
    plt.axis("off")

    # show the second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(imageB)
    plt.axis("off")

    # show the images
    plt.show()


'''
Compare histograms used images with same size.
'''


def compare_histograms(first_image, second_image):
    first_histogram = cv2.calcHist(first_image, [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(src=first_histogram, dst=first_histogram).flatten()
    second_histogram = cv2.calcHist(second_image, [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(src=second_histogram, dst=second_histogram).flatten()

    histogram = cv2.compareHist(first_histogram, second_histogram, cv2.HISTCMP_HELLINGER)

    print("___________________________")
    print("Result of comparing " + str(histogram))


img = cv2.imread("hacker4e.jpg")
img2 = cv2.imread("warped.jpg")
sift = cv2.xfeatures2d.SIFT_create()

keypoints, descriptor = sift.detectAndCompute(img, None)
keypoints2, descriptor2 = sift.detectAndCompute(img2, None)

index_param = dict(algorithm=0, trees=5)
search_param = dict();
flan = cv2.FlannBasedMatcher(index_param, search_param)

matches = flan.knnMatch(descriptor, descriptor2, k=2);

# Initialize lists
list_kp1 = []
list_kp2 = []

good_points = []
for m, n in matches:
    if m.distance < 0.2 * n.distance:
        good_points.append(m)
        # Get the matching keypoints for each of the images
        img1_idx = m.queryIdx
        img2_idx = m.trainIdx

        # x - columns
        # y - rows
        # Get the coordinates
        (x1, y1) = keypoints[img1_idx].pt
        (x2, y2) = keypoints2[img2_idx].pt

        # Append to each list
        list_kp1.append((x1, y1))
        list_kp2.append((x2, y2))

result = cv2.drawMatches(img, keypoints, img2, keypoints2, good_points, None)
cv2.imshow("Result", cv2.resize(result, None, fx=0.2, fy=0.2))
cv2.waitKey(0)
cv2.destroyAllWindows()

print(len(good_points))
print(list_kp1)
print(list_kp2)

print(len(list_kp1))
print(len(list_kp2))

for index in range(len(list_kp1)):
    print(index, list_kp1[index][0] / list_kp2[index][0])
    print(index, list_kp1[index][1] / list_kp2[index][1])
    print("-----------------------------")

resized_image = cv2.resize(img2, None, fx=0.53, fy=0.52)

print("__________________________")
print(img.shape[:2])
print("__________________________")

print(resized_image.shape[:2])

height, width, channels = resized_image.shape

resized_image = cv2.resize(img2, (1166, 1654))

cv2.imwrite("rezized.jpg", resized_image)

img_new = cv2.imread("hacker4e.jpg")
img2_new = cv2.imread("rezized.jpg")

compare_images(img_new, img2_new, "PDF vs. Image")
compare_histograms(img_new, img2_new)
