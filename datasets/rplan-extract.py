import copy
import os
import cv2
import numpy as np
import random
import shutil
from tqdm import *
np.set_printoptions(threshold=np.inf)

assert os.path.exists('rplandata'), 'path not exist'
assert os.path.exists('rplandata/Data'), 'path not exist'

# At this point you need to confirm that the floorplan_dataset folder (containing 80,788 images) is in the rplandata/Data directory
assert os.path.exists('rplandata/Data/floorplan_dataset'), 'path not exist'

# all data
all_data = os.listdir(r'rplandata/Data/floorplan_dataset')

# Iterate over each file name fn in all_data, remove the last five characters of the file name and convert it to an integer.
# If the file name is not 'list.txt', add it to the ids list
ids = [int(fn[:-4]) for fn in all_data]
# print(ids)

# Create a path to place the images
if not os.path.exists('rplandata/Data/3-channel-semantics-256'):
    os.mkdir('rplandata/Data/3-channel-semantics-256')
if not os.path.exists('rplandata/Data/1-channel-semantics-256'):
    os.mkdir('rplandata/Data/1-channel-semantics-256')
if not os.path.exists('rplandata/Data/bin_imgs'):
    os.mkdir('rplandata/Data/bin_imgs')
if not os.path.exists('rplandata/Data/e_imgs'):
    os.mkdir('rplandata/Data/e_imgs')
if not os.path.exists('rplandata/Data/e_imgs_filteredv1'):
    os.mkdir('rplandata/Data/e_imgs_filteredv1')
if not os.path.exists('rplandata/Data/e_imgs_filteredv2'):
    os.mkdir('rplandata/Data/e_imgs_filteredv2')
if not os.path.exists('rplandata/Data/e_imgs_filteredv3'):
    os.mkdir('rplandata/Data/e_imgs_filteredv3')

# Iterate through the list of ids
for i, id in tqdm(enumerate(ids)):
    # Use OpenCV to read the file. The file path is rplandata/Data/floorplan_dataset/ plus the string form of the id and the file extension '.png'
    origin_img = cv2.imread('rplandata/Data/floorplan_dataset/' + str(id) + '.png', -1)[:, :, 1]


    # Assign the processed image to the semantics variable
    semantics = origin_img
    # Save the semantics image to the path 'rplandata/Data/3-channel-semantics-256/', and the file name is id plus '.png'
    cv2.imwrite('rplandata/Data/3-channel-semantics-256/' + str(id) + '.png', semantics)
    # Get the first channel of the semantics image (index 0), which is usually a grayscale image.
    semantics2 = origin_img
    # Save the single-channel image to the path 'rplandata/Data/1-channel-semantics-256/', and the file name is id plus '.png'
    cv2.imwrite('rplandata/Data/1-channel-semantics-256/' + str(id) + '.png', semantics2)

    # # Set all pixels with a value greater than or equal to 14 in the first channel to red
    origin_img[np.where(origin_img[:, :] >= 14)] = 255
    # # Set all pixels with a value less than or equal to 13 in the first channel to black
    origin_img[np.where(origin_img[:, :] <= 13)] = 0
    # Extract the first channel of the processed image
    bin_img = origin_img
    # Save the channel to the 'bin_imgs/' path
    cv2.imwrite('rplandata/Data/bin_imgs/' + str(id) + '.png', bin_img)

    # Assign the binary image to the dst variable
    dst = bin_img
    # Create a 3x3 all-1 convolution kernel
    kernel_erode = np.ones((3, 3))
    # Use the corrosion operation to process the dst image and assign the result to dst again
    dst = cv2.erode(dst, kernel_erode, iterations=1)



def fix_cv2_bug(bin_img):
    # Fix a bug in cv2 where it returns a new image by creating a deep copy of bin_img and incrementing it by 1
    return copy.deepcopy(bin_img) + 1

def isvalid(img):
    kernel = np.ones((2, 2), dtype=img.dtype) * 255
    # Iterate over each 2x2 region in the image, if it is the same as the core, the image is invalid
    for i in range(0, 255):
        for j in range(0, 255):
            if (img[i:i + 2, j:j + 2] == kernel).all():
                return False
    # If no region is found that is identical to the core, the image is valid
    return True

def isvalid2(img):
    # Traverse every pixel in the image, except for the edge
    for i in range(1, 255):
        for j in range(1, 255):
            # If the current pixel value is 255 (white)
            if img[i, j] == 255:
                # Check if the sum of four adjacent pixels is 255 or 0, if so, the image is invalid
                if img[i - 1, j] + img[i , j - 1] + img[i + 1, j] + img[i, j + 1] == 255 or \
                        img[i - 1, j] + img[i , j - 1] + img[i + 1, j] + img[i, j + 1] == 0:
                    return False
    # If no invalid condition is detected, the image is valid
    return True







# Get all files under 'bin_imgs' directory
bin_imgs = os.listdir('rplandata/Data/bin_imgs')
# Initialize counter
count = 0
# Iterate over files
for fn in tqdm(bin_imgs):
    # Increment file counter
    count += 1
    # Initialize final image buffer
    final = None
    # Read binary image in grayscale mode
    bin_img = cv2.imread('rplandata/Data/bin_imgs/' + fn, cv2.IMREAD_GRAYSCALE)
    # Debug (disabled): print count, filename, shape, random pixel value
    # print(count, fn, bin_img.shape, bin_img[random.randint(0, 255), random.randint(0, 255)])
    # Optionally write raw loaded image to disk for inspection

    # cv2.imwrite('./rplandata/Data/t0_' + fn, bin_img)
    # Initialize life variable
    life = 31
    # Loop while life >= 0
    while life >= 0:
        # Decrement life
        life -= 1
        # Get connected component count
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(fix_cv2_bug(bin_img), connectivity=8)
        # print('num', num, stats)
        # Attempt to simplify image via erosion
        kernel_erode = np.ones((3, 3)) # Structuring element for erosion
        eroded = cv2.erode(bin_img, kernel_erode, iterations=1) # Single erosion iteration

        # Count connected components in eroded image
        num_e, labels_e, stats_e, centroids_e = cv2.connectedComponentsWithStats(fix_cv2_bug(eroded), connectivity=8)
        # If erosion collapses image to all zeros, treat original as final
        if np.sum(eroded) == 0:
            # Keep original (non-eroded) image as final representation
            final = bin_img
            # Remove isolated white pixels (noise) to smooth black regions
            unflattened_patterns_4 = [[[0, 255, 0], [0, 0, 0], [0, 0, 0]],
                                      [[0, 0, 0], [255, 0, 0], [0, 0, 0]],
                                      [[0, 0, 0], [0, 0, 0], [0, 255, 0]],
                                      [[0, 0, 0], [0, 0, 255], [0, 0, 0]],
                                      ]
            for i in range(1, 255):
                for j in range(1, 255):
                    # If local 3x3 patch matches any pattern, zero it out
                    if final[i - 1: i + 2, j - 1: j + 2].tolist() in unflattened_patterns_4:
                        final[i - 1: i + 2, j - 1: j + 2] = 0
            # Persist final simplified image

            cv2.imwrite('./rplandata/Data/e_imgs/t999_' + fn, final)
            break # Exit lifecycle loop
        else:
            if num_e < num:
                # If erosion decreases connected component count, ignore erosion attempt
                # Use 3x3 sliding window kernel to identify segments with width >= 3 white pixels
                thick_white = np.zeros((256, 256), dtype=bin_img.dtype)
                for i in range(1, 255):
                    for j in range(1, 255):
                        if (bin_img[i - 1: i + 2, j - 1: j + 2] == 255).all():
                            thick_white[i - 1: i + 2, j - 1: j + 2] = 1

                # Copy original image and add wide white regions to derive 1‑pixel-wide white lines
                bin_img_copy = copy.deepcopy(bin_img)
                thin_white = bin_img_copy + thick_white

                # Dilate thin white lines and add back; pixels with value 254 may appear so re-binarize
                kernel_dilate = np.ones((3, 3))
                dilated = cv2.dilate(thin_white, kernel_dilate, iterations=1)
                bin_img += dilated
                ret, bin_img = cv2.threshold(bin_img, thresh=128, maxval=255, type=cv2.THRESH_BINARY)

                # Smooth contours
                unflattened_patterns_3 = [[[255, 255, 255], [255, 0, 255], [0, 0, 0]],
                                        [[0, 0, 0], [255, 0, 255], [255, 255, 255]],
                                        [[255, 255, 0], [255, 0, 0], [255, 255, 0]],
                                        [[0, 255, 255], [0, 0, 255], [0, 255, 255]],

                                          [[255, 255, 255], [255, 0, 0], [0, 0, 0]],
                                          [[255, 255, 255], [0, 0, 255], [0, 0, 0]],
                                          [[0, 0, 0], [0, 0, 255], [255, 255, 255]],
                                          [[0, 0, 0], [255, 0, 0], [255, 255, 255]],
                                          [[255, 255, 0], [255, 0, 0], [255, 0, 0]],
                                          [[255, 0, 0], [255, 0, 0], [255, 255, 0]],
                                          [[0, 255, 255], [0, 0, 255], [0, 0, 255]],
                                          [[0, 0, 255], [0, 0, 255], [0, 255, 255]],

                                          [[255, 255, 0], [255, 0, 0], [0, 0, 0]],
                                          [[0, 255, 255], [0, 0, 255], [0, 0, 0]],
                                          [[0, 0, 0], [0, 0, 255], [0, 255, 255]],
                                          [[0, 0, 0], [255, 0, 0], [255, 255, 0]],
                                          ]
                for i in range(1, 255):
                    for j in range(1, 255):
                        if bin_img[i - 1: i + 2, j - 1: j + 2].tolist() in unflattened_patterns_3:
                            bin_img[i, j] = 255
                # Smooth contours
                unflattened_patterns_5 = [[[255, 0, 255], [255, 255, 255], [255, 255, 255]],
                                          [[255, 255, 255], [0, 255, 255], [255, 255, 255]],
                                          [[255, 255, 255], [255, 255, 0], [255, 255, 255]],
                                          [[255, 255, 255], [255, 255, 255], [255, 0, 255]],
                                          ]
                for i in range(1, 255):
                    for j in range(1, 255):
                        if bin_img[i - 1: i + 2, j - 1: j + 2].tolist() in unflattened_patterns_5:
                            bin_img[i - 1: i + 2, j - 1: j + 2] = 255

            else:
                # If erosion did not reduce connected components, continue next loop with the eroded image
                bin_img = copy.deepcopy(eroded)



# Use 2x2 white kernel to filter erroneous data (75350 -> 71814)

for fn in tqdm(os.listdir('rplandata/Data/e_imgs')):
    # Copy filtered images to new directory
    shutil.copy('rplandata/Data/e_imgs/' + fn, 'rplandata/Data/e_imgs_filteredv1/' + fn.replace('t999_', ''))
count = 0
remove1_count = 0
for fn in tqdm(os.listdir('rplandata/Data/e_imgs_filteredv1')):
    count += 1
    # Print current processed file count and filename
    # print(count, fn)
    img = cv2.imread('rplandata/Data/e_imgs_filteredv1/' + fn, cv2.IMREAD_GRAYSCALE)
    if not isvalid(img):
    # If image invalid (contains 2x2 white block => erosion incomplete) delete and count
        os.remove('rplandata/Data/e_imgs_filteredv1/' + fn)
        remove1_count += 1
    # Print number of removed files
        # print(remove1_count)


# Filter topological errors (dead ends) (71814 -> 71763)

for fn in tqdm(os.listdir('rplandata/Data/e_imgs_filteredv1')):
    shutil.copy('rplandata/Data/e_imgs_filteredv1/' + fn, 'rplandata/Data/e_imgs_filteredv2/' + fn)
count = 0
remove2_count = 0
for fn in tqdm(os.listdir('rplandata/Data/e_imgs_filteredv2')):
    count += 1
    # print(count, fn)
    img = cv2.imread('rplandata/Data/e_imgs_filteredv2/' + fn, cv2.IMREAD_GRAYSCALE)
    if not isvalid2(img):
        os.remove('rplandata/Data/e_imgs_filteredv2/' + fn)
        remove2_count += 1
        # print(remove2_count)



# Ensure consistency with original topology (8-connectivity) so semantics and door info preserved (71763 unchanged)

for fn in tqdm(os.listdir('rplandata/Data/e_imgs_filteredv2')):
    shutil.copy('rplandata/Data/e_imgs_filteredv2/' + fn, 'rplandata/Data/e_imgs_filteredv3/' + fn)
count = 0
remove3_count = 0
for fn in tqdm(os.listdir('rplandata/Data/e_imgs_filteredv3')):
    count += 1
    # print(count, fn)
    img = cv2.imread('rplandata/Data/e_imgs_filteredv3/' + fn, cv2.IMREAD_GRAYSCALE)
    img_ori = cv2.imread('rplandata/Data/bin_imgs/' + fn, cv2.IMREAD_GRAYSCALE)
    if not (cv2.connectedComponentsWithStats(fix_cv2_bug(img), connectivity=8)[0] == cv2.connectedComponentsWithStats(fix_cv2_bug(img_ori), connectivity=8)[0]):
        os.remove('rplandata/Data/e_imgs_filteredv3/' + fn)
        remove3_count += 1
        # print(remove3_count)






# Extract structure graph, interior doors, boundary, front door, room semantics, etc.
# Define structure graph dict: key=file name; value=graph mapping (x1,y1)->[up,left,down,right] neighbor (xi,yi) or (-1,-1) if none
count = 0
structure_graphs = {}

# Get coordinate sequence between two corner points
def get_coords(corner1, corner2):
    # If the two corners are in the same column
    if corner1[0] == corner2[0]:
        if corner1[1] < corner2[1]:
            return [(corner1[0], i) for i in range(corner1[1], corner2[1] + 1)]
        elif corner1[1] > corner2[1]:
            return [(corner1[0], i) for i in range(corner2[1], corner1[1] + 1)]
        else:
            assert 0
    # If the two corners are in the same row
    elif corner1[1] == corner2[1]:
        if corner1[0] < corner2[0]:
            return [(j, corner1[1]) for j in range(corner1[0], corner2[0] + 1)]
        elif corner1[0] > corner2[0]:
            return [(j, corner1[1]) for j in range(corner2[0], corner1[0] + 1)]
        else:
            assert 0
    else:
        assert 0

# Function to determine whether a coordinate sequence forms a boundary
def is_edge_func2(coords, corners, img):
    for coord in coords:
        if img[coord[1], coord[0]] == 0 or (coord in corners and coord != coords[0] and coord != coords[-1]):
            return False
    return True


# Iterate filtered images to extract structure graph
for fn in tqdm(os.listdir('rplandata/Data/e_imgs_filteredv3')):
    count += 1
    # print(count, fn)
    img = cv2.imread('rplandata/Data/e_imgs_filteredv3/' + fn, cv2.IMREAD_GRAYSCALE)
    try:
    # Initialize corner list
        corners_L = []
        corners_T = []
        corners_X = []
    # Extract corner points
        for i in range(1, 255):
            for j in range(1, 255):
                # Ignore I-shaped intersections; only consider L, T, and X intersections
                if img[i, j] == 255:
                    # L
                    if img[i - 1, j] + img[i , j - 1] + img[i + 1, j] + img[i, j + 1] == 254 and \
                            img[i - 1, j] + img[i + 1, j] == 255:
                        corners_L.append((j, i))
                    # T
                    elif img[i - 1, j] + img[i , j - 1] + img[i + 1, j] + img[i, j + 1] == 253:
                        corners_T.append((j, i))
                    # X
                    elif img[i - 1, j] + img[i , j - 1] + img[i + 1, j] + img[i, j + 1] == 252:
                        corners_X.append((j, i))
                    else:
                        continue
                else:
                    continue
    # Merge all corner point lists
        corners = []
        corners.extend(corners_L)
        corners.extend(corners_T)
        corners.extend(corners_X)
    # Extract boundaries
        edges = []
        for corner1 in corners:
            for corner2 in corners:
                if corner1 != corner2:
                    # Skip if the two corner points are not in the same row or column
                    if not ((corner1[0] == corner2[0]) or (corner1[1] == corner2[1])):
                        continue
                    else:
                        # Get the coordinate sequence between the two corner points
                        coords = get_coords(corner1, corner2)
                        # If not the minimal wall segment
                        if not is_edge_func2(coords, corners, img):
                            continue
                        else:
                            # Add the coordinate sequence to the boundary list
                            edges.append((corner1, corner2))
                else:
                    continue
    # Convert boundaries to a structure graph
        structure_graph = {}
        for corner in corners:
            # Get neighboring points
            adjacents = {}
            for edge in edges:
                # Determine direction
                if edge[0] == corner or edge[1] == corner:
                    e_l = list(edge)
                    e_l.remove(corner)
                    adjacent = e_l[0]
                    # up
                    if adjacent[0] == corner[0] and adjacent[1] < corner[1]:
                        adjacents['up'] = adjacent
                    # down
                    elif adjacent[0] == corner[0] and adjacent[1] > corner[1]:
                        adjacents['down'] = adjacent
                    # left
                    elif adjacent[1] == corner[1] and adjacent[0] < corner[0]:
                        adjacents['left'] = adjacent
                    # right
                    elif adjacent[1] == corner[1] and adjacent[0] > corner[0]:
                        adjacents['right'] = adjacent
                    else:
                        assert 0
            # Add neighboring point info to the structure graph
            adjacents_list = []
            for direction in ['up', 'left', 'down', 'right']:
                if direction in adjacents.keys():
                    adjacents_list.append(adjacents[direction])
                else:
                    adjacents_list.append((-1, -1))
            structure_graph[corner] = adjacents_list
    # Add the structure graph to the dictionary
        structure_graphs[int(fn[:-4])] = structure_graph
    except:
        pass

# Save the structure graph dictionary as a .npy file
np.save('rplandata/Data/structure_graphs.npy', structure_graphs)


# Load to inspect
b = np.load('rplandata/Data/structure_graphs.npy', allow_pickle=True).item()
print(b)