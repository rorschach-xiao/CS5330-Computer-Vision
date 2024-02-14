# Lab02 Report

## Techniques
To segment the leaf area in the given image, I explored five traditional computer vision techniques: 

- **Global/Local thresholding**: Local thresholding(adaptive thresholding) can help separate the background and foreground area, while
Furthermore, a global threshold is applied to filter green area in the foreground.
- **Canny edge detection**: Canny edge detector is used to generate an detailed edge map for each leaves. Contours are filtered on shape, size and hierarchy.
- **Morphological operations**: Opening and closing refine the leaves mask by removing small noise and connecting gap regions.
- **RG chromaticity segmentation**: the image is converted to RG color space. A reference leaf patch is used to estimate RG mean and standard deviation. 
A Gaussian distribution mask is generated based on the patch statistics.

The main goal of all techniques is to isolate the leaf regions from the background. Shape, size, and color cues are used to 
distinguish the leaves. Thresholding relies on intensity contrast. Edge detection looks for leaf boundaries. RG chromaticity uses the color signature of the leaf patch.

## Implementations
I built three pipelines utilizing the techniques I mentioned above.

### **Pipeline1**: thresholding based segmentation
  - convert image from BRG space to gray scale
  ```python
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  ```
  - apply adaptive thresholding to segment foreground area
  ```python
    thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                          cv2.THRESH_BINARY_INV, 271, 1)
  ```
  - extract the contours from the mask. I used `RETR_CCOMP` mode here to further exclude the inner gap areas.
  ```python
    contours, hierarchy = cv2.findContours(np.uint8(thresh), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
  ```
  
  - filter contours based on two conditions: 
    1. bounding rectangle is nearly a square
    2. contour area is larger than 1000 and smaller than 15000
  ```python
    filtered_contours = []
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if w > 0 and 0.6 < h / w < 1.4 and 1000 < area < 15000:
            filtered_contours.append(contour)
  ```
 
  - fill the filtered contours
  ```python
    filled_mask = np.zeros_like(img_gray)
    cv2.fillPoly(filled_mask, filtered_contours, 255)
  ```
  
  - apply green mask to further filter out the leaf area
  ```python
    lower_green = np.array([3, 20, 30])  # green upper-bound
    upper_green = np.array([75, 93, 101])  # green lower-bound
    mask = cv2.inRange(img, lower_green, upper_green)
  ```
  
  - apply morphology algorithm to further denoise.
  ```python
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = np.uint8(mask / 255.0)
    filled_mask = filled_mask * mask
    filled_mask = cv2.morphologyEx(filled_mask, cv2.MORPH_OPEN, kernel)
  ```
  
  
### **Pipeline2**: Edge detection based segmentation
  - convert the image from BGR to gray scale
  ```python
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  ```
  - use canny operator to detect the edge map. 
    - note: low threshold can help capture more details
  ```python
    edge_map = cv2.Canny(img_gray, 32, 75, apertureSize=3, L2gradient=True)
  ```
  - apply close operation to the edge map to fill the inner holes.
  ```python
    kernel = np.ones((5, 5), np.uint8)
    edge_map = cv2.morphologyEx(edge_map, cv2.MORPH_CLOSE, kernel, iterations=6)
  ```
  - extract the contour in `RETR_CCOMP` mode.
  ```python
    ret, thresh = cv2.threshold(edge_map, 80, 255, cv2.THRESH_BINARY)
    edge_contours, hierarchy = cv2.findContours(np.uint8(thresh), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
  ```
  - filter the edge contours based on 3 conditions: 
    1. bounding rectangle is nearly a square;
    2. contour area is larger than 1000 and smaller than 15000;
    3. only the first level contour(outer contour).
  ```python
    edge_filtered_contours = []
    for i, contour in enumerate(edge_contours):
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if w > 0 and 0.6 < h / w < 1.4 and 1000 < area < 15000 and hierarchy[0][i][3] == -1:
            edge_filtered_contours.append(contour)
  ```
  
  - fill all the filtered contours
  ```python
    edge_filled_mask = np.zeros_like(img_gray)
    cv2.fillPoly(edge_filled_mask, edge_filtered_contours, 255)
  ```

  
### **Pipeline3**: RG chromaticity based segmentation
  - convert image to rg space
  ```python
    img_r = rg_chromaticity(img, 2)
    img_g = rg_chromaticity(img, 1)
  ```
  - convert reference patch to rg space
  ```python
    patch = img[patch_coord[0]:patch_coord[1], patch_coord[2]:patch_coord[3]]
    patch_r = rg_chromaticity(patch, 2)
    patch_g = rg_chromaticity(patch, 1)
  ```
  
  - calculate the statistic stds and means for the reference patch
  ```python
    mean_r = np.mean(patch_r.flatten())
    mean_g = np.mean(patch_g.flatten())
    std_r = np.std(patch_r.flatten())
    std_g = np.std(patch_g.flatten())
  ```
  
  - generate mask using gaussian distribution
  ```python
    masked_img_r = gaussian(img_r, mean_r, std_r)
    masked_img_g = gaussian(img_g, mean_g, std_g)
    final_mask = masked_img_r * masked_img_g
  ```
  
  - binarize the mask
  ```python
    binary_mask = np.uint8(final_mask > final_mask.mean())
  ```

  - apply close operation to denoise
  ```python
    kernel = np.ones((5, 5), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask * 255.0, cv2.MORPH_CLOSE, kernel)
  ```

  - use filtered contours to further denoise
  ```python
    contours, hierarchy = cv2.findContours(np.uint8(binary_mask), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = []
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if (w > 30 or h > 30) and 200 < area < 15000 and hierarchy[0][i][3] == -1:
            filtered_contours.append(contour)
  ```

  - fill the filtered contours
  ```python
    filled_mask = np.zeros_like(img_r)
    cv2.fillPoly(filled_mask, filtered_contours, 255)
  ```

  

  
## Challenges
1. Similar color - Thresholding based method is hard to separate leaves and handwriting characters when they have similar color.
   - solution: make some assumptions on the shape of the leaves. Most of the leaves have bounding rectangle that is nearly
   to a square and larger areas. After applying these filtering rules, some handwriting characters can be filtered.
2. Leaf variability - Differences in leaf shape, size, color in this experiment scenario make it hard to have a universal approach.
   - solution: use adaptive thresholding method to extract all foreground areas first, then use carefully tuned parameters and more strict 
   rules to further extract the leaves areas.
3. Advanced edge detection techniques, like Canny, incline to be affected by the subtle textures, like veins in the leaves.
   - solution: use dilation operation to expand the contour edges of the leaves so that most of the outer edges can form
   closed areas. Then apply `cv2.find_contours` techniques to further extract the outer contours of the leaves.

  
  

  
