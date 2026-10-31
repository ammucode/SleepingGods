import cv2
import numpy as np

def crop_arbitrary_quad(img, corners, target_width, target_height):
    """
    Crops an arbitrary 4-corner region and straightens/resizes it.
    """
    
    # 1. Define your arbitrary source corners (Top-Left, Top-Right, Bottom-Right, Bottom-Left)
    src_pts = np.array(corners, dtype="float32")
    
    # 2. Define the destination corners (the clean rectangular output shape)
    dst_pts = np.array([
        [0, 0],
        [target_width - 1, 0],
        [target_width - 1, target_height - 1],
        [0, target_height - 1]
    ], dtype="float32")
    
    # 3. Calculate the Transformation Matrix & Warp Perspective (Crop + Resize)
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_img = cv2.warpPerspective(img, matrix, (target_width, target_height))

    return warped_img

def make_glowy_g(img):
    img = np.copy(img)
    # 1. Get grayscale brightness (0 to 255)
    grayscale = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)

    # 2. Make black areas thicker (Erosion shrinks white, expanding the black areas)
    kernel = np.ones((9,9), np.uint8)  # Increase size (e.g., 7x7) to make black even thicker
    thicker_black = cv2.erode(grayscale, kernel, iterations=1)

    # 3. Blur the result to get smooth transitions
    blurred = cv2.GaussianBlur(thicker_black, (15, 15), 0)

    # 4. Convert to float and normalize to 0.0 - 1.0
    normalized = blurred.astype(np.float32) / 255.0

    # 5. Apply your power curve (e.g., power of 4)
    curve = np.power(normalized, 4.0)

    # 6. Invert and scale back to 0-255 uint8
    new_alpha = (1.0 - curve) * 255
    img[:, :, 3] = new_alpha.astype(np.uint8)
    return img

def make_glowy_g2(img):
    img = np.copy(img)
    # 2. Convert color channels to grayscale (0=black, 255=white)
    grayscale = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)

    # 3. Create the base Alpha map using the power curve
    # We work on raw pixels first: White becomes 0 Alpha, Black becomes 255 Alpha
    normalized = grayscale.astype(np.float32) / 255.0
    curve = np.power(normalized, 4.0)
    alpha_base = ((1.0 - curve) * 255).astype(np.uint8)

    # 4. THICKEN THE BLACK LINES (Dilate the Alpha channel)
    # Since the black lines are now represented by high alpha values (255), 
    # cv2.dilate will physically expand the opacity outward and fill inner spaces.
    kernel_size = 7  # Increase this (e.g., 11, 15) to make lines thicker/fill larger gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    thick_alpha = cv2.dilate(alpha_base, kernel, iterations=1)

    # 5. BLUR THE ALPHA ONLY (For smooth transitions without color bleeding)
    # This creates a smooth fade into transparency without creating a white halo.
    blur_size = 15  # Must be an odd number. Controls how soft the edge fade is.
    smooth_alpha = cv2.GaussianBlur(thick_alpha, (blur_size, blur_size), 0)

    # 6. Assign the processed alpha mask back to the image
    img[:, :, 3] = smooth_alpha
    return img

def make_contrast(img):
    img = np.copy(img)
    
    # 2. Get grayscale and threshold to isolate just the black lines (0 = black, 255 = white)
    grayscale = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    _, binary_lines = cv2.threshold(grayscale, 180, 255, cv2.THRESH_BINARY)

    # 3. Calculate distance from every pixel to the nearest black line
    # For binary_lines, 0 (black) represents the lines, so it calculates distance moving outward
    distance = cv2.distanceTransform(binary_lines, cv2.DIST_L2, 5)

    # 4. Define line thickness and falloff in exact pixels
    line_thickness = 2.0  # How many pixels to extend the solid black line outward
    blur_radius = 1.0     # How many pixels wide the smooth alpha fade should be

    # 5. Create the Alpha mask based on distance
    # Pixels within line_thickness get 255 alpha (solid).
    # Pixels further away fade out linearly based on the blur_radius.
    alpha_float = 255.0 - ((distance - line_thickness) / blur_radius * 255.0)
    alpha_mask = np.clip(alpha_float, 0, 255).astype(np.uint8)

    # 6. FORCE COLOR CHANNELS TO BLACK WHERE THE MASK IS ACTIVE
    # This completely destroys the white background color data under the alpha mask,
    # removing the white glow completely.
    img[alpha_mask > 0, 0:3] = 255
    img[alpha_mask > 50, 0] = alpha_mask[alpha_mask>50]
    img[alpha_mask > 50, 1] = alpha_mask[alpha_mask>50]
    img[alpha_mask > 50, 2] = alpha_mask[alpha_mask>50]

    # 7. Apply the clean alpha mask
    img[:, :, 3] = alpha_mask
    return img

def make_glowy(img,dmin=0.5,dmax=4):
    # 1. Get grayscale brightness (0 to 255)
    grayscale = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(grayscale, 190, 255, cv2.THRESH_BINARY)

    distance = cv2.distanceTransform(bw, cv2.DIST_L2, 5)

    alpha = np.interp(distance, [dmin,dmax], [255,150], left=255, right=0)
    color = np.interp(distance, [dmin, (dmin+dmax)/2], [0,210])
    weight = np.interp(distance, [dmin, (dmin+dmax)/2, dmax], [0.5, 0.4, 0.2])
    full_mask = distance<dmax

    img_new = np.stack((color,color,color,alpha), axis=2)
    img_new[bw == 0,0:3] = img[bw == 0,0:3]
    for i in range(3):
        img_new[full_mask,i] = img[full_mask,i]*(1-weight[full_mask])+color[full_mask]*weight[full_mask]
    for i in range(3):
        img_new[grayscale < color,i] = img[grayscale<color,i]


    # return np.stack((color,color,color,alpha), axis=2)
    return img_new 


def save_alpha(name, alpha):
    alpha_pic = np.stack((alpha*255,alpha*255,alpha*255,np.ones(alpha.shape)*255), axis=2)
    # print(alpha_pic.shape)
    cv2.imwrite(name, alpha_pic[:,:,:,0])

# --- EXECUTION PIPELINE ---

# Define the coordinates of your 4 arbitrary corners on 'source.jpg'
# Format: [ [x1, y1], [x2, y2], [x3, y3], [x4, y4] ]


pagemap_img = cv2.imread("src/pagemap_large.jpg")
pagemap_img = cv2.cvtColor(pagemap_img, cv2.COLOR_BGR2BGRA)
# border centers
# pagemap_region_rows = np.array([2,159,316,470,626,783])
# pagemap_region_cols = np.array([2,300,593,891])
# border edges
pagemap_region_rows = np.array([0,3,158,160,315,317,469,471,625,627,783,786])*4
pagemap_region_cols = np.array([0,4,299,302,592,594,890,894])*4
pagemap_coords = np.dstack(np.meshgrid(pagemap_region_rows, pagemap_region_cols, indexing='ij'))
# print("pagemap_coords\n", pagemap_coords)
# print(f"Goal: {pagemap_img.shape}")

fullmap_img = cv2.imread("src/fullmap.jpg")
fullmap_img = cv2.cvtColor(fullmap_img, cv2.COLOR_BGR2BGRA)
fullmap_region_rows = np.array([0,1765,3513,5200,7028])
fullmap_region_cols = np.array([0,3307,6609,9900])
# fullmap_coords = np.dstack(np.meshgrid(fullmap_region_rows, fullmap_region_cols, indexing='ij'))
fullmap_coords = np.zeros((4,3,2,2))  # per chunk, corners
fullmap_coords[0,0] = np.array([(0,3301),(1769,3291)])
fullmap_coords[0,1] = np.array([(50,3330),(1800,6550)])
fullmap_coords[0,2] = np.array([(0,6609),(1765,9900)])
fullmap_coords[1,0] = np.array([(1765,80),(3513,3440)])
fullmap_coords[1,1] = np.array([(1765,3270),(3513,6540)])
fullmap_coords[1,2] = np.array([(1765,6609),(3513,9900)])
fullmap_coords[2,0] = np.array([(3513,30),(5180,3250)])
fullmap_coords[2,1] = np.array([(3520,3250),(5250,6580)])
fullmap_coords[2,2] = np.array([(3500,6550),(5300,9900)])
fullmap_coords[3,0] = np.array([(5150,80),(6850,3350)])
fullmap_coords[3,1] = np.array([(5200,3230),(7028,6650)])
fullmap_coords[3,2] = np.array([(5200,6570),(6980,9900)])
# print("fullmap coords\n", fullmap_coords)
# print(f"From: {fullmap_img.shape}")

basemap_img = cv2.imread("src/basemap.jpg")
basemap_img = cv2.cvtColor(basemap_img, cv2.COLOR_BGR2BGRA)
basemap_region_rows = np.array([0,1765,3513,5200,7028])
basemap_region_cols = np.array([0,3307,6609,9900])
# basemap_coords = np.dstack(np.meshgrid(basemap_region_rows, basemap_region_cols, indexing='ij'))
basemap_coords = np.zeros((4,3,2,2))  # per chunk, corners
basemap_coords[0,0] = np.array([(0,0),(280,510)])
basemap_coords[0,1] = np.array([(5,480),(280,980)])
basemap_coords[0,2] = np.array([(0,985),(280,1500)])
basemap_coords[1,0] = np.array([(268,0),(540,505)])
basemap_coords[1,1] = np.array([(272,475),(540,990)])
basemap_coords[1,2] = np.array([(268,985),(540,1500)])
basemap_coords[2,0] = np.array([(535,0),(795,495)])
basemap_coords[2,1] = np.array([(540,475),(810,990)])
basemap_coords[2,2] = np.array([(547,975),(815,1500)])
basemap_coords[3,0] = np.array([(790,15),(1055,515)])
basemap_coords[3,1] = np.array([(810,475),(1085,1000)])
basemap_coords[3,2] = np.array([(805,985),(1080,1500)])
# print("basemap coords\n", basemap_coords)
# print(f"From: {basemap_img.shape}")

# colormap_img = fullmap_img
# colormap_region_rows = fullmap_region_rows
# colormap_region_cols = fullmap_region_cols
# colormap_coords = fullmap_coords
colormap_img = basemap_img
colormap_region_rows = basemap_region_rows
colormap_region_cols = basemap_region_cols
colormap_coords = basemap_coords

# Step 1 & 2: Crop with arbitrary corners & resize to 400x500
target = np.zeros(pagemap_img.shape, dtype=np.uint8)
blanks = [(0,0),(0,2),(1,2)]
for chunk_r, chunk_c in ((r,c) for r in range(5) for c in range(3)):
    dest_r = (pagemap_region_rows[chunk_r*2+1],pagemap_region_rows[chunk_r*2+2])
    dest_c = (pagemap_region_cols[chunk_c*2+1],pagemap_region_cols[chunk_c*2+2])
    # print(f"\n\n[{chunk_r}, {chunk_c}]:\n", np.dstack(np.meshgrid(dest_r, dest_c, indexing='ij')))
    target_width = dest_c[1]-dest_c[0]
    target_height = dest_r[1]-dest_r[0]
    if chunk_r < len(colormap_coords) and chunk_c < len(colormap_coords[chunk_r]):
        # corners = colormap_coords[chunk_r:chunk_r+2,chunk_c:chunk_c+2]
        # print(f"colormap_coords[{chunk_r},{chunk_c}] = np.array([({corners[0,0][0]},{corners[0,0][1]}),({corners[1,1][0]},{corners[1,1][1]})])")
        # corners = np.copy(corners)

        bounds = colormap_coords[chunk_r,chunk_c]
        corners = np.stack(np.meshgrid(bounds[:, 0], bounds[:, 1], indexing='ij'), axis=-1)

        # print(f"colormap_coords[{chunk_r},{chunk_c}]: {corners.shape}\n", corners)
        corners = [corners[0,0,::-1],corners[0,1,::-1],corners[1,1,::-1],corners[1,0,::-1]]

        chunk = crop_arbitrary_quad(
            colormap_img,
            corners,
            target_width,
            target_height
        )
        blank=False
    else:
        chunk = np.zeros((target_height, target_width, 4), dtype=np.uint8)
        blank=True

    chunk = cv2.cvtColor(chunk, cv2.COLOR_BGR2BGRA)
    # cv2.imwrite(f"debug/pages/color_chunk_{chunk_r}_{chunk_c}.png", chunk)

    if (chunk_r,chunk_c) in blanks or blank:
        chunk[:,:,:] = 0
        # chunk = crop_arbitrary_quad(
        #     pagemap_img,
        #     [(dest_c[0],dest_r[0]),(dest_c[1],dest_r[0]),(dest_c[1],dest_r[1]),(dest_c[0],dest_r[1])],
        #     target_width,
        #     target_height
        # )

    target[dest_r[0]:dest_r[1],dest_c[0]:dest_c[1],:] = chunk
cv2.imwrite("debug/target.png", target)
print(f"Got: {target.shape}")
target_alpha = (target[:, :, 3] / 255).reshape(target.shape[:2] + (1,))
save_alpha("debug/target_alpha.png", target_alpha)

base = make_glowy(pagemap_img)
cv2.imwrite("debug/base.png", base)
base_alpha = (base[:, :, 3] / 255).reshape(base.shape[:2] + (1,))
save_alpha("debug/base_alpha.png", base_alpha)

overlay = ((target[:,:,:3] * (1 - base_alpha)) + (base[:, :, :3] * base_alpha)).astype(np.uint8)
overlay = np.stack([overlay[:,:,0],overlay[:,:,1],overlay[:,:,2],np.ones(overlay[:,:,0].shape)*255],axis=2)
print(overlay.shape)
cv2.imwrite("debug/overlay.png", overlay)

bright = 0
final = ((pagemap_img[:,:,:3] * (1 - target_alpha)) + ((overlay[:, :, :3]+bright) * target_alpha)).astype(np.uint8)
final = np.stack([final[:,:,0],final[:,:,1],final[:,:,2],np.ones(final[:,:,0].shape)*255],axis=2)
print(final.shape)
cv2.imwrite("debug/final.png", final)
cv2.imwrite("map.png", final)


# # 4. Blend the images
# foo = final[:,:,:3]+20
# cv2.imwrite("foo.png", foo)

