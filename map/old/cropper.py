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

# --- EXECUTION PIPELINE ---

# Define the coordinates of your 4 arbitrary corners on 'source.jpg'
# Format: [ [x1, y1], [x2, y2], [x3, y3], [x4, y4] ]


pagemap_img = cv2.imread("pagemap.png")
pagemap_img = cv2.cvtColor(pagemap_img, cv2.COLOR_BGR2BGRA)
pagemap_region_rows = np.array([2,159,316,470,626,783])
pagemap_region_cols = np.array([2,300,593,891])
pagemap_coords = np.dstack(np.meshgrid(pagemap_region_rows, pagemap_region_cols, indexing='ij'))
# print("pagemap_coords\n", pagemap_coords)
print(f"Goal: {pagemap_img.shape}")

basemap_img = cv2.imread("basemap.png")
basemap_img = cv2.cvtColor(basemap_img, cv2.COLOR_BGR2BGRA)
# basemap_region_rows = np.array([0,214,434,647,868])
# basemap_region_cols = np.array([0,393,796,1200])
# manual align
basemap_region_rows = np.array([0,214,434,647,868])
basemap_region_cols = np.array([0,393,796,1200])
basemap_coords = np.dstack(np.meshgrid(basemap_region_rows, basemap_region_cols, indexing='ij'))
# basemap_coords[0:2,1,1] = 385
# basemap_coords = np.array([[(0,0),(0,384),(0,795),(0,1199)],
#                          [(214,0),(,),(,),(,1199)],
#                          [(,0),(,),(,),(,1199)],
#                          [(,0),(,),(,),(,1199)],
#                          [(,0),(,),(,),(,1199)],
#                          [(867,0),(867,),(867,),(867,1199)]])
# print("basemap coords\n", basemap_coords)
print(f"From: {basemap_img.shape}")

# Step 1 & 2: Crop with arbitrary corners & resize to 400x500
chunks = []
for chunk_r in range(5):
    chunks.append([])
    chunk_row = chunks[-1]
    for chunk_c in range(3):
        target_width = pagemap_region_cols[chunk_c+1]-pagemap_region_cols[chunk_c]
        target_height = pagemap_region_rows[chunk_r+1]-pagemap_region_rows[chunk_r]
        if chunk_r < len(basemap_coords)-1:
            corners = basemap_coords[chunk_r:chunk_r+2,chunk_c:chunk_c+2]
            # print(f"chunk[{chunk_r},{chunk_c}]:\n", corners)
            corners = np.copy(corners)
            if chunk_r == 0 and chunk_c == 1:
                corners[:,0,1] = 385
            corners[1,:,0] -= 1  # bottom row's row value is exclusive
            corners[:,1,1] -= 1  # right col's col value is exclusive
            corners = [corners[0,0,::-1],corners[0,1,::-1],corners[1,1,::-1],corners[1,0,::-1]]

            chunk = crop_arbitrary_quad(
                basemap_img,
                corners,
                target_width,
                target_height
            )
        else:
            chunk = np.zeros((target_height, target_width, 4), dtype=np.uint8)
        
        chunk = cv2.cvtColor(chunk, cv2.COLOR_BGR2BGRA)
        cv2.imwrite(f"pages/base_chunk_{chunk_r}_{chunk_c}.png", chunk)
        chunk_row.append(chunk)

for blank_r, blank_c in [(0,0),(0,2),(1,2)]:
    # chunks[blank_r][blank_c] = cv2.cvtColor(chunks[blank_r][blank_c], cv2.COLOR_BGR2BGRA)
    chunks[blank_r][blank_c][:,:,:] = 0


# Step 3: Stitch them side-by-side (Horizontal concatenation)
# Use cv2.vconcat() instead if you want to stitch them top-to-bottom
stitched_result = cv2.vconcat([
    cv2.hconcat(chunk_row)
    for chunk_row in chunks
])
print(f"Got: {stitched_result.shape}")

# Save the final image
cv2.imwrite("basemap_warped.png", stitched_result)

#  [(0,0),(stitched_result.shape[1],0),(stitched_result.shape[1],stitched_result.shape[0]),(0,stitched_result.shape[0])], pagemap_img.width, pagemap_img.height)
# scaled = cv2.resize(stitched_result, (pagemap_img.shape[1], pagemap_img.shape[0]))
# cv2.imwrite("basemap_warped_and_scaled.png", scaled)
# print(f"Cheated to: {scaled.shape}")
padded = np.zeros(pagemap_img.shape, dtype=np.uint8)
padded[pagemap_region_rows[0]:pagemap_region_rows[-1],pagemap_region_cols[0]:pagemap_region_cols[-1],:] = stitched_result
cv2.imwrite("basemap_warped_and_padded.png", padded)
print(f"Cheated to: {padded.shape}")


# 3. Define the blending ratio (alpha for img1, beta for img2)
# The sum of alpha and beta usually equals 1.0 to preserve brightness
alpha = 0.3  # 70% of Image 1
beta = 0.7   # 30% of Image 2
gamma = 0    # Scalar added to each sum (brightness offset)

# 4. Blend the images
overlay = cv2.addWeighted(pagemap_img, alpha, padded, beta, gamma)
cv2.imwrite("overlay.png", overlay)

