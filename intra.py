import numpy as np
import cv2

# --- Các hàm chuyển đổi màu ---
def rgb_to_yuv(rgb_image):
    rgb_image = rgb_image.astype(np.float64)
    yuv_image = np.zeros_like(rgb_image)
    yuv_image[:,:,0] = 0.299*rgb_image[:,:,0] + 0.587*rgb_image[:,:,1] + 0.114*rgb_image[:,:,2]
    yuv_image[:,:,1] = -0.147*rgb_image[:,:,0] - 0.289*rgb_image[:,:,1] + 0.436*rgb_image[:,:,2]
    yuv_image[:,:,2] = 0.615*rgb_image[:,:,0] - 0.515*rgb_image[:,:,1] - 0.100*rgb_image[:,:,2]
    yuv_image[:,:,1:] += 128
    return yuv_image.astype(np.uint8)

def yuv_to_rgb(yuv_image):
    yuv_image = yuv_image.astype(np.float64)
    yuv_image[:,:,1:] -= 128
    rgb_image = np.zeros_like(yuv_image)
    rgb_image[:,:,0] = yuv_image[:,:,0] + 1.13983*yuv_image[:,:,2]
    rgb_image[:,:,1] = yuv_image[:,:,0] - 0.39465*yuv_image[:,:,1] - 0.58060*yuv_image[:,:,2]
    rgb_image[:,:,2] = yuv_image[:,:,0] + 2.03211*yuv_image[:,:,1]
    return np.clip(rgb_image, 0, 255).astype(np.uint8)

def yuv_to_yuv422(yuv_image):
    y_channel = yuv_image[:,:,0]
    return y_channel


# --- Hàm tính SAD (Sum of Absolute Differences)---
def sad(original, predicted):
    return np.sum(np.abs(original.astype(np.int16) - predicted.astype(np.int16)))


# --- Các hàm Intra Prediction (9 modes) ---
def intra_prediction(block, predictor, mode):
    predicted_block = np.zeros_like(block, dtype=np.uint8)
    if mode == 0:  # Vertical
        predicted_block[:] = predictor[1:5]
    elif mode == 1:  # Horizontal
        for i in range(4):
          predicted_block[i,:] = predictor[9+i]
    elif mode == 2:  # DC
        s1 = np.sum(predictor[1:5])
        s2 = np.sum(predictor[9:13])
        if (np.any(predictor[1:5] != 128) and np.any(predictor[9:13] != 128)):
          s = s1 + s2
        elif (np.all(predictor[1:5] == 128) and np.any(predictor[9:13] != 128)):
          s = 2 * s2
        elif (np.any(predictor[1:5] != 128) and np.all(predictor[9:13] == 128)):
          s = 2 * s1
        else:
          s = 128 * 8
        s = (s+4) >> 3
        predicted_block.fill(s)
    elif mode == 3:  # Diagonal Down-Left
        predicted_block[0]  = ((predictor[1] + predictor[3] + (predictor[2] << 1) + 2) >> 2)
        predicted_block[1]  = ((predictor[2] + predictor[4] + (predictor[3] << 1) + 2) >> 2)
        predicted_block[4]  = ((predictor[2] + predictor[4] + (predictor[3] << 1) + 2) >> 2)
        predicted_block[2]  = ((predictor[3] + predictor[5] + (predictor[4] << 1) + 2) >> 2)
        predicted_block[5]  = ((predictor[3] + predictor[5] + (predictor[4] << 1) + 2) >> 2)
        predicted_block[8]  = ((predictor[3] + predictor[5] + (predictor[4] << 1) + 2) >> 2)
        predicted_block[3]  = ((predictor[4] + predictor[6] + (predictor[5] << 1) + 2) >> 2)
        predicted_block[6]  = ((predictor[4] + predictor[6] + (predictor[5] << 1) + 2) >> 2)
        predicted_block[9]  = ((predictor[4] + predictor[6] + (predictor[5] << 1) + 2) >> 2)
        predicted_block[12] = ((predictor[4] + predictor[6] + (predictor[5] << 1) + 2) >> 2)
        predicted_block[7]  = ((predictor[5] + predictor[7] + (predictor[6] << 1) + 2) >> 2)
        predicted_block[10] = ((predictor[5] + predictor[7] + (predictor[6] << 1) + 2) >> 2)
        predicted_block[13] = ((predictor[5] + predictor[7] + (predictor[6] << 1) + 2) >> 2)
        predicted_block[11] = ((predictor[6] + predictor[8] + (predictor[7] << 1) + 2) >> 2)
        predicted_block[14] = ((predictor[6] + predictor[8] + (predictor[7] << 1) + 2) >> 2)
        predicted_block[15] = ((predictor[7] + 3 * predictor[8] + 2) >> 2)
    elif mode == 4:  # Diagonal Down-Right
        predicted_block[12] = ((predictor[12] + predictor[10] + (predictor[11] << 1) + 2) >> 2)
        predicted_block[8]  = ((predictor[11] + predictor[9] + (predictor[10] << 1) + 2) >> 2)
        predicted_block[13] = ((predictor[11] + predictor[9] + (predictor[10] << 1) + 2) >> 2)
        predicted_block[4]  = ((predictor[10] + predictor[1] + (predictor[9] << 1) + 2) >> 2)
        predicted_block[9]  = ((predictor[10] + predictor[1] + (predictor[9] << 1) + 2) >> 2)
        predicted_block[14] = ((predictor[10] + predictor[1] + (predictor[9] << 1) + 2) >> 2)
        predicted_block[0]  = ((predictor[0] + predictor[9] + (predictor[1] << 1) + 2) >> 2)
        predicted_block[5]  = ((predictor[0] + predictor[9] + (predictor[1] << 1) + 2) >> 2)
        predicted_block[10] = ((predictor[0] + predictor[9] + (predictor[1] << 1) + 2) >> 2)
        predicted_block[15] = ((predictor[0] + predictor[9] + (predictor[1] << 1) + 2) >> 2)
        predicted_block[1]  = ((predictor[0] + predictor[2] + (predictor[1] << 1) + 2) >> 2)
        predicted_block[6]  = ((predictor[0] + predictor[2] + (predictor[1] << 1) + 2) >> 2)
        predicted_block[11] = ((predictor[0] + predictor[2] + (predictor[1] << 1) + 2) >> 2)
        predicted_block[2]  = ((predictor[1] + predictor[3] + (predictor[2] << 1) + 2) >> 2)
        predicted_block[7]  = ((predictor[1] + predictor[3] + (predictor[2] << 1) + 2) >> 2)
        predicted_block[3]  = ((predictor[1] + predictor[3] + (predictor[2] << 1) + 2) >> 2)
    elif mode == 5:  # Vertical-Right
        predicted_block[0]  = ((predictor[0] + predictor[1] + 1) >> 1)
        predicted_block[9]  = ((predictor[0] + predictor[1] + 1) >> 1)
        predicted_block[1]  = ((predictor[1] + predictor[2] + 1) >> 1)
        predicted_block[10] = ((predictor[1] + predictor[2] + 1) >> 1)
        predicted_block[2]  = ((predictor[2] + predictor[3] + 1) >> 1)
        predicted_block[11] = ((predictor[2] + predictor[3] + 1) >> 1)
        predicted_block[3]  = ((predictor[3] + predictor[4] + 1) >> 1)
        predicted_block[4]  = ((predictor[1] + predictor[9] + (predictor[0] << 1) + 2) >> 2)
        predicted_block[13] = ((predictor[1] + predictor[9] + (predictor[0] << 1) + 2) >> 2)
        predicted_block[5]  = ((predictor[0] + predictor[2] + (predictor[1] << 1) + 2) >> 2)
        predicted_block[14] = ((predictor[0] + predictor[2] + (predictor[1] << 1) + 2) >> 2)
        predicted_block[6]  = ((predictor[1] + predictor[3] + (predictor[2] << 1) + 2) >> 2)
        predicted_block[15] = ((predictor[1] + predictor[3] + (predictor[2] << 1) + 2) >> 2)
        predicted_block[7]  = ((predictor[2] + predictor[4] + (predictor[3] << 1) + 2) >> 2)
        predicted_block[8]  = ((predictor[0] + predictor[10] + (predictor[9] << 1) + 2) >> 2)
        predicted_block[12] = ((predictor[9] + predictor[11] + (predictor[10] << 1) + 2) >> 2)
    elif mode == 6:  # Horizontal-Down
        predicted_block[0]  = ((predictor[0] + predictor[9] + 1) >> 1)
        predicted_block[6]  = ((predictor[0] + predictor[9] + 1) >> 1)
        predicted_block[1]  = ((predictor[1] + predictor[9] + (predictor[0] << 1) + 2) >> 2)
        predicted_block[7]  = ((predictor[1] + predictor[9] + (predictor[0] << 1) + 2) >> 2)
        predicted_block[2]  = ((predictor[0] + predictor[2] + (predictor[1] << 1) + 2) >> 2)
        predicted_block[3]  = ((predictor[1] + predictor[3] + (predictor[2] << 1) + 2) >> 2)
        predicted_block[4]  = ((predictor[9] + predictor[10] + 1) >> 1)
        predicted_block[10] = ((predictor[9] + predictor[10] + 1) >> 1)
        predicted_block[5]  = ((predictor[0] + predictor[11] + (predictor[10] << 1) + 2) >> 2)
        predicted_block[11] = ((predictor[0] + predictor[11] + (predictor[10] << 1) + 2) >> 2)
        predicted_block[8]  = ((predictor[10] + predictor[11] + 1) >> 1)
        predicted_block[14] = ((predictor[10] + predictor[11] + 1) >> 1)
        predicted_block[9]  = ((predictor[9] + predictor[11] + (predictor[10] << 1) + 2) >> 2)
        predicted_block[15] = ((predictor[9] + predictor[11] + (predictor[10] << 1) + 2) >> 2)
        predicted_block[12] = ((predictor[11] + predictor[12] + 1) >> 1)
        predicted_block[13] = ((predictor[10] + predictor[12] + (predictor[11] << 1) + 2) >> 2)
    elif mode == 7:  # Vertical-Left
        predicted_block[0]  = ((predictor[1] + predictor[2] + 1) >> 1)
        predicted_block[1]  = ((predictor[2] + predictor[3] + 1) >> 1)
        predicted_block[8]  = ((predictor[2] + predictor[3] + 1) >> 1)
        predicted_block[2]  = ((predictor[3] + predictor[4] + 1) >> 1)
        predicted_block[9]  = ((predictor[3] + predictor[4] + 1) >> 1)
        predicted_block[3]  = ((predictor[4] + predictor[5] + 1) >> 1)
        predicted_block[10] = ((predictor[4] + predictor[5] + 1) >> 1)
        predicted_block[11] = ((predictor[5] + predictor[6] + 1) >> 1)
        predicted_block[4]  = ((predictor[1] + predictor[3] + (predictor[2] << 1) + 2) >> 2)
        predicted_block[5]  = ((predictor[2] + predictor[4] + (predictor[3] << 1) + 2) >> 2)
        predicted_block[12] = ((predictor[2] + predictor[4] + (predictor[3] << 1) + 2) >> 2)
        predicted_block[6]  = ((predictor[3] + predictor[5] + (predictor[4] << 1) + 2) >> 2)
        predicted_block[13] = ((predictor[3] + predictor[5] + (predictor[4] << 1) + 2) >> 2)
        predicted_block[7]  = ((predictor[4] + predictor[6] + (predictor[5] << 1) + 2) >> 2)
        predicted_block[14] = ((predictor[4] + predictor[6] + (predictor[5] << 1) + 2) >> 2)
        predicted_block[15] = ((predictor[5] + predictor[7] + (predictor[6] << 1) + 2) >> 2)
    elif mode == 8:  # Horizontal-Up
        predicted_block[0]  = ((predictor[9] + predictor[10] + 1) >> 1)
        predicted_block[1]  = ((predictor[9] + predictor[11] + (predictor[10] << 1) + 2) >> 2)
        predicted_block[2]  = ((predictor[10] + predictor[11] + 1) >> 1)
        predicted_block[3]  = ((predictor[10] + predictor[12] + (predictor[11] << 1) + 2) >> 2)
        predicted_block[4] =  predicted_block[2]
        predicted_block[5] = predicted_block[3]
        predicted_block[6] = ((predictor[11] + predictor[12] + 1) >> 1)
        predicted_block[7]  = ((predictor[11] + (3 * predictor[12]) + 2) >> 2)
        predicted_block[8] =  predicted_block[6]
        predicted_block[9] =  predicted_block[7]
        predicted_block[10] = predictor[12]
        predicted_block[11] = predictor[12]
        predicted_block[12] = predictor[12]
        predicted_block[13] = predictor[12]
        predicted_block[14] = predictor[12]
        predicted_block[15] = predictor[12]

    return predicted_block
# --- Hàm dự đoán intra 4x4 ---
def intra4x4_prediction(block, ul, u, ur, l, block_size=4):
    min_sad = float('inf')
    best_mode = -1
    predicted_block = np.zeros_like(block, dtype=np.uint8)

    # Create predictor array
    predictor = np.zeros(17,dtype=np.uint8)

    # Check whether neighbors are available
    if u is not None:
        predictor[1:5] = u[-1,:]
        up_available = True
    else:
      predictor[1:5] = 128
      up_available = False

    if ur is not None:
      predictor[5:9] = ur[-1,:]
      up_right_available = True
    else:
        predictor[5:9] = 128
        up_right_available = False

    if l is not None:
      predictor[9:13] = l[:,-1]
      left_available = True
    else:
      predictor[9:13] = 128
      left_available = False


    if up_available and left_available and ul is not None:
      predictor[0] = ul[-1,-1]
      all_available = True
    else:
       predictor[0] = 128
       all_available = False


    for mode in range(9):
        if ((not up_available   and (mode == 0)) or
            (not left_available and (mode == 1)) or
            ((not up_available or not up_right_available) and (mode == 3)) or
            ((not up_available or not left_available) and (mode == 4)) or
            ((not up_available or not left_available) and (mode == 5)) or
            ((not up_available or not left_available) and (mode == 6)) or
            ((not up_available or not up_right_available) and (mode == 7)) or
            (not left_available and (mode == 8))):
            continue

        current_predicted_block = intra_prediction(block,predictor,mode)
        current_sad = sad(block, current_predicted_block)

        if current_sad < min_sad:
           min_sad = current_sad
           best_mode = mode
           predicted_block = current_predicted_block


    return min_sad, best_mode, predicted_block


# --- Main function ---
def process_image_with_intra4x4(frame):
    yuv_image = rgb_to_yuv(frame)
    y_channel = yuv_to_yuv422(yuv_image)
    BLOCK_SIZE = 4
    height, width = y_channel.shape
    best_mode_map = np.zeros((height // BLOCK_SIZE, width // BLOCK_SIZE), dtype=np.uint8)
    predicted_y_image = np.zeros_like(y_channel)

    for i in range(0, height, BLOCK_SIZE):
        for j in range(0, width, BLOCK_SIZE):
            block = y_channel[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]

            # Determine neighbors
            ul_block = y_channel[i-BLOCK_SIZE:i,j-BLOCK_SIZE:j] if i >= BLOCK_SIZE and j >= BLOCK_SIZE else None
            u_block = y_channel[i-BLOCK_SIZE:i, j:j+BLOCK_SIZE] if i >= BLOCK_SIZE else None
            ur_block = y_channel[i-BLOCK_SIZE:i, j+BLOCK_SIZE:j+2*BLOCK_SIZE] if i >= BLOCK_SIZE and j < width - BLOCK_SIZE else None
            l_block = y_channel[i:i+BLOCK_SIZE, j-BLOCK_SIZE:j] if j >= BLOCK_SIZE else None

            min_sad, best_mode, predicted_block = intra4x4_prediction(block, ul_block, u_block, ur_block, l_block)

            best_mode_map[i // BLOCK_SIZE, j // BLOCK_SIZE] = best_mode
            predicted_y_image[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE] = predicted_block

    yuv_image_predicted = np.zeros_like(yuv_image)
    yuv_image_predicted[:,:,0] = predicted_y_image
    yuv_image_predicted[:,:,1:] = yuv_image[:,:,1:]

    rgb_image_predicted = yuv_to_rgb(yuv_image_predicted)
    return frame, predicted_y_image, rgb_image_predicted, best_mode_map


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    ret, frame = cap.read()
    cap.release()

    if not ret:
      raise Exception("Failed to capture frame")

    frame, predicted_y_image, rgb_image_predicted, best_mode_map = process_image_with_intra4x4(frame)

    cv2.imshow("Original", frame)
    cv2.imshow("Y Prediction", predicted_y_image.astype(np.uint8))
    cv2.imshow("Predicted Image", rgb_image_predicted)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("Best Mode Map:\n", best_mode_map)