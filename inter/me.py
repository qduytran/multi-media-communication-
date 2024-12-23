import cv2
import numpy as np
import av

def capture_video(output_path="input.mp4", duration=5):
    """Thu video từ webcam và lưu vào file MP4."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Không thể mở webcam")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    start_time = cv2.getTickCount()
    while (cv2.getTickCount() - start_time) / cv2.getTickFrequency() < duration:
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            cv2.imshow('Camera', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def convert_mp4_to_yuv_y(input_path, output_path="yuv_y_output.yuv"):
    """Chuyển đổi MP4 sang YUV và chỉ lưu lại kênh Y."""
    container = av.open(input_path)
    output_file = open(output_path, 'wb')
    first_frame = None
    for frame in container.decode(video=0):
        if frame.format == 'yuv420p':
            yuv_frame = frame.to_ndarray(format='yuv420p')
            if yuv_frame.ndim == 3:
              y_channel = yuv_frame[0, :, :]  # Kênh Y là kênh đầu tiên
            else:
                y_channel = yuv_frame
        else:
            yuv_frame = frame.to_ndarray(format='gray')
            y_channel = yuv_frame
        if first_frame is None:
            first_frame = y_channel
        y_channel.tofile(output_file)
    output_file.close()
    return first_frame


def block_matching(prev_frame, current_frame, block_size=8, search_range=16):
    """Thực hiện Motion Estimation bằng thuật toán Block Matching."""
    height, width = current_frame.shape
    motion_vectors = np.zeros((height // block_size, width // block_size, 2), dtype=int)

    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            best_match = (0, 0)
            min_sad = float('inf')

            for dy in range(-search_range, search_range + 1):
                for dx in range(-search_range, search_range + 1):
                    ny = y + dy
                    nx = x + dx

                    if 0 <= ny < height - block_size and 0 <= nx < width - block_size:
                        current_block = current_frame[y:y + block_size, x:x + block_size]
                        ref_block = prev_frame[ny:ny + block_size, nx:nx + block_size]
                        sad = np.sum(np.abs(current_block - ref_block))

                        if sad < min_sad:
                            min_sad = sad
                            best_match = (dx, dy)
            motion_vectors[y // block_size, x // block_size] = best_match

    return motion_vectors


def motion_compensation(prev_frame, motion_vectors, block_size=8):
    """Thực hiện Motion Compensation sử dụng MV đã tìm được."""
    height, width = prev_frame.shape
    compensated_frame = np.zeros_like(prev_frame, dtype=np.uint8)

    for y in range(0, height // block_size):
        for x in range(0, width // block_size):
            dx, dy = motion_vectors[y, x]
            ny = y * block_size + dy
            nx = x * block_size + dx

            if 0 <= ny < height - block_size and 0 <= nx < width - block_size:
                compensated_frame[y * block_size:(y + 1) * block_size, x * block_size:(x + 1) * block_size] = \
                    prev_frame[ny:ny + block_size, nx:nx + block_size]

    return compensated_frame


def visualize_motion_vectors(frame, motion_vectors, block_size=8, scale=10):
    """Trực quan hóa các Motion Vector trên frame."""
    height, width = frame.shape
    vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    for y in range(0, height // block_size):
        for x in range(0, width // block_size):
            dx, dy = motion_vectors[y, x]
            start_x = x * block_size + block_size // 2
            start_y = y * block_size + block_size // 2
            end_x = start_x + dx * scale
            end_y = start_y + dy * scale
            cv2.arrowedLine(vis, (start_x, start_y), (end_x, end_y), (0, 0, 255), 1)

    return vis

def reconstruct_video(input_yuv_path, first_frame, output_path="reconstructed.mp4", output_mv_path="motion_vectors.mp4"):
    """Khôi phục video từ chuỗi Y và Motion Vector."""
    with open(input_yuv_path, 'rb') as f:
        y_bytes = f.read()

    height, width = first_frame.shape

    frame_size = height * width

    frame_list = []

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))
    out_mv = cv2.VideoWriter(output_mv_path, fourcc, 20.0, (width, height)) # Writer for MV video
    
    for i in range(0, len(y_bytes), frame_size):
        frame_bytes = y_bytes[i:i+frame_size]
        if len(frame_bytes) == frame_size:
          gray_frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(height, width)

          if not frame_list:
              first_frame = gray_frame
              frame_list.append(first_frame)

              bgr_frame = cv2.cvtColor(first_frame, cv2.COLOR_GRAY2BGR)
              out.write(bgr_frame)
              out_mv.write(bgr_frame)

              continue
          
          current_frame = gray_frame

          if frame_list:
              prev_frame = frame_list[-1]
              motion_vectors = block_matching(prev_frame, current_frame)
              compensated_frame = motion_compensation(prev_frame, motion_vectors)
              frame_list.append(compensated_frame)
              
              vis_mv = visualize_motion_vectors(current_frame, motion_vectors)
              out_mv.write(vis_mv)

              bgr_frame = cv2.cvtColor(compensated_frame, cv2.COLOR_GRAY2BGR)
              out.write(bgr_frame)


    
    out.release()
    out_mv.release()
    
# --- Main ---
if __name__ == "__main__":
    # 1. Thu video từ camera
    # capture_video()

    # 2. Chuyển đổi MP4 sang YUV và tách kênh Y
    first_frame = convert_mp4_to_yuv_y("input.mp4")
    
    # 6. Reconstruct video
    reconstruct_video("yuv_y_output.yuv", first_frame)

    # # Ví dụ cho quá trình xử lý frame (để hiểu về flow hơn), ví dụ này chỉ xử lý thuật toán với 2 frame đầu tiên
    # with open("yuv_y_output.yuv", 'rb') as f:
    #     y_bytes = f.read()
    
    # container = av.open("input.mp4")
    # first_frame_from_mp4 = next(container.decode(video=0)).to_ndarray(format="gray")
    # height, width = first_frame_from_mp4.shape
    
    # frame_size = height * width
    
    # try:
       
    #   frame_bytes = y_bytes[0:frame_size]
    #   first_frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(height, width)

    #   frame_bytes = y_bytes[frame_size:frame_size*2]
    #   second_frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(height, width)


    #   # 3. Motion Estimation
    #   motion_vectors = block_matching(first_frame, second_frame)

    #   # 4. Motion Compensation
    #   compensated_frame = motion_compensation(first_frame, motion_vectors)
    #   cv2.imshow("Compensated Frame", compensated_frame)

    #   # 5. Hiển thị MV
    #   vis_mv = visualize_motion_vectors(second_frame, motion_vectors)
    #   cv2.imshow("Motion Vectors", vis_mv)
    #   cv2.waitKey(0)
    #   cv2.destroyAllWindows()
    # except StopIteration:
    #     print("Not enough frame")