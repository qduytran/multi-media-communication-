%đọc ảnh màu rgb
image = imread("C:\Users\Hi\Downloads\meme.jpg");

%chuyển sang ảnh xám
image_gray = rgb2gray(image);

laplacian_mask = [-1 -1 -1;
                 -1  8 -1;
                  -1 -1 -1];

laplacian_result = conv2(double(image_gray), laplacian_mask, 'same');

figure;
subplot(1, 2, 1);
imshow(image_gray);
title('Original Image');

subplot(1, 2, 2);
imshow(uint8(laplacian_result));  % Convert back to uint8 for display
title('Laplacian Filter Result');