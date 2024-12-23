%đọc ảnh màu rgb
image = imread("C:\Users\Hi\OneDrive - vnu.edu.vn\Pictures\Screenshots\Ảnh chụp màn hình 2023-05-04 225401.png");

%chuyển sang ảnh xám
image_gray = rgb2gray(image);

%biến đổi fourier ảnh xám 2-D Fast Fourier Transform 
F = fft2(image_gray);

%dịch thành phần tần số 0 về trung tâm của phổ tần số
F_shifted = fftshift(F)

%tính biên độ phổ tần số
F_magnitude = abs(F_shifted)

%hiển thị phổ tần số của ảnh đầu vào 
imshow(log(1+F_magnitude), [])

%thao tác cắt ảnh
[rows, cols] = size(F_shifted);
row_center = round(rows / 2);
col_center = round(cols / 2);
row_cut = round(rows / 5);  % Giữ lại 40% theo chiều dọc
col_cut = round(cols / 5);  % Giữ lại 40% theo chiều ngang
F_cropped = F_shifted;
F_cropped(1:row_center-row_cut, :) = 0;  % Cắt tần số cao phía trên
F_cropped(row_center+row_cut:end, :) = 0;  % Cắt tần số cao phía dưới
F_cropped(:, 1:col_center-col_cut) = 0;  % Cắt tần số cao bên trái
F_cropped(:, col_center+col_cut:end) = 0;  % Cắt tần số cao bên phải
F_magnitude_cut = abs(F_cropped)

% Dịch ngược phổ
F_inv_shifted = ifftshift(F_cropped);  
% Biến đổi Fourier ngược
I_reconstructed = ifft2(F_inv_shifted);  

%Lấy phần thực của ảnh khôi phục, mục đích để hiển thị 
I_reconstructed = real(I_reconstructed);

figure();
subplot(2,3,1); imshow(image, []); title('Ảnh gốc');
subplot(2,3,2); imshow(image_gray, []); title('Ảnh xám ban đầu');
subplot(2,3,3); imshow(I_reconstructed, []); title('Ảnh sau khi cắt và biến đổi ngược');
subplot(2,3,4); imshow(log(1+F_magnitude_cut), []); title('Ảnh phổ bị cắt');
subplot(2,3,5); imshow(log(1+F_magnitude), []); title('Ảnh phổ ban đầu');
print(gcf,'image_ifft.png','-dpng','-r300');