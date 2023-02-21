import cv2
import numpy as np
import matplotlib.pyplot as plt


def hog(img, cell_size=8, block_size=2, num_bins=9):
    # 计算图像的梯度和方向
    gradient_values_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    gradient_values_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    gradient_magnitude = cv2.addWeighted(gradient_values_x, 0.5, gradient_values_y, 0.5, 0)
    gradient_angle = cv2.phase(gradient_values_x, gradient_values_y, angleInDegrees=True)

    # 将角度映射到9个方向中的一个
    bin_width = 360 // num_bins
    gradient_angle = ((gradient_angle + bin_width / 2) % 360) // bin_width

    # 计算每个单元格的梯度直方图
    num_cells_x = img.shape[1] // cell_size
    num_cells_y = img.shape[0] // cell_size
    cell_histograms = np.zeros((num_cells_y, num_cells_x, num_bins))

    for i in range(num_cells_y):
        for j in range(num_cells_x):
            for y in range(i * cell_size, (i + 1) * cell_size):
                for x in range(j * cell_size, (j + 1) * cell_size):
                    bin_idx = int(gradient_angle[y, x])
                    cell_histograms[i, j, bin_idx] += gradient_magnitude[y, x]

    # 计算每个块的梯度特征向量，并进行L2归一化
    num_blocks_x = (num_cells_x - block_size) + 1
    num_blocks_y = (num_cells_y - block_size) + 1
    block_vectors = np.zeros((num_blocks_y, num_blocks_x, block_size ** 2 * num_bins), dtype=np.float32)

    for i in range(num_blocks_y):
        for j in range(num_blocks_x):
            block_vector = block_vectors[i, j, :]
            block_histogram = cell_histograms[i:i + block_size, j:j + block_size, :].flatten()
            block_vector[:] = block_histogram / np.sqrt(np.sum(block_histogram ** 2) + 1e-5)

    # 将所有块的梯度特征向量串联起来作为最终的HOG特征向量
    hog_vector = block_vectors.flatten()

    return hog_vector


# 读入图片
img = cv2.imread('img/juli.jpg', cv2.IMREAD_GRAYSCALE)

# 提取HOG特征向量
hog_vector = hog(img)

# 将HOG特征向量可视化为图像
num_cells_x = img.shape[1] // 8
num_cells_y = img.shape[0] // 8
num_bins = 9
hist = np.zeros((num_cells_y, num_cells_x, num_bins))

for i in range(num_cells_y):
    for j in range(num_cells_x):
        hist_values = hog_vector[(i * num_cells_x + j) * num_bins:(i * num_cells_x + j + 1) * num_bins]
        hist[i, j, :] = hist_values / np.sqrt(np.sum(hist_values ** 2) + 1e-5)

hist = np.minimum(hist, 0.2)

angle_unit = 360 / num_bins
im = np.zeros((num_cells_y * cell_size, num_cells_x * cell_size))
for i in range(num_cells_y):
    for j in range(num_cells_x):
        for k in range(num_bins):
            x = int(cell_size * (j + 0.5))
            y = int(cell_size * (i + 0.5))
            angle = (k * angle_unit + angle_unit / 2) % 360
            dx = cell_size * 0.5 * np.cos(np.deg2rad(angle))
            dy = cell_size * 0.5 * np.sin(np.deg2rad(angle))
            x1 = max(int(x - dx), 0)
            y1 = max(int(y - dy), 0)
            x2 = min(int(x + dx), im.shape[1] - 1)
            y2 = min(int(y + dy), im.shape[0] - 1)
            if hist[i, j, k] > 0:
                im[y1:y2, x1:x2] += hist[i, j, k]

plt.imshow(im, cmap='gray')
plt.show()

# 保存HOG特征向量可视化的图像
cv2.imwrite('hog_visualization.jpg', im * 255)

