from PIL import Image
import numpy as np
from scipy.fftpack import dct, idct


def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')


def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')


def process_color_image(image_path, seed, p, n, mode='scramble'):
    np.random.seed(seed)
    image = Image.open(image_path)
    data = np.array(image, dtype=np.float32)

    height, width, channels = data.shape
    processed_data = np.zeros_like(data)

    data /= 255.0

    # Разбиение изображения на блоки размером 8x8
    for channel in range(channels):
        for i in range(0, height, 8):
            for j in range(0, width, 8):
                block = data[i:i+8, j:j+8, channel]
                if block.shape[0] < 8 or block.shape[1] < 8:
                    continue  # Пропускаем блоки, которые меньше 8x8
                dct_block = dct2(block)
                B = np.where(np.random.rand(8, 8) < p, 1, -1)  # Матрица Бернулли
                if mode == 'scramble':
                    dct_block[n:, n:] *= B[n:, n:]
                elif mode == 'descramble':
                    dct_block[n:, n:] /= B[n:, n:]
                processed_block = idct2(dct_block)
                processed_data[i:i+8, j:j+8, channel] = processed_block

    processed_data = np.clip(processed_data, 0, 1) * 255
    processed_image = Image.fromarray(processed_data.astype(np.uint8))
    return processed_image


scrambled_image = process_color_image('ib43.png', seed=423333, p=0.1, n=1, mode='scramble')
scrambled_image.save('scrambled.png')


descrambled_image = process_color_image('scrambled.png', seed=423333, p=0.1, n=1, mode='descramble')
descrambled_image.save('restored.png')


def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1] * imageA.shape[2])
    return err


def psnr(imageA, imageB):
    mse_value = mse(imageA, imageB)
    if mse_value == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse_value))
    return psnr


# Загрузка изображений
original_img = np.array(Image.open('ib43.png'))
scrambled_img = np.array(Image.open('scrambled.png'))

# Вычисление PSNR
psnr_value = psnr(original_img, scrambled_img)
print(f"PSNR between original and scrambled images: {psnr_value} dB")
