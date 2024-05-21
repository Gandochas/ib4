from PIL import Image
import numpy as np
from scipy.fftpack import dct, idct

# Функция для вычисления 2D DCT (дискретное косинусное преобразование) блока
def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

# Функция для вычисления 2D IDCT (обратное дискретное косинусное преобразование) блока
def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

# Функция для обработки цветного изображения, включая зашифровку и расшифровку
def process_color_image(image_path, seed, p, n, mode='scramble'):
    # Установка случайного начального состояния для генератора случайных чисел
    np.random.seed(seed)

    # Загрузка изображения и преобразование его в массив numpy
    image = Image.open(image_path)
    data = np.array(image, dtype=np.float32)

    # Получение размеров изображения и количество каналов
    height, width, channels = data.shape
    processed_data = np.zeros_like(data)

    # Нормализация данных до диапазона [0, 1]
    data /= 255.0

    # Разбиение изображения на блоки размером 8x8 и их обработка
    for channel in range(channels):
        for i in range(0, height, 8):
            for j in range(0, width, 8):
                block = data[i:i+8, j:j+8, channel]
                if block.shape[0] < 8 or block.shape[1] < 8:
                    continue  # Пропускаем блоки, которые меньше 8x8
                dct_block = dct2(block)

                # Создание матрицы Бернулли с заданной вероятностью p
                B = np.where(np.random.rand(8, 8) < p, 1, -1)

                # Зашифровка или расшифровка блока в зависимости от режима
                if mode == 'scramble':
                    dct_block[n:, n:] *= B[n:, n:]
                elif mode == 'descramble':
                    dct_block[n:, n:] /= B[n:, n:]

                # Применение обратного DCT к обработанному блоку
                processed_block = idct2(dct_block)
                processed_data[i:i+8, j:j+8, channel] = processed_block

    # Возвращение данных в диапазон [0, 255] и создание изображения из обработанных данных
    processed_data = np.clip(processed_data, 0, 1) * 255
    processed_image = Image.fromarray(processed_data.astype(np.uint8))
    return processed_image

# Применение функции для зашифровки изображения
scrambled_image = process_color_image('ib43.png', seed=423333, p=0.1, n=1, mode='scramble')
scrambled_image.save('scrambled.png')

# Применение функции для расшифровки изображения
descrambled_image = process_color_image('scrambled.png', seed=423333, p=0.1, n=1, mode='descramble')
descrambled_image.save('restored.png')

# Функция для вычисления среднеквадратичной ошибки (MSE) между двумя изображениями
def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1] * imageA.shape[2])
    return err

# Функция для вычисления пиксельного соотношения сигнал/шум (PSNR) между двумя изображениями
def psnr(imageA, imageB):
    mse_value = mse(imageA, imageB)
    if mse_value == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse_value))
    return psnr

# Загрузка оригинального и зашифрованного изображений
original_img = np.array(Image.open('ib43.png'))
scrambled_img = np.array(Image.open('scrambled.png'))

# Вычисление PSNR
psnr_value = psnr(original_img, scrambled_img)
print(f"PSNR between original and scrambled images: {psnr_value} dB")
