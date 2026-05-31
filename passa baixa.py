import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregar imagem em escala de cinza
imagem = cv2.imread('segunda_prova\provanicolas1.jpeg', 0)

# Aplicar Transformada de Fourier
fourier = np.fft.fft2(imagem)

# Centralizar frequências da imagem
fourier_centralizada = np.fft.fftshift(fourier)

# Obter dimensões da imagem
altura, largura = imagem.shape

# Encontrar centro da imagem
centro_y = altura // 2
centro_x = largura // 2

# Criar máscara do filtro
filtro_passa_baixa = np.zeros(
    (altura, largura),
    np.uint8
)

# Definir tamanho do filtro
raio_filtro = 30

# Criar região circular no centro
for linha in range(altura):
    for coluna in range(largura):

        distancia = (
            (linha - centro_y) ** 2 +
            (coluna - centro_x) ** 2
        )

        if distancia <= raio_filtro ** 2:
            filtro_passa_baixa[linha, coluna] = 1

# Aplicar filtro na Fourier
fourier_filtrada = (
    fourier_centralizada * filtro_passa_baixa
)

# Aplicar Transformada Inversa
fourier_inversa = np.fft.ifftshift(
    fourier_filtrada
)

imagem_final = np.fft.ifft2(
    fourier_inversa
)

# Converter valores complexos
imagem_final = np.abs(imagem_final)

# Criar espectros
espectro_original = 20 * np.log(
    np.abs(fourier_centralizada) + 1
)

espectro_filtrado = 20 * np.log(
    np.abs(fourier_filtrada) + 1
)

# Mostrar resultados
plt.figure(figsize=(12, 8))

# Imagem original
plt.subplot(221)
plt.imshow(imagem, cmap='gray')
plt.title('Imagem Original')
plt.axis('off')

# Espectro da Fourier
plt.subplot(222)
plt.imshow(espectro_original, cmap='gray')
plt.title('Espectro de Fourier')
plt.axis('off')

# Máscara do filtro
plt.subplot(223)
plt.imshow(filtro_passa_baixa, cmap='gray')
plt.title('Filtro Passa-Baixa')
plt.axis('off')

# Imagem filtrada
plt.subplot(224)
plt.imshow(imagem_final, cmap='gray')
plt.title('Imagem Filtrada')
plt.axis('off')

plt.tight_layout()
plt.show()