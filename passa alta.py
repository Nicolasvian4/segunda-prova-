import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregar imagem em escala de cinza-
img_original = cv2.imread('segunda_prova\provanicolas1.jpeg', 0)

# Aplicar Transformada de Fourier
fft_imagem = np.fft.fft2(img_original)

# Centralizar frequências
fft_central = np.fft.fftshift(
    fft_imagem
)

# Tamanho da imagem
total_linhas, total_colunas = img_original.shape

# Encontrar centro
ponto_y = total_linhas // 2
ponto_x = total_colunas // 2

# Criar máscara do filtro passa-alta
mascara_passa_alta = np.ones(
    (total_linhas, total_colunas),
    dtype=np.uint8
)

# Definir região de corte
raio = 35

# Remover frequências baixas
for linha in range(total_linhas):
    for coluna in range(total_colunas):

        distancia = (
            (linha - ponto_y) ** 2 +
            (coluna - ponto_x) ** 2
        )

        if distancia < raio ** 2:
            mascara_passa_alta[linha, coluna] = 0

# Aplicar filtro
fft_filtrada = (
    fft_central * mascara_passa_alta
)

# Aplicar Fourier inversa
fft_normal = np.fft.ifftshift(
    fft_filtrada
)

imagem_bordas = np.fft.ifft2(
    fft_normal
)

# Converter valores complexos
imagem_bordas = np.abs(
    imagem_bordas
)

# Gerar espectros
espectro_fft = 20 * np.log(
    np.abs(fft_central) + 1
)

espectro_resultado = 20 * np.log(
    np.abs(fft_filtrada) + 1
)

# Mostrar resultados
plt.figure(figsize=(12, 8))

# Imagem original
plt.subplot(2, 2, 1)
plt.imshow(img_original, cmap='gray')
plt.title('Imagem Original')
plt.axis('off')

# Espectro de Fourier
plt.subplot(2, 2, 2)
plt.imshow(espectro_fft, cmap='gray')
plt.title('Espectro de Fourier')
plt.axis('off')

# Máscara do filtro
plt.subplot(2, 2, 3)
plt.imshow(mascara_passa_alta, cmap='gray')
plt.title('Filtro Passa-Alta')
plt.axis('off')

# Resultado final
plt.subplot(2, 2, 4)
plt.imshow(imagem_bordas, cmap='gray')
plt.title('Realce de Bordas')
plt.axis('off')

plt.tight_layout()
plt.show()