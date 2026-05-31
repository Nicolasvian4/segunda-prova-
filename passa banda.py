import cv2
import numpy as np
import matplotlib.pyplot as plt

# Ler imagem em escala de cinza
imagem_cinza = cv2.imread('segunda_prova\provanicolas1.jpeg', 0)

# Aplicar Transformada de Fourier
transformada = np.fft.fft2(imagem_cinza)

# Centralizar frequências
transformada_shift = np.fft.fftshift(
    transformada
)

# Dimensões da imagem
linhas, colunas = imagem_cinza.shape

# Coordenadas centrais
meio_y = linhas // 2
meio_x = colunas // 2

# Criar máscara do filtro
filtro_banda = np.zeros(
    (linhas, colunas),
    np.uint8
)

# Definir limites do filtro
raio_menor = 20
raio_maior = 60

# Criar região do passa-banda
for y in range(linhas):
    for x in range(colunas):

        valor_distancia = (
            (y - meio_y) ** 2 +
            (x - meio_x) ** 2
        )

        if (
            raio_menor ** 2 <
            valor_distancia <
            raio_maior ** 2
        ):

            filtro_banda[y, x] = 1

# Aplicar filtro
fourier_filtrada = (
    transformada_shift * filtro_banda
)

# Transformada inversa
fourier_normal = np.fft.ifftshift(
    fourier_filtrada
)

imagem_processada = np.fft.ifft2(
    fourier_normal
)

# Remover parte imaginária
imagem_processada = np.abs(
    imagem_processada
)

# Criar espectros
espectro_inicial = 20 * np.log(
    np.abs(transformada_shift) + 1
)

espectro_final = 20 * np.log(
    np.abs(fourier_filtrada) + 1
)

# Mostrar imagens
plt.figure(figsize=(12, 8))

# Imagem original
plt.subplot(221)
plt.imshow(imagem_cinza, cmap='gray')
plt.title('Imagem Original')
plt.axis('off')

# Fourier original
plt.subplot(222)
plt.imshow(espectro_inicial, cmap='gray')
plt.title('Espectro de Fourier')
plt.axis('off')

# Máscara do filtro
plt.subplot(223)
plt.imshow(filtro_banda, cmap='gray')
plt.title('Filtro Passa-Banda')
plt.axis('off')

# Resultado final
plt.subplot(224)
plt.imshow(imagem_processada, cmap='gray')
plt.title('Imagem Filtrada')
plt.axis('off')

plt.tight_layout()
plt.show()