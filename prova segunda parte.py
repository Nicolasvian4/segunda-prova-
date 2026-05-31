import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregar a imagem
imagem_original = cv2.imread('segunda_prova\provanicolas1.jpeg', 0)

# Verificar se a imagem foi carregada corretamente
if imagem_original is None:
    print("Erro ao carregar a imagem.")
    exit()

# Obter dimensões da imagem
altura, largura = imagem_original.shape

# Criar padrão senoidal diagonal
eixo_x = np.arange(largura)
eixo_y = np.arange(altura)

matriz_x, matriz_y = np.meshgrid(eixo_x, eixo_y)

# Frequência do ruído
frequencia_ruido = 0.15

# Criar ruído periódico
ruido_periodico = 50 * np.sin(
    2 * np.pi * frequencia_ruido * (matriz_x + matriz_y)
)

# Adicionar ruído à imagem
imagem_com_ruido = imagem_original + ruido_periodico

# Limitar valores entre 0 e 255
imagem_com_ruido = np.clip(
    imagem_com_ruido,
    0,
    255
).astype(np.uint8)

# Aplicar Transformada de Fourier
transformada_fourier = np.fft.fft2(imagem_com_ruido)

transformada_centralizada = np.fft.fftshift(
    transformada_fourier
)

# Criar espectro da imagem
espectro_original = 20 * np.log(
    np.abs(transformada_centralizada) + 1
)

# Criar máscara do filtro Notch
mascara_filtro = np.ones(
    (altura, largura),
    np.uint8
)

centro_y, centro_x = altura // 2, largura // 2

# Coordenadas dos ruídos no espectro
lista_pontos = [
    (centro_y - 80, centro_x - 80),
    (centro_y - 60, centro_x - 60),
    (centro_y - 40, centro_x - 40),
    (centro_y - 20, centro_x - 20),

    (centro_y + 20, centro_x + 20),
    (centro_y + 40, centro_x + 40),
    (centro_y + 60, centro_x + 60),
    (centro_y + 80, centro_x + 80)
]

# Criar regiões de bloqueio no espectro
raio_filtro = 8

for ponto in lista_pontos:

    posicao_y, posicao_x = ponto

    for linha in range(altura):
        for coluna in range(largura):

            distancia = (
                (linha - posicao_y) ** 2 +
                (coluna - posicao_x) ** 2
            )

            if distancia <= raio_filtro ** 2:
                mascara_filtro[linha, coluna] = 0

# Aplicar filtro na Transformada de Fourier
transformada_filtrada = (
    transformada_centralizada * mascara_filtro
)

# Criar espectro filtrado
espectro_filtrado = 20 * np.log(
    np.abs(transformada_filtrada) + 1
)

# Aplicar Transformada Inversa de Fourier
transformada_inversa = np.fft.ifftshift(
    transformada_filtrada
)

imagem_recuperada = np.fft.ifft2(
    transformada_inversa
)

imagem_recuperada = np.abs(
    imagem_recuperada
)

# Exibir resultados
plt.figure(figsize=(12, 10))

# Imagem com ruído
plt.subplot(221)
plt.imshow(imagem_com_ruido, cmap='gray')
plt.title('Imagem Original')
plt.axis('off')

# Imagem filtrada
plt.subplot(222)
plt.imshow(imagem_recuperada, cmap='gray')
plt.title('Imagem Final')
plt.axis('off')

# Espectro original
plt.subplot(223)
plt.imshow(espectro_original, cmap='gray')
plt.title('Espectro da Imagem Original')
plt.axis('off')

# Espectro filtrado
plt.subplot(224)
plt.imshow(espectro_filtrado, cmap='gray')
plt.title('Espectro da Imagem Final')
plt.axis('off')

plt.tight_layout()
plt.show()