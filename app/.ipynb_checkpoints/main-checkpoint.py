import os
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from google.colab import drive
from google.colab import files
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
drive.mount('/content/drive/')
from skimage import io
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2hsv
from app.imgProcess.Gray import processGray
from app.imgProcess.Rgb import processRgb
from app.imgProcess.Hsv import processHsv

path = '/content/drive/MyDrive/semestre8/tpi/databaseImagesTrab1'

def process_image(escolha, image, numImgsReturn):
    switch = {
        1: lambda: processGray(image, numImgsReturn),
        2: lambda: processRgb(image, numImgsReturn),
        3: lambda: processHsv(image, numImgsReturn)
    }
    return switch.get(escolha, lambda: "Escolha inválida")()

while True:
  try:
    uploaded = files.upload()
    file_name = list(uploaded.keys())[0]

    print("escolha como quer o processamento da imagem sendo: 1 - niveis de cinza / 2 - RGB / 3 - HSV / 4 - Sair")
    escolha = int(input())

    if escolha == 4:
      print("Opcao de saida escolhida. Até logo!")
      break  # Sai do loop se o usuário escolher parar

    print("escolha quantas imagens quer retornar")
    numImgsReturn = int(input())

    if escolha == 1:
      image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    else:
      image = cv2.imread(file_name, cv2.IMREAD_COLOR)

    result = process_image(escolha, image, numImgsReturn)
    print(result)
  except Exception as e:
    print(f"Algo inesperado aconteceu ou a imagem não pode ser processada {e}. Até logo!")
    break