
from PIL import Image

import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from imgProcess.gray import processGray
from imgProcess.rgb import processRgb
from imgProcess.hsv import processHsv
import tkinter as tk
from tkinter import filedialog
from config import path

def process_image(escolha, image, numImgsReturn):
  switch = {
      1: lambda: processGray.processGray(image, numImgsReturn),
      2: lambda: processRgb.processRgb(image, numImgsReturn),
      3: lambda: processHsv.processHsv(image, numImgsReturn)
  }
  return switch.get(escolha, lambda: "Escolha inválida")()

while True:
  try:

    print("escolha como quer o processamento da imagem sendo: 1 - niveis de cinza / 2 - RGB / 3 - HSV / 4 - Sair")
    escolha = int(input())

    if escolha == 4:
      print("Opcao de saida escolhida. Até logo!")
      break  # Sai do loop se o usuário escolher parar

    file_name = filedialog.askopenfilename(
        title="Selecione um arquivo de imagem",
        filetypes=[("Arquivos de Imagem", "*.jpg *.jpeg *.png *.bmp *.gif")]
    )

    #file_name = filedialog.askopenfilenames() # show an "Open" dialog box and return the path to the selected file
    print(file_name)
    
    #uploaded = files.upload()
    #file_name = list(uploaded.keys())[0]

    print("escolha quantas imagens quer retornar")
    numImgsReturn = int(input())

    if escolha == 1:
      image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    else:
      image = cv2.imread(file_name, cv2.IMREAD_COLOR)

    print(escolha, image, numImgsReturn)

    result = process_image(escolha, image, numImgsReturn)
    print(result)
  except Exception as e:
    print(f"Algo inesperado aconteceu ou a imagem não pode ser processada {e}. Até logo!")
    break