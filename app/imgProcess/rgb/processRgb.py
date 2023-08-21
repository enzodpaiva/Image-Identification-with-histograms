import numpy as np
import matplotlib.pyplot as plt
import os
from config import path
import cv2
from skimage import io
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2hsv

def calcDistancePdfRgb(pdf1, pdf2):
  distance_r = np.linalg.norm(pdf1[0] - pdf2[0])
  distance_g = np.linalg.norm(pdf1[1] - pdf2[1])
  distance_b = np.linalg.norm(pdf1[2] - pdf2[2])
  return distance_r + distance_g + distance_b

def returnNumImagesSimilarRgb(pdfs_list, input_pdf, num):
    distances = [calcDistancePdfRgb(input_pdf, pdf) for pdf in pdfs_list]
    most_similar = np.argsort(distances)[:num]
    return most_similar

def calcRGBPdf(image):
  # Separa os canais de cores (R, G, B)
  r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]

  # Calcula os histogramas para cada canal
  hist_r, bins_r = np.histogram(r, bins=256, range=[0, 256])
  hist_g, bins_g = np.histogram(g, bins=256, range=[0, 256])
  hist_b, bins_b = np.histogram(b, bins=256, range=[0, 256])

  # Calcula as PDFs para cada canal
  pdf_r = hist_r / np.sum(hist_r)
  pdf_g = hist_g / np.sum(hist_g)
  pdf_b = hist_b / np.sum(hist_b)

  return pdf_r, pdf_g, pdf_b

def isImageMatchRgb(image1, image2, threshold=0.9):
  # Calcule a PDF em RGB para ambas as imagens
  pdf_r1, pdf_g1, pdf_b1 = calcRGBPdf(image1)
  pdf_r2, pdf_g2, pdf_b2 = calcRGBPdf(image2)

  # Calcule a distância entre as PDFs
  distance_r = calcDistancePdfRgb(pdf_r1, pdf_r2)
  distance_g = calcDistancePdfRgb(pdf_g1, pdf_g2)
  distance_b = calcDistancePdfRgb(pdf_b1, pdf_b2)

  # Você pode ajustar o limiar de distância conforme necessário
  return distance_r < threshold and distance_g < threshold and distance_b < threshold

def processRgb(image, numImageUserWant):
  # Calcula a PDF em RGB para a imagem de entrada
  pdf_r, pdf_g, pdf_b = calcRGBPdf(image)

  # Lista para armazenar as PDFs das imagens da pasta
  pdfs_list = []
  image_names = []

  # Loop através de cada arquivo de imagem na pasta
  for filename in os.listdir(path):
    # Carrega a imagem usando OpenCV
    image_path = os.path.join(path, filename)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Carrega a imagem em formato RGB

    # Calcula a PDF em RGB
    pdf_r_i, pdf_g_i, pdf_b_i = calcRGBPdf(image)

        # Armazena as PDFs em uma lista
    pdfs_list.append((pdf_r_i, pdf_g_i, pdf_b_i))
    image_names.append(filename)

  most_similar = returnNumImagesSimilarRgb(pdfs_list, (pdf_r, pdf_g, pdf_b), numImageUserWant)

  correct_images = []

  plt.figure(figsize=(15, 5))
  for i, idx in enumerate(most_similar):
    image_path = os.path.join(path, image_names[idx])
    similar_image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    plt.subplot(1, numImageUserWant, i + 1)
    plt.imshow(cv2.cvtColor(similar_image, cv2.COLOR_BGR2RGB))  # Converte de BGR para RGB
    plt.title(f"Imagem {i+1} ({image_names[idx]})")

    if isImageMatchRgb(image, similar_image):
      correct_images.append(similar_image)

  plt.tight_layout()
  plt.show()
  percentage_correct_rgb = (len(correct_images) / numImageUserWant) * 100
  print(f"Porcentagem de Classificação Correta em RGB: {percentage_correct_rgb:.2f}%")