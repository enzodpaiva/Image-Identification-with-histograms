import numpy as np
import matplotlib.pyplot as plt
import os
from config import path
import cv2
from skimage import io
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2hsv

def calcHSVPdf(image):
  hsv_image = rgb2hsv(image)  # Converte a imagem de RGB para HSV
  h, s, v = hsv_image[:, :, 0], hsv_image[:, :, 1], hsv_image[:, :, 2]

  hist_h = np.histogram(h, bins=256, range=[0, 1])[0]
  hist_s = np.histogram(s, bins=256, range=[0, 1])[0]
  hist_v = np.histogram(v, bins=256, range=[0, 1])[0]

  pdf_h = hist_h / np.sum(hist_h)
  pdf_s = hist_s / np.sum(hist_s)
  pdf_v = hist_v / np.sum(hist_v)

  return pdf_h, pdf_s, pdf_v

def calcDistancePdfHSV(pdf1, pdf2):
  distance_h = np.linalg.norm(pdf1[0] - pdf2[0])
  distance_s = np.linalg.norm(pdf1[1] - pdf2[1])
  distance_v = np.linalg.norm(pdf1[2] - pdf2[2])
  return (distance_h + distance_s + distance_v) / 3.0

def returnNumImagesSimilarHSV(pdfs_list, input_pdf, num):
  distances = [calcDistancePdfHSV(input_pdf, pdf) for pdf in pdfs_list]
  most_similar = np.argsort(distances)[:num]
  return most_similar

def isImageMatchHSV(image1, image2, threshold=0.95):
  # Calcule as PDFs HSV para ambas as imagens
  pdf_h1, pdf_s1, pdf_v1 = calcHSVPdf(image1)
  pdf_h2, pdf_s2, pdf_v2 = calcHSVPdf(image2)

  # Calcule a distância entre as PDFs
  distance_h = calcDistancePdfHSV(pdf_h1, pdf_h2)
  distance_s = calcDistancePdfHSV(pdf_s1, pdf_s2)
  distance_v = calcDistancePdfHSV(pdf_v1, pdf_v2)

  # Você pode ajustar o limiar de distância conforme necessário
  return distance_h < threshold and distance_s < threshold and distance_v < threshold

def processHsv(image, numImageUserWant):
  pdf_h, pdf_s, pdf_v = calcHSVPdf(image)

  pdfs_list = []
  image_names = []

  # Loop através de cada arquivo de imagem na pasta
  for filename in os.listdir(path):
    # Carrega a imagem usando OpenCV
    image_path = os.path.join(path, filename)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Carrega a imagem em formato RGB

    # Calcula a PDF em HSV
    pdf_h_i, pdf_s_i, pdf_v_i = calcHSVPdf(image)

    # Armazena as PDFs em uma lista
    pdfs_list.append((pdf_h_i, pdf_s_i, pdf_v_i))
    image_names.append(filename)

  most_similar = returnNumImagesSimilarHSV(pdfs_list, (pdf_h, pdf_s, pdf_v), numImageUserWant)

  correct_images = []
  plt.figure(figsize=(15, 5))
  for i, idx in enumerate(most_similar):
    image_path = os.path.join(path, image_names[idx])
    similar_image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    plt.subplot(1, numImageUserWant, i + 1)
    plt.imshow(cv2.cvtColor(similar_image, cv2.COLOR_BGR2RGB))  # Converte de BGR para RGB
    plt.title(f"Imagem {i+1} ({image_names[idx]})")

    # Verifique se a imagem retornada é uma correspondência correta e adicione à lista
    if isImageMatchHSV(image, similar_image):
      correct_images.append(similar_image)

  plt.tight_layout()
  plt.show()

  # Calcule a porcentagem de classificação correta em HSV
  percentage_correct_hsv = (len(correct_images) / numImageUserWant) * 100
  print(f"Porcentagem de Classificação Correta em HSV: {percentage_correct_hsv:.2f}%")