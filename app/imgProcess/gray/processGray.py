import numpy as np
import matplotlib.pyplot as plt
import os
from config import path
import cv2

def calcGrayPdf(image):
  hist, bins = np.histogram(image, bins=256, range=[0, 256])
  pdf = hist / np.sum(hist)

  return pdf

  #distances = [np.linalg.norm(input_pdf - pdf) for pdf in pdfs_list]
def calcDistancePdfGray(pdfs_list, pdf_uploaded):
  distances = []
  for pdf in pdfs_list:
    #Calcula a distância euclidiana entre as PDFs
    distance = np.linalg.norm(pdf_uploaded - pdf)
    distances.append(distance)

  return distances
  
def returnNumImagesSimilarGray(pdfs_list, pdf_uploaded, num):
  distances = calcDistancePdfGray(pdfs_list, pdf_uploaded)
  most_similar = np.argsort(distances)[:num]
  return most_similar

def isImageMatchGray(image1, image2, threshold=0.8):
  # Calcule a PDF em RGB para ambas as imagens
  pdf1 = calcGrayPdf(image1)
  pdf2 = calcGrayPdf(image2)

  # Calcule a distância entre as PDFs
  #distance = calcDistancePdfGray(pdf1, pdf2)
  distance = np.linalg.norm(pdf1 - pdf2)

  # Você pode ajustar o limiar de distância conforme necessário
  return distance < threshold

def processGray(image, numImageUserWant):
  pdf_uploaded = calcGrayPdf(image)
  pdfs_list = []
  image_names = []

  for filename in os.listdir(path):
    image_path = os.path.join(path, filename)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    pdf = calcGrayPdf(image)
    pdfs_list.append(pdf)
    image_names.append(filename)

  correct_images = []
  # Mostra as imagens mais parecidas
  plt.figure(figsize=(15, 5))
  most_similar = returnNumImagesSimilarGray(pdfs_list, pdf_uploaded, numImageUserWant)
  for i, idx in enumerate(most_similar):
    image_path = os.path.join(path, image_names[idx])
    similar_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    plt.subplot(1, numImageUserWant, i + 1)
    plt.imshow(similar_image, cmap='gray')
    plt.title(f"Imagem {i+1} - {image_names[idx]}")

    if isImageMatchGray(image, similar_image):
      correct_images.append(similar_image)

  plt.tight_layout()
  plt.show()
  percentage_correct_gray = (len(correct_images) / numImageUserWant) * 100
  print(f"Porcentagem de Classificação Correta em Tons de Cinza: {percentage_correct_gray:.2f}%")