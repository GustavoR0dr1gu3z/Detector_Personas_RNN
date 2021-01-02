# -*- coding: utf-8 -*-
import cv2
import numpy as np


    #Ingresamos la imagen que servira como entrenamiento a la neurona (IMAGEN UNO H)
img = cv2.imread("/home/gustavo/Descargas/ProyectoFinal/ImagenesOriginalesO/unoH.jpg", 0) #Leemos la imagen
    #Para redimensionar la imagen y hacer que sean menos los pieles del entrenamiento,
    #tilizaremos la opcion de opencv "resize" para definir la medida de la imagen
primerF = cv2.resize(img, (100, 300))#(ancho->columnas)(largo->Renglones)
cv2.imwrite("/home/gustavo/Descargas/ProyectoFinal/ImagenesReducidasO/1/H/unoHReducida.jpg",primerF)
#Ahora ya aplicado estos filtros haremos la rotación de imagenes    
    #Rotacion 90 grados
# Primero obtendremos largo y ancho de la imagen
height, width = primerF.shape[0:2]
imR = cv2.getRotationMatrix2D((width/2, height/2),90,.30)
rot90 = cv2.warpAffine(primerF, imR, (width, height))
cv2.imwrite("/home/gustavo/Descargas/ProyectoFinal/ImagenesRotadasO/1/H/Imagen_90_G.jpg",rot90)
    
    #Rotacion 180 grados
# Primero obtendremos largo y ancho de la imagen
height, width = primerF.shape[0:2]
imR = cv2.getRotationMatrix2D((width/2, height/2),180,1)
rot180 = cv2.warpAffine(primerF, imR, (width, height))
cv2.imwrite("/home/gustavo/Descargas/ProyectoFinal/ImagenesRotadasO/1/H/Imagen_180_G.jpg",rot180)

    #Rotacion 270 grados
# Primero obtendremos largo y ancho de la imagen
height, width = primerF.shape[0:2]
imR = cv2.getRotationMatrix2D((width/2, height/2),270,.30)
rot270 = cv2.warpAffine(primerF, imR, (width, height))
cv2.imwrite("/home/gustavo/Descargas/ProyectoFinal/ImagenesRotadasO/1/H/Imagen_270_G.jpg",rot270)
        
    #Rotacion 360 grados
# Primero obtendremos largo y ancho de la imagen
height, width = primerF.shape[0:2]
imR = cv2.getRotationMatrix2D((width/2, height/2),360,1)
rot360 = cv2.warpAffine(primerF, imR, (width, height))
cv2.imwrite("/home/gustavo/Descargas/ProyectoFinal/ImagenesRotadasO/1/H/Imagen_360_G.jpg",rot360)



# Imagenes rotadas 1/Hombre
img0 = cv2.imread("/home/gustavo/Descargas/ProyectoFinal/ImagenesRotadasO/1/H/Imagen_90_G.jpg",0)
img1 = cv2.imread("/home/gustavo/Descargas/ProyectoFinal/ImagenesRotadasO/1/H/Imagen_180_G.jpg",0)
img2 = cv2.imread("/home/gustavo/Descargas/ProyectoFinal/ImagenesRotadasO/1/H/Imagen_270_G.jpg",0)
img3 = cv2.imread("/home/gustavo/Descargas/ProyectoFinal/ImagenesRotadasO/1/H/Imagen_360_G.jpg",0)

"""
# Imagenes rotadas 1/Mujer
img0 = cv2.imread("/home/gustavo/Descargas/ProyectoFinal/ImagenesRotadasO/1/M/Imagen_90_G.jpg",0)
img1 = cv2.imread("/home/gustavo/Descargas/ProyectoFinal/ImagenesRotadasO/1/M/Imagen_180_G.jpg",0)
img2 = cv2.imread("/home/gustavo/Descargas/ProyectoFinal/ImagenesRotadasO/1/M/Imagen_270_G.jpg",0)
img3 = cv2.imread("/home/gustavo/Descargas/ProyectoFinal/ImagenesRotadasO/1/M/Imagen_360_G.jpg",0)
"""

"""
# Imagenes rotadas 2/Hombre
img0 = cv2.imread("/home/gustavo/Descargas/ProyectoFinal/ImagenesRotadasO/2/H/Imagen_90_G.jpg",0)
img1 = cv2.imread("/home/gustavo/Descargas/ProyectoFinal/ImagenesRotadasO/2/H/Imagen_180_G.jpg",0)
img2 = cv2.imread("/home/gustavo/Descargas/ProyectoFinal/ImagenesRotadasO/2/H/Imagen_270_G.jpg",0)
img3 = cv2.imread("/home/gustavo/Descargas/ProyectoFinal/ImagenesRotadasO/2/H/Imagen_360_G.jpg",0)
"""

"""
# Imagenes rotadas 2/Mujer
img0 = cv2.imread("/home/gustavo/Descargas/ProyectoFinal/ImagenesRotadasO/2/M/Imagen_90_G.jpg",0)
img1 = cv2.imread("/home/gustavo/Descargas/ProyectoFinal/ImagenesRotadasO/2/M/Imagen_180_G.jpg",0)
img2 = cv2.imread("/home/gustavo/Descargas/ProyectoFinal/ImagenesRotadasO/2/M/Imagen_270_G.jpg",0)
img3 = cv2.imread("/home/gustavo/Descargas/ProyectoFinal/ImagenesRotadasO/2/M/Imagen_360_G.jpg",0)
"""

"""
# Imagenes rotadas 3/Hombre
img0 = cv2.imread("/home/gustavo/Descargas/ProyectoFinal/ImagenesRotadasO/3/H/Imagen_90_G.jpg",0)
img1 = cv2.imread("/home/gustavo/Descargas/ProyectoFinal/ImagenesRotadasO/3/H/Imagen_180_G.jpg",0)
img2 = cv2.imread("/home/gustavo/Descargas/ProyectoFinal/ImagenesRotadasO/3/H/Imagen_270_G.jpg",0)
img3 = cv2.imread("/home/gustavo/Descargas/ProyectoFinal/ImagenesRotadasO/3/H/Imagen_360_G.jpg",0)
"""

"""
# Imagenes rotadas 3/Mujer
img0 = cv2.imread("/home/gustavo/Descargas/ProyectoFinal/ImagenesRotadasO/3/M/Imagen_90_G.jpg",0)
img1 = cv2.imread("/home/gustavo/Descargas/ProyectoFinal/ImagenesRotadasO/3/M/Imagen_180_G.jpg",0)
img2 = cv2.imread("/home/gustavo/Descargas/ProyectoFinal/ImagenesRotadasO/3/M/Imagen_270_G.jpg",0)
img3 = cv2.imread("/home/gustavo/Descargas/ProyectoFinal/ImagenesRotadasO/3/M/Imagen_360_G.jpg",0)
"""

"""
# Imagenes rotadas 4/Hombre
img0 = cv2.imread("/home/gustavo/Descargas/ProyectoFinal/ImagenesRotadasO/4/H/Imagen_90_G.jpg",0)
img1 = cv2.imread("/home/gustavo/Descargas/ProyectoFinal/ImagenesRotadasO/4/H/Imagen_180_G.jpg",0)
img2 = cv2.imread("/home/gustavo/Descargas/ProyectoFinal/ImagenesRotadasO/4/H/Imagen_270_G.jpg",0)
img3 = cv2.imread("/home/gustavo/Descargas/ProyectoFinal/ImagenesRotadasO/4/H/Imagen_360_G.jpg",0)
"""

"""
# Imagenes rotadas 4/Mujer
img0 = cv2.imread("/home/gustavo/Descargas/ProyectoFinal/ImagenesRotadasO/4/M/Imagen_90_G.jpg",0)
img1 = cv2.imread("/home/gustavo/Descargas/ProyectoFinal/ImagenesRotadasO/4/M/Imagen_180_G.jpg",0)
img2 = cv2.imread("/home/gustavo/Descargas/ProyectoFinal/ImagenesRotadasO/4/M/Imagen_270_G.jpg",0)
img3 = cv2.imread("/home/gustavo/Descargas/ProyectoFinal/ImagenesRotadasO/4/M/Imagen_360_G.jpg",0)
"""

"""
# Imagenes rotadas 5/Hombre
img0 = cv2.imread("/home/gustavo/Descargas/ProyectoFinal/ImagenesRotadasO/5/H/Imagen_90_G.jpg",0)
img1 = cv2.imread("/home/gustavo/Descargas/ProyectoFinal/ImagenesRotadasO/5/H/Imagen_180_G.jpg",0)
img2 = cv2.imread("/home/gustavo/Descargas/ProyectoFinal/ImagenesRotadasO/5/H/Imagen_270_G.jpg",0)
img3 = cv2.imread("/home/gustavo/Descargas/ProyectoFinal/ImagenesRotadasO/5/H/Imagen_360_G.jpg",0)
"""

"""
# Imagenes rotadas 5/Mujer
img0 = cv2.imread("/home/gustavo/Descargas/ProyectoFinal/ImagenesRotadasO/5/M/Imagen_90_G.jpg",0)
img1 = cv2.imread("/home/gustavo/Descargas/ProyectoFinal/ImagenesRotadasO/5/M/Imagen_180_G.jpg",0)
img2 = cv2.imread("/home/gustavo/Descargas/ProyectoFinal/ImagenesRotadasO/5/M/Imagen_270_G.jpg",0)
img3 = cv2.imread("/home/gustavo/Descargas/ProyectoFinal/ImagenesRotadasO/5/M/Imagen_360_G.jpg",0)
"""

"""
# Imagenes rotadas 6/Hombre
img0 = cv2.imread("/home/gustavo/Descargas/ProyectoFinal/ImagenesRotadasO/6/H/Imagen_90_G.jpg",0)
img1 = cv2.imread("/home/gustavo/Descargas/ProyectoFinal/ImagenesRotadasO/6/H/Imagen_180_G.jpg",0)
img2 = cv2.imread("/home/gustavo/Descargas/ProyectoFinal/ImagenesRotadasO/6/H/Imagen_270_G.jpg",0)
img3 = cv2.imread("/home/gustavo/Descargas/ProyectoFinal/ImagenesRotadasO/6/H/Imagen_360_G.jpg",0)
"""

"""
# Imagenes rotadas 6/Mujer
img0 = cv2.imread("/home/gustavo/Descargas/ProyectoFinal/ImagenesRotadasO/6/M/Imagen_90_G.jpg",0)
img1 = cv2.imread("/home/gustavo/Descargas/ProyectoFinal/ImagenesRotadasO/6/M/Imagen_180_G.jpg",0)
img2 = cv2.imread("/home/gustavo/Descargas/ProyectoFinal/ImagenesRotadasO/6/M/Imagen_270_G.jpg",0)
img3 = cv2.imread("/home/gustavo/Descargas/ProyectoFinal/ImagenesRotadasO/6/M/Imagen_360_G.jpg",0)
"""




# Convertir las imagenes a un arreglo de numpy 

img00 = np.array(img0)
img11 = np.array(img1)
img22 = np.array(img2)
img33 = np.array(img3)


# APlanar el arreglo a un vector de 30,000 x 1 

img000 = img00.reshape([30000,1]) #Tamaño dinámico
img111 = img11.reshape([30000,1]) #Tamaño dinámico
img222 = img22.reshape([30000,1]) #Tamaño dinámico
img333 = img33.reshape([30000,1]) #Tamaño dinámico

# Dividir cada elemento entre 255 para mejor eficiencia

X0 = img000/255
X1 = img111/255
X2 = img222/255
X3 = img333/255