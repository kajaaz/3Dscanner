from __future__ import print_function
import cv2
from common import draw_str
from common import splitfn
import numpy as np
import sys
import getopt
from glob import glob
import math
import datetime
import os
from itertools import groupby
from operator import itemgetter
from sympy.combinatorics.graycode import GrayCode, bin_to_gray, gray_to_bin
import pdb

cwd = os.getcwd()

valeurs_couleurs = [0,1,2] #RGB

gray = True #Si on decode en mode Gray

def flipSide():
    global CInst
    CInst[0], CInst[1] = CInst[1], CInst[0]
    
def takePicture(CInst_tab, SImg, num, Side):
    if CInst_tab[0][2] == Side:
        index = 0
    else:
        index = 1
    SImg[Side].append(CInst_tab[index][1])

def delimPict(SImg, Npt, Spt):
    pass

def takeFrame(CInst_tab, flip):
    for camera_instance in CInst_tab:
        camera_instance[3], camera_instance[1] = camera_instance[0].read()
        if flip == True:
            camera_instance[1] = cv2.flip(camera_instance[1], 1)

def traitement_NB(img):
    global threshold
    res_blanc = np.zeros((h, w, 3), dtype=np.int8)
    res_noir = np.zeros((h, w, 3), dtype=np.int8)
    output = np.zeros((h, w, 1), dtype=np.int8)

    ret, dst = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    bleu, vert, rouge = cv2.split(dst)
    
    bleu /= 255
    vert /= 255
    rouge /= 255
    
    res_blanc = np.logical_and(bleu, vert)
    res_blanc = np.logical_and(res_blanc, rouge) #on garde seulement le blanc
    res_blanc = res_blanc.astype(np.uint8)
    
    res_noir = np.logical_and(1-bleu, 1-vert)
    res_noir = np.logical_and(res_noir, 1-rouge) #on garde seulement le noir
    res_noir = res_noir.astype(np.uint8)
    
    output = (np.logical_and(res_noir, 1-res_blanc)) * 255
    output = output.astype(np.uint8)
    
    return output
    
def tabID_NB(SImg):
    output = np.zeros((SImg[0][0].shape[0], SImg[0][0].shape[1], len(SImg)), dtype=np.uint16)
    cnt = 0
    print(output.shape)
    for L in SImg: #pour chaque camera
        print("Traitement camera", cnt)
        output[:,:,cnt] = dPackBits(np.dstack(([traitement_NB(L[i]) for i in range(len(L))])) / 255, 2)[:,:,0]
        cnt += 1
    print("fini")
    #print(list((output[0][0][0], output[0][0][1])))
    #print(list((bin(output[0][0][0]), bin(output[0][0][1]))))
    #return output[delim[1][0]:delim[2][0]][delim[1][1]:delim[2][1]]
    return output
    #output.shape = (h, w, 2)
    #output correspons aux identifiants des points en base 10

#permet de concatener des bits suivant l'axe Z
def dPackBits(arrays, n):
    output = np.zeros((arrays.shape[0], arrays.shape[1], 1), dtype=np.uint16)
    #print("output.shape=", output.shape)
    #print("arrays.shape=", arrays.shape)
    if(gray == True): #Si on decode en Gray
        for x in range(arrays.shape[0]):
            for y in range(arrays.shape[1]):
                output[x][y] = int(gray_to_bin(''.join(str(x) for x in arrays[x][y])), base=n)
    else: #Sinon en binaire classique
        for x in range(arrays.shape[0]):
            for y in range(arrays.shape[1]):
                output[x][y] = int(''.join(str(x) for x in arrays[x][y]), base=n)
            #pout chaque x,y on transforme les coordonnees sur Z en un string et on le transforme en int depuis la base n
            #timeit.timeit("[int(''.join(str(x) for x in [i for i in range(8)]))]", number=1000000) -> 3.8µs par appel
    #µprint("packbits ->", output[0][0])
    return output

def traitement_RGB(img):
    ret, dst = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    bleu, vert, rouge = cv2.split(dst)
    #img2 = cv2.adaptiveThreshold(r, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)    
    
    r = rouge / 255 # 0 ou 1 (cast en int)
    v = vert / 255
    b = bleu / 255
    
    r = np.logical_and(r, 255-vert) #on exclut le vert
    r = np.logical_and(r, 255-bleu) #et le bleu
    r = r.astype(np.uint8) #on le transforme en uint8
    r *= 255 #et on le remet de 0 à 255
    
    v = np.logical_and(v, 255-rouge)
    v = np.logical_and(v, 255-bleu)
    v = v.astype(np.uint8)
    v*= 255
    
    b = np.logical_and(b, 255-vert)
    b = np.logical_and(b, 255-rouge)
    b = b.astype(np.uint8)
    b *= 255
    
    rouge = cv2.cvtColor(r, cv2.COLOR_GRAY2RGB) #conversion en image couleur RGB pour l'affichage
    vert = cv2.cvtColor(v, cv2.COLOR_GRAY2RGB) #cela reste quand même une image en n
    bleu = cv2.cvtColor(b, cv2.COLOR_GRAY2RGB)
    return (rouge, vert, bleu)
    
def formattage_RGB(imgs): #renvoie une seule image avec les valeurs correspondant aux couleurs
    output = np.zeros((h,w,1), dtype = np.uint8) + 3
    output -= (3-valeurs_couleurs[0])*imgs[0]
    output -= (3-valeurs_couleurs[1])*imgs[1]
    output -= (3-valeurs_couleurs[2])*imgs[2]
    return output
    
def milieu(x):
    return (x[0]+0.5,x[1]+0.5)

def translation1(x):
    if x[0] <360 and x[1]<640:
        return (-(360-x[0]),-(640-x[1]))
    if x[0] <360 and x[1]>640:
        return (-(360-x[0]),x[1]-640)
    if x[0] >360 and x[1]<640:
        return (x[0]-360,-(640-x[1]))
    if x[0]>360 and x[1]>640:
        return (x[0]-360,x[1]-640)

def inversionbase(x):
    return (-x[0],-x[1])

#on combine les 3 fonctions pour transformer les coordonnées x,y en su,sv

def invcoord(x):
    a=milieu(x)
    b=translation1(a)
    return inversionbase(b)
    
def trouverPaires(tabID, valeur):
    S = np.where(tabID[:,:,1] == valeur)
    #renvoie les coordonnées des identifiants égaux à valeur
    #S[0] = [x1, x2, x3, ...] (S[0] est un numpy array)
    #S[1] = [y1, y2, y3, ...] (S[1] est un numpy array)
    L = []
    for x in range(len(S[0])): #reformattage de S
        #L.append(invcoord([S[0][x], S[1][x]]))
        L.append([S[0][x], S[1][x]])
    return L
    #L = [ [x1, y1], [x2, y2], [x3, y3], ... ]
    #chaque xi, yi est un int
    
def ppdist(p1,p2):
    #distance focale
    f=0.05
    #facteur d'agrandissement
    k=20.9
    #coordonnées du centre optique
    cu=0.0
    cv=0.0
    #non orthogonalité des lignes?
    suv=0
    
    #matrice 1 et 2
    A=np.array([[k,suv,cu],[0.0,k,cu],[0.0, 0.0, 1.0]])
    B=np.array([[f, 0.0, 0.0, 0.0],[0.0, f, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    AB=np.dot(A,B)
    
    # NR1---------matrice de rotation (axe y) et de translation
    theta=20*np.pi/180
    tx=0.5
    ty=0.0
    tz=-0.15
    
    RT=np.array([[np.cos(theta),0.0, np.sin(theta),tx],[0.0, 1.0, 0.0,ty],[-np.sin(theta), 0.0, np.cos(theta),tz],[0.0, 0.0, 0.0, 1.0]])
    
    #produit matriciel
    ABRT=np.dot(AB,RT)
    
    C=np.linalg.pinv(ABRT)
    
    #obtenir les cordonnées en 3D:
    m=p1[0]
    n=p1[1]
    s=1
    
    plan=np.array([m,n,s])
    
    space=np.dot(C,plan)
    space1=space/space[3]
    
    # NR2---------matrice de rotation (axe y) et de translation
    theta2=(-20)*np.pi/180
    tx2=-0.5
    ty=0.0
    tz=-0.15
    
    RT2=np.array([[np.cos(theta2),0.0, np.sin(theta2),tx2],[0.0, 1.0, 0.0,ty],[-np.sin(theta2), 0.0, np.cos(theta2),tz],[0.0, 0.0, 0.0, 1.0]])
    
    #produit matriciel
    ABRT2=np.dot(AB,RT2)
    
    C2=np.linalg.pinv(ABRT2)
    
    #obtenir les cordonnées en 3D:
    m2=p2[0]
    n2=p2[1]
    s2=1
    
    plan2=np.array([m2,n2,s2])
    
    space2=np.dot(C2,plan2)
    space12=space2/space2[3]
    #____________________________________________________________________________________________
    
    #coordonnées du centre optique 1 et 2
    tx=0.5
    tx2=-0.5
    ty=0.0
    tz=-0.15
    
    #coordonnée du centre optique utiles pour le vect directeur
    c1=np.array((tx,ty,tz))
    c2=np.array((tx2,ty,tz))
    
    #cordonnées des 2 points UN EXEMPLE
    # space1=np.array[ 1.69705175],
    #        [-1.31288453],
    #        [ 1.14804771],
    #        [ 1.        ]))
    # space12=np.array(([-1.69499111],
    #        [ 1.31421291],
    #        [ 1.15676431],
    #        [ 1.        ]))
    
    #vecteur en 3D (pas 4D) pour produit vectoriel
    
    spaceA=np.array((space1[0],space1[1],space1[2]))
    spaceB=np.array((space12[0],space12[1],space12[2]))
    
    #vecteur directeur Cam1 et Cam2
    
    v1=spaceA-c1
    v2=spaceB-c2
    
#    #__________________________________________________________________________________________
#    
#    #produit vectoriel
#    n= np.cross(v1,v2)
#    
#    #vecteur P1P2 obtenu à partir des 2 points dans l'espace
#    
#    p1p2=np.array((space12[0]-space1[0],space12[1]-space1[1],space12[2]-space1[2]))
#    
#    
#    #point D, plus petite distance entre les deux droites (0 si intersection)
#    
#    D=abs((n[0]*p1p2[0]+n[1]*p1p2[1]+n[2]*p1p2[2])/(np.sqrt((n[0])**2+(n[1])**2+(n[2])**2)))
    return (spaceA, v1, spaceB, v2)
    
def produitScalaire(U, V):
    return float(U[0])*float(V[0]) + float(U[1])*float(V[1]) + float(U[2])*float(V[2])
    
def normeCarre(U):
    return produitScalaire(U, U)
    
#A = np.array((1.0, 1.0, 1.0))
#B = np.array((2.0, -1.0, 4.0))
#U = np.array((1.0, 5.0, -3.0))
#V = np.array((5.0, 3.0, -4.0))

def pointMilieu((A, U, B, V)):
    BA = np.array((A[0]-B[0], A[1]-B[1], A[2]-B[2]))
    t2 = (produitScalaire(U, BA) * produitScalaire(U, V) - produitScalaire(V, BA) * normeCarre(U)) / (produitScalaire(U, V)**2-normeCarre(U)*normeCarre(V))
    t = (produitScalaire(U, BA) * normeCarre(V) - produitScalaire(V, BA) * produitScalaire(U, V)) / (produitScalaire(U, V)**2-normeCarre(U)*normeCarre(V))
    M = A + U*t
    M2 = B + V*t2
    P = 0.5*(M+M2)
    return P
    
def zonecom(SImg):
    global delim
    CImg = [[], []]
    for i in range(len(SImg)):
        for j in range(len(SImg[i])):
            #crop_SImg[i][j] = SImg[i][j][delim[2][1]:delim[1][1],delim[2][0]:delim[1][0]]
            CImg[i].append(SImg[i][j][delim[1][1]:delim[2][1], delim[1][0]:delim[2][0]])
    return CImg



def milieu(id):
    nomid = [[],[]]
    groupid = [[],[]]
    for i in range (id.shape[2]):
        for k in range(id.shape[0]):
            for j in range (id.shape[1]):
                if id[k][j][i] not in nomid[i]:
                    nomid[i].append(id[k][j][i])
                    groupid[i].append([id[k][j][i],k,j])  #on ajoute la liste de l'identifiant et de ses coordonnées associées
                else:
                    a=nomid[i].index(id[k][j][i])
                    groupid[i][a].append([id[k][j][i],k,j])
    
    #voisin de A
    
    for i in range(len(groupid)):
        for j in range(len(groupid[i])):
            for k in range (len(groupid[i][j])):
                a=[groupid[i][j][k][1] , groupid[i][j][k][2]]  #on prend les coordonées d'un point et si elles sont trop éloignées des cordonnées des autres points de la zone, on le sort de la zone
                identif = groupe[i][j][0]
                
                #savoir si le point est dans un coin
                
                
                if a[0]=0 and a[1]=0: #dans le coin haut gauche
                    if id[1][0] != identif or id[0][1] != identif or id[1][1] != identif:
                        b=groupid[i][j]
                        groupid[i].pop(j)  
                        groupid.append([b])
                        
                if a[0]=0 and a[1]=id.shape[1]: #dans le coin haut droit 
                    if id[0][id.shape[1]-1] != identif or id[1][id.shape[1]-1] != identif or id[1][id.shape[1]] != identif:
                        b=groupid[i][j]
                        groupid[i].pop(j)  
                        groupid.append([b])
                        
                if a[0]=id.shape[0] and a[1]=0: #dans le coin bas gauche
                    if id[id.shape[0]-1][0] != identif or id[id.shape[0]-1][1] != identif or id[id.shape[0]][1] != identif:
                        b=groupid[i][j]
                        groupid[i].pop(j)  
                        groupid.append([b])
                        
                if a[0]=id.shape[0] and a[1]=id.shape[1]: #dans le coin bas droit 
                    if id[id.shape[0]][id.shape[1]] != identif or id[id.shape[0]-1][id.shape[1]-1] != identif or id[id.shape[0]-1][id.shape[1]] != identif:
                        b=groupid[i][j]
                        groupid[i].pop(j)  
                        groupid.append([b])
                        
                #savoir si le coin est sur un bord strict
                            
                if a[0]=0 and a[1] != 0 and a[1] != id.shape[1]: #sur le bord haut de l'image, pas dans les coins
                    c=0
                    for m in [a[0],a[0]+1]:   
                        for n in [a[1]-1,a[1],a[1]+1]:
                            if id[m][n] = identif:
                                c +=1
                            if c <= 1: # s'il n'y a pas de voisin avec le même identifiant, on l'enlève
                                b=groupid[i][j]
                                groupid[i].pop(j)  
                                groupid.append([b])
                            
                            
                if a[0]=id.shape[0] and a[1] != 0 and a[1] != id.shape[1] :
                    c=0
                    for m in [a[0]-1,a[0]]:   #sur le bord bas de l'image, pas dans les coins
                        for n in [a[1]-1,a[1],a[1]+1]:
                            if id[m][n] = identif:
                                c +=1
                            if c <= 1: # s'il n'y a pas de voisin avec le même identifiant, on l'enlève
                                b=groupid[i][j]
                                groupid[i].pop(j)  
                                groupid.append([b])
                            
                if a[1]=0 and a[0] != 0 and a[0] != id.shape[0]: 
                    c=0
                    for m in [a[0]-1,a[0],a[0]+1]:   #sur le bord gauche de l'image, pas dans les coins
                        for n in [a[1],a[1]+1]:
                            if id[m][n] = identif:
                                c +=1
                            if c <= 1: # s'il n'y a pas de voisin avec le même identifiant, on l'enlève
                                b=groupid[i][j]
                                groupid[i].pop(j)  
                                groupid.append([b])
                                
                if a[1]=id.shape[1] and a[0] != 0 and a[0] != id.shape[0]:
                    c=0
                    for m in [a[0]-1,a[0],a[0]+1]:   #sur le bord droit de l'image, pas dans les coins
                        for n in [a[1]-1,a[1]]:
                            if id[m][n] = identif:
                                c +=1
                            if c <= 1: # s'il n'y a pas de voisin avec le même identifiant, on l'enlève
                                b=groupid[i][j]
                                groupid[i].pop(j)  
                                groupid.append([b])
    
    #la plus grosse zone
    
    for i in range(len(groupid)):
        for j in range(len(groupid[i])):
            for k in range(len(groupid[i])):
                if groupid[i][j][0] == groupid[i][k][0]: #pour 2 zones correspondants au même identifiant
                    if len(groupeid[i][j])>len(groupid[i][k]):
                        groupeid[i].pop(k) #on enlève la plus petite zone


    #definir le barycentre
    
    BImg = [[],[]]
    Sx=0
    Sy=0
    for i in range (len(groupid)):
        Sx=0
        Sy=0
        for j in range (len(groupid[i])):
            Sx=0
            Sy=0
            for k in range(len(groupid[i][j])
                Sx += groupid[i][j][k][1]
                Sy += groupid[i][j][k][2]
            
            xg=Sx/len(groupid[i][j])  #coordonnées du barycentre de chaque zone
            yg=Sy/len(groupid[i][j])
        
            BImg[i].append([groupid[i][j][k][0],xg,yg])  #on obtient la liste des barycentres de chaque zone pour chaque image
    
    return BImg 

#fonction appelée quand la souris envoie un événement
def mouseCallBack(event,x,y,flags,param):
    global delim, img
    if event == cv2.EVENT_LBUTTONDBLCLK: #si in double clique
        if(delim[0] == 0): #1er point
            print("1er point:", 2*x, 2*y)
            delim[1][0], delim[1][1] = 2*x, 2*y
            delim[0] += 1
        else: #2eme point
            print("2eme point:", 2*x, 2*y)
            delim[2][0], delim[2][1] = 2*x, 2*y
            delim[0] = 0

cv2.namedWindow("GCam") #création de la fenêtre d'affichage
cv2.namedWindow("res") #création de la fenêtre d'affichage
w, h = (1280, 720)
cv2.setMouseCallback('GCam', mouseCallBack)

Frame1 = np.zeros((h, w, 3), dtype=np.int16)
Frame2 = np.zeros((h, w, 3), dtype=np.int16)
Side1 = 0
Side2 = 1
rval1 = False
rval2 = False
flip1 = False
flip2 = False

CInst = [ [cv2.VideoCapture(0), Frame1, Side1, rval1, flip1], 
          [cv2.VideoCapture(2), Frame2, Side2, rval2, flip2] ]
SImg = [[], []]

for camera_instance in CInst: #initialisation des instances des caméras
    camera_instance[0].set(3, w)
    camera_instance[0].set(4, h)
    
    if camera_instance[0].isOpened():
        camera_instance[3], camera_instance[1] = camera_instance[0].read()
    else:
        camera_instance[3] = False

running = 0
flip = False
numImg = 0
threshold = 100

r = np.zeros((h, w, 3), dtype=np.int16)
v = np.zeros((h, w, 3), dtype=np.int16)
b = np.zeros((h, w, 3), dtype=np.int16)

delim = [1, [0,0], [0,0]]
img = 0

while all(cam[3] == True for cam in CInst): #tant qu'on récupère des images des deux caméras
    e1 = cv2.getTickCount() #debut mesure du temps
    takeFrame(CInst, flip)
    img = np.concatenate((cv2.resize(CInst[0][1], (0,0), fx=0.5, fy=0.5), cv2.resize(CInst[1][1], (0,0), fx=0.5, fy=0.5)), axis=1)
#        grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) #on en fait une copie en NB
    cv2.imshow("res", traitement_NB(img))
    e2 = cv2.getTickCount() #fin mesure temps execution
    time = (e2 - e1)/ cv2.getTickFrequency() #calcul du temps écoulé
    draw_str(img, (20, 20), 'FPS=%d' % int(1.0/time)) #on affiche le nombre de FPS
    draw_str(img, (20,40), 'Seuil=%d' % threshold)
    img = cv2.rectangle(img, (delim[1][0]/2, delim[1][1]/2), (delim[2][0]/2, delim[2][1]/2), (0, 255, 0))
    img = cv2.rectangle(img, (delim[1][0]/2 + w/2, delim[1][1]/2), (delim[2][0]/2 + w/2, delim[2][1]/2), (0, 255, 255))
    cv2.imshow("GCam", img) #on affiche l'image
    
    key = cv2.waitKey(2) #on attend qu'une touche soit pressée (2ms timeout)
    if key == ord('s'): #on enregistre l'image courante avec la touche S
        draw_str(traitement_NB(CInst[0][1]), (20, 50), 'SAUVEGARDE')
        now = str(datetime.datetime.now())
        now = now.replace(":",".")
        now = now.replace(" ",".")
        cv2.imwrite(cwd + "/sauvegarde_A_" + now + ".png", traitement_NB(CInst[0][1]))
        print("Image enregistrée sous :" + cwd + "/sauvegarde_A_" + now + ".png")
        
        draw_str(traitement_NB(CInst[1][1]), (20, 50), 'SAUVEGARDE')
        now = str(datetime.datetime.now())
        now = now.replace(":",".")
        now = now.replace(" ",".")
        cv2.imwrite(cwd + "/sauvegarde_B_" + now + ".png", traitement_NB(CInst[1][1]))
        print("Image enregistrée sous :" + cwd + "/sauvegarde_B_" + now + ".png")
        
    elif key == ord('l'): #On charge les images enregistrees
        for i in range(1,9):
            SImg[0].append(cv2.imread('save/original/A_0' + str(i) + '.png',cv2.IMREAD_COLOR))
            SImg[1].append(cv2.imread('save/original/B_0' + str(i) + '.png',cv2.IMREAD_COLOR))
        print("Chargement terminé.")
        
    elif key == ord('t'): #appui sur t sauvegarde une image des deux cameras
        takePicture(CInst, SImg, numImg, 0)
        takePicture(CInst, SImg, numImg, 1)
        numImg += 1
        print("Paire d'images n°" + str(numImg) + " capturée.")
        
    elif key == ord('d'): #decodage NB
    
        #CROP Images
        CImg = zonecom(SImg)
        print(len(CImg))
        print(len(CImg[0]))
        print(CImg[0][0].shape)
    
        #identifiant composé des images verticales puis horizontales (ex: vvvvvhhhhhh)
        ID = tabID_NB(CImg)
        
        now = str(datetime.datetime.now()) #Date du jour pour le nom des fichiers
        now = now.replace(":",".")
        now = now.replace(" ",".")
        
        img1 = ID[:, :, 0] #Images associees au tableau d'identifiants
        img2 = ID[:, :, 1]
        img1 = img1.astype(np.uint8)
        img2 = img2.astype(np.uint8)
        img1 =  cv2.cvtColor(img1,cv2.COLOR_GRAY2RGB)
        img2 =  cv2.cvtColor(img2,cv2.COLOR_GRAY2RGB)
        
        cv2.imshow('image_ID_A',img1)
        cv2.imshow('image_ID_B',img2)
        cv2.waitKey(0)
        
        assert(img1.shape == img2.shape)
        
        bar = milieu(ID) #Calcul des barycentres des zones

        #Association des paires des barycentres
        paires = []
        for x in range(len(bar)):
                paires.append(trouverPaires(bar, bar[x]))
        
        #Calcul des points 3D et enregistrement
        file = open("3DPoints_" + now + ".txt", "w")
        file.write("ply") #Creation du fichier 3D, format PLY
        file.write("format ascii 1.0")
        file.write("element vertex " + str(len(paires)))
        file.write("property float32 x")
        file.write("property float32 y")
        file.write("property float32 z")
        file.write("end_header")
        
        point = [0, 0, 0]
        for paire in paires:
            point = pointMilieu(ppdist(paire[0], paire[1]))
            file.write(str(point[0]) + " " + str(point[1]) + " " + str(point[2]))

        file.close()
        
        
        #cv2.imwrite(cwd + "/tabID_A_" + now + ".png", img1)
        #print("Image enregistrée sous :" + cwd + "/tabID_A_" + now + ".png")
        #cv2.imwrite(cwd + "/tabID_B_" + now + ".png", img2)
        #print("Image enregistrée sous :" + cwd + "/tabID_B_" + now + ".png")
        
#        print(bin(ID[227,158,0]))
#        L = trouverPaires(ID, ID[227,158,0])
#        
#        L.sort(key = itemgetter(1))
#        groups = groupby(L, itemgetter(1))
#        print([[item for item in data] for (key, data) in groups])
        
        print("Fin du decodage")
            
        #P =  np.zeros((h, w, 1), dtype=object)
        #for x in range(w):
        #    for y in range(h):
        #        L = list(trouverPaires(ID, ID[x][y][0]))
                #L.insert(0, invcoord([x, y]))
                #P[x][y] = L #dans le systeme de coordonnees su, sv
#                print(len(L))
        
        #print(P.shape)
            #choisir les deux points dont les coordonnées sont les plus proches et les passer dans la fonction ppdistance mais du coup avec les coordonnées du point
       
        #P.insert(0, [x,y])
        #retourne la liste des coordonnées des identifiants égaux à 0b1101
        #print(P)
        
    elif key == ord('c'): #on efface les images enregistrees
        SImg = [[], []]
        numImg = 0
        print("Images effacées")
        
    elif key == ord('p'): #on augmente le seuil
        threshold += 1
        
    elif key == ord('m'): #on diminue le seuil
        threshold -= 1
        
    elif key == ord('f'):
        flipSide()
        
    elif key == 27: # Sortie avec ESC
        break

for camera_instance in CInst:
    camera_instance[0].release() #on libère le flux vidéo
cv2.destroyAllWindows()
