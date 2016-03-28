# -*- coding: utf-8 -*-
"""
@author: Branislav Cogic
"""

import numpy as np
import math
import cv2 as cv2
import matplotlib.pyplot as plt
import scipy as sc
import collections
from scipy.spatial import distance
from sklearn.cluster import KMeans

def nadji_konture_determinante(pogodne_konture, frame_bin, frame):
    potencijalne_granice_determinante = []
    rbr = 0
    for contour in pogodne_konture:
        rotated_contour = rotate_contour(contour, frame_bin)
        rbr = rbr+1
        x,y,w,h = cv2.boundingRect(rotated_contour) #koordinate i velicina granicnog pravougaonika
        #area = cv2.contourArea(contour)
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(frame, [box], 0, (0,0,255), 2)
        rect = cv2.minAreaRect(rotated_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(frame, [box], 0, (0,255,255), 2)
        if (h/w > 5) and h>30 and w>5:
            potencijalne_granice_determinante.append(contour)
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(frame, [box], 0, (0,255,0), 2)
            
    print "Broj potencijalnih granica pre mergovanja = ", len(potencijalne_granice_determinante)
    potencijalne_granice_determinante = merge_regions(potencijalne_granice_determinante)
    potencijalne_granice_determinante = merge_intersected(potencijalne_granice_determinante)
    print "Broj potencijalnih granica nakon mergovanja = ", len(potencijalne_granice_determinante)
    if len(potencijalne_granice_determinante) != 2:
        return
        
    # NALAZENJE PRAVOUGAONIKA KOJI PREDSTALJVA REGION DETERMINANTE
    granica_determinante = np.ndarray((len(potencijalne_granice_determinante[0])+len(potencijalne_granice_determinante[1]), 1,2), dtype=np.int32)
    granica_determinante = np.concatenate((potencijalne_granice_determinante[0], potencijalne_granice_determinante[1]))    
    rectDet = cv2.minAreaRect(granica_determinante)
    boxDet = cv2.boxPoints(rectDet)
    boxDet = np.int0(boxDet)
    cv2.drawContours(frame, [boxDet], 0, (255,0,0), 2)
    konture_determinante = []
    for contour in pogodne_konture:
        #rotated_contour = rotate_contour(contour, frame_bin)
        x,y,w,h = cv2.boundingRect(rotated_contour) #koordinate i velicina granicnog pravougaonika
        #area = cv2.contourArea(contour)
        rectCont = cv2.minAreaRect(contour)
        indicator, vertices = cv2.rotatedRectangleIntersection(rectDet, rectCont)
        print indicator
        if indicator>0:
            konture_determinante.append(contour)
            
    # Rotiranje kontura unutar determinante oko centra determinante
    centerDet, sizeDet, angleDet = cv2.minAreaRect(granica_determinante)
    print "Centar determinante je u    :   ", centerDet
    rotirane_konture = []
    index = 0
    centri_x = []
    for contour in konture_determinante:
        xt,yt,h,w = cv2.boundingRect(contour)
        rectCont = cv2.minAreaRect(contour)
        centerCont, sizeCont, angleCont = rectCont
        region_points = []
        for i in range (xt,xt+h):
            for j in range(yt,yt+w):
                dist = cv2.pointPolygonTest(contour,(i,j),False)
                if dist>=0 and frame_bin[j,i]==255: # da li se tacka nalazi unutar konture?
                    region_points.append([i,j])
        cx,cy = centerDet
#        height, width = sizeCont
#        if width<height:
#            angle+=90
        # Rotiranje svake tačke regiona oko centra rotacije
        alpha = np.pi/2 - abs(np.radians(angleDet))
        region_points_rotated = np.ndarray((len(region_points), 1,2), dtype=np.int32)
        for i, point in enumerate(region_points):
            x = point[0]
            y = point[1]
            
            #TODO 1 - izračunati koordinate tačke nakon rotacije
            rx = np.sin(alpha)*(x-cx) - np.cos(alpha)*(y-cy) + cx
            ry = np.cos(alpha)*(x-cx) + np.sin(alpha)*(y-cy) + cy
            region_points_rotated[i] = [rx,ry]
        rectRotatedCont = cv2.minAreaRect(region_points_rotated)
        boxRotatedCont = cv2.boxPoints(rectRotatedCont)
        boxRotatedCont = np.int0(boxRotatedCont)
        cv2.drawContours(frame, [boxRotatedCont], 0, (255,255,255), 2)
        rotirane_konture.append(region_points_rotated)
        
    print "Broj kontura u determinanti = ", len(konture_determinante)
    
    k_means = KMeans(n_clusters=math.sqrt(len(konture_determinante)-2))
    k_means.fit(konture_determinante)
def merge_intersected(contours):
    ret_val = []
    merged_index = [] #lista indeksa kontura koje su već spojene sa nekim
    
    for i,contour1 in enumerate(contours): #slova
        if i in merged_index:
            continue
        rect1 = cv2.minAreaRect(contour1)
        for j,contour2 in enumerate(contours): #kukice
            if j in merged_index or i == j:
                continue
            rect2 = cv2.minAreaRect(contour2)
            
            #TODO 2 - izvršiti spajanje kukica iznad slova
            #spajanje dva niza je moguće obaviti funkcijom np.concatenate((contour1,contour2))
            indicator, vertices = cv2.rotatedRectangleIntersection(rect1, rect2)
            if indicator>0:
                #spajanje kontura
                ret_val.append(np.concatenate((contour1,contour2)))
                merged_index.append(i)
                merged_index.append(j)
    #svi regioni koji se nisu ni sa kim spojili idu u listu kontura, bez spajanja
    for idx,contour in enumerate(contours):
        if idx not in merged_index:
            ret_val.append(contour)
        
    return ret_val
    
def rotate_contour(contour, frame_bin):
    center, size, angle = cv2.minAreaRect(contour)
    xt,yt,h,w = cv2.boundingRect(contour)
    region_points = []
    for i in range (xt,xt+h):
        for j in range(yt,yt+w):
            dist = cv2.pointPolygonTest(contour,(i,j),False)
            if dist>=0 and frame_bin[j,i]==255: # da li se tacka nalazi unutar konture?
                region_points.append([i,j])
    cx,cy = center
    height, width = size
    if width<height:
        angle+=90
    # Rotiranje svake tačke regiona oko centra rotacije
    alpha = np.pi/2 - abs(np.radians(angle))
    region_points_rotated = np.ndarray((len(region_points), 1,2), dtype=np.int32)
    for i, point in enumerate(region_points):
        x = point[0]
        y = point[1]
        
        #TODO 1 - izračunati koordinate tačke nakon rotacije
        rx = np.sin(alpha)*(x-cx) - np.cos(alpha)*(y-cy) + cx
        ry = np.cos(alpha)*(x-cx) + np.sin(alpha)*(y-cy) + cy
        region_points_rotated[i] = [rx,ry]
    return region_points_rotated

def izaberi_konture(contour_borders):
    pogodne_konture = []
    ### PRVO NADJEMO USPRAVNE LINIJE
    #for contour in contour_borders:
    #    x,y,w,h = cv2.boundingRect(contour) #koordinate i velicina granicnog pravougaonika
    #    area = cv2.contourArea(contour)
        
    
    for contour in contour_borders:
        x,y,w,h = cv2.boundingRect(contour) #koordinate i velicina granicnog pravougaonika
        area = cv2.contourArea(contour)
        
        # PRONACI REGIONE KOJI PREDSTAVLJAJU USPRAVNE LINIJE DETERMINANTE
        if (h/w > 5):
            pogodne_konture.append(contour)
            print "USPRAVNA LAJNA"
            
        # PRONACI REGIONE IZMEDJU LINIJA 
        
        # ZA SLOVA
        #elif area > 20 and area < 10000 and h < 400 and h > 10 and w > 10 and w<400:
        #    pogodne_konture.append(contour)
        # ZA GRANICE DETERMINANTE
        #elif area > 10 and h>50 and h< 400 and w<20:
        #    pogodne_konture.append(contour)
    pogodne_konture = merge_regions(pogodne_konture)
    return pogodne_konture
    """
    for contour in contour_borders:
        i = i + 1
        print "Kontura br ", i, ": ", contour
        x,y,w,h = cv2.boundingRect(contour) #koordinate i velicina granicnog pravougaonika
        area = cv2.contourArea(contour)
        
        # PRONACI REGIONE KOJI PREDSTAVLJAJU USPRAVNE LINIJE DETERMINANTE
        if (h/w > 10):
            region = frame_bin[y:y+h+1,x:x+w+1];
            regions_dic[x] = cv2.resize(region,(28,28), interpolation = cv2.INTER_LANCZOS4)
            cv2.drawContours(frame, contour, -1, (0,0,255), 1)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        # PRONACI REGIONE IZMEDJU LINIJA       
        
        
        
        # ZA SLOVA
        if area > 20 and area < 10000 and h < 400 and h > 10 and w > 10 and w<400:
            # kopirati [y:y+h+1, x:x+w+1] sa binarne slike i smestiti u novu sliku
            # označiti region pravougaonikom na originalnoj slici (image_orig) sa rectangle funkcijom
            region = frame_bin[y:y+h+1,x:x+w+1];
            regions_dic[x] = cv2.resize(region,(28,28), interpolation = cv2.INTER_LANCZOS4)
            cv2.drawContours(frame, contour, -1, (0,0,255), 1)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        # ZA GRANICE DETERMINANTE
        elif area > 10 and h>50 and h< 400 and w<20:
            region = frame_bin[y:y+h+1,x:x+w+1];
            regions_dic[x] = cv2.resize(region,(28,28), interpolation = cv2.INTER_LANCZOS4)
            cv2.drawContours(frame, contour, -1, (0,0,255), 1)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    """
def merge_regions(contours):
    '''Funkcija koja vrši spajanje kukica i kvačica sa osnovnim karakterima
    Args:
        contours: skup svih kontura (kontura - niz tacaka bele boje)
    Return:
        ret_val: skup kontura sa spojenim kukicama i kvacicama'''
    ret_val = []
    merged_index = [] #lista indeksa kontura koje su već spojene sa nekim
    
    for i,contour1 in enumerate(contours): #slova
        if i in merged_index:
            continue
        min_x1 = min(contour1[:,0,0])
        max_x1 = max(contour1[:,0,0])
        min_y1 = min(contour1[:,0,1])
        max_y1 = max(contour1[:,0,1])
        for j,contour2 in enumerate(contours): #kukice
            if j in merged_index or i == j:
                continue
            min_x2 = min(contour2[:,0,0])
            max_x2 = max(contour2[:,0,0])
            min_y2 = min(contour2[:,0,1])
            max_y2 = max(contour2[:,0,1])
            
            #TODO 2 - izvršiti spajanje kukica iznad slova
            #spajanje dva niza je moguće obaviti funkcijom np.concatenate((contour1,contour2))
            if (min_y1<min_y2) and (max_y1>max_y2) and (min_x1<min_x2) and (max_x1>max_x2):
                #spajanje kontura
                ret_val.append(np.concatenate((contour1,contour2)))
                merged_index.append(i)
                merged_index.append(j)
    #svi regioni koji se nisu ni sa kim spojili idu u listu kontura, bez spajanja
    for idx,contour in enumerate(contours):
        if idx not in merged_index:
            ret_val.append(contour)
        
    return ret_val
    
def crtaj_konture(contours, frame, frame_bin):
    regions_dic = {}
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour) #koordinate i velicina granicnog pravougaonika
        region = frame_bin[y:y+h+1,x:x+w+1];
        #regions_dic[x] = cv2.resize(region,(28,28), interpolation = cv2.INTER_LANCZOS4)
        cv2.drawContours(frame, contour, -1, (0,0,255), 2)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

def regioni_od_interesa(frame, frame_bin):
    img, contour_borders, hierarchy = cv2.findContours(frame_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = izaberi_konture(contour_borders)
    contours = merge_regions(contours)
    contours = merge_intersected(contours)
    #crtaj_konture(contours, frame, frame_bin)
    sorted_regions = [] # lista sortiranih regiona po x osi (sa leva na desno)
    #contours2, contour_angles, contour_centers, contour_sizes = centri_uglovi_velicine(contours, frame_bin)
    #rotated_contours = []
    #rotated_contours = rotate_regions(contours2, contour_angles, contour_centers, contour_sizes)
    determinant_contours = nadji_konture_determinante(contours, frame_bin, frame)
    #crtaj_konture(rotated_contours, frame, frame_bin)
    
    '''
    for contour in rotated_contours:
        i = i + 1
        print "Kontura br ", i, ": ", contour
        x,y,w,h = cv2.boundingRect(contour) #koordinate i velicina granicnog pravougaonika
        area = cv2.contourArea(contour)
        
        # PRONACI REGIONE KOJI PREDSTAVLJAJU USPRAVNE LINIJE DETERMINANTE
        if (h/w > 10):
            region = frame_bin[y:y+h+1,x:x+w+1];
            regions_dic[x] = cv2.resize(region,(28,28), interpolation = cv2.INTER_LANCZOS4)
            cv2.drawContours(frame, contour, -1, (0,0,255), 1)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        # PRONACI REGIONE IZMEDJU LINIJA       
        
        
        
        # ZA SLOVA
        if area > 20 and area < 10000 and h < 400 and h > 10 and w > 10 and w<400:
            # kopirati [y:y+h+1, x:x+w+1] sa binarne slike i smestiti u novu sliku
            # označiti region pravougaonikom na originalnoj slici (image_orig) sa rectangle funkcijom
            region = frame_bin[y:y+h+1,x:x+w+1];
            regions_dic[x] = cv2.resize(region,(28,28), interpolation = cv2.INTER_LANCZOS4)
            cv2.drawContours(frame, contour, -1, (0,0,255), 1)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        # ZA GRANICE DETERMINANTE
        elif area > 10 and h>50 and h< 400 and w<20:
            region = frame_bin[y:y+h+1,x:x+w+1];
            regions_dic[x] = cv2.resize(region,(28,28), interpolation = cv2.INTER_LANCZOS4)
            cv2.drawContours(frame, contour, -1, (0,0,255), 1)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    '''
    
    #sorted_regions_dic = collections.OrderedDict(sorted(regions_dic.items()))
    #sorted_regions = sorted_regions_dic.values()
    return frame, sorted_regions
    
# Rotiranje regiona
def rotate_regions(contours,angles,centers,sizes):
    '''Funkcija koja vrši rotiranje regiona oko njihovih centralnih tačaka
    Args:
        contours: skup svih kontura [kontura1, kontura2, ..., konturaN]
        angles:   skup svih uglova nagiba kontura [nagib1, nagib2, ..., nagibN]
        centers:  skup svih centara minimalnih pravougaonika koji su opisani 
                  oko kontura [centar1, centar2, ..., centarN]
        sizes:    skup parova (height,width) koji predstavljaju duzine stranica minimalnog
                  pravougaonika koji je opisan oko konture [(h1,w1), (h2,w2), ...,(hN,wN)]
    Return:
        ret_val: rotirane konture'''
    ret_val = []
    for idx, contour in enumerate(contours):
        angle = angles[idx]
        cx,cy = centers[idx]
        height, width = sizes[idx]
        if width<height:
            angle+=90
            
        # Rotiranje svake tačke regiona oko centra rotacije
        alpha = np.pi/2 - abs(np.radians(angle))
        region_points_rotated = np.ndarray((len(contour), 1,2), dtype=np.int32)
        for i, point in enumerate(contour):
            x = point[0]
            y = point[1]
            
            #TODO 1 - izračunati koordinate tačke nakon rotacije
            rx = np.sin(alpha)*(x-cx) - np.cos(alpha)*(y-cy) + cx
            ry = np.cos(alpha)*(x-cx) + np.sin(alpha)*(y-cy) + cy
            region_points_rotated[i] = [rx,ry]
        ret_val.append(region_points_rotated)
    #print ret_val
    return ret_val
    
def centri_uglovi_velicine(contour_borders, frame_bin):
    contours = []
    contour_angles = []
    contour_centers = []
    contour_sizes = []
    for contour in contour_borders:
        center, size, angle = cv2.minAreaRect(contour)
        xt,yt,h,w = cv2.boundingRect(contour)
        region_points = []
        for i in range (xt,xt+h):
            for j in range(yt,yt+w):
                dist = cv2.pointPolygonTest(contour,(i,j),False)
                if dist>=0 and frame_bin[j,i]==255: # da li se tacka nalazi unutar konture?
                    region_points.append([i,j])
        contour_centers.append(center)
        contour_angles.append(angle)
        contour_sizes.append(size)
        contours.append(region_points)
    return contours, contour_angles, contour_centers, contour_sizes