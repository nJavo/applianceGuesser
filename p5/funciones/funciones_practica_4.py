import os
import sys
import numpy as np
from scipy.fftpack import fft, fftshift, ifft
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
from datetime import datetime


from ctypes import *
from funciones.dwfconstants import *
import time


from funciones.util import plot_confusion_matrix
from funciones.funciones_practica_2 import nombres_electrodomesticos

# importar funciones del módulo de aprendizaje sklearn
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier as RF, AdaBoostClassifier as ABC

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier as RF, AdaBoostClassifier as ABC
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel

#--------------------------------------------------------

# COMPLETAR 
# AGREGAR FUNCIONES DE P4 QUE FALTEN, CORREGIR IMPORTS SI ES NECESARIO
# 







#---------------------------------------------------------


def clasificar(clasificador, caracteristicas, etiquetas, subconjuntos ):
    '''
    Recibe un clasificador ya creado, las caracteristicas, 
    las etiquetas y los indicadores de subconjuntos. 
    Devuelve las tasas de acierto y las predicciones
    '''
    
    # Si se usa una sóla característica, forzar que sea un vector columna
    if caracteristicas.ndim == 1:
        caracteristicas = caracteristicas[:,np.newaxis]
    
    # cantidad de subconjuntos
    cantidad_subconjuntos = len(np.unique(subconjuntos))
    
    # inicializar arrays para guardar los resultados
    accuracies = np.empty((cantidad_subconjuntos))
    y_predictions = np.empty((caracteristicas.shape[0]), dtype=np.uint8)

    start = datetime.now()
    #para cada subconjunto
    for i in range(cantidad_subconjuntos):
        id_subconjunto = np.unique(subconjuntos)[i]
        print('%d/%d fold...\t tiempo: %ds'%(id_subconjunto,cantidad_subconjuntos,(datetime.now()-start).seconds), end='\r', flush=True)

        # separar los datos en entrenamiento y test
        indices_test = np.where(subconjuntos==id_subconjunto)[0]
        indices_train = np.where(subconjuntos!=id_subconjunto)[0]

        X_train = caracteristicas[indices_train,:]
        y_train = etiquetas[indices_train]

        X_test = caracteristicas[indices_test,:]
        y_test = etiquetas[indices_test]

        # entrenar el clasificador
        clf = clasificador # solo un nombre más corto
        clf.fit(X_train,y_train)

        #obtener las predicciones sobre el conjunto de test
        y_pred=clf.predict(X_test)
        # obtener la tasa de acierto
        acc = clf.score(X_test,y_test)

        # guardar predicciones y tasa de acierto para el fold
        accuracies[i]=acc
        y_predictions[indices_test] = y_pred
    
    return accuracies, y_predictions


def mostrar_performance(accuracies, y_predictions, etiquetas):
    print('Acierto medio = {:.2f}'.format(np.mean(accuracies)*100))

    # Graficar non-normalized confusion matrix
    y_pred = y_predictions.astype(int)
    y_test = etiquetas.astype(int)

    plot_confusion_matrix(y_test, y_pred, 
                          classes=nombres_electrodomesticos,
                          title='Matriz de confusión')

# VI IMAGE
#
# Adaptado de:
# [1] Gao, Jingkun, et al. "Plaid: a public dataset of high-resoultion 
# electrical appliance measurements for load identification research: 
# demo abstract." proceedings of the 1st ACM Conference on Embedded 
# Systems for Energy-Efficient Buildings. ACM, 2014.
# 


def get_img_from_VI(V, I, width, hard_threshold=False, para=.5):
    '''Get images from VI, hard_threshold, set para as threshold to cut off,5-10
    soft_threshold, set para to .1-.5 to shrink the intensity'''
    
    d = V.shape[0]
    # doing interploation if number of points is less than width*2
    if d<2* width:
        assert False
        newV = np.hstack([V, V[0]])
        newI = np.hstack([I, I[0]])
        oldt = np.linspace(0,d,d+1)
        newt = np.linspace(0,d,2*width)
        I = np.interp(newt,oldt,newI)
        V = np.interp(newt,oldt,newV)
    # get the size resolution of mesh given width
    d_c = (np.amax(I) - np.amin(I)) / width
    d_v = (np.amax(V) - np.amin(V)) / width
    
    #  find the index where the VI goes through in current-voltage axis
    ind_c = np.floor((I-np.amin(I))/d_c).astype(int)
    ind_v = np.floor((V-np.amin(V))/d_v).astype(int)
    ind_c[ind_c==width] = width-1
    ind_v[ind_v==width] = width-1  # ok
    
    Img = np.zeros((width,width))
    
    for i in range(len(I)):
        Img[ind_c[i],width-ind_v[i]-1] += 1 # why backwards?
    
    if hard_threshold:
        Img[Img<para] = 0
        Img[Img!=0] = 1
        return Img
    else:
        return (Img/np.max(Img))**para



def adquirir_dos_canales_con_cuenta_regresiva_v2(cantidad_de_muestras=50000, frecuencia_de_muestreo=25000,
                                              rango_canal_0=10, rango_canal_1=10,
                                              generar_sinusoide=False,
                                              amplitud_sinusoide=1,
                                              salida_digital_ajustable=False,
                                              salida_digital_0=0,
                                              salida_digital_1=0,
                                              voltaje_saturacion=4.5,
                                              delay_en_segundos_hasta_adquisicion_definitiva=7):
    '''
    Adquiere las señales conectadas a los canales ch0 y ch1 del osciloscopio.

    Opcionalmente genera una sinusoide por el primer canal del generador de señales por
    si se quiere enviar esta señal al osciloscopio.

    Devuelve las muestras adquiridas como dos arrays de numpy.
    '''

    # COMPLETAR
    if sys.platform.startswith("win"):
        dwf = cdll.dwf
    elif sys.platform.startswith("darwin"):
        dwf = cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")
    else:
        dwf = cdll.LoadLibrary("libdwf.so")

    global cAvailable, cLost, cCorrupted, fLost, fCorrupted

    # declare ctype variables
    hdwf = c_int()
    sts = c_byte()
    hzAcq = c_double(frecuencia_de_muestreo)
    nSamples = cantidad_de_muestras
    rgdSamples_ch0 = (c_double * nSamples)()
    rgdSamples_ch1 = (c_double * nSamples)()

    cAvailable = c_int()
    cLost = c_int()
    cCorrupted = c_int()
    fLost = 0
    fCorrupted = 0

    # open device----------------------------------------------------------
    print('---------------------------------------------------------------')
    print("Abriendo el dispositivo AD2")
    dwf.FDwfDeviceOpen(c_int(-1), byref(hdwf))

    if hdwf.value == hdwfNone.value:
        szerr = create_string_buffer(512)
        dwf.FDwfGetLastErrorMsg(szerr)
        print(str(szerr.value))
        print("ERROR: No se pudo abrir el dispositivo")
        quit()

    # POWER Vss, Vdd -----------------------------------------------------
    print('Prendiendo fuentes +5,-5')
    # set up analog IO channel nodes
    # enable positive supply
    dwf.FDwfAnalogIOChannelNodeSet(hdwf, c_int(0), c_int(0), c_double(True))
    # set voltage to 5 V
    dwf.FDwfAnalogIOChannelNodeSet(hdwf, c_int(0), c_int(1), c_double(5.0))
    # enable negative supply
    dwf.FDwfAnalogIOChannelNodeSet(hdwf, c_int(1), c_int(0), c_double(True))
    # set voltage to -5 V
    dwf.FDwfAnalogIOChannelNodeSet(hdwf, c_int(1), c_int(1), c_double(-5.0))
    # master enable
    dwf.FDwfAnalogIOEnableSet(hdwf, c_int(True))

    # GENERADOR ---------------------------------------------------------
    if generar_sinusoide:
        print("Generando sinusoide de amplitud {} V".format(amplitud_sinusoide))
        dwf.FDwfAnalogOutNodeEnableSet(hdwf, c_int(0), AnalogOutNodeCarrier, c_bool(True))
        dwf.FDwfAnalogOutNodeFunctionSet(hdwf, c_int(0), AnalogOutNodeCarrier, funcSine)
        dwf.FDwfAnalogOutNodeFrequencySet(hdwf, c_int(0), AnalogOutNodeCarrier, c_double(50))
        dwf.FDwfAnalogOutNodeAmplitudeSet(hdwf, c_int(0), AnalogOutNodeCarrier, c_double(amplitud_sinusoide))
        dwf.FDwfAnalogOutConfigure(hdwf, c_int(0), c_bool(True))

    # DIGITAL IO --------------------------------------------------------
    dwRead = c_uint32()  # para leer las entradas digitales
    # chequear valores
    salida_digital_0 = 0 if salida_digital_0 == 0 else 1
    salida_digital_1 = 0 if salida_digital_1 == 0 else 1
    if salida_digital_ajustable:
        print('Salidas digitales se van a ajustar con la señal del canal 0')
        salida_digital_0 = 0
        salida_digital_1 = 0
    else:
        print('Salidas digitales fijas')
    mascara_salidas_digitales = salida_digital_1 * 2 + salida_digital_0

    # enable output/mask on 8 LSB IO pins, from DIO 0 to 7
    dwf.FDwfDigitalIOOutputEnableSet(hdwf, c_int(0x00FF))
    # set value on enabled IO pins
    dwf.FDwfDigitalIOOutputSet(hdwf, c_int(mascara_salidas_digitales))
    # fetch digital IO information from the device
    dwf.FDwfDigitalIOStatus(hdwf)
    # read state of all pins, regardless of output enable
    dwf.FDwfDigitalIOInputStatus(hdwf, byref(dwRead))

    # print(dwRead as bitfield (32 digits, removing 0b at the front)
    print('Salidas digitales sin ajustar [D1 D0] = [{}]'.format(bin(dwRead.value)[2:].zfill(16)[14:]))

    # PREPARACION DE LA ADQUISICION -------------------------------------
    # set up acquisition
    dwf.FDwfAnalogInChannelEnableSet(hdwf, c_int(0), c_bool(True))
    dwf.FDwfAnalogInChannelRangeSet(hdwf, c_int(0), c_double(rango_canal_0))
    dwf.FDwfAnalogInChannelEnableSet(hdwf, c_int(1), c_bool(True))
    dwf.FDwfAnalogInChannelRangeSet(hdwf, c_int(1), c_double(rango_canal_1))

    dwf.FDwfAnalogInAcquisitionModeSet(hdwf, acqmodeRecord)
    dwf.FDwfAnalogInFrequencySet(hdwf, hzAcq)
    dwf.FDwfAnalogInRecordLengthSet(hdwf, c_double(nSamples / hzAcq.value))  # -1 infinite record length

    def adquirir_muestras(num_muestras):
        global cAvailable, cLost, cCorrupted, fLost, fCorrupted
        cSamples = 0
        while cSamples < num_muestras:
            dwf.FDwfAnalogInStatus(hdwf, c_int(1), byref(sts))
            if cSamples == 0 and (sts == DwfStateConfig or sts == DwfStatePrefill or sts == DwfStateArmed):
                # Acquisition not yet started.
                continue
            dwf.FDwfAnalogInStatusRecord(hdwf, byref(cAvailable), byref(cLost), byref(cCorrupted))
            cSamples += cLost.value
            if cLost.value:
                fLost = 1
            if cCorrupted.value:
                fCorrupted = 1
            if cAvailable.value == 0:
                continue
            if cSamples + cAvailable.value > num_muestras:
                cAvailable = c_int(num_muestras - cSamples)

            dwf.FDwfAnalogInStatusData(hdwf, c_int(0), byref(rgdSamples_ch0, sizeof(c_double) * cSamples),
                                       cAvailable)  # get channel 1 data
            dwf.FDwfAnalogInStatusData(hdwf, c_int(1), byref(rgdSamples_ch1, sizeof(c_double) * cSamples),
                                       cAvailable)  # get channel 2 data
            cSamples += cAvailable.value

        if fLost:
            print("Pérdida de muestras! Reduzca la frecencia")
        if fCorrupted:
            print("Corrupción de muestras! Reduzca la frecencia")
        muestras_ch0 = np.fromiter(rgdSamples_ch0, dtype=np.float)[:num_muestras]
        muestras_ch1 = np.fromiter(rgdSamples_ch1, dtype=np.float)[:num_muestras]
        return (muestras_ch0, muestras_ch1)

    # SONDEO PARA DETERMINAR LA GANANCIA ADECUADA--------------------------
    if salida_digital_ajustable:
        print('---------------------------------------------------------------')
        print('Calibrando la ganancia adecuada ...')



        for mascara_salidas_digitales in [2,1,0]:
            # setear las nuevas salidas digitales
            dwf.FDwfDigitalIOOutputSet(hdwf, c_int(mascara_salidas_digitales))
            time.sleep(1)
            dwf.FDwfAnalogInConfigure(hdwf, c_int(0), c_int(1))  # inicio de la adquisici'on
            muestras_ch0, muestras_ch1 = adquirir_muestras(int(frecuencia_de_muestreo*0.5))  # adquisici'on de 0.5 segundo
            detected_amplitude = np.percentile(muestras_ch0, 99.9)
            if detected_amplitude < voltaje_saturacion:
                #encontramos la ganancia adecuada ya que no está saturada
                break;
        # fetch digital IO information from the device
        dwf.FDwfDigitalIOStatus(hdwf)
        # read state of all pins, regardless of output enable
        dwf.FDwfDigitalIOInputStatus(hdwf, byref(dwRead))

        print('Orden de ganancia determinada: ~ {}'.format( 10 ** int(mascara_salidas_digitales+1)))
        print('Salidas digitales ajustadas   [D1 D0] = [{}]'.format( bin(dwRead.value)[2:].zfill(16)[14:] ) )

    # wait at least 2 seconds for the offset to stabilize
    # print('Adquisición dentro de 3s ... ')
    # time.sleep(1)

    
    print('---------------------------------------------------------------')
    for i in range(delay_en_segundos_hasta_adquisicion_definitiva):
        print('Iniciando adquisición en {:04d} segundos'.format(delay_en_segundos_hasta_adquisicion_definitiva-i), end='\r', flush=True)
        time.sleep(1)
    
    # print("Starting oscilloscope")
    dwf.FDwfAnalogInConfigure(hdwf, c_int(0), c_int(1))
    print('\nINICIO !', end=None)
    muestras_ch0, muestras_ch1 = adquirir_muestras(nSamples)

    if generar_sinusoide:
        dwf.FDwfAnalogOutReset(hdwf, c_int(0))

    dwf.FDwfDeviceCloseAll()

    print("Grabación finalizada")

    detected_amplitude = np.percentile(muestras_ch0, 99.9)
    if detected_amplitude > voltaje_saturacion:
        print('ERROR: Ganancia excesiva. Señal del canal 0 saturada !!!')

    return muestras_ch0, muestras_ch1, mascara_salidas_digitales
