
import os
import numpy as np
import pandas as pd
from scipy.fftpack import fft, fftshift, ifft
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
import time


#estilo de las gráficas
plt.style.use('ggplot')



frecuencia_muestreo = 30000 #Frecuencia de muestreo en Hz
frecuencia_linea = 60    #Frecuencia de línea en Hz
muestras_por_ciclo = int(frecuencia_muestreo/frecuencia_linea)

nombres_electrodomesticos = ['Air Conditioner',
                         'Compact Fluorescent Lamp',
                         'Fan',
                         'Fridge',
                         'Hairdryer',
                         'Heater',
                         'Incandescent Light Bulb',
                         'Laptop',
                         'Microwave',
                         'Vacuum',
                         'Washing Machine']

nombres_abreviados_electrodomesticos = ['AirC','CFL','Fan','Frid','Hair','Heat','ILB','Lapt','MWave','Vacc','Wash']


# ubicación del directorio  de la base PLAID que contiene los ".csv"
PLAID_csv_directory = "./bdd/PLAID/CSV"  #PONER EL CAMINO ADECUADO

# archivo con la metadata
archivo_metadata = './data/meta1_simple.csv'


#--------------------------------------------------------

# COMPLETAR
# Agregar las funciones de la práctica 2

def cargar_metadata():
    '''
    Carga la informacion del archivo "meta1_simple.csv" a la variable metadata
    '''
    # COMPLETAR
    filename='./data/meta1_simple.csv'
    raw_data = open(filename)
    metadata = np.loadtxt(raw_data, delimiter=",",skiprows=1)
    return metadata


#display(cargar_metadata())
metadata=cargar_metadata()

#Defino a la variable metadata global para no tener problemas al llamarla, sino no la reconoce

#------------------------------------------------------------------------------
# Todas las funciones 'get' suponen la pre-existencia de la variable 'metadata'
#------------------------------------------------------------------------------


def get_cantidad_electrodomesticos():
    '''Devuelve la cantidad de electrodomésticos en la base'''
    # COMPLETAR
    return np.shape(metadata)[0]
#Devuelve el largo de esa columna
#si pongo np.shape(metadata)[0] devuelve el largo de la fila, va por dimensiones



def get_tipo(id_electrodomestico):
    '''Devuelve el tipo de electrodoméstico'''
    # COMPLETAR
    return metadata[id_electrodomestico-1,1] 

#SE LE PASA id_electrodomestico-1 debido a que saltamos el primero.

def get_casa( id_electrodomestico):
    '''Devuelve la casa del electrodoméstico'''
    # COMPLETAR
    return metadata[id_electrodomestico-1,2]



def get_nombre(id_electrodomestico):
    '''Devuelve el nombre del electrodoméstico '''
    # COMPLETAR
    tipo=int(get_tipo(id_electrodomestico))
    return nombres_electrodomesticos[tipo]
    
    
def get_nombre_abreviado(id_electrodomestico):
    '''Devuelve el nombre abreviado del electrodoméstico '''
    # COMPLETAR
    tipo=int(get_tipo(id_electrodomestico))
    return nombres_abreviados_electrodomesticos[tipo]
    
def get_nombre_archivo(id_electrodomestico):
    '''Devuelve el camino completo al archivo correspondiente al electrodoméstico'''
    # COMPLETAR
    camino=PLAID_csv_directory + "/"+ str(id_electrodomestico) + '.csv'
    return camino
#LE PONEMOS DOS BARRAS YA QUE SOLO CON LA PRIMERA SE ROMPE.     
    
def get_ids():
    '''Devuelve un array con los ids de todos los electrodomésticos de la base'''
    # COMPLETAR
    array=np.arange(0,get_cantidad_electrodomesticos())
    return array
    
def get_ids_por_tipo(tipo):
    '''Devuelve los ids correspondientes a cierto tipo'''
    # COMPLETAR
    lista=list()
    for i in range(1,get_cantidad_electrodomesticos()):
        if get_tipo(i)==tipo:
            lista.append(i) 
    return lista
#REMIRAR 


def get_ids_por_casa(casa):
    '''Devuelve los ids correspondientes a cierta casa'''
    # COMPLETAR
    lista_casa=list()
    for i in range(0,get_cantidad_electrodomesticos()):
        if get_casa(i)==casa:
            lista_casa.append(i) 
    return lista_casa

def cargar_VI_por_ciclos(nombre_archivo, 
                         frecuencia_muestreo=30000,
                         frecuencia_linea=60,
                         ciclos_a_cargar=1e100,  #por defecto se cargan todos los ciclos posibles
                         ciclos_a_saltear=0):    #por defecto se carga desde el inicio
    '''
    Carga un cierto numero de ciclos de una señal I,V guardada en un archivo
    Devuelve las señales I y V como  2 arrays Nx1
    
    Importante: se debe asegurar que se carga un número entero de ciclos (el número final
    de ciclos cargados podría eventualmente ser menor a 'ciclos_a_cargar')
    '''
    # COMPLETAR
    archivo_ohm = open(nombre_archivo)
    
    t=time.time()
    datos_ohm = np.loadtxt(archivo_ohm, delimiter=",",skiprows=0)
    print('np.load',time.time()-t)
    
    
    I=list()
    V=list()
    
    muestras_a_saltear=int(ciclos_a_saltear*(frecuencia_muestreo/frecuencia_linea))
    muestras_a_cargar=int(ciclos_a_cargar*(frecuencia_muestreo/frecuencia_linea))
    
    if muestras_a_saltear>np.shape(datos_ohm)[0]:
        return (I,V)
    if muestras_a_saltear+muestras_a_cargar>=np.shape(datos_ohm)[0]:
        muestras_a_cargar=np.shape(datos_ohm)[0]-muestras_a_saltear
        

    
    #Prints para entender
    '''
    print('ciclos_a_saltear:',ciclos_a_saltear)
    print('ciclos_a_cargar:',ciclos_a_cargar)
    print('Entonces cargo muestras de la:',muestras_a_saltear,'  A la:',muestras_a_saltear+muestras_a_cargar)    
    '''
    for i in range(muestras_a_saltear,muestras_a_saltear+muestras_a_cargar):
        I.append(datos_ohm[i,0])
        V.append(datos_ohm[i,1]) 

    return (I,V)


def cargar_VI_por_ciclos_optimizado(nombre_archivo, 
                         frecuencia_muestreo=30000,
                         frecuencia_linea=60,
                         ciclos_a_cargar=1e100,  #por defecto se cargan todos los ciclos posibles
                         ciclos_a_saltear=0):    #por defecto se carga desde el inicio
    '''
    Carga un cierto numero de ciclos de una señal I,V guardada en un archivo
    Devuelve las señales I y V como  2 arrays Nx1
    
    Importante: se debe asegurar que se carga un número entero de ciclos (el número final
    de ciclos cargados podría eventualmente ser menor a 'ciclos_a_cargar')
    '''
    # COMPLETAR
    
    I=list()
    V=list()
    muestras_a_saltear=int(ciclos_a_saltear*(frecuencia_muestreo/frecuencia_linea))
    muestras_a_cargar=int(ciclos_a_cargar*(frecuencia_muestreo/frecuencia_linea))
    
    
    
    archivo_ohm = open(nombre_archivo)
    
    t=time.time()
    datos_ohm = np.loadtxt(archivo_ohm, delimiter=",",skiprows=muestras_a_saltear)
    
    

    if muestras_a_cargar>=np.shape(datos_ohm)[0]:
        muestras_a_cargar=np.shape(datos_ohm)[0]
        

    for i in range(0,muestras_a_cargar):
        I.append(datos_ohm[i,0])
        V.append(datos_ohm[i,1]) 

        
    
    #Prints para entender
    '''
    print('ciclos_a_saltear:',ciclos_a_saltear)
    print('ciclos_a_cargar:',ciclos_a_cargar)
    print('Entonces cargo muestras de la:',muestras_a_saltear,'  A la:',muestras_a_saltear+muestras_a_cargar)    
    '''
    return (I,V)

def generar_vector_tiempo(numero_muestras, frecuencia_muestreo=30000):
    '''
    Genera un vector de tiempo del largo solicitado
    '''
    duracion = (numero_muestras-1)/frecuencia_muestreo   #duración de la señal en segundos(regla de 3)
    vector_tiempo = np.linspace(0,duracion,numero_muestras)
    #Numero de muestras a generar, se crea un vector de 0 a duracion con numero de muestras la que le digamos 
    #REMIRAR PORQUE EL -1
    
    return vector_tiempo


def graficar_I_V(I, V, frecuencia_muestreo=30000, fignum=None, limitex=[0,(500-1)/frecuencia_muestreo]):
    '''
    Genera un vector de tiempos T adecuado y grafica por separado
    las señales de corriente I(T) y voltaje V(T) que se le pasan.
    Se supone que I y V son de igual largo
    
    Si se le pasa un fignum, grafica I sobre la figura (fignum) y V sobre la 
    figura (fignum+1). De lo contrario crea dos figuras nuevas.
    '''
    # COMPLETAR

    T = generar_vector_tiempo(len(I), frecuencia_muestreo)
    
    if not fignum is None:
        plt.figure(fignum)
    else:
        plt.figure()
        
    plt.plot(T,I,'.')
    plt.xlim(limitex[0],limitex[1])
    plt.title('Diagrama I(T)')
    plt.xlabel('t(s)')
    plt.ylabel('I(A)')
    
    
    if not fignum is None:
        plt.figure(fignum+1)
    else:
        plt.figure()
    
    plt.plot(T,V,'.')
    plt.xlim(limitex[0],limitex[1])
    plt.title('Diagrama V(T)')
    plt.xlabel('t(s)')
    plt.ylabel('V(V)')
    


def graficar_diagrama_VI(I, V, fignum=None):
    '''
    Grafica I vs. V 
    
    Si se le pasa un fignum, grafica el diagrama sobre la 
    figura (fignum). De lo contrario crea una figura nueva.
    '''
    if not fignum is None:
        plt.figure(fignum)
    else:
        plt.figure()
        
    plt.plot(V,I,'.-')
    plt.title('Diagrama V-I')
    plt.xlabel('V(V)')
    plt.ylabel('I(A)')
    
def promediar_ciclos(S, frecuencia_muestreo=30000, frecuencia_linea=60):
    '''
    Promedia los ciclos de la señal S. Si la señal no tiene un número entero de ciclos,
    las muestras del final correspondientes a un ciclo incompleto no se tienen en cuenta
    Devuelve el ciclo promedio.
    Entrada:
        S     array Nx1
    Salida
        ciclo      array Cx1    con C=frecuencia_muestreo/frecuencia_linea
    '''
    # COMPLETAR

    #CREO UNA LISTA VACIA A LA QUE LE VOY A IR CONCATENANDO LOS RESULTADOS DE CADA CICLO PROMEDIADO (500 MUESTRAS)
    array_prom = np.array([])

    largo_ciclo = int(frecuencia_muestreo/frecuencia_linea)
    cant_ciclos = int((np.shape(S)[0])/largo_ciclo)
    
    #HACEMOS UN FOR PARA RECORRER CADA MUESTRA DE CADA CICLO EN SIMULTANEO
    for muestra in range(0, largo_ciclo):
        prom_muestra = np.array([])
        
        for ciclo in range(0, cant_ciclos):
            prom_muestra = np.append(prom_muestra, S[muestra + ciclo * largo_ciclo])
        
        array_prom = np.append(array_prom, np.mean(prom_muestra))
    
    return array_prom

def calcular_indices_para_alinear_ciclo(ciclo):
    '''
    Alinea un ciclo de manera que la señal inicie en el cruce por cero ascendente.
    Devuelve los indices que hacen el ordenamiento
    Ejemplo de uso:
    indices = calcular_indices_para_alinear_ciclo(ciclo)
    ciclo_alineado = ciclo[indices]
    '''
    cantidad_muestras = len(ciclo)
    
    ix = np.argsort(np.abs(ciclo))
    j = 0
    while True:
        if ix[j]<muestras_por_ciclo-1 and ciclo[ix[j]+1]>ciclo[ix[j]]:
            real_ix = ix[j]
            break
        else:
            j += 1
    
    indices_ordenados = np.hstack( (np.arange(real_ix,cantidad_muestras),
                                    np.arange(0,real_ix)) )
    return indices_ordenados


def alinear_ciclo_I_V(ciclo_I, ciclo_V):
    '''
    Devuelve los ciclos I y V alineados tal que la señal 
    de voltaje inicie en el cruce por cero ascendente
    '''
    # COMPLETAR
    
    indices = calcular_indices_para_alinear_ciclo(ciclo_V)
    V_shift = ciclo_V[indices]
    I_shift = ciclo_I[indices]
    
    return (I_shift, V_shift)
    

def get_ciclo_I_V_promedio_alineado(I,V,frecuencia_muestreo=30000, frecuencia_linea=60):
    '''
    Dadas las señales I y V, calcula los ciclos promedio y los alinea
    Devuelve los ciclos alineados ciclo_I_alineado y ciclo_v_alineado
    '''
    #COMPLETAR
    prom_sin_alin_I=promediar_ciclos(I)
    prom_sin_alin_V=promediar_ciclos(V)
    
    (I,V)=alinear_ciclo_I_V(prom_sin_alin_I,prom_sin_alin_V)
    return (I,V)
    
def generar_vector_frecuencia(numero_muestras, frecuencia_muestreo=30000, centrar_frecuencia=True):
    '''
    Genera un vector de frecuencias del largo especificado
    If centrar_frecuencia==True (por defecto)
        salida es un array   [-Fm/2.....Fm/2-Fm/numero_muestras]
    else
        salida es un array   [0.....Fm-Fm/numero_muestras]
    '''
    step_frecuencia = frecuencia_muestreo/numero_muestras 
    vector_frecuencia = np.arange(0,frecuencia_muestreo,step_frecuencia)
    if centrar_frecuencia:
        vector_frecuencia = vector_frecuencia - frecuencia_muestreo/2
    
    return vector_frecuencia

def graficar_FI_FV(I,V, frecuencia_muestreo=30000, centrar_frecuencia=True, fignum=None):
    '''
    Genera un vector de frecuencia adecuado.
    Grafica el modulo de la transformada de I y de V en figuras separadas
    
    Si se le pasa un fignum, grafica FI sobre la figura (fignum) y FV sobre la 
    figura (fignum+1). De lo contrario crea dos figuras nuevas.
    '''
    numero_muestras = len(I)
    vector_frecuencia = generar_vector_frecuencia(numero_muestras, frecuencia_muestreo, centrar_frecuencia)
    
    # COMPLETAR
    
    I_fft = fft(I)
    V_fft = fft(V)
    
    if centrar_frecuencia:
        I_fft = fftshift(I_fft)
        V_fft = fftshift(V_fft)
        
    if not fignum is None:
        plt.figure(fignum)
    else:
        plt.figure()
     
    plt.plot(vector_frecuencia,np.abs(I_fft))
    plt.title('I_FFT')
    plt.xlabel('n')
    plt.ylabel('Transformada')
                 
    if not fignum is None:
        plt.figure(fignum+1)
    else:
        plt.figure()
    
    plt.plot(vector_frecuencia,np.abs(V_fft))
    plt.title('V_FFT')
    plt.xlabel('n')
    plt.ylabel('Transformada')
    
    

    

def graficar_espectrograma_I_V(I,V, frecuencia_muestreo=30000, largo_ventana=256, fignum=None):
    '''
    Grafica el espectrograma de I y de V en figuras separadas
    
    Si se le pasa un fignum, grafica el espectrograma de I sobre la figura (fignum)
    y el de V sobre la figura (fignum+1). De lo contrario crea dos figuras nuevas.
    '''
    I_array=np.array(I)
    V_array=np.array(V)
    
    
    f,t,SI = spectrogram(I_array,fs=frecuencia_muestreo, nperseg=largo_ventana)
    f,t,SV = spectrogram(V_array,fs=frecuencia_muestreo, nperseg=largo_ventana)
    
    
    if not fignum is None:
        plt.figure(fignum)
    else:
        plt.figure()
    
    plt.pcolormesh(t,f, SI)
    plt.title('Espectrograma de I')
    plt.ylim(-1,500)
    plt.ylabel('Frecuencia [Hz]')
    plt.xlabel('Tiempo [s]')
    
    if not fignum is None:
        plt.figure(fignum)
    else:
        plt.figure()
    

    plt.pcolormesh(t, f, SV)
    plt.title('Espectrograma de V')
    plt.ylim(-1,500)
    plt.ylabel('Frecuencia [Hz]')
    plt.xlabel('Tiempo [s]')
    
def calcular_THD(S, frecuencia_muestreo=30000, frecuencia_linea=60):
    
    numero_muestras = len(S)
    FS = fft(S)
        
    #np.sqrt(2) para que sea rms,  el dos viene porque tenemos frecuencias positivas y negativas 
    #luego de esto solo vamos a usar las frecuencias positivas
    FS_rms = np.abs(FS)/numero_muestras  * 2/np.sqrt(2)  
    
    #EL 2 APARECE PORQUE HAY FRECUENCIAS NEGATIVAS Y POSITIVAS 
    #EL /SQRT(2) APARECE PORQUE ES RMS
    #SE DIVIDE LA FFT SOBRE N YA QUE ESA ES LA RELACION QUE HAY ENTRE LA FFT Y LA TRANSF FOURIER DISCRETA
    
    
    
    step_frecuencia = frecuencia_muestreo/numero_muestras 
    indice_frecuencia_fundamental = int(frecuencia_linea / step_frecuencia)
    
    indices_armonicos = np.arange(2*indice_frecuencia_fundamental,
                                  numero_muestras//2,
                                  indice_frecuencia_fundamental) 
    
    indices_armonicos_y_frecuencia_cero = np.concatenate(([0],indices_armonicos))
    
    distorsion = np.sqrt(np.sum(FS_rms[indices_armonicos_y_frecuencia_cero]**2)) / FS_rms[indice_frecuencia_fundamental]
    return distorsion

    
def calcular_potencia_media(I,V, frecuencia_muestreo=30000, frecuencia_linea=60):
    '''
    Calcula la potencia media
    '''
    # COMPLETAR
    I=np.asarray(I)
    V=np.asarray(V)
    I=fft(I)
    V=fft(V)
    
    P=np.zeros(len(V))
    for x in range(len(V)):
        P[x]=np.real(V[x]*np.conj(I[x])) 
        #EL CONJUGADO ES POR DEF,Y TOMAMOS LA PARTE REAL PARA QUE APAREZCA EL COSENO DE FI 
    potencia_media=np.sum(P)/(len(V))
    
    return potencia_media
    
    
def calcular_potencia_IEEE_1459_2010(I,V, frecuencia_muestreo=30000, frecuencia_linea=60):
    '''
    Calcula la potencia para señales I,V que pueden ser no-sinusoidales.
    Se supone que las señales I y V tienen un número entero de períodos.
    
    Los cálculos se realizan en frecuencia. Para esta implementación se consideran las componentes 
    correspondientes a la continua, la frecuencia fundamental y sus armónicos. No se tienen en cuenta 
    otras componentes de frecuencia intermedias.
    
    La función devuelve: S, S_11, S_H, S_N, P, P_11, P_H, Q_11, D_I, D_V, D_H, N, THD_V, THD_I
    S        Apparent power 
    S_11     Fundamental apparent power
    S_H      Harmonic apparent power
    S_N      Non-fundamental apparent power
    P        Active power
    P_11     Fundamental active power
    P_H      Harmonics active power
    Q_11     Fundamental reactive power
    N        Non-active apparent power
    
    D_I      Current distortion power
    D_V      Voltage distortion power 
    D_H      Harmonic distortion power
    THD_V    Total harmonic distortion for voltage
    THD_I    Total harmonic distortion for current
    '''
    numero_muestras = len(V)
    
    
    THD_V = calcular_THD(V)
    THD_I = calcular_THD(I) 

    V = fft(V)
    I = fft(I)
    
    # rms
    
    V_rms = V/numero_muestras * 2/np.sqrt(2)
    I_rms = I/numero_muestras * 2/np.sqrt(2)
    
    step_frecuencia = frecuencia_muestreo/numero_muestras 
    indice_frecuencia_fundamental = int(frecuencia_linea / step_frecuencia)
    
    
    indices_armonicos = np.arange(0, numero_muestras//2, indice_frecuencia_fundamental) 
    indices_armonicos_y_frecuencia_cero = np.concatenate(([0],indices_armonicos))

    #MODULO DE LOS ARMONICOS
    V_armonicos = np.abs(V_rms[indices_armonicos])
    I_armonicos = np.abs(I_rms[indices_armonicos])
    
    V_DC=(V_armonicos[0])
    V_FUND =(V_armonicos[1])
    
    I_DC = (I_armonicos[0])
    I_FUND = (I_armonicos[1]) 
    VH = np.sqrt(V_DC**2 + np.sum(V_armonicos[2:]**2))
    IH = np.sqrt(I_DC**2 + np.sum(I_armonicos[2:]**2))
  
    theta_H = np.angle(V[indices_armonicos_y_frecuencia_cero]) - np.angle(I[indices_armonicos_y_frecuencia_cero])
    
    
    #EXPRESIONES DE LAS POTENCIAS 
    
    S_H = IH*VH
    
    D_I = V_FUND*IH
    
    D_V = VH*I_FUND
    
    S_11=(V_FUND)*(I_FUND)
    #S=np.sum((V_I_en_Armonicos_confund))
    S = np.sqrt((S_11)**2 + (D_I)**2 + (D_V)**2 + (S_H)**2)
    
    Q_11=np.imag(V_rms[indices_armonicos[1]]*np.conj(I_rms[indices_armonicos[1]]))
    
    
    P_11 = np.real(V_rms[indices_armonicos[1]]*np.conj(I_rms[indices_armonicos[1]]))
    
    #P_11 = np.abs(V_rms[indices_armonicos[1]])*np.abs(I_rms[indices_armonicos[1]])*np.angle(V_rms[indices_armonicos[1]]*np.conj(I_rms[indices_armonicos[1]]))
    
    #print('esta es la activa',P_11)
    #print('esta es la Vfund',V_FUND)
    #print('esta es la Ifund',I_FUND)
    #print('esta es Varm',V_armonicos)
    #print('esta es Iarm',I_armon_con_fundamental)
    #print('indices armonicos',indices_armonicos)
    #print('Array IH',IH)
    #print('Array VH',VH)
    
    #PH = np.sum(np.abs((V_rms[indices_armonicos_y_frecuencia_cero])*I_rms[indices_armonicos_y_frecuencia_cero]*np.cos(theta_H)))
    
    #P=(P_11 + PH)
    
    P=calcular_potencia_media(I_rms[indices_armonicos],V_rms[indices_armonicos])
    
    PH=P-P_11
        
    S_11=V_FUND*I_FUND
    
    N=np.sqrt(np.abs(S**2 - P**2))
    #tomo el valor absoluto para que no tire problema
    
    #Current distortion power
    #D_I=V_armon*I_en_arm_sin_fund
    #D_I=0
    #Voltage distortion power 
    #D_V=V_en_arm_sin_fund*I_FUND     
    #D_V=0
    #Harmonic distortion power
    #D_H=V_en_arm_sin_fund*I_en_arm_sin_fund
    D_H = np.sqrt(np.abs((S_H)**2 - (PH)**2))
    #!REMIRAR SU CALCULO!
    S_N=np.sqrt(S**2 - S_11**2)    
    
    
    
    
    # COMPLETAR
    
    
    return round(S,2), round(S_11,2), round(S_H,2), round(S_N,2), round(P,2), round(P_11,2), round(PH,2), round(Q_11,2), round(D_I,2), round(D_V,2), round(D_H,2), round(N,2), round(THD_V,2), round(THD_I,2)
