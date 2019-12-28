# SoftwareDMNN Tesis de Omarsaurio UV 2019

# * para compilar un ejecutable .exe:
# * instalar libreria de compilacion (si no la tiene):
# pip install pyinstaller
# * luego ubique la consola en la carpeta de proyecto:
# cd ruta_de_carpeta_sin_incluir_el_.py
# * la carpeta contiene:
# * img*.png (* de 0 a 37), TutorialDMNN.pdf, SoftwareDMNN.py, icono.ico, wait.gif, pitido.mp3
# * nota: los ... abajo son todos los archivos extras: --add-data "archivo.algo";"."
# pyinstaller -y -F -w -i "icono.ico" --add-data "img0.png";"." ... "SoftwareDMNN.py"
# * se incluye la funcion Compilador(img) para generar el comando, ingresar parametro 38

# se incluye la funcion generica EjecutarRed(entradas, pesW, numK) para proyectos externos

# importar librerias necesarias de Python
import sys
import os
from io import StringIO
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QPushButton, QLabel,\
    QGridLayout, QHBoxLayout, QVBoxLayout, QTableWidget, QComboBox, QTableWidgetItem,\
    QLineEdit, QProgressBar, QCheckBox, QMessageBox, QGroupBox, QSizePolicy
from PyQt5.QtCore import Qt, QMargins, QThread, QTimer, QObject
from PyQt5.QtGui import QPixmap, QColor, QIcon, QPalette, QBrush, QMovie
from PyQt5.QtChart import QChartView, QScatterSeries, QLineSeries
import numpy as np
import xml.etree.ElementTree as ET
from playsound import playsound
import time

# la funcion principal o inicializadora
def main():
    # para que al ejecutar en forma .exe, se hallen los assets
    if getattr(sys, "frozen", False):
        os.chdir(sys._MEIPASS)

    app = QApplication(sys.argv)
    # obtener la altura de la pantalla y escalarla para inicializar
    screenH = app.desktop().screenGeometry().height() * 0.8

    #instanciar la clase maestra
    motor = Motor()

    # instanciar todas las GUIs del programa, quedan invisibles
    modulo = []
    modulo.append(Problema())
    modulo.append(Inicializacion())
    modulo.append(Entreno(0))
    modulo.append(Entreno(1))
    modulo.append(Entreno(2))
    modulo.append(PostEntreno())
    modulo.append(Analisis())

    # instanciar la GUI principal, escalarla y hacerla visible
    menu = Menu()
    menu.modulo = modulo
    menu.motor = motor
    menu.Inicializar()
    menu.resize(screenH * 1.6, screenH)
    menu.show()

    # agregar a las GUIs secundarias la id de la principal y maestra
    for i in range(len(modulo)):
        modulo[i].menu = menu
        modulo[i].modulo = modulo
        modulo[i].motor = motor

    # poner la hoja de estilos para todas las GUIs
    app.setStyleSheet(Estilos(screenH))

    # ejecutar aplicacion hasta que no haya ventana visible
    sys.exit(app.exec_())

# a continuacion las clases mas importantes

class Motor(QObject):
    def __init__(self):
        QObject.__init__(self)
        # matriz que guarda todos los patrones de entreno, shape es (P, x.size + 1)
        self.patrones = np.zeros((3, 2))

        # numero de patrones que seran de entreno, validacion y testeo
        self.numPEVT = np.ones(3, dtype=int)

        # numero de los baches que seran de entreno y validacion
        self.numBEV = np.ones(2, dtype=int)

        # apodos que reciben las clases y titulo del problema
        self.apodos = [""]
        self.entradas = [""]
        self.titulo = "???"

        # principal red DMNN
        self.laRed = Red()

        # verdadero si la red coincide con las entradas y salidas
        self.redOk = False

        # adecuacion lineal de patrones
        self.adecuaH = np.ones(1)
        self.adecuaL = -np.ones(1)
        self.adecuaN = 0.0

        # matrices de confusion
        self.matrizE = np.zeros((2, 2), dtype=int)
        self.matrizV = np.zeros((2, 2), dtype=int)
        self.matrizT = np.zeros((2, 2), dtype=int)
        self.matrizN = np.zeros((2, 2), dtype=int)
        self.matrizG = np.zeros((2, 2), dtype=int)

        # curvas de entrenamiento: 0:E, 1:V, 2:T, 3:D
        self.curvas = np.zeros((1, 4))
        # id del vector curvas donde se halla el menor V
        self.mejV = 0

        # informacion de ciclos de entrenamiento, 0:actual, 1:total
        self.runData = np.zeros(2, dtype=int)

        # informacion de las variables del entreno
        # SGD: 0:inhibir, 1:%cambioA, 2:ecu%A, 3:A, 4:B, 5:mini-bache, 6:toler
        # DE: 0:inhibir, 1:%cambioH, 2:ecu%H, 3:H, 4:C, 5:(1)si-unoAuno, 6:toler
        # PSO: 0:inhibir, 1:%cambioC3, 2:ecu%C3, 3:C3, 4:C1, 5:C2, 6:toler
        self.trainData = np.zeros(7)

        # guardara id de las particulas PSO o la poblacion DE
        self.agentes = []

        # el mejor agente
        self.mejor = 0

        # escalamiento para graficar numero de dendritas, y escala para errores
        self.escDen = 1.0

        # los dos ejes X,Y a ser graficados
        self.grXY = np.array([0, 0])

    def CrearRedGradiente(self):
        self.agentes = []
        self.mejor = 0
        self.agentes.append(LaGradiente())
        self.agentes[0].IniciaLaGradiente(self.laRed, self.patrones)

    def CrearPoblacion(self, total):
        self.agentes = []
        self.mejor = 0
        for n in range(total):
            self.agentes.append(Individuo())
            if n != 0:
                self.agentes[n].IniciaIndividuo(self.laRed, 0.05)
            else:
                self.agentes[n].IniciaIndividuo(self.laRed, 0.0)

    def CrearParticulas(self, total):
        self.agentes = []
        self.mejor = 0
        for n in range(total):
            self.agentes.append(Particula())
            if n != 0:
                self.agentes[n].IniciaParticula(self.laRed, 0.05, 0.02)
            else:
                self.agentes[n].IniciaParticula(self.laRed, 0.0, 0.04)

    def CicloEntrenar(self, id, tipo):
        # tipo: 0:SGD, 1:DE, 2:PSO
        dim = self.patrones.shape[1] - 1

        # estructura para ver los mejores errores promedio de entreno
        proE = np.zeros(len(self.agentes))

        # si el entreno apenas inicia
        if self.laRed.error == -1.0:

            # guardar la red en .txt
            self.laRed.CrearXML(dim, int(self.patrones[:, dim].max() + 1), "AutoSavePreEntreno.xml",
                                self.adecuaH, self.adecuaL, self.titulo, self.apodos, self.entradas,
                                self.adecuaN)

            # calcular el error inicial para la red principal
            proV = 0.0
            for b in range(int(np.ceil(self.numPEVT[1] / self.numBEV[1]))):
                BV = self.patrones[(b * self.numBEV[1] + self.numPEVT[0]): min(self.numPEVT[:2].sum(),
                                                                               b * self.numBEV[1] + self.numBEV[1] +
                                                                               self.numPEVT[0]), :]
                self.laRed.errorRL(BV)
                proV += self.laRed.error

            # obtener el promedio
            proV /= np.ceil(self.numPEVT[1] / self.numBEV[1])
            self.laRed.error = proV

            # calcular el error inicial para los agentes
            for b in range(int(np.ceil(self.numPEVT[0] / self.numBEV[0]))):
                BE = self.patrones[(b * self.numBEV[0]): min(self.numPEVT[0],
                                                             b * self.numBEV[0] + self.numBEV[0]), :]
                # acumular el error de entreno inicial
                for n in range(len(self.agentes)):
                    self.agentes[n].errorRL(BE)
                    proE[n] += self.agentes[n].error

            # obtener el promedio y mejor agente
            proE /= np.ceil(self.numPEVT[0] / self.numBEV[0])
            for n in range(len(self.agentes)):
                self.agentes[n].errorB = proE[n]
            self.mejor = np.argmin(proE)

            # hallar la escala para el numero de dendritas
            self.escDen = min(self.laRed.error, self.agentes[self.mejor].errorB)
            self.escDen = np.power(self.escDen, np.e)

        # poner el parametro cambiante en su valor inicial
        self.trainData[2] = self.trainData[3]

        # ciclo de entrenamiento
        okey = True
        while self.runData[0] < self.runData[1]:

            # verificar si se dio orden de pausa o stop
            if self.FinEntreno(id):
                okey = False
                break

            # contador de iteraciones (epocas)
            self.runData[0] += 1

            # mezclar patrones al azar
            np.random.shuffle(self.patrones[:self.numPEVT[0], :])
            np.random.shuffle(self.patrones[self.numPEVT[0]:self.numPEVT[:2].sum(), :])

            # cambiar el parametro cambiante
            if self.trainData[1] != 0.0:
                self.trainData[2] = 0.1 + 0.9 * np.exp(-np.power(self.runData[0], 2.0) /
                                                       (self.trainData[1] * 2.0 * np.power(self.runData[1], 2.0)))
                self.trainData[2] *= self.trainData[3]

            # optimizar numero de dendritas
            if np.random.rand() < self.trainData[0]:
                self.OptimizaRed(self.agentes[np.random.randint(len(self.agentes))],
                                 self.patrones[:self.numPEVT[0], :], self.trainData[6])

            # reiniciar estructura para ver el promedio de error de entreno
            proE *= 0.0

            # ciclo de los baches para entrenamiento
            for b in range(int(np.ceil(self.numPEVT[0] / self.numBEV[0]))):

                # crear el bache de entreno como tal
                BE = self.patrones[(b * self.numBEV[0]): min(self.numPEVT[0],
                                                             b * self.numBEV[0] + self.numBEV[0]), :]

                # ejecutar operaciones para cada tipo de entreno
                if tipo == 0:
                    self.OperaSGD(BE)
                elif tipo == 1:
                    self.OperaDE(BE)
                else:
                    self.OperaPSO(BE)

                # acumular el error de entreno
                for n in range(len(self.agentes)):
                    proE[n] += self.agentes[n].errorB

            # obtener el promedio y mejor agente
            proE /= np.ceil(self.numPEVT[0] / self.numBEV[0])
            self.mejor = np.argmin(proE)

            # estructura para hallar promedio de validacion
            proV = 0.0

            # ciclo de los baches para validacion
            for b in range(int(np.ceil(self.numPEVT[1] / self.numBEV[1]))):
                BV = self.patrones[(b * self.numBEV[1] + self.numPEVT[0]): min(self.numPEVT[:2].sum(),
                    b * self.numBEV[1] + self.numBEV[1] + self.numPEVT[0]), :]
                self.agentes[self.mejor].errorRL(BV)
                proV += self.agentes[self.mejor].error

            # obtener el promedio y copiar la mejor red
            proV /= np.ceil(self.numPEVT[1] / self.numBEV[1])
            if proV <= self.laRed.error or self.numPEVT[1] == 1:
                self.laRed.error = proV
                self.laRed.CopiarRed(self.agentes[self.mejor])
                self.mejV = self.curvas.shape[0] - 1

            # poner estadisticas de entreno
            self.EstadisticasCiclo(proE, proV)

        # guardar las redes en .txt
        self.laRed.CrearXML(dim, int(self.patrones[:, dim].max() + 1), "AutoSavePosEntrenoV.xml",
                            self.adecuaH, self.adecuaL, self.titulo, self.apodos, self.entradas,
                            self.adecuaN)
        if len(self.agentes) > 0:
            self.agentes[self.mejor].CrearXML(dim, int(self.patrones[:, dim].max() + 1),
                                              "AutoSavePosEntrenoE.xml", self.adecuaH, self.adecuaL,
                                              self.titulo, self.apodos, self.entradas, self.adecuaN)

        # si finalizo sin detenerse por botones volver en pausa
        if okey:
            id.estado = 1

    def EstadisticasCiclo(self, proE, proV):
        # crear pedazo de nuevo ciclo de datos
        Vest = np.zeros((1, 4))
        # poner errores ya obtenidos E y V
        Vest[0, 0] = proE[self.mejor]
        Vest[0, 1] = proV
        # calcular el error faltante T
        self.agentes[self.mejor].errorRL(self.patrones[self.numPEVT[:2].sum():, :])
        Vest[0, 2] = self.agentes[self.mejor].error
        # calcular escala logaritmica de los tres primeros datos
        Vest[0, :3] = np.power(Vest[0, :3], np.e)
        # calcular el numero de dendritas porcentual y escalado
        act = np.where(self.laRed.actK, 1, 0).sum()
        Vest[0, 3] = (act / self.laRed.actK.size) * self.escDen
        # limitar los extremos de los valores y agregar a matriz
        Vest = np.clip(Vest, 0.0, self.escDen * 1.25)
        self.curvas = np.concatenate((self.curvas, Vest), axis=0)

    def OptimizaRed(self, red, BE, toler):

        # calcular el actual error CM de la red
        red.errorCM(BE)
        if red.error <= toler:

            # elegir la dendria a activar o desactivar
            den = np.random.randint(red.actK.size)

            # compara el error sin la caja, con el original
            if red.actK[den]:
                red.actK[den] = False
                if red.ClaseVacia():
                    red.actK[den] = True
                else:
                    red.errorCM(BE)
                    if red.error > toler:
                        red.actK[den] = True

    def OperaSGD(self, BE):

        # hacer ciclo del mini-bache
        for mb in range(int(np.ceil(BE.shape[0] / self.trainData[5]))):

            # crear el mini-bache de entreno como tal
            MBE = BE[(mb * int(self.trainData[5])): min(BE.shape[0],
                                                        (mb + 1) * int(self.trainData[5])), :]

            # opera el vector de inercia antes de sumarle las derivaciones
            self.agentes[0].impU *= self.trainData[4] * (MBE.shape[0] / self.numPEVT[0])

            # hacer ciclo para hallar todas las componentes de derivacion
            for p in range(MBE.shape[0]):
                self.agentes[0].EjecutarSuave(True, MBE[p, :])
                self.agentes[0].EjecutarSuave(False, MBE[p, :])

            # actualizar los pesos W de la red
            self.agentes[0].pesW += self.agentes[0].norA * self.trainData[2] * self.agentes[0].impU

        # prevenir valores inadecuados de la red
        self.agentes[0].PrevenirSolape()
        self.agentes[0].PrevenirEscape()

        # verificar si la red es mejor
        self.agentes[0].errorRL(BE)
        if self.agentes[0].error < self.agentes[0].errorB:
            self.agentes[0].errorB = self.agentes[0].error

    def OperaDE(self, BE):

        # hacer ciclo que crea la nueva generacion
        for n in range(len(self.agentes)):
            self.agentes[n].CrearHijo(self.agentes, self.trainData[2], self.trainData[4])
            self.agentes[n].errorRL(BE)
            self.agentes[n].CambioW()

        # hacer ciclo de seleccion de los mas aptos
        if self.trainData[5] == 1.0:
            for n in range(len(self.agentes)):
                if self.agentes[n].error < self.agentes[n].errorB:
                    self.agentes[n].errorB = self.agentes[n].error
                    self.agentes[n].CambioW()
        else:
            tot = len(self.agentes)
            vpesW = np.zeros((tot, self.laRed.pesW.size))
            vactK = np.ones((tot, self.laRed.actK.size)) > 0
            error = np.zeros(tot)
            vpesW[0, :] = self.agentes[self.mejor].pesW.copy()
            vactK[0, :] = self.agentes[self.mejor].actK.copy()
            error[0] = self.agentes[self.mejor].errorB
            obt = [self.mejor]
            for n in range(1, tot):
                A = np.random.randint(tot * 2)
                while A in obt:
                    A = np.random.randint(tot * 2)
                obt.append(A)
                B = np.random.randint(tot * 2)
                while B in obt:
                    B = np.random.randint(tot * 2)
                obt.append(B)
                if A >= tot:
                    eA = self.agentes[A - tot].error
                else:
                    eA = self.agentes[A].errorB
                if B >= tot:
                    eB = self.agentes[B - tot].error
                else:
                    eB = self.agentes[B].errorB
                if eA < eB:
                    error[n] = eA
                    if A >= tot:
                        vpesW[n, :] = self.agentes[A - tot].otrW.copy()
                        vactK[n, :] = self.agentes[A - tot].actK.copy()
                    else:
                        vpesW[n, :] = self.agentes[A].pesW.copy()
                        vactK[n, :] = self.agentes[A].actK.copy()
                else:
                    error[n] = eB
                    if B >= tot:
                        vpesW[n, :] = self.agentes[B - tot].otrW.copy()
                        vactK[n, :] = self.agentes[B - tot].actK.copy()
                    else:
                        vpesW[n, :] = self.agentes[B].pesW.copy()
                        vactK[n, :] = self.agentes[B].actK.copy()
            for n in range(tot):
                self.agentes[n].pesW = vpesW[n, :]
                self.agentes[n].actK = vactK[n, :]
                self.agentes[n].errorB = error[n]

        # seleccionar al mejor
        for n in range(len(self.agentes)):
            if self.agentes[n].errorB < self.agentes[self.mejor].errorB:
                self.mejor = n

    def OperaPSO(self, BE):

        # hacer ciclo que ejecute la fisica de las particulas
        for n in range(len(self.agentes)):
            self.agentes[n].CalcularPaso(self.trainData[4], self.trainData[5],
                           self.trainData[2], self.agentes[self.mejor])
            self.agentes[n].errorRL(BE)
            if self.agentes[n].error < self.agentes[n].errorB:
                self.agentes[n].errorB = self.agentes[n].error
                self.agentes[n].besW = self.agentes[n].pesW.copy()

        # seleccionar al mejor
        for n in range(len(self.agentes)):
            if self.agentes[n].errorB < self.agentes[self.mejor].errorB:
                self.mejor = n

    def FinEntreno(self, id):
        if id.estado == 0 or id.estado == 1:
            res = True
            if id.estado == 0:
                # guardar la red en .txt
                dim = self.patrones.shape[1] - 1
                self.agentes[self.mejor].CrearXML(dim, int(self.patrones[:, dim].max() + 1),
                                                  "AutoSavePosEntrenoE.xml", self.adecuaH, self.adecuaL,
                                                  self.titulo, self.apodos, self.entradas, self.adecuaN)
                self.agentes = []
                self.runData[0] = 0
        else:
            res = False
        return res

    def MatricesConfusion(self):
        dim = self.patrones.shape[1] - 1
        cla = self.laRed.numK.size
        self.matrizE = np.zeros((cla, cla), dtype=int)
        self.matrizV = np.zeros((cla, cla), dtype=int)
        self.matrizT = np.zeros((cla, cla), dtype=int)
        for p in range(self.numPEVT[0]):
            rea = int(self.patrones[p, dim])
            pre = np.argmax(self.laRed.EjecutarRed(self.patrones[p, :dim]))
            self.matrizE[rea, pre] += 1
        for p in range(self.numPEVT[1]):
            rea = int(self.patrones[self.numPEVT[0] + p, dim])
            pre = np.argmax(self.laRed.EjecutarRed(self.patrones[self.numPEVT[0] + p, :dim]))
            self.matrizV[rea, pre] += 1
        for p in range(self.numPEVT[2]):
            rea = int(self.patrones[self.numPEVT[:2].sum() + p, dim])
            pre = np.argmax(self.laRed.EjecutarRed(self.patrones[self.numPEVT[:2].sum() + p, :dim]))
            self.matrizT[rea, pre] += 1
        self.matrizN = (self.matrizV + self.matrizT).copy()
        self.matrizG = (self.matrizE + self.matrizV + self.matrizT).copy()

    def ExportarMatrices(self):
        fileDir, _ = QFileDialog.getSaveFileName(caption="Guardar Matrices", filter="XML File (*.xml)")
        if fileDir:
            datos = ET.Element("dataset")
            datos.text = "Matrices de Confusion DMNN: " + self.titulo
            cla = self.laRed.numK.size

            llave = ET.SubElement(datos, "record")
            item = ET.SubElement(llave, "Reales")
            item.text = "Entreno"
            for p in range(cla):
                ET.SubElement(llave, "P" + str(p))
            for r in range(cla):
                llave = ET.SubElement(datos, "record")
                item = ET.SubElement(llave, "Reales")
                item.text = "R" + str(r) + ": " + self.apodos[r]
                for p in range(cla):
                    item = ET.SubElement(llave, "P" + str(p))
                    item.text = str(self.matrizE[r, p])

            llave = ET.SubElement(datos, "record")
            item = ET.SubElement(llave, "Reales")
            item.text = "Validacion"
            for p in range(cla):
                ET.SubElement(llave, "P" + str(p))
            for r in range(cla):
                llave = ET.SubElement(datos, "record")
                item = ET.SubElement(llave, "Reales")
                item.text = "R" + str(r) + ": " + self.apodos[r]
                for p in range(cla):
                    item = ET.SubElement(llave, "P" + str(p))
                    item.text = str(self.matrizV[r, p])

            llave = ET.SubElement(datos, "record")
            item = ET.SubElement(llave, "Reales")
            item.text = "Testeo"
            for p in range(cla):
                ET.SubElement(llave, "P" + str(p))
            for r in range(cla):
                llave = ET.SubElement(datos, "record")
                item = ET.SubElement(llave, "Reales")
                item.text = "R" + str(r) + ": " + self.apodos[r]
                for p in range(cla):
                    item = ET.SubElement(llave, "P" + str(p))
                    item.text = str(self.matrizT[r, p])

            llave = ET.SubElement(datos, "record")
            item = ET.SubElement(llave, "Reales")
            item.text = "Dendritas"
            for p in range(cla):
                ET.SubElement(llave, "P" + str(p))
            llave = ET.SubElement(datos, "record")
            item = ET.SubElement(llave, "Reales")
            item.text = str(np.where(self.laRed.actK, 1, 0).sum())
            for p in range(cla):
                ET.SubElement(llave, "P" + str(p))

            file = open(fileDir, "w")
            file.write(ET.tostring(datos).decode())
            file.close()

    def ExportarResultados(self):
        fileDir, _ = QFileDialog.getSaveFileName(caption="Guardar Resultados",
                                                 filter="XML File (*.xml);;Text File (*.txt)")
        if fileDir:
            if ".xml" in fileDir:
                self.ResultadosXML(fileDir)
            else:
                self.ResultadosTXT(fileDir)

    def ResultadosXML(self, fileDir):
        datos = ET.Element("dataset")
        datos.text = "Resultados DMNN: " + self.titulo
        dim = self.patrones.shape[1] - 1

        for p in range(self.patrones.shape[0]):
            llave = ET.SubElement(datos, "record")
            item = ET.SubElement(llave, "Deseado")
            item.text = str(int(self.patrones[p, dim]))
            item = ET.SubElement(llave, "Obtenido")
            item.text = str(np.argmax(self.laRed.EjecutarRed(self.patrones[p, :dim])))

        file = open(fileDir, "w")
        file.write(ET.tostring(datos).decode())
        file.close()

    def ResultadosTXT(self, fileDir):
        file = open(fileDir, "w")
        file.write("Resultados DMNN: " + self.titulo + "\nDeseado, Obtenido\n")
        dim = self.patrones.shape[1] - 1

        txx = ""
        for p in range(self.patrones.shape[0]):
            txx += str(int(self.patrones[p, dim])) + "," + \
                   str(np.argmax(self.laRed.EjecutarRed(self.patrones[p, :dim]))) + "\n"

        file.write(txx)
        file.close()

    def PartirPatrones(self, E, V, BE, BV):
        tot = self.patrones.shape[0]
        self.numPEVT[0] = max(1, min(tot - 2, int(tot * E)))
        self.numPEVT[1] = max(1, min(tot - self.numPEVT[0] - 1, int(tot * V)))
        self.numPEVT[2] = tot - self.numPEVT[:2].sum()
        self.numBEV[0] = max(1, min(int(BE), self.numPEVT[0]))
        self.numBEV[1] = max(1, min(int(BV), self.numPEVT[1]))
        # devolvera falso si en patrones de entreno no estan todas las clases
        return self.ClasesEntrenadas()

    def MezclarExportar(self, sinNorma):
        np.random.shuffle(self.patrones)
        if sinNorma:
            fileDir, _ = QFileDialog.getSaveFileName(caption="Guardar Patrones", filter="XML File (*.xml)")
            if fileDir:
                datos = ET.Element("Patrones")
                llave = ET.SubElement(datos, "Titulo")
                llave.text = self.titulo

                dim = self.patrones.shape[1] - 1
                cla = int(self.patrones[:, dim].max() + 1)
                llave = ET.SubElement(datos, "Dimension")
                item = ET.SubElement(llave, "Entradas")
                item.text = str(dim)
                item = ET.SubElement(llave, "Clases")
                item.text = str(cla)
                item = ET.SubElement(llave, "TotalPatrones")
                item.text = str(self.patrones.shape[0])

                llave = ET.SubElement(datos, "NombresSalidas")
                for i in range(len(self.apodos)):
                    item = ET.SubElement(llave, "A" + str(i))
                    item.text = self.apodos[i]

                llave = ET.SubElement(datos, "NombresEntradas")
                for i in range(len(self.entradas)):
                    item = ET.SubElement(llave, "N" + str(i))
                    item.text = self.entradas[i]

                llave = ET.SubElement(datos, "ValoresDatos")
                for p in range(self.patrones.shape[0]):
                    pat = ET.SubElement(llave, "P" + str(p))
                    for i in range(dim):
                        item = ET.SubElement(pat, "E" + str(i))
                        item.text = str(self.patrones[p, i])
                    item = ET.SubElement(pat, "S")
                    item.text = str(int(self.patrones[p, dim]))

                llave = ET.SubElement(datos, "ParticionPatrones")
                item = ET.SubElement(llave, "Entreno")
                item.text = str(self.numPEVT[0])
                item = ET.SubElement(llave, "Validacion")
                item.text = str(self.numPEVT[1])
                item = ET.SubElement(llave, "Testeo")
                item.text = str(self.numPEVT[2])

                llave = ET.SubElement(datos, "ParticionBaches")
                item = ET.SubElement(llave, "Entreno")
                item.text = str(self.numBEV[0])
                item = ET.SubElement(llave, "Validacion")
                item.text = str(self.numBEV[1])

                file = open(fileDir, "w")
                file.write(ET.tostring(datos).decode())
                file.close()

    def ImportarPatrones(self):
        nota = 0
        fileDir, _ = QFileDialog.getOpenFileName(caption="Abrir Patrones",
                                                 filter="XML File (*.xml);;Text File (*.txt)")
        if fileDir:
            if ".xml" in fileDir:
                nota = self.AbrirXML(fileDir)
                if nota == 1:
                    nota = 2
            else:
                nota = self.AbrirTXT(fileDir)
        return nota

    def AbrirXML(self, archivo):
        file = open(archivo, "r")
        raiz = ET.fromstring(file.read())
        file.close()

        if raiz.tag == "Patrones":
            self.titulo = raiz.find("Titulo").text
            dim = int(raiz.find("Dimension").find("Entradas").text)
            self.apodos = self.ExtraeTags(raiz, "NombresSalidas")
            self.entradas = self.ExtraeTags(raiz, "NombresEntradas")
            self.patrones = np.zeros((int(raiz.find("Dimension").find("TotalPatrones").text), dim + 1))

            self.numPEVT[0] = int(raiz.find("ParticionPatrones").find("Entreno").text)
            self.numPEVT[1] = int(raiz.find("ParticionPatrones").find("Validacion").text)
            self.numPEVT[2] = int(raiz.find("ParticionPatrones").find("Testeo").text)
            self.numBEV[0] = int(raiz.find("ParticionBaches").find("Entreno").text)
            self.numBEV[1] = int(raiz.find("ParticionBaches").find("Validacion").text)

            for p in range(self.patrones.shape[0]):
                ss = raiz.find("ValoresDatos").find("P" + str(p))
                for i in range(dim):
                    self.patrones[p, i] = float(ss.find("E" + str(i)).text)
                self.patrones[p, dim] = float(ss.find("S").text)

            if self.patrones.ndim != 2:
                self.patrones = np.zeros((3, 2))
                nota = -2
            elif self.patrones.shape[1] < 2 or self.patrones.shape[0] < 3:
                self.patrones = np.zeros((3, 2))
                nota = -2
            elif not self.ClasesPatronadas():
                self.patrones = np.zeros((3, 2))
                nota = -3
            else:
                self.adecuaH = np.ones(dim)
                self.adecuaL = -np.ones(dim)
                self.adecuaN = 0.0
                nota = 1
        else:
            nota = -1
        return nota

    def ExtraeTags(self, raiz, llave):
        vec = []
        item = {"NombresSalidas": "A", "NombresEntradas": "N"}
        i = 0
        while True:
            n = raiz.find(llave).find(item[llave] + str(i))
            if n == None:
                break
            else:
                vec.append(n.text)
                i += 1
        return vec

    def AbrirTXT(self, archivo):
        file = open(archivo, "r")
        data = file.readlines()
        file.close()
        if "Patrones: " in data[0]:
            # comienza a procesarse la informacion
            tx = StringIO("".join(data[3:]))
            self.patrones = np.loadtxt(tx, delimiter=",")
            if self.patrones.ndim != 2:
                self.patrones = np.zeros((3, 2))
                nota = -2
            elif self.patrones.shape[1] < 2 or self.patrones.shape[0] < 3:
                self.patrones = np.zeros((3, 2))
                nota = -2
            elif not self.ClasesPatronadas():
                self.patrones = np.zeros((3, 2))
                nota = -3
            else:
                # leer etiquetas en caso de que marche bien
                dim = self.patrones.shape[1] - 1
                cla = int(self.patrones[:, dim].max() + 1)
                # primera linea es el titulo del problema
                self.titulo = data[0].replace("\n", "").replace("Patrones: ", "")
                # segunda linea son los nombres de las salidas o clases
                apo = data[1].replace("Salidas:", "").replace(" ", "").replace("\n", "").split(",")
                if len(apo) == cla:
                    self.apodos = apo
                else:
                    self.apodos = []
                    for m in range(cla):
                        self.apodos.append("")
                # tercera linea son los nombres de las entradas
                apo = data[2].replace("Entradas:", "").replace(" ", "").replace("\n", "").split(",")
                if len(apo) == dim:
                    self.entradas = apo
                else:
                    self.entradas = []
                    for i in range(dim):
                        self.entradas.append("")
                # poner valores por defecto para normalizacion
                self.adecuaH = np.ones(dim)
                self.adecuaL = -np.ones(dim)
                self.adecuaN = 0.0
                nota = 1
        else:
            nota = -1
        return nota

    def ClasesPatronadas(self):
        res = True
        dim = self.patrones.shape[1] - 1
        cla = int(self.patrones[:, dim].max() + 1)
        for m in range(cla):
            if not m in self.patrones[:, dim]:
                res = False
                break
        return res

    def ClasesEntrenadas(self):
        res = True
        dim = self.patrones.shape[1] - 1
        cla = int(self.patrones[:, dim].max() + 1)
        for m in range(cla):
            if not m in self.patrones[:self.numPEVT[0], dim]:
                res = False
                break
        return res

    def AdecuarPatrones(self, normal, general, zscore):
        dim = self.patrones.shape[1] - 1
        if zscore == 1:
            self.adecuaN = -1.0
            if not general:
                promedio = np.zeros(dim) + np.mean(self.patrones[:, :dim])
                desviEst = np.zeros(dim) + np.std(self.patrones[:, :dim])
            else:
                promedio = np.mean(self.patrones[:, :dim], axis=0)
                desviEst = np.std(self.patrones[:, :dim], axis=0)
            self.adecuaH = promedio
            self.adecuaL = desviEst
            self.patrones[:, :dim] = (self.patrones[:, :dim] - promedio) / desviEst
        else:
            self.adecuaN = normal
            mn = -normal
            Mn = normal
            if not general:
                m = np.zeros(dim) + np.min(self.patrones[:, :dim])
                M = np.zeros(dim) + np.max(self.patrones[:, :dim])
            else:
                m = np.min(self.patrones[:, :dim], axis=0)
                M = np.max(self.patrones[:, :dim], axis=0)
            self.adecuaH = M
            self.adecuaL = m
            self.patrones[:, :dim] = ((self.patrones[:, :dim] - m) / (M - m)) * (Mn - mn) + mn

class Red():
    def __init__(self):
        # guarda los pesos sinapticos de toda la red
        self.pesW = np.array([0.0])

        # guarda el numero de dendritas para cada neurona
        self.numK = np.array([0])

        # dice si la dendrita esta activa o no
        self.actK = np.array([True])

        # guarda el error calculado para esta red
        self.error = 0.0

        # guarda el error asociado a la mejor solucion hallada
        self.errorB = 0.0

        # maximo valor que pueden tener los pesos sinapticos
        self.lim = 1000.0

    def EjecutarRed(self, entradas):
        X = entradas.copy()
        while X.size < self.pesW.size / 2:
            X = np.hstack((X, entradas))
        W = self.pesW.copy().reshape(-1, 2)
        WH = W[:, 0] - X
        WL = X - W[:, 1]
        Wmki = np.minimum(WH, WL)
        Wmki = Wmki.reshape(-1, entradas.size)
        Smk = Wmki.min(axis=1)
        Smk = np.where(self.actK, Smk, -1000000.0)
        Zm = np.zeros(self.numK.size)
        n = 0
        for m in range(Zm.size):
            Zm[m] = Smk[n:(n + self.numK[m])].max()
            n += self.numK[m]
        Zm = np.exp(Zm)
        Ym = Zm / min(Zm.sum(), 1000000.0)
        return Ym

    def errorCM(self, patrones):
        self.error = 0.0
        dim = patrones.shape[1] - 1
        for p in range(patrones.shape[0]):
            if np.argmax(self.EjecutarRed(patrones[p, :dim])) != patrones[p, dim]:
                self.error += 1.0
        self.error /= patrones.shape[0]

    def errorRL(self, patrones):
        self.error = 0.0
        dim = patrones.shape[1] - 1
        for p in range(patrones.shape[0]):
            self.error += -np.log10(np.clip(self.EjecutarRed(patrones[p, :dim])[int(patrones[p, dim])],
                                            0.000001, 1.0)) / 6.0
        self.error /= patrones.shape[0]

    def KmediasItera(self, patrones, clusters, dimension):
        if clusters != 0:
            self.Kmedias(patrones, clusters, dimension)
        else:
            errorV = np.ones(2)
            cc = 0
            res = True
            while res:
                res = False
                cc += 1
                self.Kmedias(patrones, cc, dimension)
                self.errorCM(patrones)
                if max(errorV[0] - self.error, errorV[1] - errorV[0]) > 0.01:
                    res = True
                errorV[1] = errorV[0]
                errorV[0] = self.error

    def Kmedias(self, patrones, clusters, dimension):
        dim = patrones.shape[1] - 1

        # calcula la dimension de las cajas
        mrg = (patrones[:, :dim].max(axis=0) - patrones[:, :dim].min(axis=0)) * dimension * 0.25

        # crea vector que asocia el patron al centroide
        Ap = np.ones(patrones.shape[0], dtype=int) * -1

        # crea matriz que guardara los centroides
        Cen = np.zeros((clusters, dim))

        # crear la red DMNN haciendo internamente el K means para cada clase m
        self.numK = np.zeros(int(patrones[:, dim].max() + 1), dtype=int)
        for m in range(self.numK.size):
            self.numK[m] = np.where(patrones[:, dim] == m, 1, 0).sum()
        mayor = self.numK.max()
        for m in range(self.numK.size):
            self.numK[m] = max(1, min(self.numK[m], int(np.ceil((self.numK[m] / mayor) * clusters))))
        self.actK = np.ones(self.numK.sum()) > 0
        self.pesW = np.array([])
        for m in range(self.numK.size):

            # escoge los centroides al azar
            enlist = [-1]
            for c in range(self.numK[m]):
                uno = np.random.randint(patrones.shape[0])
                while patrones[uno, dim] != m or uno in enlist:
                    uno = np.random.randint(patrones.shape[0])
                enlist.append(uno)
                Cen[c, :] = patrones[uno, :dim]

            # hacer ciclo del K medias
            cambio = True
            while cambio:
                cambio = False

                # asociar patrones al centroide mas cercano
                for p in range(patrones.shape[0]):
                    if patrones[p, dim] == m:
                        dist = np.sqrt(np.power(Cen[:self.numK[m], :] - patrones[p, :dim], 2.0).sum(axis=1))
                        Ap[p] = np.argmin(dist)
                    else:
                        Ap[p] = -1

                # mover centroides al promedio de sus patrones asociados
                viejo = Cen[:self.numK[m], :].copy()
                Cen[:self.numK[m], :] *= 0.0
                for c in range(self.numK[m]):
                    n = 0.0
                    for p in range(patrones.shape[0]):
                        if Ap[p] == c:
                            Cen[c, :] += patrones[p, :dim]
                            n += 1.0
                    Cen[c, :] /= (n if n != 0 else 0.000001)
                if (False in (viejo == Cen[:self.numK[m], :])):
                    cambio = True

            # crear las cajas para la clase m, utilizando la dimension dada
            vH = (Cen[:self.numK[m], :] + mrg).ravel()
            vL = (Cen[:self.numK[m], :] - mrg).ravel()
            self.pesW = np.concatenate((self.pesW, np.dstack((vH, vL)).ravel()))

    def DyC(self, patrones, margen, unir):
        dim = patrones.shape[1] - 1

        # calcular margen
        mrg = (patrones[:, :dim].max(axis=0) - patrones[:, :dim].min(axis=0)) * margen * 0.05

        # crear la primera caja que abarca todos los patrones
        pertenece = np.array([-1])
        vH = patrones[:, :dim].max(axis=0)
        vL = patrones[:, :dim].min(axis=0)
        cajasH = np.atleast_2d(vH + (vH - vL) * 0.05).copy()
        cajasL = np.atleast_2d(vL - (vH - vL) * 0.05).copy()

        # hacer ciclo de division de cajas
        while -1 in pertenece:
            for c in range(pertenece.size):
                if pertenece[c] == -1:

                    # verifica que solo un tipo de patron este en la caja
                    res = -2
                    for p in range(patrones.shape[0]):
                        den = True
                        if True in np.hstack((patrones[p, :dim] > cajasH[c, :], patrones[p, :dim] < cajasL[c, :])):
                            den = False
                        if den:
                            if res == -2:
                                res = patrones[p, dim]
                            elif res != patrones[p, dim]:
                                res = -1
                                break

                    # decide que hacer segun la respuesta anterior
                    if res != -1:
                        # se asocia la caja a la clase y no se le afecta mas
                        pertenece[c] = res
                    else:
                        # desactiva la caja y crea un nuevo sistema
                        pertenece[c] = -2
                        nuevasH = np.atleast_2d(cajasH[c, :]).copy()
                        nuevasL = np.atleast_2d(cajasL[c, :]).copy()

                        # iterativamente se duplican y dividen
                        for i in range(dim):
                            dl = abs(nuevasH[0, i] - nuevasL[0, i]) / 2.0
                            dl = min(dl + mrg[i], dl * 1.9)
                            for h in range(pow(2, i)):
                                nuevasH = np.concatenate((nuevasH, np.atleast_2d(nuevasH[h, :]).copy()), axis=0)
                                nuevasL = np.concatenate((nuevasL, np.atleast_2d(nuevasL[h, :]).copy()), axis=0)
                                nuevasH[h, i] -= dl
                                nuevasL[-1, i] += dl

                        # se agregan las nuevas cajas a las originales
                        pertenece = np.concatenate((pertenece, np.ones(nuevasH.shape[0], dtype=int) * -1))
                        cajasH = np.concatenate((cajasH, nuevasH), axis=0)
                        cajasL = np.concatenate((cajasL, nuevasL), axis=0)

        # hacer ciclo de union de hipercajas
        if unir:
            cambio = True
            while cambio:
                cambio = False
                for c in range(pertenece.size):
                    if pertenece[c] != -2:
                        for cc in range(c, pertenece.size):
                            if pertenece[c] == pertenece[cc] and c != cc:

                                # forma la caja mas grande posible entre ambas
                                nuevasH = np.where(cajasH[c, :] > cajasH[cc, :], cajasH[c, :], cajasH[cc, :])
                                nuevasL = np.where(cajasL[c, :] < cajasL[cc, :], cajasL[c, :], cajasL[cc, :])

                                # verifica que solo un tipo de patron este en la caja
                                res = True
                                for p in range(patrones.shape[0]):
                                    if patrones[p, dim] != pertenece[c]:
                                        den = True
                                        if True in np.hstack(
                                                (patrones[p, :dim] > nuevasH, patrones[p, :dim] < nuevasL)):
                                            den = False
                                        if den:
                                            res = False
                                            break

                                # desactivar una caja y redimensionar la otra
                                if res:
                                    cambio = True
                                    pertenece[cc] = -2
                                    cajasH[c, :] = nuevasH.copy()
                                    cajasL[c, :] = nuevasL.copy()

        # crear la red DMNN resultante
        self.numK = np.zeros(int(patrones[:, dim].max() + 1), dtype=int)
        self.pesW = np.array([])
        for m in range(self.numK.size):
            k = 0
            for c in range(pertenece.size):
                if pertenece[c] == m:
                    self.pesW = np.concatenate((self.pesW, np.dstack((cajasH[c, :], cajasL[c, :])).ravel()))
                    k += 1
            self.numK[m] = k
        self.actK = np.ones(self.numK.sum()) > 0

    def ImportarRed(self, entradas, clases, adecuaH, adecuaL, adecuaN):
        nota = 0
        fileDir, _ = QFileDialog.getOpenFileName(caption="Abrir Red",
                                                 filter="XML File (*.xml);;Text File (*.txt)")
        if fileDir:
            if ".xml" in fileDir:
                nota = self.LeerXML(entradas, clases, adecuaH, adecuaL, fileDir, adecuaN)
            else:
                nota = self.LeerTXT(entradas, clases, adecuaH, adecuaL, fileDir, adecuaN)
        return nota

    def LeerXML(self, entradas, clases, adecuaH, adecuaL, archivo, adecuaN):
        file = open(archivo, "r")
        raiz = ET.fromstring(file.read())
        file.close()

        if raiz.tag == "DMNN":
            dim = np.zeros(2, dtype=int)
            dim[0] = int(raiz.find("Dimension").find("Entradas").text)
            dim[1] = int(raiz.find("Dimension").find("Clases").text)

            self.pesW = self.ExtraeTags(raiz, "Pesos", True)
            self.actK = self.ExtraeTags(raiz, "Activas", False) > 0
            self.numK = self.ExtraeTags(raiz, "DendritasPorClase", False)
            norH = self.ExtraeTags(raiz, "NormalizacionH", True)
            norL = self.ExtraeTags(raiz, "NormalizacionL", True)
            norN = float(raiz.find("NormalizacionN").text)

            if dim[0] == entradas and dim[1] == clases:
                if norN == 0.0:
                    nota = 1
                elif False in (norH == adecuaH) or False in (norL == adecuaL):
                    nota = 2
                elif norN == adecuaN:
                    nota = 1
                else:
                    nota = 2
            else:
                nota = -2
        else:
            nota = -1
        return nota

    def ExtraeTags(self, raiz, llave, esFloat):
        if esFloat:
            vec = np.array([])
        else:
            vec = np.array([], dtype=int)
        item = {"NormalizacionH": "H", "NormalizacionL": "L", "Pesos": "W",
                "DendritasPorClase": "C", "Activas": "T"}
        i = 0
        while True:
            n = raiz.find(llave).find(item[llave] + str(i))
            if n == None:
                break
            else:
                if esFloat:
                    vec = np.concatenate([vec, np.array([float(n.text)])])
                else:
                    vec = np.concatenate([vec, np.array([int(n.text)], dtype=int)])
                i += 1
        return vec

    def LeerTXT(self, entradas, clases, adecuaH, adecuaL, archivo, adecuaN):
        file = open(archivo, "r")
        data = file.readlines()
        file.close()

        if len(data) >= 16 and "DMNN: " in data[0]:
            dim = np.fromstring(data[2], dtype=int, sep=",")
            self.pesW = np.fromstring(data[4], sep=",")
            self.numK = np.fromstring(data[6], dtype=int, sep=",")
            self.actK = np.fromstring(data[8], dtype=int, sep=",") > 0
            norH = np.fromstring(data[10], sep=",")
            norL = np.fromstring(data[12], sep=",")
            norN = float(data[13].replace("NormalizacionN: ", ""))

            if dim[0] == entradas and dim[1] == clases:
                if norN == 0.0:
                    nota = 1
                elif False in (norH == adecuaH) or False in (norL == adecuaL):
                    nota = 2
                elif norN == adecuaN:
                    nota = 1
                else:
                    nota = 2
            else:
                nota = -2
        else:
            nota = -1
        return nota

    def ExportarRed(self, entradas, clases, adecuaH, adecuaL, titulo, apodos, nombresE, adecuaN):
        fileDir, _ = QFileDialog.getSaveFileName(caption="Guardar Red",
                                                 filter="XML File (*.xml);;Text File (*.txt)")
        if fileDir:
            if ".xml" in fileDir:
                self.CrearXML(entradas, clases, fileDir, adecuaH, adecuaL, titulo, apodos, nombresE, adecuaN)
            else:
                self.CrearTXT(entradas, clases, fileDir, adecuaH, adecuaL, titulo, apodos, nombresE, adecuaN)

    def CrearXML(self, entradas, clases, archivo, adecuaH, adecuaL, titulo, apodos, nombresE, adecuaN):
            datos = ET.Element("DMNN")
            llave = ET.SubElement(datos, "Titulo")
            llave.text = titulo

            llave = ET.SubElement(datos, "Dimension")
            item = ET.SubElement(llave, "Entradas")
            item.text = str(entradas)
            item = ET.SubElement(llave, "Clases")
            item.text = str(clases)

            llave = ET.SubElement(datos, "NombresSalidas")
            for i in range(len(apodos)):
                item = ET.SubElement(llave, "A" + str(i))
                item.text = apodos[i]

            llave = ET.SubElement(datos, "NombresEntradas")
            for i in range(len(nombresE)):
                item = ET.SubElement(llave, "N" + str(i))
                item.text = nombresE[i]

            llave = ET.SubElement(datos, "NormalizacionH")
            for i in range(adecuaH.size):
                item = ET.SubElement(llave, "H" + str(i))
                item.text = str(adecuaH[i])

            llave = ET.SubElement(datos, "NormalizacionL")
            for i in range(adecuaL.size):
                item = ET.SubElement(llave, "L" + str(i))
                item.text = str(adecuaL[i])

            llave = ET.SubElement(datos, "NormalizacionN")
            llave.text = str(adecuaN)

            llave = ET.SubElement(datos, "Pesos")
            for i in range(self.pesW.size):
                item = ET.SubElement(llave, "W" + str(i))
                item.text = str(self.pesW[i])

            llave = ET.SubElement(datos, "Activas")
            for i in range(self.actK.size):
                item = ET.SubElement(llave, "T" + str(i))
                item.text = ("1" if self.actK[i] else "0")

            llave = ET.SubElement(datos, "DendritasPorClase")
            for i in range(self.numK.size):
                item = ET.SubElement(llave, "C" + str(i))
                item.text = str(self.numK[i])

            file = open(archivo, "w")
            file.write(ET.tostring(datos).decode())
            file.close()

    def CrearTXT(self, entradas, clases, archivo, adecuaH, adecuaL, titulo, apodos, nombresE, adecuaN):
        file = open(archivo, "w")
        file.write("DMNN: " + titulo + "\n")

        file.write("Dimension: Entradas, Clases\n")
        txx = np.array2string(np.array([entradas, clases]), separator=",").replace("\n", "")
        file.write(txx.replace(" ", "").replace("[", "").replace("]", "") + "\n")

        file.write("Pesos\n")
        for w in self.pesW[:(self.pesW.size - 1)]:
            file.write(str(w) + ",")
        file.write(str(self.pesW[-1]) + "\n")

        file.write("DendritasPorClase\n")
        txx = np.array2string(self.numK, separator=",").replace("\n", "")
        file.write(txx.replace(" ", "").replace("[", "").replace("]", "") + "\n")

        file.write("Activas\n")
        for a in self.actK[:(self.actK.size - 1)]:
            file.write(("1" if a else "0") + ",")
        file.write(("1" if self.actK[-1] else "0") + "\n")

        file.write("NormalizacionH\n")
        txx = np.array2string(adecuaH, separator=",").replace("\n", "")
        file.write(txx.replace(" ", "").replace("[", "").replace("]", "") + "\n")

        file.write("NormalizacionL\n")
        txx = np.array2string(adecuaL, separator=",").replace("\n", "")
        file.write(txx.replace(" ", "").replace("[", "").replace("]", "") + "\n")

        file.write("NormalizacionN: " + str(adecuaN) + "\n")

        txx = ""
        for i in range(len(apodos) - 1):
            txx += apodos[i] + ", "
        txx += apodos[-1] + "\n"
        file.write("NombresSalidas: " + txx)

        txx = ""
        for i in range(len(nombresE) - 1):
            txx += nombresE[i] + ", "
        txx += nombresE[-1] + "\n"
        file.write("NombresEntradas: " + txx)

        file.close()

    def UnirDendritas(self, patrones, toler):
        param = int(self.pesW.size / (1 if self.numK.sum() == 0 else self.numK.sum()))

        # ver el error actual de la red
        self.errorCM(patrones)
        if self.error <= toler:

            # crear vector de apoyo para operacion interna
            bas = []
            bas.append(np.dstack((np.ones(int(param / 2)), np.zeros(int(param / 2)))).ravel())
            bas.append(abs(bas[0] - 1.0))

            # hacer ciclo de union de hipercajas, dividido segun clase m
            mOperado = []
            for mmm in range(self.numK.size):

                    # se eligen las clases al azar para no sesgar por orden
                    mOperado.append(-1)
                    m = -1
                    while m in mOperado:
                        m = np.random.randint(self.numK.size)
                    mOperado[-1] = m
                    n = self.numK[:m].sum()

                    # se reordenan las dendritas al azar para no sesgar por orden
                    binAct = np.where(self.actK, 1, 0)
                    pedazoW = self.pesW[(n * param):((n + self.numK[m]) * param)].copy()
                    revu = np.vstack((binAct[n:(n + self.numK[m])],
                                      np.arange(self.numK[m], dtype=int)))
                    np.random.shuffle(revu.T)
                    binAct[n:(n + self.numK[m])] = revu[0, :]
                    self.actK = binAct > 0
                    pedazoW = pedazoW.reshape(-1, param)[revu[1, :], :].ravel()
                    self.pesW[(n * param):((n + self.numK[m]) * param)] = pedazoW

                    # revisar cada caja de la clase actual
                    for k in range(self.numK[m]):
                        if self.actK[n]:

                            # verificar si la caja se puede unir con las siguientes
                            nn = 0
                            for kk in range(k, self.numK[m]):
                                if self.actK[n + nn] and kk != k:

                                    # desactivar la caja secundaria kk
                                    self.actK[n + nn] = False

                                    # formar la caja grande entre ambas
                                    ant = self.pesW[(n * param):((n + 1) * param)].copy()
                                    sec = self.pesW[((n + nn) * param):((n + nn + 1) * param)].copy()
                                    H = np.maximum(ant, sec) * bas[0]
                                    L = np.minimum(ant, sec) * bas[1]
                                    self.pesW[(n * param):((n + 1) * param)] = H + L

                                    # comparar el error de la gran caja con el original
                                    self.errorCM(patrones)
                                    if self.error > toler:
                                        self.actK[n + nn] = True
                                        self.pesW[(n * param):((n + 1) * param)] = ant
                                nn += 1
                        n += 1

    def QuitarDendritas(self, patrones, toler):
        dim = patrones.shape[1] - 1

        # crear un vector de prioridad segun area/volumen/hipervolumen de cajas
        priori = self.pesW.reshape(-1, 2).copy()
        priori = (priori[:, 0] - priori[:, 1]).reshape(-1, dim)
        priori = np.prod(priori, axis=1)
        limite = priori.max() * 2.0
        priori = np.where(self.actK, priori, limite)

        # ver el error actual de la red
        self.errorCM(patrones)
        if self.error <= toler:

            # hacer ciclo para ver que pasa al quitar cada caja
            for j in range(priori.size):
                menor = priori.argmin()
                if priori[menor] != limite:
                    priori[menor] = limite

                    # compara el error sin la caja, con el original
                    self.actK[menor] = False
                    if self.ClaseVacia():
                        self.actK[menor] = True
                    else:
                        self.errorCM(patrones)
                        if self.error > toler:
                            self.actK[menor] = True

    def ClaseVacia(self):
        res = False
        unos = np.where(self.actK, 1, 0)
        for m in range(self.numK.size):
            if unos[self.numK[:m].sum():self.numK[:(m + 1)].sum()].sum() == 0:
                res = True
                break
        return res

    def EliminarInhibidas(self):
        res = False
        param = int(self.pesW.size / (1 if self.numK.sum() == 0 else self.numK.sum()))
        NnumK = np.zeros(self.numK.size, dtype=int)
        NpesW = np.array([])
        n = 0
        for m in range(self.numK.size):
            kt = 0
            for k in range(self.numK[m]):
                if self.actK[n]:
                    NpesW = np.concatenate((NpesW, self.pesW[(n * param):((n + 1) * param)]))
                    kt += 1
                n += 1
            NnumK[m] = kt
        self.pesW = NpesW.copy()
        self.numK = NnumK.copy()
        self.actK = np.ones(self.numK.sum()) > 0
        if self.numK.sum() == 0:
            res = True
        return res

    def Deshinibir(self):
        self.actK = np.ones(self.numK.sum()) > 0

    def PrevenirSolape(self):
        for i in range(0, self.pesW.size, 2):
            if self.pesW[i] < self.pesW[i + 1]:
                pro = self.pesW[i:(i + 2)].mean()
                self.pesW[i] = pro + 0.05
                self.pesW[i + 1] = pro - 0.05

    def PrevenirEscape(self):
        self.pesW = np.clip(self.pesW, -self.lim, self.lim)

    def AleatorizarRed(self, porcent):
        esc = np.zeros(self.pesW.size)
        for i in range(0, esc.size, 2):
            esc[i] = self.pesW[i] - self.pesW[i + 1]
            esc[i + 1] = esc[i]
        self.pesW += (np.random.rand(esc.size) * 2.0 - 1.0) * esc * porcent

    def CopiarRed(self, id):
        self.pesW = id.pesW.copy()
        self.numK = id.numK.copy()
        self.actK = id.actK.copy()
        self.lim = id.lim

class LaGradiente(Red):
    def __init__(self):
        Red.__init__(self)
        # vector que conserva el impulso
        self.impU = np.array([0.0])

        # guarda la normalizacion para las componentes A de la operacion
        self.norA = np.array([0.0])

    def IniciaLaGradiente(self, id, patrones):
        self.pesW = id.pesW.copy()
        self.numK = id.numK.copy()
        self.actK = id.actK.copy()
        self.lim = id.lim
        self.impU = np.zeros(self.pesW.size)
        dim = patrones.shape[1] - 1
        vec = patrones[:, :dim].max(axis=0) - patrones[:, :dim].min(axis=0)
        vec = vec / vec.max()
        vec = np.dstack((vec, -vec)).ravel()
        self.norA = vec
        while self.norA.size < self.pesW.size:
            self.norA = np.concatenate([self.norA, vec])

    def EjecutarSuave(self, deseada, entradas):
        # ejecutar la red normalmente
        dim = entradas.size - 1
        X = entradas[:dim].copy()
        while X.size < self.pesW.size / 2:
            X = np.hstack((X, entradas[:dim]))
        W = self.pesW.copy().reshape(-1, 2)
        WH = W[:, 0] - X
        WL = X - W[:, 1]
        Wmki = np.minimum(WH, WL)
        Wmki = Wmki.reshape(-1, dim)
        Smk = Wmki.min(axis=1)
        Smk = np.where(self.actK, Smk, -1000000.0)
        Zm = np.zeros(self.numK.size)
        n = 0
        for m in range(Zm.size):
            Zm[m] = Smk[n:(n + self.numK[m])].max()
            n += self.numK[m]
        Zm = np.exp(Zm)
        Ym = Zm / min(Zm.sum(), 1000000.0)

        # regresar por la ejecucion para hallar los m,k,i,l correspondientes
        # m ya es conocido desde un principio, absoluto
        if deseada:
            vm = int(entradas[-1])
        else:
            vm = np.argmax(Ym)
        # hallamos k, relativo a Smk
        n = 0
        for m in range(Zm.size):
            if m == vm:
                vk = n + np.argmax(Smk[n:(n + self.numK[m])])
                break
            n += self.numK[m]
        # hallamos i, relativo a Wmki
        vi = ((vk + 1) * Wmki.shape[1]) - (Wmki.shape[1] - (np.argmin(Wmki[vk, :]) + 1)) - 1
        # hallamos l, relativo a 0:H o 1:L
        vl = np.argmin(np.array([WH[vi], WL[vi]]))
        # finalmente la variable uno tendra al indice relativo a pesW
        uno = ((vi + 1) * 2) - (2 - (vl + 1)) - 1

        # agregar al W hallado la derivada correspondiente
        if deseada:
            self.impU[uno] += (1.0 - Ym[vm]) / np.log(10.0)
        else:
            self.impU[uno] -= (1.0 - Ym[vm]) / np.log(10.0)

class Individuo(Red):
    def __init__(self):
        Red.__init__(self)
        # guarda otros pesos W creados
        self.otrW = np.array([0.0])

    def IniciaIndividuo(self, id, porcent):
        self.pesW = id.pesW.copy()
        self.numK = id.numK.copy()
        self.actK = id.actK.copy()
        self.lim = id.lim
        self.AleatorizarRed(porcent)
        self.otrW = self.pesW.copy()

    def CrearHijo(self, ids, H, C):
        self.otrW = self.pesW.copy()
        self.otrA = self.actK.copy()
        X = []
        for j in range(3):
            uno = np.random.randint(len(ids))
            while ids[uno] == self or uno in X:
                uno = np.random.randint(len(ids))
            X.append(uno)
        S = np.random.rand(self.pesW.size) < C
        self.pesW = np.where(S, ids[X[0]].pesW + H * (ids[X[1]].pesW - ids[X[2]].pesW), self.pesW)
        self.PrevenirSolape()
        self.PrevenirEscape()

    def CambioW(self):
        antW = self.pesW.copy()
        self.pesW = self.otrW
        self.otrW = antW

class Particula(Red):
    def __init__(self):
        Red.__init__(self)
        # guarda las velocidades de la particula
        self.velW = np.array([0.0])

        # guarda la mejor posicion de la particula
        self.besW = np.array([0.0])

    def IniciaParticula(self, id, porcent, porcevel):
        esc = np.zeros(id.pesW.size)
        for i in range(0, esc.size, 2):
            esc[i] = id.pesW[i] - id.pesW[i + 1]
            esc[i + 1] = esc[i]
        self.velW = (np.random.rand(esc.size) * 2.0 - 1.0) * esc * porcevel
        self.pesW = id.pesW.copy()
        self.numK = id.numK.copy()
        self.actK = id.actK.copy()
        self.lim = id.lim
        self.AleatorizarRed(porcent)
        self.besW = self.pesW.copy()

    def CalcularPaso(self, c1, c2, c3, mejor):
        if self == mejor:
            s = c1 * np.random.rand() * (self.besW - self.pesW)
        else:
            s = c1 * np.random.rand() * (self.besW - self.pesW) +\
                c2 * np.random.rand() * (mejor.besW - self.pesW)
        self.velW *= c3
        self.velW += s
        self.pesW += self.velW
        self.PrevenirSolape()
        self.PrevenirEscape()

# a continuacion las GUIs y sus hilos

class Menu(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.setWindowTitle("SoftwareDMNN-Menú")
        self.setWindowIcon(QIcon("img12.png"))
        self.setObjectName("menu")
        self.version = "1.0.0"
        # array que guardara la id de las otras GUIs
        self.modulo = None
        # id de la clase maestra del softare
        self.motor = None

        # guardara el id de la clase que refresca las graficas
        self.eterno = None

        # definir todos los botones
        self.boton = []
        self.boton.append(QPushButton(QIcon("img14.png"), "Problema"))
        self.boton.append(QPushButton(QIcon("img15.png"), "Inicialización"))
        self.boton.append(QPushButton(QIcon("img16.png"), "Gradiente"))
        self.boton.append(QPushButton(QIcon("img17.png"), "Evolutivo"))
        self.boton.append(QPushButton(QIcon("img18.png"), "Partículas"))
        self.boton.append(QPushButton(QIcon("img19.png"), "Post-Entreno"))
        self.boton.append(QPushButton(QIcon("img20.png"), "Análisis"))
        self.boton.append(QPushButton(QIcon("img22.png"), "Acerca de..."))
        self.boton.append(QPushButton(QIcon("img21.png"), "Tutorial (pdf)"))

        # activar las ayudas de texto para los botones
        self.boton[0].setToolTip(MyToolTip(0))
        self.boton[1].setToolTip(MyToolTip(1))
        self.boton[2].setToolTip(MyToolTip(2))
        self.boton[3].setToolTip(MyToolTip(3))
        self.boton[4].setToolTip(MyToolTip(4))
        self.boton[5].setToolTip(MyToolTip(5))
        self.boton[6].setToolTip(MyToolTip(6))
        self.boton[7].setToolTip(MyToolTip(7))
        self.boton[8].setToolTip(MyToolTip(8))

        # poner nombres a los botones para aplicar estilos
        self.boton[0].setObjectName("m_problema")
        self.boton[1].setObjectName("m_inicializacion")
        self.boton[2].setObjectName("m_gradiente")
        self.boton[3].setObjectName("m_evolutivo")
        self.boton[4].setObjectName("m_particulas")
        self.boton[5].setObjectName("m_postentreno")
        self.boton[6].setObjectName("m_analisis")
        self.boton[7].setObjectName("b_menu")
        self.boton[8].setObjectName("b_menu")

        # conectar los botones a su funcion correspondiente
        self.boton[0].clicked.connect(self.toProblema)
        self.boton[1].clicked.connect(self.toInicializacion)
        self.boton[2].clicked.connect(self.toSGD)
        self.boton[3].clicked.connect(self.toDE)
        self.boton[4].clicked.connect(self.toPSO)
        self.boton[5].clicked.connect(self.toPostEntreno)
        self.boton[6].clicked.connect(self.toAnalisis)
        self.boton[7].clicked.connect(self.forInformacion)
        self.boton[8].clicked.connect(self.forAyuda)

        # crear el contenedor secundario
        rejilla = QGridLayout()

        # poner el escalado por defecto de la rejilla, vertical
        rejilla.setRowStretch(0, 3)
        rejilla.setRowStretch(1, 1)
        rejilla.setRowStretch(2, 1)
        rejilla.setRowStretch(3, 2)
        rejilla.setRowStretch(4, 1)
        rejilla.setRowStretch(5, 2)
        rejilla.setRowStretch(6, 1)
        rejilla.setRowStretch(7, 2)
        rejilla.setRowStretch(8, 3)
        rejilla.setRowStretch(9, 2)

        # poner el escalado por defecto de la rejilla, horizontal
        rejilla.setColumnStretch(0, 7)
        rejilla.setColumnStretch(1, 6)
        rejilla.setColumnStretch(2, 1)
        rejilla.setColumnStretch(3, 1)
        rejilla.setColumnStretch(4, 1)
        rejilla.setColumnStretch(5, 6)
        rejilla.setColumnStretch(6, 1)
        rejilla.setColumnStretch(7, 1)
        rejilla.setColumnStretch(8, 1)
        rejilla.setColumnStretch(9, 6)
        rejilla.setColumnStretch(10, 7)

        # agregar los botones al lugar correspondiente en la rejilla
        rejilla.addWidget(self.boton[0], 3, 1)
        rejilla.addWidget(self.boton[1], 5, 1)
        rejilla.addWidget(self.boton[2], 3, 5)
        rejilla.addWidget(self.boton[3], 5, 5)
        rejilla.addWidget(self.boton[4], 7, 5)
        rejilla.addWidget(self.boton[5], 5, 9)
        rejilla.addWidget(self.boton[6], 7, 9)
        sublay = QVBoxLayout()
        sublay.addWidget(self.boton[8])
        sublay.addWidget(self.boton[7])
        rejilla.addLayout(sublay, 8, 1)

        # pintar las lineas en cruz
        pix = QPixmap("img0.png")
        xtt = QLabel()
        xtt.setPixmap(pix)
        rejilla.addWidget(xtt, 5, 3)
        xtt = QLabel()
        xtt.setPixmap(pix)
        rejilla.addWidget(xtt, 5, 7)

        # pintar la flecha hacia abajo al inicio
        sublay = QHBoxLayout()
        pix = QPixmap("img11.png")
        xtt = QLabel()
        xtt.setPixmap(pix)
        sublay.addWidget(xtt)
        xtt = QLabel()
        xtt.setPixmap(pix)
        sublay.addWidget(xtt)
        pix = QPixmap("img1.png")
        xtt = QLabel()
        xtt.setPixmap(pix)
        sublay.addWidget(xtt)
        pix = QPixmap("img11.png")
        xtt = QLabel()
        xtt.setPixmap(pix)
        sublay.addWidget(xtt)
        rejilla.addLayout(sublay, 4, 1)

        # pintar la flecha hacia abajo al final
        sublay = QHBoxLayout()
        pix = QPixmap("img11.png")
        xtt = QLabel()
        xtt.setPixmap(pix)
        sublay.addWidget(xtt)
        pix = QPixmap("img1.png")
        xtt = QLabel()
        xtt.setPixmap(pix)
        sublay.addWidget(xtt)
        pix = QPixmap("img11.png")
        xtt = QLabel()
        xtt.setPixmap(pix)
        sublay.addWidget(xtt)
        xtt = QLabel()
        xtt.setPixmap(pix)
        sublay.addWidget(xtt)
        rejilla.addLayout(sublay, 6, 9)

        # pintar las flechas hacia la derecha
        pix = QPixmap("img2.png")
        xtt = QLabel()
        xtt.setPixmap(pix)
        rejilla.addWidget(xtt, 7, 4)
        xtt = QLabel()
        xtt.setPixmap(pix)
        rejilla.addWidget(xtt, 5, 4)
        xtt = QLabel()
        xtt.setPixmap(pix)
        rejilla.addWidget(xtt, 3, 4)
        xtt = QLabel()
        xtt.setPixmap(pix)
        rejilla.addWidget(xtt, 5, 8)
        xtt = QLabel()
        xtt.setPixmap(pix)
        rejilla.addWidget(xtt, 7, 8)

        # pintar las lineas en horizontal
        pix = QPixmap("img3.png")
        xtt = QLabel()
        xtt.setPixmap(pix)
        rejilla.addWidget(xtt, 5, 2)
        xtt = QLabel()
        xtt.setPixmap(pix)
        rejilla.addWidget(xtt, 1, 4)
        xtt = QLabel()
        xtt.setPixmap(pix)
        rejilla.addWidget(xtt, 1, 6)
        xtt = QLabel()
        xtt.setPixmap(pix)
        rejilla.addWidget(xtt, 3, 6)
        xtt = QLabel()
        xtt.setPixmap(pix)
        rejilla.addWidget(xtt, 5, 6)
        xtt = QLabel()
        xtt.setPixmap(pix)
        rejilla.addWidget(xtt, 7, 6)

        # pintar las lineas en vertical
        pix = QPixmap("img4.png")
        xtt = QLabel()
        xtt.setPixmap(pix)
        rejilla.addWidget(xtt, 6, 3)
        xtt = QLabel()
        xtt.setPixmap(pix)
        rejilla.addWidget(xtt, 4, 3)
        xtt = QLabel()
        xtt.setPixmap(pix)
        rejilla.addWidget(xtt, 2, 3)
        xtt = QLabel()
        xtt.setPixmap(pix)
        rejilla.addWidget(xtt, 2, 7)
        xtt = QLabel()
        xtt.setPixmap(pix)
        rejilla.addWidget(xtt, 4, 7)
        xtt = QLabel()
        xtt.setPixmap(pix)
        rejilla.addWidget(xtt, 6, 7)

        # pintar las lineas en T hacia la derecha
        pix = QPixmap("img5.png")
        xtt = QLabel()
        xtt.setPixmap(pix)
        rejilla.addWidget(xtt, 3, 3)

        # pintar las lineas en T hacia la izquierda
        pix = QPixmap("img6.png")
        xtt = QLabel()
        xtt.setPixmap(pix)
        rejilla.addWidget(xtt, 3, 7)

        # pintar las lineas en angulo superior izquierdo
        pix = QPixmap("img7.png")
        xtt = QLabel()
        xtt.setPixmap(pix)
        rejilla.addWidget(xtt, 1, 3)

        # pintar las lineas en angulo superior derecho
        pix = QPixmap("img9.png")
        xtt = QLabel()
        xtt.setPixmap(pix)
        rejilla.addWidget(xtt, 1, 7)

        # pintar las lineas en T hacia arriba
        pix = QPixmap("img8.png")
        xtt = QLabel()
        xtt.setPixmap(pix)
        rejilla.addWidget(xtt, 7, 7)

        # pintar las lineas en angulo inferior izquierdo
        pix = QPixmap("img10.png")
        xtt = QLabel()
        xtt.setPixmap(pix)
        rejilla.addWidget(xtt, 7, 3)

        # poner en las esquinas de la rejilla el espaciador invisible
        pix = QPixmap("img11.png")
        xtt = QLabel()
        xtt.setPixmap(pix)
        rejilla.addWidget(xtt, 0, 0)
        xtt = QLabel()
        xtt.setPixmap(pix)
        rejilla.addWidget(xtt, 0, 10)
        xtt = QLabel()
        xtt.setPixmap(pix)
        rejilla.addWidget(xtt, 8, 0)
        xtt = QLabel()
        xtt.setPixmap(pix)
        rejilla.addWidget(xtt, 8, 10)

        # pintar las lineas horizontales largas en la parte superior
        pix = QPixmap("img3.png")
        sublay = QHBoxLayout()
        for i in range(4):
            xtt = QLabel()
            xtt.setPixmap(pix)
            sublay.addWidget(xtt)
        rejilla.addLayout(sublay, 1, 5)

        # crear el contenedor principal, agregar titulo y contenedor secundario
        hiplay = QVBoxLayout()
        hiplay.addStretch(2)
        subhori = QHBoxLayout()
        subhori.addStretch(6)
        pix = QPixmap("img34.png")
        xtt = QLabel()
        xtt.setPixmap(pix)
        subhori.addWidget(xtt)
        xtt = QLabel("Testeador de Red Neuronal Artificial Morfológica de Dendritas\nOmarsaurio - 2019")
        xtt.setAlignment(Qt.AlignCenter)
        xtt.setObjectName("titulo")
        subhori.addWidget(xtt)
        pix = QPixmap("img35.png")
        xtt = QLabel()
        xtt.setPixmap(pix)
        subhori.addWidget(xtt)
        subhori.addStretch(6)
        hiplay.addLayout(subhori)
        hiplay.addStretch(1)
        hiplay.addLayout(rejilla)
        hiplay.addStretch(2)

        # agregar el contenedor principal a la ventana
        self.setLayout(hiplay)

    def Inicializar(self):
        # clase que refresca la informacion del entreno en hilos
        self.eterno = QTimer()
        self.eterno.timeout.connect(self.modulo[2].Refresca)
        self.eterno.timeout.connect(self.modulo[3].Refresca)
        self.eterno.timeout.connect(self.modulo[4].Refresca)
        self.eterno.start(1000)

        # definir estado de los botones
        self.Activacion(0)
        FondoPSI(self)

    def resizeEvent(self, size):
        FondoPSI(self)

    def toProblema(self):
        self.modulo[0].show()
        self.modulo[0].setGeometry(self.geometry())
        FondoPSI(self.modulo[0])
        self.hide()

    def toInicializacion(self):
        self.modulo[1].show()
        self.modulo[1].setGeometry(self.geometry())
        self.modulo[1].InfoIni()
        FondoPSI(self.modulo[1])
        self.hide()

    def toSGD(self):
        self.modulo[2].show()
        self.modulo[2].setGeometry(self.geometry())
        self.modulo[2].InfoIni()
        FondoPSI(self.modulo[2])
        self.hide()

    def toDE(self):
        self.modulo[3].show()
        self.modulo[3].setGeometry(self.geometry())
        self.modulo[3].InfoIni()
        FondoPSI(self.modulo[3])
        self.hide()

    def toPSO(self):
        self.modulo[4].show()
        self.modulo[4].setGeometry(self.geometry())
        self.modulo[4].InfoIni()
        FondoPSI(self.modulo[4])
        self.hide()

    def toPostEntreno(self):
        self.modulo[5].show()
        self.modulo[5].setGeometry(self.geometry())
        self.modulo[5].InfoIni()
        FondoPSI(self.modulo[5])
        self.hide()

    def toAnalisis(self):
        self.modulo[6].show()
        self.modulo[6].setGeometry(self.geometry())
        self.modulo[6].CalcularTodo()
        FondoPSI(self.modulo[6])
        self.hide()

    def forInformacion(self):
        tx = "SoftwareDMNN creado por Omar Jordán Jordán\n" \
             "Universidad del Valle  - Cali - Colombia - 2019\n" \
             "Versión " + self.version + " - código abierto y libre distribución\n" \
             "Contacto: ojorcio@gmail.com\n\n" \
             "Programa hecho para administrar la red neuronal " \
             "artificial morfológica de dendritas (DMNN), incluye:\n" \
             "- Inicialización por K-medias.\n" \
             "- Inicialización por División de Hipercajas (DyC).\n" \
             "- Entreno por Gradiente Descendente Estocástico (SGD).\n" \
             "- Entreno por Evolución Diferencial (DE).\n" \
             "- Entreno por Enjambre de Partículas (PSO).\n" \
             "- Optimización de Dendritas: Unión y Eliminación.\n" \
             "- Métricas de desempeño: Matriz de Confusión y ROC."
        QMessageBox.about(self, "Acerca de SoftwareDMNN", tx)

    def forAyuda(self):
        os.startfile("TutorialDMNN.pdf")

    def Activacion(self, codigo):
        # activa o desactiva los widgets del software
        # 0: no hay patrones, estado inicial o nuevos patrones incorrectos
        # 1: no inicializo red, patrones correctos o nueva red incorrecta
        # 2: todo listo, red inicializada o importada

        if codigo == 0:

            # menu
            for i in range(1, 7):
                self.boton[i].setEnabled(False)

            # motor
            self.motor.redOk = False

            # problema
            self.modulo[0].elTitulo.setText("...")
            self.modulo[0].boton[2].setEnabled(False)
            self.modulo[0].boton[3].setEnabled(False)
            self.modulo[0].boton[4].setEnabled(False)
            self.modulo[0].selectClas.setEnabled(False)
            self.modulo[0].norgen.setEnabled(False)
            self.modulo[0].norzmm.setEnabled(False)
            for i in range(2):
                self.modulo[0].ejes[i].setEnabled(False)
                self.modulo[0].ejes[i].clear()
                self.modulo[0].ejes[i].insertItem(0, "")
                self.modulo[0].ejes[i].setCurrentIndex(0)
            for i in range(len(self.modulo[0].escribe)):
                self.modulo[0].escribe[i].setEnabled(False)
            self.modulo[0].display[0].setText("Patrones: 0")
            self.modulo[0].display[1].setText("Entradas: 0")
            self.modulo[0].display[2].setText("Clases: 0")
            self.modulo[0].display[3].setText("BacheE: 0 / 0")
            self.modulo[0].display[4].setText("BacheV: 0 / 0")
            self.modulo[0].display[5].setText("Testeo: 0")
            self.modulo[0].selectClas.clear()
            self.modulo[0].selectClas.insertItem(0, "clase 0")
            self.modulo[0].selectClas.setCurrentIndex(0)
            self.modulo[0].escribe[4].setText("")
            self.modulo[0].Colorear(-1)

        elif codigo == 1:

            # menu
            for i in range(2, 7):
                self.boton[i].setEnabled(False)
            self.boton[1].setEnabled(True)

            # motor
            self.motor.redOk = False

            # problema
            self.modulo[0].boton[2].setEnabled(True)
            self.modulo[0].boton[3].setEnabled(True)
            self.modulo[0].boton[4].setEnabled(True)
            self.modulo[0].selectClas.setEnabled(True)
            self.modulo[0].norgen.setEnabled(True)
            self.modulo[0].norzmm.setEnabled(True)
            for i in range(2):
                self.modulo[0].ejes[i].setEnabled(True)
            for i in range(len(self.modulo[0].escribe)):
                self.modulo[0].escribe[i].setEnabled(True)

            # inicializacion
            self.modulo[1].boton[2].setEnabled(False)
            self.modulo[1].display[3].setText("Dendritas: 0")
            self.modulo[1].display[4].setText("Inhibidas: 0")
            self.modulo[1].display[5].setText("ECM: ?")

        elif codigo == 2:

            # menu
            for i in range(1, 7):
                self.boton[i].setEnabled(True)

            # motor
            self.motor.redOk = True

            # inicializacion
            self.modulo[1].boton[2].setEnabled(True)

class Problema(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.setWindowTitle("SoftwareDMNN-Problema")
        self.setWindowIcon(QIcon("img12.png"))
        self.setObjectName("problema")
        # id de GUI del menu principal, para volver
        self.menu = None
        # array que guardara la id de las otras GUIs
        self.modulo = None
        # id de la clase maestra del softare
        self.motor = None

        # definir los botones de la GUI con su nombre e indice
        self.boton = []
        self.boton.append(QPushButton(QIcon("img13.png"), "Menú"))
        self.boton.append(QPushButton(QIcon("img23.png"), "Importar Patrones"))
        self.boton.append(QPushButton(QIcon("img26.png"), "Calcular Porcentajes"))
        self.boton.append(QPushButton(QIcon("img27.png"), "Normalizar"))
        self.boton.append(QPushButton(QIcon("img25.png"), "Mezclar y Exportar"))

        # activar las ayudas de texto para los botones
        self.boton[0].setToolTip(MyToolTip(9))
        self.boton[1].setToolTip(MyToolTip(10))
        self.boton[2].setToolTip(MyToolTip(11))
        self.boton[3].setToolTip(MyToolTip(12))
        self.boton[4].setToolTip(MyToolTip(13))

        # poner nombres a los botones para el estilo
        self.boton[0].setObjectName("m_menu")
        for i in range(1, len(self.boton)):
            self.boton[i].setObjectName("b_problema")

        # conectar los botones a su funcion correspondiente
        self.boton[0].clicked.connect(self.toMenu)
        self.boton[1].clicked.connect(self.ImportPatrons)
        self.boton[2].clicked.connect(self.CalcuPorcents)
        self.boton[3].clicked.connect(self.PreProcess)
        self.boton[4].clicked.connect(self.MezclaExp)

        # definir el checkbox para normalizacion general
        self.norgen = QCheckBox("Relativa")
        self.norgen.setChecked(True)
        self.norgen.setToolTip(MyToolTip(14))

        # definir el combobox para normalizacion zscore o minmax
        self.norzmm = QComboBox()
        self.norzmm.insertItem(0, "Min-Max")
        self.norzmm.insertItem(1, "Z-Score")
        self.norzmm.setCurrentIndex(1)
        self.norzmm.setToolTip(MyToolTip(67))

        # definir el selectClas de clase a apodar
        self.selectClas = QComboBox()
        self.selectClas.insertItem(0, "clase 0")
        self.selectClas.setCurrentIndex(0)
        self.selectClas.activated.connect(self.cambioSelectClas)

        # definir el cuadro que cambia de color
        self.colorin = QLabel("")
        self.colorin.setAlignment(Qt.AlignCenter)
        self.colorin.setFixedWidth(16)
        self.setAutoFillBackground(True)
        self.Colorear(-1)

        # definir los dos selectores de ejes a graficar
        self.ejes = []
        for i in range(2):
            self.ejes.append(QComboBox())
            self.ejes[i].insertItem(0, "")
            self.ejes[i].setCurrentIndex(0)
            self.ejes[i].activated.connect(self.cambioEjes)

        # definir las cajas de escritura
        self.escribe = []
        self.escribe.append(QLineEdit("80.00"))
        self.escribe.append(QLineEdit("15.00"))
        self.escribe.append(QLineEdit("128"))
        self.escribe.append(QLineEdit("1024"))
        self.escribe.append(QLineEdit(""))
        self.escribe.append(QLineEdit("1"))

        # activar las ayudas de texto para las cajas de escritura
        self.escribe[0].setToolTip(MyToolTip(15))
        self.escribe[1].setToolTip(MyToolTip(16))
        self.escribe[2].setToolTip(MyToolTip(17))
        self.escribe[3].setToolTip(MyToolTip(18))
        self.escribe[4].setToolTip(MyToolTip(19))
        self.escribe[5].setToolTip(MyToolTip(20))

        # modificar propiedades de las cajas de escritura
        for i in range(len(self.escribe)):
            self.escribe[i].setAlignment(Qt.AlignCenter)

        # limitar la longitud de los textos y su formato
        for i in range(2):
            self.escribe[i].setInputMask("00.00;")
            self.escribe[i].setFixedWidth(150)
        for i in range(2, 4):
            self.escribe[i].setInputMask("0000000;")
            self.escribe[i].setFixedWidth(150)
        self.escribe[4].setInputMask("nnnnnnnnnnnnnnnnnnnnnnnn;")
        self.escribe[5].setInputMask("000;")
        self.escribe[5].setFixedWidth(150)

        # conectar las cajas de escritura
        self.escribe[0].textEdited.connect(self.txEntreno)
        self.escribe[1].textEdited.connect(self.txValidacion)
        self.escribe[2].textEdited.connect(self.txBacheE)
        self.escribe[3].textEdited.connect(self.txBacheV)
        self.escribe[4].textEdited.connect(self.txApodo)

        # definir el coso que guardara el titulo del problema
        self.elTitulo = QLabel("...")
        self.elTitulo.setAlignment(Qt.AlignCenter)

        # definir los textos que seran cambiados con codigo
        self.display = []
        self.display.append(QLabel("Patrones:"))
        self.display.append(QLabel("Entradas:"))
        self.display.append(QLabel("Clases:"))
        self.display.append(QLabel("BacheE:"))
        self.display.append(QLabel("BacheV:"))
        self.display.append(QLabel("Testeo:"))

        # modificar propiedades de los textos cambiantes
        for i in range(len(self.display)):
            self.display[i].setAlignment(Qt.AlignLeft)

        # crear el contenedor principal
        rejilla = QGridLayout()

        # poner el escalado por defecto de la rejilla, vertical
        rejilla.setRowStretch(0, 1)
        rejilla.setRowStretch(1, 100)
        rejilla.setRowStretch(2, 1)

        # poner el escalado por defecto de la rejilla, horizontal
        rejilla.setColumnStretch(0, 1)
        rejilla.setColumnStretch(1, 100)
        rejilla.setColumnStretch(2, 1)
        rejilla.setColumnStretch(3, 25)
        rejilla.setColumnStretch(4, 1)

        # crear los dos contenedores secundarios
        zizq = QVBoxLayout()
        zder = QVBoxLayout()

        # agregar el titulo del problema
        subhori = QHBoxLayout()
        xtt = QLabel("Eje X   .")
        xtt.setAlignment(Qt.AlignLeft)
        subhori.addWidget(xtt)
        subhori.addWidget(self.elTitulo)
        xtt = QLabel(".   Eje Y")
        xtt.setAlignment(Qt.AlignRight)
        subhori.addWidget(xtt)
        subhori.setStretch(1, QSizePolicy.Maximum)
        zizq.addLayout(subhori)

        # agregar los selectores de ejes a graficar y la info de grafica
        subhori = QHBoxLayout()
        subhori.addWidget(self.ejes[0])
        xtt = QLabel("(● Ent, ⬛ Val, ◯ Test)")
        xtt.setAlignment(Qt.AlignCenter)
        subhori.addWidget(xtt)
        subhori.addWidget(self.ejes[1])
        zizq.addLayout(subhori)

        # agregar la grafica al panel izquierdo
        self.figura = QChartView()
        self.figura.chart().setDropShadowEnabled(False)
        self.figura.chart().setMargins(QMargins(0, 0, 0, 0))
        zizq.addWidget(self.figura)

        # agregar los textos cambiantes
        subhori = QHBoxLayout()
        subhori.addWidget(self.display[0])
        subhori.addWidget(self.display[1])
        subhori.addWidget(self.display[2])
        zizq.addLayout(subhori)
        subhori = QHBoxLayout()
        subhori.addWidget(self.display[3])
        subhori.addWidget(self.display[4])
        subhori.addWidget(self.display[5])
        zizq.addLayout(subhori)

        # agregar el titulo del submenu y el boton de volver al menu
        subhori = QHBoxLayout()
        self.boton[0].setFixedWidth(150)
        subhori.addWidget(self.boton[0])
        xtt = QLabel("Problema")
        xtt.setStyleSheet("background-color: rgb(218,220,217);")
        xtt.setAlignment(Qt.AlignCenter)
        xtt.setObjectName("subtitulo")
        subhori.addWidget(xtt)
        zder.addLayout(subhori)

        # comprimir lo de la derecha hacia abajo
        zder.addStretch(1)

        # agregar packete
        pack = QGroupBox("Archivos")
        subpack = QVBoxLayout()
        subpack.addSpacing(10)
        #
        # agregar boton de importar patrones
        subpack.addWidget(self.boton[1])
        #
        pack.setLayout(subpack)
        zder.addWidget(pack)

        # agregar packete
        pack = QGroupBox("Manipulación de Patrones")
        subpack = QVBoxLayout()
        subpack.addSpacing(10)
        #
        # agregar el par escribe/texto para entreno
        subhori = QHBoxLayout()
        subhori.addWidget(self.escribe[0])
        xtt = QLabel("% Patrones Entreno")
        xtt.setAlignment(Qt.AlignCenter)
        subhori.addWidget(xtt)
        subpack.addLayout(subhori)
        #
        # agregar el par escribe/texto para validacion
        subhori = QHBoxLayout()
        subhori.addWidget(self.escribe[1])
        xtt = QLabel("% Patrones Validación")
        xtt.setAlignment(Qt.AlignCenter)
        subhori.addWidget(xtt)
        subpack.addLayout(subhori)
        #
        # agregar el par escribe/texto para bache entreno
        subhori = QHBoxLayout()
        subhori.addWidget(self.escribe[2])
        xtt = QLabel("Bache Entreno")
        xtt.setAlignment(Qt.AlignCenter)
        subhori.addWidget(xtt)
        subpack.addLayout(subhori)
        #
        # agregar el par escribe/texto para bache validacion
        subhori = QHBoxLayout()
        subhori.addWidget(self.escribe[3])
        xtt = QLabel("Bache Validación")
        xtt.setAlignment(Qt.AlignCenter)
        subhori.addWidget(xtt)
        subpack.addLayout(subhori)
        #
        # agregar el boton de mezcla y calculo d eporcentajes
        subhori = QHBoxLayout()
        subhori.addWidget(self.boton[4])
        subhori.addWidget(self.boton[2])
        subpack.addLayout(subhori)
        #
        pack.setLayout(subpack)
        zder.addWidget(pack)

        # agregar packete
        pack = QGroupBox("Edición de Etiquetas")
        subpack = QVBoxLayout()
        subpack.addSpacing(10)
        #
        # agregar trio color/escribe/selectClas para apodo
        subhori = QHBoxLayout()
        subhori.addWidget(self.colorin)
        subhori.addWidget(self.escribe[4])
        subhori.addWidget(self.selectClas)
        subpack.addLayout(subhori)
        #
        pack.setLayout(subpack)
        zder.addWidget(pack)

        # agregar packete
        pack = QGroupBox("Pre-procesamiento")
        subpack = QVBoxLayout()
        subpack.addSpacing(10)
        #
        # agregar caja de escritura de minmax
        self.escribe[5].setVisible(False)
        subpack.addWidget(self.escribe[5])
        #
        # agregar combo zscore-minmax y check abs-rel
        subhori = QHBoxLayout()
        subhori.addWidget(self.norzmm)
        subhori.addWidget(self.norgen)
        subpack.addLayout(subhori)
        #
        # agregar boton de pre procesamiento
        subpack.addWidget(self.boton[3])
        #
        pack.setLayout(subpack)
        zder.addWidget(pack)

        # comprimir lo de la derecha hacia arriba
        zder.addStretch(1)

        # poner los contenedores secundarios en el principal
        rejilla.addLayout(zizq, 1, 1)
        rejilla.addLayout(zder, 1, 3)

        # poner en las esquinas de la rejilla el espaciador invisible
        pix = QPixmap("img11.png")
        xtt = QLabel()
        xtt.setPixmap(pix)
        rejilla.addWidget(xtt, 0, 0)
        xtt = QLabel()
        xtt.setPixmap(pix)
        rejilla.addWidget(xtt, 2, 0)
        xtt = QLabel()
        xtt.setPixmap(pix)
        rejilla.addWidget(xtt, 0, 2)
        xtt = QLabel()
        xtt.setPixmap(pix)
        rejilla.addWidget(xtt, 2, 2)
        xtt = QLabel()
        xtt.setPixmap(pix)
        rejilla.addWidget(xtt, 0, 4)
        xtt = QLabel()
        xtt.setPixmap(pix)
        rejilla.addWidget(xtt, 2, 4)

        # agregar el contenedor principal a la ventana
        self.setLayout(rejilla)

    def toMenu(self):
        self.menu.show()
        self.menu.setGeometry(self.geometry())
        FondoPSI(self.menu)
        self.hide()

    def resizeEvent(self, size):
        FondoPSI(self)

    def ImportPatrons(self):
        res = self.motor.ImportarPatrones()
        if res > 0:
            self.menu.Activacion(1)

            # poner informacion sobre los patrones
            self.elTitulo.setText(self.motor.titulo)
            self.display[0].setText("Patrones: " + str(self.motor.patrones.shape[0]))
            dim = self.motor.patrones.shape[1] - 1
            self.display[1].setText("Entradas: " + str(dim))
            cla = int(self.motor.patrones[:, dim].max() + 1)
            self.display[2].setText("Clases: " + str(cla))

            # poner las entradas en los selectores de ejes
            self.ejes[0].clear()
            self.ejes[1].clear()
            for i in range(dim):
                self.ejes[0].insertItem(i, self.motor.entradas[i])
                self.ejes[1].insertItem(i, self.motor.entradas[i])
            self.motor.grXY[0] = 0
            self.motor.grXY[1] = (1 if dim >= 2 else 0)
            self.ejes[0].setCurrentIndex(0)
            self.ejes[1].setCurrentIndex(self.motor.grXY[1])

            # calcular division de los patrones y poner la informacion
            if res == 1:
                if self.escribe[0].text() == "00.00":
                    self.escribe[0].setText("80.00")
                self.PartirBaches(True)
            else:
                for i in range(2):
                    self.escribe[i].setText("00.00")
                for i in range(2, 4):
                    self.escribe[i].setText("0")
                self.PartirBaches(False)

            # poner las clases en el selectClas y apodo
            self.selectClas.clear()
            for m in range(cla):
                self.selectClas.insertItem(m, "clase " + str(m))
            self.selectClas.setCurrentIndex(0)
            self.escribe[4].setText(self.motor.apodos[0])
            self.Colorear(0)

            QMessageBox.about(self, "Éxito", "Éxito:\n"
                                             "importación completada adecuadamente")

        elif res == -1:
            QMessageBox.about(self, "Error", "Error:\n"
                                             "cabecera de archivo inválida")
        elif res == -2:
            QMessageBox.about(self, "Error", "Error:\n"
                                             "dimensión o número de datos inadecuados")
        elif res == -3:
            QMessageBox.about(self, "Error", "Error:\n"
                                             "índice de clase ausente, asegurese iniciar desde 0")
        if res < 0:
            self.menu.Activacion(0)
            self.figura.chart().removeAllSeries()

    def Colorear(self, clase):
        if clase != -1:
            ccc = Colores(clase, False)
        else:
            ccc = QColor(Qt.white)
            ccc.setAlpha(0)
        ccc = "{r}, {g}, {b}, {a}".format(r=ccc.red(), g=ccc.green(), b=ccc.blue(), a=ccc.alpha())
        self.colorin.setStyleSheet("background-color: rgba(" + ccc + ");")

    def MezclaExp(self):
        self.motor.MezclarExportar(self.boton[3].isEnabled())
        QMessageBox.about(self, "Éxito", "Éxito:\n"
                                         "set de datos aleatorizado")
        self.Graficar()

    def CalcuPorcents(self):
        self.PartirBaches(True)

    def PartirBaches(self, calcu):
        if calcu:
            # obtener datos de la GUI
            dat = []
            for i in range(2):
                dat.append(float(self.escribe[i].text() if self.escribe[i].text() != "." else "0.0"))
                dat[i] = (100.0 if dat[i] == 99.99 else dat[i]) / 100.0
            for i in range(2, 4):
                dat.append(float(self.escribe[i].text() if self.escribe[i].text() != "" else "1"))
            # ciclo que previene particion inadecuada
            res = False
            while not res:
                dat[1] = min(dat[1], 1.0 - dat[0])
                res = self.motor.PartirPatrones(dat[0], dat[1], dat[2], dat[3])
                dat[0] += 0.01
                if dat[0] > 1.0:
                    break
            if res:
                QMessageBox.about(self, "Éxito", "Éxito:\n"
                                                 "particiónes y baches calculados")
            else:
                QMessageBox.about(self, "Advertencia", "Advertencia:\n"
                                                       "las particiónes y baches tuvieron dificultades")

        self.InfoParticion()

    def InfoParticion(self):
        # poner informacion sobre los porcentajes
        self.boton[2].setStyleSheet("")
        self.display[3].setText("BacheE: " + str(self.motor.numBEV[0]) + " / " + str(self.motor.numPEVT[0]))
        self.display[4].setText("BacheV: " + str(self.motor.numBEV[1]) + " / " + str(self.motor.numPEVT[1]))
        self.display[5].setText("Testeo: " + str(self.motor.numPEVT[2]))
        self.Graficar()

    def PreProcess(self):
        self.menu.Activacion(1)
        self.boton[3].setEnabled(False)
        self.norgen.setEnabled(False)
        self.norzmm.setEnabled(False)
        self.escribe[5].setEnabled(False)
        dat = float(self.escribe[5].text() if self.escribe[5].text() != "" else "1")
        dat = (dat if dat > 0.0 else 1.0)
        self.motor.AdecuarPatrones(dat, self.norgen.isChecked(), self.norzmm.currentIndex())
        QMessageBox.about(self, "Éxito", "Éxito:\n"
                                         "normalización aplicada, ya no puede exportar el set")
        self.Graficar()

    def cambioEjes(self):
        self.motor.grXY[0] = self.ejes[0].currentIndex()
        self.motor.grXY[1] = self.ejes[1].currentIndex()
        self.Graficar()

    def cambioSelectClas(self):
        self.escribe[4].setText(self.motor.apodos[self.selectClas.currentIndex()])
        self.Colorear(self.selectClas.currentIndex())

    def txEntreno(self):
        self.boton[2].setStyleSheet("background-color: rgb(239, 172, 122);")

    def txValidacion(self):
        self.boton[2].setStyleSheet("background-color: rgb(239, 172, 122);")

    def txBacheE(self):
        self.boton[2].setStyleSheet("background-color: rgb(239, 172, 122);")

    def txBacheV(self):
        self.boton[2].setStyleSheet("background-color: rgb(239, 172, 122);")

    def txApodo(self):
        self.motor.apodos[self.selectClas.currentIndex()] = self.escribe[4].text()

    def Graficar(self):
        self.figura.chart().removeAllSeries()
        GPatrones(self)
        GAxes(self)

class Inicializacion(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.setWindowTitle("SoftwareDMNN-Inicialización")
        self.setWindowIcon(QIcon("img12.png"))
        self.setObjectName("inicializacion")
        # id de GUI del menu principal, para volver
        self.menu = None
        # array que guardara la id de las otras GUIs
        self.modulo = None
        # id de la clase maestra del softare
        self.motor = None

        # guarda el estado habilitado de los widgets
        self.estadoH = []

        # clase que ejecuta los algoritmos importantes en hilos
        self.elHilo = HiloProcesos2()
        self.elHilo.origen = self
        self.elHilo.finished.connect(self.FinHilo)

        # definir los botones de la GUI con su nombre e indice
        self.boton = []
        self.boton.append(QPushButton(QIcon("img13.png"), "Menú"))
        self.boton.append(QPushButton(QIcon("img23.png"), "Importar Red"))
        self.boton.append(QPushButton(QIcon("img24.png"), "Exportar Red"))
        self.boton.append(QPushButton(QIcon("img28.png"), "Ejecutar K-medias"))
        self.boton.append(QPushButton(QIcon("img28.png"), "Ejecutar Divide y Conquista"))

        # activar las ayudas de texto para los botones
        self.boton[0].setToolTip(MyToolTip(9))
        self.boton[1].setToolTip(MyToolTip(22))
        self.boton[2].setToolTip(MyToolTip(23))
        self.boton[3].setToolTip(MyToolTip(24))
        self.boton[4].setToolTip(MyToolTip(25))

        # poner nombres a los botones para el estilo
        self.boton[0].setObjectName("m_menu")
        for i in range(1, len(self.boton)):
            self.boton[i].setObjectName("b_inicializacion")

        # conectar los botones a su funcion correspondiente
        self.boton[0].clicked.connect(self.toMenu)
        self.boton[1].clicked.connect(self.ImportRed)
        self.boton[2].clicked.connect(self.ExportRed)
        self.boton[3].clicked.connect(self.Kmedias)
        self.boton[4].clicked.connect(self.DyC)

        # definir el selector de grafica
        self.selector = QComboBox()
        self.selector.insertItem(0, "           Patrones")
        self.selector.insertItem(1, "           Patrones + Cajas")
        self.selector.insertItem(2, "           Patrones + Superficie")
        self.selector.insertItem(3, "           Patrones + Cajas + Superficie")
        self.selector.setCurrentIndex(1)
        self.selector.activated.connect(self.cambioSelect)

        # definir las cajas de escritura
        self.escribe = []
        self.escribe.append(QLineEdit("0"))
        self.escribe.append(QLineEdit("20.0"))
        self.escribe.append(QLineEdit("10.0"))

        # activar las ayudas de texto para las cajas de escritura
        self.escribe[0].setToolTip(MyToolTip(26))
        self.escribe[1].setToolTip(MyToolTip(27))
        self.escribe[2].setToolTip(MyToolTip(28))

        # modificar propiedades de las cajas de escritura
        for i in range(len(self.escribe)):
            self.escribe[i].setAlignment(Qt.AlignCenter)
            self.escribe[i].setFixedWidth(150)

        # limitar la longitud de los textos y su formato
        self.escribe[0].setInputMask("000;")
        self.escribe[1].setInputMask("00.00;")
        self.escribe[2].setInputMask("00.00;")

        # conectar las cajas de escritura
        self.escribe[0].textEdited.connect(self.txClusters)
        self.escribe[1].textEdited.connect(self.txDimension)
        self.escribe[2].textEdited.connect(self.txMargen)

        # definir los textos que seran cambiados con codigo
        self.display = []
        self.display.append(QLabel("Patrones:"))
        self.display.append(QLabel("Entradas:"))
        self.display.append(QLabel("Clases:"))
        self.display.append(QLabel("Dendritas:"))
        self.display.append(QLabel("Inhibidas:"))
        self.display.append(QLabel("ECM:"))

        # modificar propiedades de los textos cambiantes
        for i in range(len(self.display)):
            self.display[i].setAlignment(Qt.AlignLeft)

        # definir el checkbox para unir en enterno por DyC
        self.unirDyC = QCheckBox("Unir Dendritas")
        self.unirDyC.setChecked(True)
        self.unirDyC.setToolTip(MyToolTip(50))

        # crear el contenedor principal
        rejilla = QGridLayout()

        # poner el escalado por defecto de la rejilla, vertical
        rejilla.setRowStretch(0, 1)
        rejilla.setRowStretch(1, 100)
        rejilla.setRowStretch(2, 1)

        # poner el escalado por defecto de la rejilla, horizontal
        rejilla.setColumnStretch(0, 1)
        rejilla.setColumnStretch(1, 100)
        rejilla.setColumnStretch(2, 1)
        rejilla.setColumnStretch(3, 25)
        rejilla.setColumnStretch(4, 1)

        # crear los dos contenedores secundarios
        zizq = QVBoxLayout()
        zder = QVBoxLayout()

        # agregar selector de grafica de grafica
        zizq.addWidget(self.selector)

        # agregar la grafica al panel izquierdo
        self.figura = QChartView()
        self.figura.chart().setDropShadowEnabled(False)
        self.figura.chart().setMargins(QMargins(0, 0, 0, 0))
        zizq.addWidget(self.figura)

        # agregar los textos cambiantes
        subhori = QHBoxLayout()
        subhori.addWidget(self.display[0])
        subhori.addWidget(self.display[1])
        subhori.addWidget(self.display[2])
        zizq.addLayout(subhori)
        subhori = QHBoxLayout()
        subhori.addWidget(self.display[3])
        subhori.addWidget(self.display[4])
        subhori.addWidget(self.display[5])
        zizq.addLayout(subhori)

        # agregar el titulo del submenu y el boton de volver al menu
        subhori = QHBoxLayout()
        self.boton[0].setFixedWidth(150)
        subhori.addWidget(self.boton[0])
        xtt = QLabel("Inicialización")
        xtt.setStyleSheet("background-color: rgb(252,250,200);")
        xtt.setAlignment(Qt.AlignCenter)
        xtt.setObjectName("subtitulo")
        subhori.addWidget(xtt)
        zder.addLayout(subhori)

        # comprimir lo de la derecha hacia abajo
        zder.addStretch(1)

        # agregar packete
        pack = QGroupBox("Archivos")
        subpack = QVBoxLayout()
        subpack.addSpacing(10)
        #
        # agregar botones de importar exportar
        subhori = QHBoxLayout()
        subhori.addWidget(self.boton[1])
        subhori.addWidget(self.boton[2])
        subpack.addLayout(subhori)
        #
        pack.setLayout(subpack)
        zder.addWidget(pack)

        # agregar packete
        pack = QGroupBox("Algoritmo 1")
        subpack = QVBoxLayout()
        subpack.addSpacing(10)
        #
        # agregar el par escribe/texto para clusters
        subhori = QHBoxLayout()
        subhori.addWidget(self.escribe[0])
        xtt = QLabel("Dendritas por Clase")
        xtt.setAlignment(Qt.AlignCenter)
        subhori.addWidget(xtt)
        subpack.addLayout(subhori)
        #
        # agregar el par escribe/texto para dimension
        subhori = QHBoxLayout()
        subhori.addWidget(self.escribe[1])
        xtt = QLabel("% Dimensión de Cajas")
        xtt.setAlignment(Qt.AlignCenter)
        subhori.addWidget(xtt)
        subpack.addLayout(subhori)
        #
        # agregar el boton de Kmedias
        subpack.addWidget(self.boton[3])
        #
        pack.setLayout(subpack)
        zder.addWidget(pack)

        # agregar packete
        pack = QGroupBox("Algoritmo 2")
        subpack = QVBoxLayout()
        subpack.addSpacing(10)
        #
        # agregar el par escribe/texto para dimension
        subpack.addWidget(self.unirDyC)
        subhori = QHBoxLayout()
        subhori.addWidget(self.escribe[2])
        xtt = QLabel("% Margen de Cajas")
        xtt.setAlignment(Qt.AlignCenter)
        subhori.addWidget(xtt)
        subpack.addLayout(subhori)
        #
        # agregar el boton de DyC
        subpack.addWidget(self.boton[4])
        #
        pack.setLayout(subpack)
        zder.addWidget(pack)

        # comprimir lo de la derecha hacia arriba
        zder.addStretch(1)

        # poner los contenedores secundarios en el principal
        rejilla.addLayout(zizq, 1, 1)
        rejilla.addLayout(zder, 1, 3)

        # poner en las esquinas de la rejilla el espaciador invisible
        pix = QPixmap("img11.png")
        xtt = QLabel()
        xtt.setPixmap(pix)
        rejilla.addWidget(xtt, 0, 0)
        xtt = QLabel()
        xtt.setPixmap(pix)
        rejilla.addWidget(xtt, 2, 0)
        self.vivo = QLabel()
        gif = QMovie("wait.gif")
        self.vivo.setMovie(gif)
        gif.start()
        self.vivo.setVisible(False)
        rejilla.addWidget(self.vivo, 0, 2)
        xtt = QLabel()
        xtt.setPixmap(pix)
        rejilla.addWidget(xtt, 2, 2)
        xtt = QLabel()
        xtt.setPixmap(pix)
        rejilla.addWidget(xtt, 0, 4)
        xtt = QLabel()
        xtt.setPixmap(pix)
        rejilla.addWidget(xtt, 2, 4)

        # agregar el contenedor principal a la ventana
        self.setLayout(rejilla)

    def toMenu(self):
        self.selector.setCurrentIndex(1)
        self.menu.show()
        self.menu.setGeometry(self.geometry())
        FondoPSI(self.menu)
        self.hide()

    def resizeEvent(self, size):
        FondoPSI(self)

    def ImportRed(self):
        entradas = self.motor.patrones.shape[1] - 1
        clases = int(self.motor.patrones[:, entradas].max() + 1)
        res = self.motor.laRed.ImportarRed(entradas, clases, self.motor.adecuaH,
                                           self.motor.adecuaL, self.motor.adecuaN)
        if res > 0:
            self.menu.Activacion(2)
            if res == 1:
                QMessageBox.about(self, "Éxito", "Éxito:\n"
                                                 "red cargada adecuadamente")
            else:
                QMessageBox.about(self, "Advertencia", "Advertencia:\n"
                                                       "los valores de normalización No coinciden")
            self.Graficar()

        elif res < 0:
            self.menu.Activacion(1)
            if res == -1:
                QMessageBox.about(self, "Error", "Error:\n"
                                                 "cabecera de archivo inválida")
            elif res == -2:
                QMessageBox.about(self, "Error", "Error:\n"
                                                 "las dimensiónes No coinciden con el problema")
            self.Graficar()

    def ExportRed(self):
        entradas = self.motor.patrones.shape[1] - 1
        clases = int(self.motor.patrones[:, entradas].max() + 1)
        self.motor.laRed.ExportarRed(entradas, clases, self.motor.adecuaH,
                                     self.motor.adecuaL, self.motor.titulo,
                                     self.motor.apodos, self.motor.entradas,
                                     self.motor.adecuaN)

    def Kmedias(self):
        self.menu.Activacion(2)
        self.boton[3].setStyleSheet("")
        self.elHilo.esKmedias = True
        self.elHilo.start()

    def DyC(self):
        self.menu.Activacion(2)
        self.boton[4].setStyleSheet("")
        self.elHilo.esKmedias = False
        self.elHilo.start()

    def cambioSelect(self):
        self.Graficar()

    def txClusters(self):
        self.boton[3].setStyleSheet("background-color: rgb(239, 172, 122);")

    def txDimension(self):
        self.boton[3].setStyleSheet("background-color: rgb(239, 172, 122);")

    def txMargen(self):
        self.boton[4].setStyleSheet("background-color: rgb(239, 172, 122);")

    def Graficar(self):
        self.figura.chart().removeAllSeries()
        if self.motor.redOk:
            GSuperficie(self)
        GPatrones(self)
        if self.motor.redOk:
            GRed(self, self.motor.laRed)
        GAxes(self)
        self.InfoRed()

    def InfoIni(self):
        self.Graficar()
        # poner informacion sobre los patrones
        self.display[0].setText("PatronesE: " + str(self.motor.numPEVT[0]))
        dim = self.motor.patrones.shape[1] - 1
        self.display[1].setText("Entradas: " + str(dim))
        cla = int(self.motor.patrones[:, dim].max() + 1)
        self.display[2].setText("Clases: " + str(cla))

    def InfoRed(self):
        # poner informacion sobre la red
        if self.motor.redOk:
            act = np.where(self.motor.laRed.actK, 1, 0).sum()
            self.display[3].setText("Dendritas: " + str(act))
            self.display[4].setText("Inhibidas: " + str(self.motor.laRed.actK.size - act))
            self.motor.laRed.errorCM(self.motor.patrones[0:self.motor.numPEVT[0], :])
            self.display[5].setText("ECM: " + str(round(self.motor.laRed.error, 6)))

    def BloquearCosas(self, block):
        if block:
            self.estadoH = []
            for i in range(len(self.boton)):
                self.estadoH.append(self.boton[i].isEnabled())
                self.boton[i].setEnabled(False)
            for i in range(len(self.escribe)):
                self.estadoH.append(self.escribe[i].isEnabled())
                self.escribe[i].setEnabled(False)
            self.estadoH.append(self.selector.isEnabled())
            self.selector.setEnabled(False)
            self.estadoH.append(self.unirDyC.isEnabled())
            self.unirDyC.setEnabled(False)
        else:
            n = 0
            for i in range(len(self.boton)):
                self.boton[i].setEnabled(self.estadoH[n])
                n += 1
            for i in range(len(self.escribe)):
                self.escribe[i].setEnabled(self.estadoH[n])
                n += 1
            self.selector.setEnabled(self.estadoH[n])
            self.unirDyC.setEnabled(self.estadoH[n + 1])

    def FinHilo(self):
        QMessageBox.about(self, "Éxito", "Éxito:\n"
                                         "inicialización ejecutada correctamente")
        self.Graficar()

class HiloProcesos2(QThread):
    def __init__(self):
        QThread.__init__(self)
        self.origen = None
        self.esKmedias = True

    def run(self):
        elId = self.origen
        elId.vivo.setVisible(True)
        elId.BloquearCosas(True)
        if self.esKmedias:
            clu = int(elId.escribe[0].text() if elId.escribe[0].text() != "" else "0")
            dim = float(elId.escribe[1].text() if elId.escribe[1].text() != "." else "0.0")
            dim = (100.0 if dim == 99.99 else dim)
            dim = (5.0 if dim < 5.0 else dim) / 100.0
            elId.motor.laRed.KmediasItera(elId.motor.patrones[0:elId.motor.numPEVT[0], :], clu, dim)
        else:
            mar = float(elId.escribe[2].text() if elId.escribe[2].text() != "." else "0.0")
            mar = (100.0 if mar == 99.99 else mar) / 100.0
            elId.motor.laRed.DyC(elId.motor.patrones[0:elId.motor.numPEVT[0], :], mar,
                                 elId.unirDyC.isChecked())
        time.sleep(1)
        elId.BloquearCosas(False)
        elId.vivo.setVisible(False)

class Entreno(QWidget):
    def __init__(self, tipo):
        QWidget.__init__(self)
        # que clase de entreno es: 0:SGD, 1:DE, 2:PSO
        self.tipo = tipo
        if tipo == 0:
            self.setWindowTitle("SoftwareDMNN-SGD")
            self.setObjectName("gradiente")
        elif tipo == 1:
            self.setWindowTitle("SoftwareDMNN-DE")
            self.setObjectName("evolutivo")
        elif tipo == 2:
            self.setWindowTitle("SoftwareDMNN-PSO")
            self.setObjectName("particulas")
        self.setWindowIcon(QIcon("img12.png"))
        # id de GUI del menu principal, para volver
        self.menu = None
        # array que guardara la id de las otras GUIs
        self.modulo = None
        # id de la clase maestra del softare
        self.motor = None

        # guarda el estado actual de la GUI: 0:stop, 1:pausa, 2:run
        self.estado = 0

        # clase que ejecuta los algoritmos importantes en hilos
        self.elHilo = HiloProcesos1()
        self.elHilo.origen = self
        self.elHilo.finished.connect(self.FinHilo)

        # dice si esta disponible para pintar
        self.libre = True

        # definir los botones de la GUI con su nombre e indice
        self.boton = []
        self.boton.append(QPushButton(QIcon("img13.png"), "Menú"))
        self.boton.append(QPushButton(QIcon("img28.png"), ""))
        self.boton.append(QPushButton(QIcon("img29.png"), ""))
        self.boton.append(QPushButton(QIcon("img30.png"), ""))
        self.boton.append(QPushButton(QIcon("img36.png"), "Por Defecto"))

        # activar las ayudas de texto para los botones
        self.boton[0].setToolTip(MyToolTip(9))
        self.boton[1].setToolTip(MyToolTip(29))
        self.boton[2].setToolTip(MyToolTip(30))
        self.boton[3].setToolTip(MyToolTip(31))
        self.boton[4].setToolTip(MyToolTip(21))

        # poner nombres a los botones para el estilo
        self.boton[0].setObjectName("m_menu")
        for i in range(1, len(self.boton)):
            if tipo == 0:
                self.boton[i].setObjectName("b_gradiente")
            elif tipo == 1:
                self.boton[i].setObjectName("b_evolutivo")
            elif tipo == 2:
                self.boton[i].setObjectName("b_particulas")

        # conectar los botones a su funcion correspondiente
        self.boton[0].clicked.connect(self.toMenu)
        self.boton[1].clicked.connect(self.Play)
        self.boton[2].clicked.connect(self.Pause)
        self.boton[3].clicked.connect(self.Stop)
        self.boton[4].clicked.connect(self.Defecto)

        # definir el selector de grafica
        self.selector = QComboBox()
        self.selector.insertItem(0, "           Patrones (entrenamiento veloz, sin refresco)")
        self.selector.insertItem(1, "           Patrones + Cajas")
        self.selector.insertItem(2, "           Patrones + Superficie")
        self.selector.insertItem(3, "           Patrones + Cajas + Superficie")
        self.selector.insertItem(4, "           Entrenamiento: roj:eE, azu:eV, mag:eT, ver:Den")
        if tipo == 1:
            self.selector.insertItem(5, "           Genes (vista de muestra)")
        elif tipo == 2:
            self.selector.insertItem(5, "           Partículas (vista de muestra)")
        self.selector.setCurrentIndex(4)
        self.selector.activated.connect(self.cambioSelect)

        # definir el check para DE o SGD
        if tipo == 1:
            self.ordenamiento = QComboBox()
            self.ordenamiento.insertItem(0, "Seleccionar Padre vs Hijo")
            self.ordenamiento.insertItem(1, "Seleccionar Par Global Azaroso")
            self.ordenamiento.setCurrentIndex(0)
            self.ordenamiento.setToolTip(MyToolTip(32))
        else:
            self.ordenamiento = None

        # definir las cajas de escritura
        self.escribe = []
        self.escribe.append(QLineEdit(""))
        self.escribe.append(QLineEdit(""))
        self.escribe.append(QLineEdit(""))
        if tipo == 0:
            self.escribe.append(QLineEdit(""))
            self.escribe.append(QLineEdit(""))
            self.escribe.append(QLineEdit(""))
            # inutil
            self.escribe.append(QLineEdit(""))
        elif tipo == 1:
            self.escribe.append(QLineEdit(""))
            self.escribe.append(QLineEdit(""))
            self.escribe.append(QLineEdit(""))
            # inutil
            self.escribe.append(QLineEdit(""))
        elif tipo == 2:
            self.escribe.append(QLineEdit(""))
            self.escribe.append(QLineEdit(""))
            self.escribe.append(QLineEdit(""))
            self.escribe.append(QLineEdit(""))
        self.escribe.append(QLineEdit(""))

        # activar las ayudas de texto para las cajas de escritura
        self.escribe[0].setToolTip(MyToolTip(34))
        self.escribe[1].setToolTip(MyToolTip(35))
        if tipo == 0:
            self.escribe[2].setToolTip(MyToolTip(36))
            self.escribe[3].setToolTip(MyToolTip(39))
            self.escribe[4].setToolTip(MyToolTip(40))
            self.escribe[5].setToolTip(MyToolTip(41))
        elif tipo == 1:
            self.escribe[2].setToolTip(MyToolTip(37))
            self.escribe[3].setToolTip(MyToolTip(42))
            self.escribe[4].setToolTip(MyToolTip(43))
            self.escribe[5].setToolTip(MyToolTip(44))
        elif tipo == 2:
            self.escribe[2].setToolTip(MyToolTip(38))
            self.escribe[3].setToolTip(MyToolTip(45))
            self.escribe[4].setToolTip(MyToolTip(46))
            self.escribe[5].setToolTip(MyToolTip(47))
            self.escribe[6].setToolTip(MyToolTip(48))
        self.escribe[7].setToolTip(MyToolTip(49))

        # modificar propiedades de las cajas de escritura
        for i in range(len(self.escribe)):
            self.escribe[i].setAlignment(Qt.AlignCenter)
            self.escribe[i].setFixedWidth(150)

        # limitar la longitud de los textos y su formato
        self.escribe[0].setInputMask("000000000;")
        self.escribe[1].setInputMask("00.00;")
        self.escribe[2].setInputMask("00.00;")
        if tipo == 0:
            self.escribe[3].setInputMask("0000000;")
            self.escribe[4].setInputMask("0.000;")
            self.escribe[5].setInputMask("0.000;")
        elif tipo == 1:
            self.escribe[3].setInputMask("0000;")
            self.escribe[4].setInputMask("0.000;")
            self.escribe[5].setInputMask("00.00;")
        elif tipo == 2:
            self.escribe[3].setInputMask("0000;")
            self.escribe[4].setInputMask("0.000;")
            self.escribe[5].setInputMask("0.000;")
            self.escribe[6].setInputMask("0.000;")
        self.escribe[7].setInputMask("0.000000;")

        # definir los textos que seran cambiados con codigo
        self.display = []
        self.display.append(QLabel(" "))
        self.display.append(QLabel("Dendr:"))
        self.display.append(QLabel("ERL:"))
        self.display.append(QLabel("ECM:"))

        # modificar propiedades de los textos cambiantes
        for i in range(len(self.display)):
            self.display[i].setAlignment(Qt.AlignLeft)
        self.display[2].setToolTip(MyToolTip(58))

        # definir la barra de progreso para el entreno
        self.progreso = QProgressBar()
        self.progreso.setMaximum(100)
        if tipo == 0:
            self.progreso.setObjectName("v_gradiente")
        elif tipo == 1:
            self.progreso.setObjectName("v_evolutivo")
        elif tipo == 2:
            self.progreso.setObjectName("v_particulas")
        self.progreso.setAlignment(Qt.AlignRight)

        # crear el contenedor principal
        rejilla = QGridLayout()

        # poner el escalado por defecto de la rejilla, vertical
        rejilla.setRowStretch(0, 1)
        rejilla.setRowStretch(1, 100)
        rejilla.setRowStretch(2, 1)

        # poner el escalado por defecto de la rejilla, horizontal
        rejilla.setColumnStretch(0, 1)
        rejilla.setColumnStretch(1, 100)
        rejilla.setColumnStretch(2, 1)
        rejilla.setColumnStretch(3, 25)
        rejilla.setColumnStretch(4, 1)

        # crear los dos contenedores secundarios
        zizq = QVBoxLayout()
        zder = QVBoxLayout()

        # agregar selector de grafica
        zizq.addWidget(self.selector)

        # agregar la grafica al panel izquierdo
        self.figura = QChartView()
        self.figura.chart().setDropShadowEnabled(False)
        self.figura.chart().setMargins(QMargins(0, 0, 0, 0))
        zizq.addWidget(self.figura)

        # agregar los textos cambiantes
        subhori = QHBoxLayout()
        subhori.addWidget(self.display[1])
        subhori.addWidget(self.display[3])
        subhori.addWidget(self.display[2])
        zizq.addLayout(subhori)

        # agregar la barra de progreso de entrenamiento
        subhori = QHBoxLayout()
        subhori.addWidget(self.progreso)
        subhori.addWidget(self.display[0])
        zizq.addLayout(subhori)

        # agregar el titulo del submenu y el boton de volver al menu
        subhori = QHBoxLayout()
        self.boton[0].setFixedWidth(150)
        self.boton[4].setFixedWidth(150)
        subsubv = QVBoxLayout()
        subsubv.addWidget(self.boton[0])
        subsubv.addWidget(self.boton[4])
        subhori.addLayout(subsubv)
        if tipo == 0:
            xtt = QLabel("Gradiente\nDescendente\nEstocástico")
            xtt.setStyleSheet("background-color: rgb(255,220,215);")
        elif tipo == 1:
            xtt = QLabel("Evolución\nDiferencial")
            xtt.setStyleSheet("background-color: rgb(209,255,207);")
        elif tipo == 2:
            xtt = QLabel("Optimización\npor Enjambre\nde Partículas")
            xtt.setStyleSheet("background-color: rgb(211,207,255);")
        xtt.setAlignment(Qt.AlignCenter)
        xtt.setObjectName("subtitulo")
        subhori.addWidget(xtt)
        zder.addLayout(subhori)

        # comprimir lo de la derecha hacia abajo
        zder.addStretch(1)

        # agregar packete
        pack = QGroupBox("Parámetros de Escalabilidad")
        subpack = QVBoxLayout()
        subpack.addSpacing(10)
        #
        # agregar el par escribe/texto
        subhori = QHBoxLayout()
        subhori.addWidget(self.escribe[3])
        if tipo == 0:
            xtt = QLabel("Mini-Bache")
        elif tipo == 1:
            xtt = QLabel("Dimensión de Población")
        elif tipo == 2:
            xtt = QLabel("Número de Partículas")
        xtt.setAlignment(Qt.AlignCenter)
        subhori.addWidget(xtt)
        subpack.addLayout(subhori)
        #
        # agregar el check para DE o SGD
        if tipo == 1:
            subpack.addWidget(self.ordenamiento)
        #
        pack.setLayout(subpack)
        zder.addWidget(pack)

        # agregar packete
        pack = QGroupBox("Parámetros del Algoritmo")
        subpack = QVBoxLayout()
        subpack.addSpacing(10)
        #
        # agregar el par escribe/texto
        subhori = QHBoxLayout()
        subhori.addWidget(self.escribe[4])
        if tipo == 0:
            xtt = QLabel("Paso alfa")
        elif tipo == 1:
            xtt = QLabel("h Escala de Mutación")
        elif tipo == 2:
            xtt = QLabel("c1 Memoria Local")
        xtt.setAlignment(Qt.AlignCenter)
        subhori.addWidget(xtt)
        subpack.addLayout(subhori)
        #
        # agregar el par escribe/texto
        subhori = QHBoxLayout()
        subhori.addWidget(self.escribe[5])
        if tipo == 0:
            xtt = QLabel("Fricción beta")
        elif tipo == 1:
            xtt = QLabel("c % Recombinación")
        elif tipo == 2:
            xtt = QLabel("c2 Memoria Global")
        xtt.setAlignment(Qt.AlignCenter)
        subhori.addWidget(xtt)
        subpack.addLayout(subhori)
        #
        # agregar el par escribe/texto
        if tipo == 2:
            subhori = QHBoxLayout()
            subhori.addWidget(self.escribe[6])
            xtt = QLabel("c3 Fricción")
            xtt.setAlignment(Qt.AlignCenter)
            subhori.addWidget(xtt)
            subpack.addLayout(subhori)
        #
        # agregar el par escribe/texto
        subhori = QHBoxLayout()
        subhori.addWidget(self.escribe[2])
        xtt = QLabel("Factor Amortiguamiento")
        xtt.setAlignment(Qt.AlignCenter)
        subhori.addWidget(xtt)
        subpack.addLayout(subhori)
        #
        pack.setLayout(subpack)
        zder.addWidget(pack)

        # agregar packete
        pack = QGroupBox("Disminuir Dendritas")
        subpack = QVBoxLayout()
        subpack.addSpacing(10)
        #
        # agregar el par escribe/texto
        subhori = QHBoxLayout()
        subhori.addWidget(self.escribe[1])
        xtt = QLabel("% Probabilidad Inhibir")
        xtt.setAlignment(Qt.AlignCenter)
        subhori.addWidget(xtt)
        subpack.addLayout(subhori)
        #
        subhori = QHBoxLayout()
        subhori.addWidget(self.escribe[7])
        xtt = QLabel("Tolerancia, ECM Máximo")
        xtt.setAlignment(Qt.AlignCenter)
        subhori.addWidget(xtt)
        subpack.addLayout(subhori)
        #
        pack.setLayout(subpack)
        zder.addWidget(pack)

        # agregar packete
        pack = QGroupBox("Ejecución")
        subpack = QVBoxLayout()
        subpack.addSpacing(10)
        #
        # agregar linea y botones de entreno
        subhori = QHBoxLayout()
        subhori.addWidget(self.escribe[0])
        subhori.addWidget(self.boton[1])
        subhori.addWidget(self.boton[2])
        subhori.addWidget(self.boton[3])
        subpack.addLayout(subhori)
        #
        pack.setLayout(subpack)
        zder.addWidget(pack)

        # comprimir lo de la derecha hacia arriba
        zder.addStretch(1)

        # poner los contenedores secundarios en el principal
        rejilla.addLayout(zizq, 1, 1)
        rejilla.addLayout(zder, 1, 3)

        # poner en las esquinas de la rejilla el espaciador invisible
        pix = QPixmap("img11.png")
        xtt = QLabel()
        xtt.setPixmap(pix)
        rejilla.addWidget(xtt, 0, 0)
        xtt = QLabel()
        xtt.setPixmap(pix)
        rejilla.addWidget(xtt, 2, 0)
        self.vivo = QLabel()
        gif = QMovie("wait.gif")
        self.vivo.setMovie(gif)
        gif.start()
        self.vivo.setVisible(False)
        rejilla.addWidget(self.vivo, 0, 2)
        xtt = QLabel()
        xtt.setPixmap(pix)
        rejilla.addWidget(xtt, 2, 2)
        xtt = QLabel()
        xtt.setPixmap(pix)
        rejilla.addWidget(xtt, 0, 4)
        xtt = QLabel()
        xtt.setPixmap(pix)
        rejilla.addWidget(xtt, 2, 4)

        # agregar el contenedor principal a la ventana
        self.setLayout(rejilla)
        self.Defecto()

    def toMenu(self):
        self.selector.setCurrentIndex(4)
        self.menu.show()
        self.menu.setGeometry(self.geometry())
        FondoPSI(self.menu)
        self.hide()

    def resizeEvent(self, size):
        FondoPSI(self)

    def Play(self):
        self.estado = 2
        if self.selector.currentIndex() == 2 or self.selector.currentIndex() == 3:
            self.selector.setCurrentIndex(1)
        self.elHilo.start()

    def Pause(self):
        self.estado = 1

    def Stop(self):
        self.estado = 0
        if not self.boton[2].isEnabled():
            self.motor.agentes = []
            self.motor.runData[0] = 0
            self.BloquearCosas()

    def cambioSelect(self):
        self.Graficar()

    def Defecto(self):
        self.escribe[0].setText("1000") # iteraciones
        self.escribe[1].setText("20.0")  # probabilidad inhibir
        self.escribe[7].setText("0.1")  # tolerancia inhibir
        if self.tipo == 0:
            self.escribe[3].setText("32")  # minibache
            self.escribe[4].setText("0.1")  # paso alfa
            self.escribe[5].setText("0.9")  # friccion beta
            self.escribe[2].setText("80.0")  # amortiguamiento
        elif self.tipo == 1:
            self.escribe[3].setText("10")  # poblacion
            self.escribe[4].setText("0.75")  # mutacion h
            self.escribe[5].setText("10.0")  # combinacion c
            self.escribe[2].setText("70.0")  # amortiguamiento
            self.ordenamiento.setCurrentIndex(0)  # padre vs hijo
        elif self.tipo == 2:
            self.escribe[3].setText("10")  # particulas
            self.escribe[4].setText("1.47")  # local c1
            self.escribe[5].setText("1.47")  # global c2
            self.escribe[6].setText("0.8")  # friccion c3
            self.escribe[2].setText("0.0")  # amortiguamiento

    def Graficar(self):
        self.libre = False
        self.figura.chart().removeAllSeries()
        if self.selector.currentIndex() == 5:
            if self.tipo == 2:
                GPesos(self, True)
            else:
                GPesos(self, False)
        elif self.selector.currentIndex() == 4:
            GEntrenamiento(self, self.motor.trainData[0] > 0.0)
        else:
            if self.estado == 0:
                GSuperficie(self)
            GPatrones(self)
            if self.estado == 2 and len(self.motor.agentes) > 0:
                GRed(self, self.motor.agentes[self.motor.mejor])
            else:
                GRed(self, self.motor.laRed)
        GAxes(self)
        self.InfoRed()
        self.libre = True

    def InfoRed(self):
        self.display[0].setText("  " + str(self.progreso.value()))
        act = np.where(self.motor.laRed.actK, 1, 0).sum()
        self.display[1].setText("Dendr: " + str(act) + "/" + str(self.motor.laRed.actK.size))
        self.display[2].setText("ERL: " + str(round(self.motor.laRed.error, 6)))

    def InfoIni(self):
        self.motor.curvas = np.zeros((1, 4))
        self.motor.mejV = 0
        self.estado = 0
        self.BloquearCosas()
        self.motor.laRed.errorRL(self.motor.patrones[self.motor.numPEVT[0]:self.motor.numPEVT[:2].sum(), :])
        self.Graficar()
        viej = self.motor.laRed.error
        self.motor.laRed.errorCM(self.motor.patrones[:self.motor.numPEVT[0], :])
        self.display[3].setText("ECM: " + str(round(self.motor.laRed.error, 6)))
        self.motor.laRed.error = viej

    def FinHilo(self):
        if self.motor.runData[0] == self.motor.runData[1]:
            playsound("pitido.mp3")
            QMessageBox.about(self, "Éxito", "Éxito:\n"
                                             "entrenamiento finalizado correctamente\n"
                                             "pulse Parar para desbloquear Menú")

    def Refresca(self):
        if not self.isHidden():
            self.progreso.setValue(self.motor.runData[0])
            if self.selector.currentIndex() != 0 and self.libre:
                self.Graficar()

    def BloquearCosas(self):
        if self.estado == 0:
            for i in range(len(self.boton)):
                self.boton[i].setEnabled(True)
            for i in range(len(self.escribe)):
                self.escribe[i].setEnabled(True)
            if self.ordenamiento != None:
                self.ordenamiento.setEnabled(True)
            self.boton[2].setEnabled(False)
            self.boton[3].setEnabled(False)
        elif self.estado == 1:
            for i in range(len(self.boton)):
                self.boton[i].setEnabled(False)
            for i in range(len(self.escribe)):
                self.escribe[i].setEnabled(True)
            if self.ordenamiento != None:
                self.ordenamiento.setEnabled(True)
            self.boton[1].setEnabled(True)
            self.boton[3].setEnabled(True)
            self.boton[4].setEnabled(True)
            if self.tipo != 0:
                self.escribe[3].setEnabled(False)
        else:
            for i in range(len(self.boton)):
                self.boton[i].setEnabled(False)
            for i in range(len(self.escribe)):
                self.escribe[i].setEnabled(False)
            if self.ordenamiento != None:
                self.ordenamiento.setEnabled(False)
            self.boton[2].setEnabled(True)
            self.boton[3].setEnabled(True)

class HiloProcesos1(QThread):
    def __init__(self):
        QThread.__init__(self)
        self.origen = None

    def run(self):
        elId = self.origen
        elId.vivo.setVisible(True)
        elId.BloquearCosas()
        elId.display[3].setText("ECM: ?")
        dim = elId.motor.patrones.shape[1] - 1

        # verificar si es la primera vez que arranca el entreno
        if len(elId.motor.agentes) == 0:

            # poner valor maximo para pesos sinapticos
            elId.motor.laRed.lim = max(abs(elId.motor.patrones[:, :dim].min()),
                                       abs(elId.motor.patrones[:, :dim].max())) * 1.1

            # crear agentes que seran modificados al entrenar
            if elId.tipo == 0:
                elId.motor.CrearRedGradiente()
            elif elId.tipo == 1:
                tot = max(4, int(elId.escribe[3].text() if elId.escribe[3].text() != "" else "0"))
                elId.motor.CrearPoblacion(tot)
            elif elId.tipo == 2:
                tot = max(2, int(elId.escribe[3].text() if elId.escribe[3].text() != "" else "0"))
                elId.motor.CrearParticulas(tot)

            # poner el numero de epocas en cero
            elId.motor.runData[0] = 0

            # reiniciar valores de las curvas
            elId.motor.curvas = np.zeros((1, 4))
            elId.motor.mejV = 0

            # reiniciar el error de la red principal
            elId.motor.laRed.error = -1.0

        # leer los parametros del entrenamiento
        # seleccion padre vs hijo DE
        if elId.tipo == 1:
            if elId.ordenamiento.currentIndex() == 0:
                elId.motor.trainData[5] = 1.0
            else:
                elId.motor.trainData[5] = 0.0

        # inhibir
        dat = float(elId.escribe[1].text() if elId.escribe[1].text() != "." else "0.0")
        dat = (100.0 if dat == 99.99 else dat) / 100.0
        elId.motor.trainData[0] = dat

        # cambio Alf, H, C3
        dat = float(elId.escribe[2].text() if elId.escribe[2].text() != "." else "0.0")
        dat = (100.0 if dat == 99.99 else dat) / 100.0
        elId.motor.trainData[1] = dat

        # Alf, H, C3
        if elId.tipo == 2:
            dat = float(elId.escribe[6].text() if elId.escribe[6].text() != "." else "0.0")
            dat = min(1.0, dat)
        else:
            dat = float(elId.escribe[4].text() if elId.escribe[4].text() != "." else "0.0")
            if dat == 0.0:
                dat = 0.001
        elId.motor.trainData[3] = dat

        # B, C, C1
        if elId.tipo == 2:
            dat = float(elId.escribe[4].text() if elId.escribe[4].text() != "." else "0.0")
            if dat == 0.0:
                dat = 0.001
        else:
            dat = float(elId.escribe[5].text() if elId.escribe[5].text() != "." else "0.0")
            if elId.tipo == 0:
                dat = min(1.0, dat)
            elif elId.tipo == 1:
                dat = (100.0 if dat == 99.99 else dat) / 100.0
                if dat == 0.0:
                    dat = max(1.0 / elId.motor.laRed.pesW.size, 0.0001)
        elId.motor.trainData[4] = dat

        # mini bache
        if elId.tipo == 0:
            dat = float(elId.escribe[3].text() if elId.escribe[3].text() != "" else "1")
            dat = max(1.0, min(dat, elId.motor.numBEV[0]))
            elId.motor.trainData[5] = dat
        # C2
        elif elId.tipo == 2:
            dat = float(elId.escribe[5].text() if elId.escribe[5].text() != "." else "0.0")
            elId.motor.trainData[5] = dat

        # tolerancia para disminuir dendritas
        dat = float(elId.escribe[7].text() if elId.escribe[7].text() != "." else "0.0")
        elId.motor.trainData[6] = dat

        # ejecutar entrenamiento
        elId.motor.runData[1] = max(1, int(elId.escribe[0].text() if elId.escribe[0].text() != "" else "0"))
        elId.progreso.setMaximum(elId.motor.runData[1])
        # codigo: 0:SGD, 1:DE, 2:PSO
        elId.motor.CicloEntrenar(elId, elId.tipo)

        # poner ECM
        viej = elId.motor.laRed.error
        elId.motor.laRed.errorCM(elId.motor.patrones[:elId.motor.numPEVT[0], :])
        elId.display[3].setText("ECM: " + str(round(elId.motor.laRed.error, 6)))
        elId.motor.laRed.error = viej

        # liberar botones
        time.sleep(1)
        elId.BloquearCosas()
        elId.vivo.setVisible(False)

class PostEntreno(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.setWindowTitle("SoftwareDMNN-PostEntreno")
        self.setWindowIcon(QIcon("img12.png"))
        self.setObjectName("postentreno")
        # id de GUI del menu principal, para volver
        self.menu = None
        # array que guardara la id de las otras GUIs
        self.modulo = None
        # id de la clase maestra del softare
        self.motor = None

        # guarda el estado habilitado de los widgets
        self.estadoH = []

        # clase que ejecuta los algoritmos importantes en hilos
        self.elHilo = HiloProcesos3()
        self.elHilo.origen = self
        self.elHilo.finished.connect(self.FinHilo)

        # definir los botones de la GUI con su nombre e indice
        self.boton = []
        self.boton.append(QPushButton(QIcon("img13.png"), "Menú"))
        self.boton.append(QPushButton(QIcon("img24.png"), "Exportar Red"))
        self.boton.append(QPushButton(QIcon("img31.png"), "Ejecutar Prueba"))
        self.boton.append(QPushButton(QIcon("img28.png"), "Unir Dendritas"))
        self.boton.append(QPushButton(QIcon("img28.png"), "Quitar Dendritas"))
        self.boton.append(QPushButton(QIcon("img32.png"), "Eliminar Inhibidas"))
        self.boton.append(QPushButton(QIcon("img23.png"), "Importar Red"))
        self.boton.append(QPushButton(QIcon("img33.png"), "Deshinibir"))

        # activar las ayudas para los botones
        self.boton[0].setToolTip(MyToolTip(9))
        self.boton[1].setToolTip(MyToolTip(23))
        self.boton[2].setToolTip(MyToolTip(51))
        self.boton[3].setToolTip(MyToolTip(52))
        self.boton[4].setToolTip(MyToolTip(53))
        self.boton[5].setToolTip(MyToolTip(55))
        self.boton[6].setToolTip(MyToolTip(22))
        self.boton[7].setToolTip(MyToolTip(54))

        # poner nombres a los botones para el estilo
        self.boton[0].setObjectName("m_menu")
        for i in range(1, len(self.boton)):
            self.boton[i].setObjectName("b_postentreno")

        # conectar los botones a su funcion correspondiente
        self.boton[0].clicked.connect(self.toMenu)
        self.boton[1].clicked.connect(self.ExportRed)
        self.boton[2].clicked.connect(self.EjecuRed)
        self.boton[3].clicked.connect(self.UnirDen)
        self.boton[4].clicked.connect(self.QuitDen)
        self.boton[5].clicked.connect(self.ElimInhi)
        self.boton[6].clicked.connect(self.ImportRed)
        self.boton[7].clicked.connect(self.DesInhi)

        # definir el selector de grafica
        self.selector = QComboBox()
        self.selector.insertItem(0, "           Patrones")
        self.selector.insertItem(1, "           Patrones + Cajas")
        self.selector.insertItem(2, "           Patrones + Superficie")
        self.selector.insertItem(3, "           Patrones + Cajas + Superficie")
        self.selector.setCurrentIndex(1)
        self.selector.activated.connect(self.cambioSelect)

        # definir chequeo de normalizar entradas de prueba
        self.normalizar = QCheckBox("Normalizar")
        self.normalizar.setChecked(True)
        self.normalizar.setToolTip(MyToolTip(57))

        # definir las cajas de escritura
        self.escribe = []
        self.escribe.append(QLineEdit("0.0"))
        self.escribe.append(QLineEdit("0.1"))

        # activar las ayudas de las cajas de escritura
        self.escribe[0].setToolTip(MyToolTip(56))
        self.escribe[1].setToolTip(MyToolTip(49))

        # modificar propiedades de las cajas de escritura
        for i in range(len(self.escribe)):
            self.escribe[i].setAlignment(Qt.AlignCenter)
        self.escribe[1].setFixedWidth(150)

        # limitar la longitud de los textos y su formato
        self.escribe[0].setMaxLength(3)
        self.escribe[1].setInputMask("0.000000;")

        # conectar las cajas de escritura
        self.escribe[0].textEdited.connect(self.txInputs)
        self.escribe[1].textEdited.connect(self.txToler)

        # definir el cuadro que cambia de color
        self.colorin = QLabel("")
        self.colorin.setAlignment(Qt.AlignCenter)
        self.colorin.setFixedWidth(16)
        self.setAutoFillBackground(True)
        self.Colorear(-1)

        # definir el texto que muestra la salida
        self.salida = QLabel("???")
        self.salida.setAlignment(Qt.AlignCenter)

        # definir los textos que seran cambiados con codigo
        self.display = []
        self.display.append(QLabel("Patrones:"))
        self.display.append(QLabel("Entradas:"))
        self.display.append(QLabel("Clases:"))
        self.display.append(QLabel("Dendritas:"))
        self.display.append(QLabel("Inhibidas:"))
        self.display.append(QLabel("ECM:"))

        # modificar propiedades de los textos cambiantes
        for i in range(len(self.display)):
            self.display[i].setAlignment(Qt.AlignLeft)

        # crear el contenedor principal
        rejilla = QGridLayout()

        # poner el escalado por defecto de la rejilla, vertical
        rejilla.setRowStretch(0, 1)
        rejilla.setRowStretch(1, 100)
        rejilla.setRowStretch(2, 1)

        # poner el escalado por defecto de la rejilla, horizontal
        rejilla.setColumnStretch(0, 1)
        rejilla.setColumnStretch(1, 100)
        rejilla.setColumnStretch(2, 1)
        rejilla.setColumnStretch(3, 25)
        rejilla.setColumnStretch(4, 1)

        # crear los dos contenedores secundarios
        zizq = QVBoxLayout()
        zder = QVBoxLayout()

        # agregar selector de grafica de grafica
        zizq.addWidget(self.selector)

        # agregar la grafica al panel izquierdo
        self.figura = QChartView()
        self.figura.chart().setDropShadowEnabled(False)
        self.figura.chart().setMargins(QMargins(0, 0, 0, 0))
        zizq.addWidget(self.figura)

        # agregar los textos cambiantes
        subhori = QHBoxLayout()
        subhori.addWidget(self.display[0])
        subhori.addWidget(self.display[1])
        subhori.addWidget(self.display[2])
        zizq.addLayout(subhori)
        subhori = QHBoxLayout()
        subhori.addWidget(self.display[3])
        subhori.addWidget(self.display[4])
        subhori.addWidget(self.display[5])
        zizq.addLayout(subhori)

        # agregar el titulo del submenu y el boton de volver al menu
        subhori = QHBoxLayout()
        self.boton[0].setFixedWidth(150)
        subhori.addWidget(self.boton[0])
        xtt = QLabel("Post-Entreno")
        xtt.setStyleSheet("background-color: rgb(211,213,189);")
        xtt.setAlignment(Qt.AlignCenter)
        xtt.setObjectName("subtitulo")
        subhori.addWidget(xtt)
        zder.addLayout(subhori)

        # comprimir lo de la derecha hacia abajo
        zder.addStretch(1)

        # agregar packete
        pack = QGroupBox("Archivos")
        subpack = QVBoxLayout()
        subpack.addSpacing(10)
        #
        # agregar botones de importar exportar
        subhori = QHBoxLayout()
        subhori.addWidget(self.boton[6])
        subhori.addWidget(self.boton[1])
        subpack.addLayout(subhori)
        #
        pack.setLayout(subpack)
        zder.addWidget(pack)

        # agregar packete
        pack = QGroupBox("Prueba Manual")
        subpack = QVBoxLayout()
        subpack.addSpacing(10)
        #
        # agregar el cuadro de escritura de entradas
        subpack.addWidget(self.escribe[0])
        #
        # agregar el boton de ejecutar
        subhori = QHBoxLayout()
        subhori.addWidget(self.boton[2])
        subhori.addWidget(self.normalizar)
        subpack.addLayout(subhori)
        #
        # agregar el par color/salida
        subhori = QHBoxLayout()
        subhori.addWidget(self.colorin)
        subhori.addWidget(self.salida)
        subpack.addLayout(subhori)
        #
        pack.setLayout(subpack)
        zder.addWidget(pack)

        # agregar packete
        pack = QGroupBox("Disminuir Dendritas")
        subpack = QVBoxLayout()
        subpack.addSpacing(10)
        #
        # agregar el par escribe/texto para tolerancia
        subhori = QHBoxLayout()
        subhori.addWidget(self.escribe[1])
        xtt = QLabel("Tolerancia, ECM Máximo")
        xtt.setAlignment(Qt.AlignCenter)
        subhori.addWidget(xtt)
        subpack.addLayout(subhori)
        #
        # agregar los botones de unir y quitar cajas
        subpack.addWidget(self.boton[3])
        subpack.addWidget(self.boton[4])
        #
        pack.setLayout(subpack)
        zder.addWidget(pack)

        # agregar packete
        pack = QGroupBox("Administrar Inhibidas")
        subpack = QVBoxLayout()
        subpack.addSpacing(10)
        #
        # agregar el boton de eliminar inhibidas y deshinibir
        subhori = QHBoxLayout()
        subhori.addWidget(self.boton[7])
        subhori.addWidget(self.boton[5])
        subpack.addLayout(subhori)
        #
        pack.setLayout(subpack)
        zder.addWidget(pack)

        # comprimir lo de la derecha hacia arriba
        zder.addStretch(1)

        # poner los contenedores secundarios en el principal
        rejilla.addLayout(zizq, 1, 1)
        rejilla.addLayout(zder, 1, 3)

        # poner en las esquinas de la rejilla el espaciador invisible
        pix = QPixmap("img11.png")
        xtt = QLabel()
        xtt.setPixmap(pix)
        rejilla.addWidget(xtt, 0, 0)
        xtt = QLabel()
        xtt.setPixmap(pix)
        rejilla.addWidget(xtt, 2, 0)
        self.vivo = QLabel()
        gif = QMovie("wait.gif")
        self.vivo.setMovie(gif)
        gif.start()
        self.vivo.setVisible(False)
        rejilla.addWidget(self.vivo, 0, 2)
        xtt = QLabel()
        xtt.setPixmap(pix)
        rejilla.addWidget(xtt, 2, 2)
        xtt = QLabel()
        xtt.setPixmap(pix)
        rejilla.addWidget(xtt, 0, 4)
        xtt = QLabel()
        xtt.setPixmap(pix)
        rejilla.addWidget(xtt, 2, 4)

        # agregar el contenedor principal a la ventana
        self.setLayout(rejilla)

    def toMenu(self):
        self.selector.setCurrentIndex(1)
        self.menu.show()
        self.menu.setGeometry(self.geometry())
        FondoPSI(self.menu)
        self.hide()

    def resizeEvent(self, size):
        FondoPSI(self)

    def ImportRed(self):
        entradas = self.motor.patrones.shape[1] - 1
        clases = int(self.motor.patrones[:, entradas].max() + 1)
        res = self.motor.laRed.ImportarRed(entradas, clases, self.motor.adecuaH,
                                           self.motor.adecuaL, self.motor.adecuaN)
        if res > 0:
            if res == 1:
                QMessageBox.about(self, "Éxito", "Éxito:\n"
                                                 "red cargada adecuadamente")
            else:
                QMessageBox.about(self, "Advertencia", "Advertencia:\n"
                                                       "los valores de normalización No coinciden")
            self.Graficar()

        elif res < 0:
            self.menu.Activacion(1)
            if res == -1:
                QMessageBox.about(self, "Error", "Error:\n"
                                                 "cabecera de archivo inválida")
            elif res == -2:
                QMessageBox.about(self, "Error", "Error:\n"
                                                 "las dimensiónes No coinciden con el problema")
            self.toMenu()

    def ExportRed(self):
        entradas = self.motor.patrones.shape[1] - 1
        clases = int(self.motor.patrones[:, entradas].max() + 1)
        self.motor.laRed.ExportarRed(entradas, clases, self.motor.adecuaH,
                                     self.motor.adecuaL, self.motor.titulo,
                                     self.motor.apodos, self.motor.entradas,
                                     self.motor.adecuaN)

    def EjecuRed(self):
        lin = self.escribe[0].text()
        # crear el vector de flotantes desde la cadena de texto
        tx = StringIO(("0" if lin == "" else lin).replace("\n", ""))
        entr = np.genfromtxt(tx, dtype=float, delimiter=",", filling_values="0")
        # redimensionarlo linealmente y ejecutar la red, si coincide con la dimension
        if entr.size == self.motor.patrones.shape[1] - 1:
            if self.normalizar.isChecked():
                if self.motor.adecuaN == -1:
                    promedio = self.motor.adecuaH
                    desviEst = self.motor.adecuaL
                    entr = entr - promedio / desviEst
                elif self.motor.adecuaN > 0.0:
                    M = self.motor.adecuaH
                    m = self.motor.adecuaL
                    Mn = self.motor.adecuaN
                    mn = -self.motor.adecuaN
                    entr = (entr / (M - m)) * (Mn - mn) + mn
            res = self.motor.laRed.EjecutarRed(entr).argmax()
            self.Colorear(res)
            self.salida.setText("clase " + str(res) + ": " + self.motor.apodos[res])

    def UnirDen(self):
        dim = self.motor.patrones.shape[1] - 1
        self.motor.laRed.CrearXML(dim, int(self.motor.patrones[:, dim].max() + 1), "AutoSavePreOptimU.xml",
                                  self.motor.adecuaH, self.motor.adecuaL, self.motor.titulo,
                                  self.motor.apodos, self.motor.entradas, self.motor.adecuaN)
        self.boton[3].setStyleSheet("")
        self.elHilo.esUnir = True
        self.elHilo.start()

    def QuitDen(self):
        dim = self.motor.patrones.shape[1] - 1
        self.motor.laRed.CrearXML(dim, int(self.motor.patrones[:, dim].max() + 1), "AutoSavePreOptimQ.xml",
                                  self.motor.adecuaH, self.motor.adecuaL, self.motor.titulo,
                                  self.motor.apodos, self.motor.entradas, self.motor.adecuaN)
        self.boton[4].setStyleSheet("")
        self.elHilo.esUnir = False
        self.elHilo.start()

    def ElimInhi(self):
        if self.motor.laRed.EliminarInhibidas():
            self.motor.laRed.Kmedias(self.motor.patrones[:self.motor.numPEVT[0], :], 1, 0.1)
            QMessageBox.about(self, "Error", "Error:\n"
                                             "sobre-eliminación, se re-estableció una red mínima")
        else:
            QMessageBox.about(self, "Éxito", "Éxito:\n"
                                             "se han eliminado definitivamente las inhibidas")
        self.Graficar()

    def DesInhi(self):
        self.motor.laRed.Deshinibir()
        QMessageBox.about(self, "Éxito", "Éxito:\n"
                                         "las dendritas inhibidas se han activado")
        self.Graficar()

    def cambioSelect(self):
        self.Graficar()

    def txToler(self):
        self.boton[3].setStyleSheet("background-color: rgb(239, 172, 122);")
        self.boton[4].setStyleSheet("background-color: rgb(239, 172, 122);")

    def txInputs(self):
        self.Colorear(-1)
        self.salida.setText("???")

    def Colorear(self, clase):
        if clase != -1:
            ccc = Colores(clase, False)
        else:
            ccc = QColor(Qt.white)
            ccc.setAlpha(0)
        ccc = "{r}, {g}, {b}, {a}".format(r=ccc.red(), g=ccc.green(), b=ccc.blue(), a=ccc.alpha())
        self.colorin.setStyleSheet("QLabel { background-color: rgba(" + ccc + "); }")

    def Graficar(self):
        self.figura.chart().removeAllSeries()
        GSuperficie(self)
        GPatrones(self)
        GRed(self, self.motor.laRed)
        GAxes(self)
        self.InfoRed()

    def InfoIni(self):
        self.Graficar()
        # poner informacion sobre los patrones
        self.display[0].setText("PatronesE: " + str(self.motor.numPEVT[0]))
        dim = self.motor.patrones.shape[1] - 1
        self.display[1].setText("Entradas: " + str(dim))
        cla = int(self.motor.patrones[:, dim].max() + 1)
        self.display[2].setText("Clases: " + str(cla))
        # poner limite
        self.escribe[0].setMaxLength(8 * dim)
        h = ""
        for i in range(dim):
            h += "0,"
        self.escribe[0].setText(h[:(len(h) - 1)])

    def InfoRed(self):
        # poner informacion sobre la red
        act = np.where(self.motor.laRed.actK, 1, 0).sum()
        self.display[3].setText("Dendritas: " + str(act))
        self.display[4].setText("Inhibidas: " + str(self.motor.laRed.actK.size - act))
        self.motor.laRed.errorCM(self.motor.patrones[:self.motor.numPEVT[0], :])
        self.display[5].setText("ECM: " + str(round(self.motor.laRed.error, 6)))

    def BloquearCosas(self, block):
        if block:
            self.estadoH = []
            for i in range(len(self.boton)):
                self.estadoH.append(self.boton[i].isEnabled())
                self.boton[i].setEnabled(False)
            for i in range(len(self.escribe)):
                self.estadoH.append(self.escribe[i].isEnabled())
                self.escribe[i].setEnabled(False)
            self.estadoH.append(self.selector.isEnabled())
            self.selector.setEnabled(False)
            self.estadoH.append(self.normalizar.isEnabled())
            self.normalizar.setEnabled(False)
        else:
            n = 0
            for i in range(len(self.boton)):
                self.boton[i].setEnabled(self.estadoH[n])
                n += 1
            for i in range(len(self.escribe)):
                self.escribe[i].setEnabled(self.estadoH[n])
                n += 1
            self.selector.setEnabled(self.estadoH[n])
            self.normalizar.setEnabled(self.estadoH[n + 1])

    def FinHilo(self):
        QMessageBox.about(self, "Éxito", "Éxito:\n"
                                         "optimización ejecutada adecuadamente")
        self.Graficar()

class HiloProcesos3(QThread):
    def __init__(self):
        QThread.__init__(self)
        self.origen = None
        self.esUnir = True

    def run(self):
        elId = self.origen
        elId.vivo.setVisible(True)
        elId.BloquearCosas(True)
        tol = float(elId.escribe[1].text() if elId.escribe[1].text() != "." else "0.0")
        if self.esUnir:
            elId.motor.laRed.UnirDendritas(elId.motor.patrones[:elId.motor.numPEVT[0], :], tol)
        else:
            elId.motor.laRed.QuitarDendritas(elId.motor.patrones[:elId.motor.numPEVT[0], :], tol)
        dim = elId.motor.patrones.shape[1] - 1
        if self.esUnir:
            elId.motor.laRed.CrearXML(dim, int(elId.motor.patrones[:, dim].max() + 1), "AutoSavePosOptimU.xml",
                                      elId.motor.adecuaH, elId.motor.adecuaL, elId.motor.titulo,
                                      elId.motor.apodos, elId.motor.entradas, elId.motor.adecuaN)
        else:
            elId.motor.laRed.CrearXML(dim, int(elId.motor.patrones[:, dim].max() + 1), "AutoSavePosOptimQ.xml",
                                      elId.motor.adecuaH, elId.motor.adecuaL, elId.motor.titulo,
                                      elId.motor.apodos, elId.motor.entradas, elId.motor.adecuaN)
        time.sleep(1)
        elId.BloquearCosas(False)
        elId.vivo.setVisible(False)

class Analisis(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.setWindowTitle("SoftwareDMNN-Análisis")
        self.setWindowIcon(QIcon("img12.png"))
        self.setObjectName("analisis")
        # id de GUI del menu principal, para volver
        self.menu = None
        # array que guardara la id de las otras GUIs
        self.modulo = None
        # id de la clase maestra del softare
        self.motor = None

        # guardara la matriz calculada
        self.matrix = np.zeros(1)

        # definir los botones de la GUI con su nombre e indice
        self.boton = []
        self.boton.append(QPushButton(QIcon("img13.png"), "Menú"))
        self.boton.append(QPushButton(QIcon("img24.png"), "Exportar Matrices de Confusión"))
        self.boton.append(QPushButton(QIcon("img31.png"), "Exportar Resultados de Entradas"))

        # activar las ayudas para los botones
        self.boton[0].setToolTip(MyToolTip(9))
        self.boton[1].setToolTip(MyToolTip(59))
        self.boton[2].setToolTip(MyToolTip(66))

        # poner nombres a los botones para el estilo
        self.boton[0].setObjectName("m_menu")
        for i in range(1, len(self.boton)):
            self.boton[i].setObjectName("b_analisis")

        # conectar los botones a su funcion correspondiente
        self.boton[0].clicked.connect(self.toMenu)
        self.boton[1].clicked.connect(self.ExportConf)
        self.boton[2].clicked.connect(self.ExportResults)

        # definir el selector de grafica
        self.selector = QComboBox()
        self.selector.insertItem(0, "           Patrones")
        self.selector.insertItem(1, "           Patrones + Cajas")
        self.selector.insertItem(2, "           Patrones + Superficie")
        self.selector.insertItem(3, "           Patrones + Cajas + Superficie")
        self.selector.insertItem(4, "           Curva ROC (Región de Convergencia)")
        self.selector.insertItem(5, "           Matriz de Confusión")
        self.selector.setCurrentIndex(5)
        self.selector.activated.connect(self.cambioSelect)

        # definir los selectores de tipo de estadisticas
        self.selEst = []
        for i in range(3):
            self.selEst.append(QComboBox())
            self.setObjectName("x_analisis")

        # poner las opciones del selector de error
        self.selEst[0].insertItem(0, "  ECM Entreno")
        self.selEst[0].insertItem(1, "  ECM Validación")
        self.selEst[0].insertItem(2, "  ECM Testeo")
        self.selEst[0].insertItem(3, "  ECM General")
        self.selEst[0].insertItem(4, "  ERL Entreno")
        self.selEst[0].insertItem(5, "  ERL Validación")
        self.selEst[0].insertItem(6, "  ERL Testeo")
        self.selEst[0].insertItem(7, "  ERL General")
        self.selEst[0].setCurrentIndex(2)
        self.selEst[0].setFixedWidth(150)

        # poner las opciones del selector de matriz
        self.selEst[1].insertItem(0, "      Estadísticas Entreno")
        self.selEst[1].insertItem(1, "      Estadísticas Validación")
        self.selEst[1].insertItem(2, "      Estadísticas Testeo")
        self.selEst[1].insertItem(3, "      Estadísticas No-Entreno")
        self.selEst[1].insertItem(4, "      Estadísticas Generales")
        self.selEst[1].setCurrentIndex(4)

        # conectar los selectores a la accion correspondiente
        self.selEst[0].activated.connect(self.cambioError)
        self.selEst[1].activated.connect(self.cambioMatriz)
        self.selEst[2].activated.connect(self.cambioClase)

        # definir los textos cambiantes para las estadisticas
        self.estadist = []
        for i in range(8):
            self.estadist.append(QLabel(""))
            self.estadist[-1].setAlignment(Qt.AlignLeft)

        # definir los textos que seran cambiados con codigo
        self.display = []
        self.display.append(QLabel("Patrones:"))
        self.display.append(QLabel("Entradas:"))
        self.display.append(QLabel("Clases:"))
        self.display.append(QLabel("Entreno:"))
        self.display.append(QLabel("Validación:"))
        self.display.append(QLabel("Testeo:"))

        # modificar propiedades de los textos cambiantes
        for i in range(len(self.display)):
            self.display[i].setAlignment(Qt.AlignLeft)

        # definir el cuadro que cambia de color
        self.colorin = QLabel("")
        self.colorin.setAlignment(Qt.AlignCenter)
        self.colorin.setFixedWidth(16)
        self.setAutoFillBackground(True)
        self.Colorear(0)

        # crear el contenedor principal
        rejilla = QGridLayout()

        # poner el escalado por defecto de la rejilla, vertical
        rejilla.setRowStretch(0, 1)
        rejilla.setRowStretch(1, 100)
        rejilla.setRowStretch(2, 1)

        # poner el escalado por defecto de la rejilla, horizontal
        rejilla.setColumnStretch(0, 1)
        rejilla.setColumnStretch(1, 100)
        rejilla.setColumnStretch(2, 1)
        rejilla.setColumnStretch(3, 25)
        rejilla.setColumnStretch(4, 1)

        # crear los dos contenedores secundarios
        zizq = QVBoxLayout()
        zder = QVBoxLayout()

        # agregar selector de grafica de grafica
        zizq.addWidget(self.selector)

        # definir grafica y agregarla al panel izquierdo
        self.figura = QChartView()
        self.figura.chart().setDropShadowEnabled(False)
        self.figura.chart().setMargins(QMargins(0, 0, 0, 0))
        self.figura.setVisible(False)
        zizq.addWidget(self.figura)

        # definir tabla y agregarla al panel izquierdo
        self.antiLoop = False
        self.tabla = QTableWidget()
        self.tabla.setAlternatingRowColors(True)
        self.tabla.cellChanged.connect(self.preservar)
        self.tabla.setVisible(False)
        zizq.addWidget(self.tabla)

        # agregar los textos cambiantes
        subhori = QHBoxLayout()
        subhori.addWidget(self.display[0])
        subhori.addWidget(self.display[1])
        subhori.addWidget(self.display[2])
        zizq.addLayout(subhori)
        subhori = QHBoxLayout()
        subhori.addWidget(self.display[3])
        subhori.addWidget(self.display[4])
        subhori.addWidget(self.display[5])
        zizq.addLayout(subhori)

        # agregar el titulo del submenu y el boton de volver al menu
        subhori = QHBoxLayout()
        self.boton[0].setFixedWidth(150)
        subhori.addWidget(self.boton[0])
        xtt = QLabel("Análisis")
        xtt.setStyleSheet("background-color: rgb(247,214,254);")
        xtt.setAlignment(Qt.AlignCenter)
        xtt.setObjectName("subtitulo")
        subhori.addWidget(xtt)
        zder.addLayout(subhori)

        # comprimir lo de la derecha hacia abajo
        zder.addStretch(1)

        # agregar packete
        pack = QGroupBox("Resultados Básicos")
        subpack = QVBoxLayout()
        subpack.addSpacing(10)
        #
        # agregar estadistica de numero de dendritas
        subhori = QHBoxLayout()
        xtt = QLabel("Dendritas:   ")
        xtt.setAlignment(Qt.AlignRight)
        xtt.setFixedWidth(150)
        subhori.addWidget(xtt)
        subhori.addWidget(self.estadist[0])
        subpack.addLayout(subhori)
        #
        # agregar estadistica de errores con selector
        subhori = QHBoxLayout()
        subhori.addWidget(self.selEst[0])
        subhori.addWidget(self.estadist[1])
        subpack.addLayout(subhori)
        #
        pack.setLayout(subpack)
        zder.addWidget(pack)

        # agregar packete
        pack = QGroupBox("Calculos Estadísticos")
        subpack = QVBoxLayout()
        subpack.addSpacing(10)
        #
        # agregar el selector de matriz
        subpack.addWidget(self.selEst[1])
        #
        # agregar estadistica numero 1
        subhori = QHBoxLayout()
        xtt = QLabel("Precisión:   ")
        xtt.setToolTip(MyToolTip(60))
        xtt.setAlignment(Qt.AlignRight)
        xtt.setFixedWidth(150)
        subhori.addWidget(xtt)
        subhori.addWidget(self.estadist[3])
        subpack.addLayout(subhori)
        #
        # agregar estadistica numero 2
        subhori = QHBoxLayout()
        xtt = QLabel("Kappa:   ")
        xtt.setToolTip(MyToolTip(65))
        xtt.setAlignment(Qt.AlignRight)
        xtt.setFixedWidth(150)
        subhori.addWidget(xtt)
        subhori.addWidget(self.estadist[2])
        subpack.addLayout(subhori)
        #
        # agregar el selector de clase
        subhori = QHBoxLayout()
        subhori.addWidget(self.colorin)
        subhori.addWidget(self.selEst[2])
        subpack.addLayout(subhori)
        #
        # agregar estadistica numero 3
        subhori = QHBoxLayout()
        xtt = QLabel("Exactitud:   ")
        xtt.setToolTip(MyToolTip(61))
        xtt.setAlignment(Qt.AlignRight)
        xtt.setFixedWidth(150)
        subhori.addWidget(xtt)
        subhori.addWidget(self.estadist[4])
        subpack.addLayout(subhori)
        #
        # agregar estadistica numero 4
        subhori = QHBoxLayout()
        xtt = QLabel("Sensibilidad:   ")
        xtt.setToolTip(MyToolTip(62))
        xtt.setAlignment(Qt.AlignRight)
        xtt.setFixedWidth(150)
        subhori.addWidget(xtt)
        subhori.addWidget(self.estadist[5])
        subpack.addLayout(subhori)
        #
        # agregar estadistica numero 5
        subhori = QHBoxLayout()
        xtt = QLabel("Especificidad:   ")
        xtt.setToolTip(MyToolTip(63))
        xtt.setAlignment(Qt.AlignRight)
        xtt.setFixedWidth(150)
        subhori.addWidget(xtt)
        subhori.addWidget(self.estadist[6])
        subpack.addLayout(subhori)
        #
        # agregar estadistica numero 6
        subhori = QHBoxLayout()
        xtt = QLabel("Valor-F:   ")
        xtt.setToolTip(MyToolTip(64))
        xtt.setAlignment(Qt.AlignRight)
        xtt.setFixedWidth(150)
        subhori.addWidget(xtt)
        subhori.addWidget(self.estadist[7])
        subpack.addLayout(subhori)
        #
        pack.setLayout(subpack)
        zder.addWidget(pack)

        # agregar packete
        pack = QGroupBox("Archivos")
        subpack = QVBoxLayout()
        subpack.addSpacing(10)
        #
        # agregar el boton de exportar matrices y resultados
        subpack.addWidget(self.boton[1])
        subpack.addWidget(self.boton[2])
        #
        pack.setLayout(subpack)
        zder.addWidget(pack)

        # comprimir lo de la derecha hacia arriba
        zder.addStretch(1)

        # poner los contenedores secundarios en el principal
        rejilla.addLayout(zizq, 1, 1)
        rejilla.addLayout(zder, 1, 3)

        # poner en las esquinas de la rejilla el espaciador invisible
        pix = QPixmap("img11.png")
        xtt = QLabel()
        xtt.setPixmap(pix)
        rejilla.addWidget(xtt, 0, 0)
        xtt = QLabel()
        xtt.setPixmap(pix)
        rejilla.addWidget(xtt, 2, 0)
        xtt = QLabel()
        xtt.setPixmap(pix)
        rejilla.addWidget(xtt, 0, 2)
        xtt = QLabel()
        xtt.setPixmap(pix)
        rejilla.addWidget(xtt, 2, 2)
        xtt = QLabel()
        xtt.setPixmap(pix)
        rejilla.addWidget(xtt, 0, 4)
        xtt = QLabel()
        xtt.setPixmap(pix)
        rejilla.addWidget(xtt, 2, 4)

        # agregar el contenedor principal a la ventana
        self.setLayout(rejilla)

    def toMenu(self):
        self.selector.setCurrentIndex(5)
        self.menu.show()
        self.menu.setGeometry(self.geometry())
        FondoPSI(self.menu)
        self.hide()

    def resizeEvent(self, size):
        FondoPSI(self)

    def CalcularTodo(self):
        self.GenerarSelectorClases()
        self.motor.MatricesConfusion()
        self.cambioError()
        self.calculaEstadisticas()
        self.pintaEstadisticas()
        self.Graficar()

        # poner informacion sobre los patrones
        self.display[0].setText("Patrones: " + str(self.motor.patrones.shape[0]))
        dim = self.motor.patrones.shape[1] - 1
        self.display[1].setText("Entradas: " + str(dim))
        cla = int(self.motor.patrones[:, dim].max() + 1)
        self.display[2].setText("Clases: " + str(cla))
        self.display[3].setText("Entreno: " + str(self.motor.numPEVT[0]))
        self.display[4].setText("Validación: " + str(self.motor.numPEVT[1]))
        self.display[5].setText("Testeo: " + str(self.motor.numPEVT[2]))

        # poner informacion sobre numero de dendritas
        self.estadist[0].setText(str(np.where(self.motor.laRed.actK, 1, 0).sum()))

    def calculaEstadisticas(self):
        # seleccionar el tipo de matriz a tratar
        opc = self.selEst[1].currentIndex()
        if opc == 0:
            matr = self.motor.matrizE
        elif opc == 1:
            matr = self.motor.matrizV
        elif opc == 2:
            matr = self.motor.matrizT
        elif opc == 3:
            matr = self.motor.matrizN
        else:
            matr = self.motor.matrizG

        # crear matriz de resultados
        cla = matr.shape[0]
        mtx = np.zeros((cla + 3, cla + 3))
        # agregar los valores de la matriz original
        mtx[:cla, :cla] = matr
        # obtener total predicho
        for x in range(cla):
            mtx[cla, x] = mtx[:cla, x].sum()
        # obtener total real
        for y in range(cla):
            mtx[y, cla] = mtx[y, :cla].sum()
        # obtener total general
        mtx[cla, cla] = mtx[:cla, :cla].sum()

        # calcular exactitud
        for x in range(cla):
            mtx[cla + 1, x] = mtx[x, x] / max(0.001, mtx[cla, x])
        # calcular sensibilidad
        for y in range(cla):
            mtx[y, cla + 1] = mtx[y, y] / max(0.001, mtx[y, cla])
        # calcular especificidad
        for x in range(cla):
            mtx[cla + 2, x] = (mtx[cla, cla] - mtx[x, cla] - mtx[cla, x] + mtx[x, x]) \
                              / max(0.001, mtx[cla, cla] - mtx[x, cla])
        # calcular valorF
        for y in range(cla):
            mtx[y, cla + 2] = 2.0 * (mtx[cla + 1, y] * mtx[y, cla + 1]) \
                              / max(0.001, mtx[cla + 1, y] + mtx[y, cla + 1])
        # calcular precision
        mtx[cla + 1, cla + 2] = mtx[:cla, :cla].diagonal().sum() / max(0.001, mtx[cla, cla])
        # calcular kappa
        op = 0.0
        for x in range(cla):
            op += mtx[cla, x] * mtx[x, cla]
        mtx[cla + 2, cla + 2] = (mtx[cla, cla] * mtx[:cla, :cla].diagonal().sum() - op) \
                                / max(0.001, np.power(mtx[cla, cla], 2.0) - op)
        # guardarla en variable persistente
        self.matrix = mtx

    def pintaEstadisticas(self):
        # seleccionar la clase a tratar
        m = self.selEst[2].currentIndex()
        cla = self.matrix.shape[0] - 3
        self.estadist[3].setText(str(round(self.matrix[cla + 1, cla + 2], 6)))
        self.estadist[2].setText(str(round(self.matrix[cla + 2, cla + 2], 6)))
        self.estadist[4].setText(str(round(self.matrix[cla + 1, m], 6)))
        self.estadist[5].setText(str(round(self.matrix[m, cla + 1], 6)))
        self.estadist[6].setText(str(round(self.matrix[cla + 2, m], 6)))
        self.estadist[7].setText(str(round(self.matrix[m, cla + 2], 6)))

    def ExportConf(self):
        self.motor.ExportarMatrices()

    def ExportResults(self):
        self.motor.ExportarResultados()

    def cambioSelect(self):
        self.Graficar()

    def cambioError(self):
        opc = self.selEst[0].currentIndex()
        # seleccionar el tramo de los patrones a tratar
        if opc == 0 or opc == 4:
            patr = self.motor.patrones[:self.motor.numPEVT[0], :]
        elif opc == 1 or opc == 5:
            patr = self.motor.patrones[self.motor.numPEVT[0]:self.motor.numPEVT[:2].sum(), :]
        elif opc == 2 or opc == 6:
            patr = self.motor.patrones[self.motor.numPEVT[:2].sum():self.motor.numPEVT.sum(), :]
        else:
            patr = self.motor.patrones
        # seleccionar entre error cuadratico o logistico
        if opc < 4:
            self.motor.laRed.errorCM(patr)
        else:
            self.motor.laRed.errorRL(patr)
        # escribir en la caja el resultado
        self.estadist[1].setText(str(round(self.motor.laRed.error, 6)))

    def preservar(self):
        if self.antiLoop:
            self.Graficar()

    def Graficar(self):
        # dibujar graficas
        if self.selector.currentIndex() < 5:
            self.figura.setVisible(True)
            self.tabla.setVisible(False)
            self.figura.chart().removeAllSeries()
            if self.selector.currentIndex() < 4:
                GSuperficie(self)
                GPatrones(self)
                GRed(self, self.motor.laRed)
            else:
                GROC(self)
            GAxes(self)

        # dibujar tablas
        else:
            self.antiLoop = False
            self.figura.setVisible(False)
            self.tabla.setVisible(True)
            mtx = self.matrix
            cla = mtx.shape[0] - 3
            self.tabla.clear()
            self.tabla.setRowCount(cla + 4)
            self.tabla.setColumnCount(cla + 4)
            self.tabla.setItem(0, 0, QTableWidgetItem("PREDICHOS 🠊 "))
            for x in range(cla):
                self.tabla.setItem(0, x + 1, QTableWidgetItem("P" + str(x)))
            for y in range(cla):
                self.tabla.setItem(y + 1, 0, QTableWidgetItem("R" + str(y) +
                                                              ": " + self.motor.apodos[y]))
            mayor = mtx[:cla, :cla].max()
            for r in range(cla + 1):
                for p in range(cla + 1):
                    it = QTableWidgetItem(str(int(mtx[r, p])))
                    if r < cla and p < cla:
                        ccc = QColor(255, 255, 255)
                        ccc.setGreen(255 - int(np.power(mtx[r, p] / mayor, 0.5) * 0.666 * 255))
                        it.setBackground(ccc)
                    else:
                        it.setBackground(QColor(210, 255, 210))
                    self.tabla.setItem(r + 1, p + 1, it)
            self.tabla.setItem(0, cla + 1, QTableWidgetItem("Total"))
            self.tabla.setItem(0, cla + 2, QTableWidgetItem("Sensibili."))
            self.tabla.setItem(0, cla + 3, QTableWidgetItem("Valor-F"))
            self.tabla.setItem(cla + 1, 0, QTableWidgetItem("Total"))
            self.tabla.setItem(cla + 2, 0, QTableWidgetItem("Exactitud"))
            self.tabla.setItem(cla + 3, 0, QTableWidgetItem("Especificidad"))
            self.tabla.setItem(cla + 2, cla + 2, QTableWidgetItem("Precisión"))
            self.tabla.setItem(cla + 3, cla + 2, QTableWidgetItem("Kappa"))
            self.tabla.setItem(cla + 2, cla + 3, QTableWidgetItem(
                str(int(mtx[cla + 1, cla + 2] * 100.0)) + "%"))
            self.tabla.setItem(cla + 3, cla + 3, QTableWidgetItem(
                str(round(mtx[cla + 2, cla + 2], 3))))
            for x in range(cla):
                self.tabla.setItem(cla + 2, x + 1, QTableWidgetItem(
                    str(int(mtx[cla + 1, x] * 100.0)) + "%"))
                self.tabla.setItem(cla + 3, x + 1, QTableWidgetItem(
                    str(int(mtx[cla + 2, x] * 100.0)) + "%"))
            for y in range(cla):
                self.tabla.setItem(y + 1, cla + 2, QTableWidgetItem(
                    str(int(mtx[y, cla + 1] * 100.0)) + "%"))
                self.tabla.setItem(y + 1, cla + 3, QTableWidgetItem(
                    str(int(mtx[y, cla + 2] * 100.0)) + "%"))
            self.tabla.resizeColumnsToContents()
            self.tabla.resizeRowsToContents()
            self.antiLoop = True

    def Colorear(self, clase):
        if clase != -1:
            ccc = Colores(clase, False)
        else:
            ccc = QColor(Qt.white)
            ccc.setAlpha(0)
        ccc = "{r}, {g}, {b}, {a}".format(r=ccc.red(), g=ccc.green(), b=ccc.blue(), a=ccc.alpha())
        self.colorin.setStyleSheet("QLabel { background-color: rgba(" + ccc + "); }")

    def cambioMatriz(self):
        self.calculaEstadisticas()
        self.pintaEstadisticas()
        if self.selector.currentIndex() > 3:
            self.Graficar()

    def cambioClase(self):
        self.pintaEstadisticas()
        self.Colorear(self.selEst[2].currentIndex())

    def GenerarSelectorClases(self):
        self.selEst[2].clear()
        for m in range(self.motor.laRed.numK.size):
            self.selEst[2].insertItem(m, "   Clase " + str(m) + ": " + self.motor.apodos[m])
        self.selEst[2].setCurrentIndex(0)
        self.Colorear(0)

# esta funcion tiene todos los tooltips (ayudas) del software
def MyToolTip(id):
    res = ""
    amortigua = "Digite valor que representa el decaimiento\n" \
                "de la variable $; en la siguiente\n" \
                "ecuación se ve este valor como (f), siendo\n" \
                "(t) el número de iteraciónes actual y (T)\n" \
                "el máximo:\n\n" \
                "= (0.9 . exp(-t² / (2 . f . T²)) + 0.1) . $\n\n" \
                "f = 0 No modificará a la variable $."
    if id == 0:  # menu - problema
        res = "Para cargar los patrones del problema\n" \
              "de clasificación y adecuarlos."
    elif id == 1:  # menu - inicializacion
        res = "Aquí se crea la red neuronal a partir de\n" \
              "uno de 2 algoritmos disponibles, también\n" \
              "se puede importar o exportar una red."
    elif id == 2:  # menu - SGD
        res = "Entrenamiento para optimizar la red con\n" \
              "el algoritmo de descenso del gradiente."
    elif id == 3:  # menu - DE
        res = "Entrenamiento para optimizar la red con\n" \
              "el algoritmo de evolución diferencial."
    elif id == 4:  # menu - PSO
        res = "Entrenamiento para optimizar la red con\n" \
              "el algoritmo de optimización por enjambre de partículas."
    elif id == 5:  # menu - postentreno
        res = "Aquí se prueba la red con datos puntuales, se\n" \
              "disminuye el número de dendritas y también se\n" \
              "puede importar o exportar una red."
    elif id == 6:  # menu - analisis
        res = "Tiene las métricas de desempeño estandarizadas,\n" \
              "matrices de confisión y ROC para calificar la red."
    elif id == 7:  # menu - informacion
        res = "Muestra la descripción del software\n" \
              "y correo de contacto de su creador."
    elif id == 8:  # menu - ayuda
        res = "Abre un archivo (pdf) que tiene dos partes:\n\n" \
              "- tutorial sobre manejo del software.\n\n" \
              "- explicación breve sobre las redes DMNN\n" \
              "  y los algoritmos aquí empleados."
    elif id == 9:  # all - menu
        res = "Retorna al menú principal\n" \
              "del software."
    elif id == 10:  # problema - importar patrones
        res = "Abre un archivo (.txt) o (.xml) que tenga\n" \
              "la matriz de patrones del problema.\n" \
              "TXT con la forma:\n\n" \
              "Patrones: titulo_del_problema\n" \
              "Salidas: c0, c1, c2\n" \
              "Entradas: e0, e1, e2\n" \
              "v, v, v, s\n" \
              "v, v, v, s\n" \
              "v, v, v, s\n\n" \
              "En caso de datos invalidos, No serán cargados.\n" \
              "XML posee mayor información, como lo es:\n" \
              "los porcentajes de entreno, validación y testeo."
    elif id == 11:  # problema - calcula porcentajes
        res = "Distribuye todos los patrones entre:\n" \
              "entreno, validación y testeo.\n\n" \
              "En base a los datos puestos en las 4 lineas de\n" \
              "abajo, el resultado se observa en los textos\n" \
              "bajo la gráfica, donde se aprecia el tamaño del\n" \
              "bache respecto al total: BacheEntreno / Entreno."
    elif id == 12:  # problema - normalizar
        res = "A veces se quiere cambiar la escala de todo\n" \
              "el set de patrones, aquí hay dos algoritmos:\n" \
              "- Min-Max\n" \
              "- Z-Score\n\n" \
              "Al pulsarse desactivará la exportación de patrones."
    elif id == 13:  # problema - mezcla y exporta
        res = "Los porcentajes solo cambian el índice a partir\n" \
              "del cual los patrones son:\n" \
              "entreno, validación o testeo.\n\n" \
              "Por eso esta función se encarga de mezclar\n" \
              "aleatoriamente el orden de estos, indispensable\n" \
              "si el set viene ordenado.\n\n" \
              "Finalmente preguntará si desea exportar los\n" \
              "patrones con el nuevo orden y porcentajes."
    elif id == 14:  # problema - check normalizacion general
        res = "Relativo modificará cada dimensión del set\n" \
              "independientemente, mientras que de lo contrario\n" \
              "tomará todo el set para hallar los únicos\n" \
              "parámetros de normalización."
    elif id == 15:  # problema - escr entreno
        res = "El porcentaje de patrones que serán de entreno,\n" \
              "recomendable un valor alto (ej: 80%), igualmente\n" \
              "de ser 100% almenos un par de patrones irán a\n" \
              "validación y testeo."
    elif id == 16:  # problema - escr validacion
        res = "El porcentaje de patrones que serán de validación,\n" \
              "este parametro se autolimita a lo que sobre del\n" \
              "entreno, y a su vez el porcentaje que sobre de ambos\n" \
              "será para testeo, ejemplo:\n\n" \
              "80% entreno, 15% validación -> 5% testeo.\n\n" \
              "Almenos un patron irá a testeo."
    elif id == 17:  # problema - escr bache entreno
        res = "De los patrones de entreno se harán sub-grupos\n" \
              "del tamaño especificado aquí, se recomienda\n" \
              "para problemas pequeños No usar baches, entonces\n" \
              "puede poner este valor muy alto (autolimitado)."
    elif id == 18:  # problema - escr bache validacion
        res = "De los patrones de validación se harán sub-grupos,\n" \
              "del tamaño especificado aquí, se recomienda\n" \
              "dado que el set de validación es mucho menos que\n" \
              "el de entreno No usar baches, entonces puede\n" \
              "poner un valor muy alto (autolimitado)."
    elif id == 19:  # problema - escr apodos
        res = "Puede editar el nombre de la clase de salida,\n" \
              "esto se guardará automaticamente.\n\n" \
              "Con el selector de la derecha puede\n" \
              "cambiar de clase."
    elif id == 20:  # problema - escr normalizacion
        res = "Valor máximo que tendrán los parámetros\n" \
              "al ser normalizados con Min-Max."
    elif id == 21:  # all - por defecto
        res = "Pone los valores de la GUI\n" \
              "en los recomendables."
    elif id == 22:  # all - importa red
        res = "Abre un archivo (.txt) o (.xml) que contiene\n" \
              "la información de una red neuronal, importará\n" \
              "la red neuronal si:\n\n" \
              "- coinciden las dimensiones del problema.\n" \
              "- coinciden los valores de normalización.\n\n" \
              "Estos últimos se especifíca si provienen de\n" \
              "cálculos Min-Max, Z-Score o No se normalizó."
    elif id == 23:  # all  exporta red
        res = "Guarda una red en formato (.txt) o (.xml),\n" \
              "procure No modificar estos archivos.\n\n" \
              "XML posee mayor información, recomendable."
    elif id == 24:  # inicializacion - Kmedias
        res = "Algoritmo de clusterización (agrupamiento)\n" \
              "donde de partida se tiene un número finito\n" \
              "de puntos (dendritas), estos se distribuyen\n" \
              "en un número finito de ciclos."
    elif id == 25:  # inicializacion - DyC
        res = "Algoritmo de segragación iterativa, creará\n" \
              "una gran hipercaja y la dividirá un número\n" \
              "finito de veces, hasta que cada hipercaja\n" \
              "tenga solo una clase dentro."
    elif id == 26:  # inicializacion escr clusters
        res = "Digite el número de dendritas (clústers)\n" \
              "que desea como máximo en cada neurona\n" \
              "(clase); el algoritmo modificará esta\n" \
              "cantidad dependiendo de la relación de\n" \
              "patrones por clase.\n\n" \
              "= 0 hallará una solución automaticamente."
    elif id == 27:  # inicializacion escr dimension
        res = "Digite el tamaño de las hiper-cajas,\n" \
              "será truncado a un mínimo de 5%, y el\n" \
              "máximo (100%) referirá a el 25% del\n" \
              "tamaño total para cada dimensión."
    elif id == 28:  # inicializacion escr margen
        res = "Digite un valor que será una pequeña\n" \
              "margen entre las hipercajas, entre\n" \
              "menor sea, menor será el error, pero\n" \
              "tomará significativamente más tiempo.\n\n" \
              "100% refiere al 10% del tamaño total\n" \
              "para cada dimensión.\n\n" \
              "= 0% error de entreno ECM = 0.0"
    elif id == 29:  # entreno - play
        res = "Correr algoritmo."
    elif id == 30:  # entreno - pause
        res = "Pausar algoritmo\n" \
              "permite cambiar parámetros."
    elif id == 31:  # entreno - stop
        res = "Parar algoritmo\n" \
              "permite regresar al menú."
    elif id == 32:  # entreno - DE - check local
        res = "Ativo hace la selección de la nueva generación\n" \
              "comparando al padre con el hijo y dejando al\n" \
              "mejor; Inactivo elige a 2 individuos al azar\n" \
              "(sean padres o hijos) y deja al mejor.\n\n" \
              "Inactivo convergerá más rápido, se recomienda\n" \
              "dejarlo Activo."
    elif id == 33:  # null
        res = "You are cast out from the heavens to the ground"
    elif id == 34:  # entreno - iteraciones
        res = "Digite número de ciclos que desea ejecutar.\n\n" \
              "Al llegar al final, si desea seguir\n" \
              "entrenando sin reiniciar, digite un\n" \
              "nuevo valor mayor y pulse Correr."
    elif id == 35:  # entreno - probabilidad inhibir
        res = "Digite parámetro porcentual, durante el\n" \
              "entreno, en cada ciclo habrá una\n" \
              "probabilidad dada por éste valor de\n" \
              "que se intente quitar una dendrita al\n" \
              "azar; es decir: 0% nunca intentará\n" \
              "disminuir dendritas (más veloz) y\n" \
              "100% lo intentará siempre (lento)."
    elif id == 36:  # entreno - SGD - amortiguamiento
        res = amortigua.replace("$", "alfa")
    elif id == 37:  # entreno - DE - amortiguamiento
        res = amortigua.replace("$", "h")
    elif id == 38:  # entreno - PSO - amortiguamiento
        res = amortigua.replace("$", "c3")
    elif id == 39:  # entreno - SGD - minibache
        res = "Del bache de entreno se tomarán grupos\n" \
              "llamados mini-baches, del tamaño dado aquí,\n" \
              "SGD puro sería = 1, mientras que un\n" \
              "valor muy alto (autolimitado) tomaría a\n" \
              "todo el bache (entreno sin mini-baches)."
    elif id == 40:  # entreno - SGD - alfa
        res = "Digite valor de parámetro encargado de la\n" \
              "modificación de los pesos sinápticos en\n" \
              "cada paso (alfa), obedece a la ecuación\n" \
              "para cada peso:\n\n" \
              "u = u . beta + de/dw\n" \
              "w = w - alfa . u"
    elif id == 41:  # entreno - SGD - beta
        res = "Digite valor de parámetro encargado de la\n" \
              "fricción (beta) de la inercia involucrada\n" \
              "en el cambio de cada peso sináptico, obedece\n" \
              "a la ecuación para cada peso:\n\n" \
              "u = u . beta + de/dw\n" \
              "w = w - alfa . u\n\n" \
              "Autolimitada a 1 como máximo.\n" \
              "= 0 sin fricción."
    elif id == 42:  # entreno - DE - poblacion
        res = "Digite cuántos individuos quiere que\n" \
              "conformen la población que se reproducirá\n" \
              "para buscar la optimización."
    elif id == 43:  # entreno - DE - h
        res = "Digite valor de parámetro encargado de la\n" \
              "fuerza de mutación, es deir, que tanto\n" \
              "cambiará un gen (peso sináptico) al crear\n" \
              "un hijo, la ecuación para cada peso es:\n\n" \
              "v = x0 + h . (x1 - x2)\n" \
              "u = v si rand < c sino n"
    elif id == 44:  # entreno - DE - c
        res = "Digite valor de parámetro encargado de el\n" \
              "porcentaje de recombinación, un valor de\n" \
              "10% indica que el hijo tendrá ese porcentaje\n" \
              "de pesos diferentes a los del padre (n), la\n" \
              "ecuación para cada peso es:\n\n" \
              "v = x0 + h . (x1 - x2)\n" \
              "u = v si rand < c sino n\n\n" \
              "= 0% será 1 / pesos sinápticos ó mínimo 0.01%"
    elif id == 45:  # entreno - PSO - particulas
        res = "Digite cuántas partículas estarán\n" \
              "interactuando para buscar la optimización."
    elif id == 46:  # entreno - PSO - c1
        res = "Digite parámetro que representa la influencia\n" \
              "de la mejor solución (b) hallada por la partícula\n" \
              "(exploración), la ecuación para un peso es:\n\n" \
              "v = v . c3 + rand . c1 . (b - p) + rand . c2 . (g - p)\n" \
              "p = p + v"
    elif id == 47:  # entreno - PSO - c2
        res = "Digite parámetro que representa la influencia\n" \
              "de la solución global (g) hallada por las partículas\n" \
              "(explotación), la ecuación para un peso es:\n\n" \
              "v = v . c3 + rand . c1 . (b - p) + rand . c2 . (g - p)\n" \
              "p = p + v"
    elif id == 48:  # entreno - PSO - c3
        res = "Digite parámetro que representa la disminución\n" \
              "de la velocidad de la partícula, se recomiendan\n" \
              "valores altos, cercanos a 1 para evitar\n" \
              "convergencia prematura, la ecuación es:\n\n" \
              "v = v . c3 + rand . c1 . (b - p) + rand . c2 . (g - p)\n" \
              "p = p + v"
    elif id == 49:  # all - tolerancia ECM
        res = "Digite valor de tolerancia, refiere al ECM máximo\n" \
              "permitido por el algoritmo de optimización de\n" \
              "dendritas (disminución); es decir que se\n" \
              "intentarán quitar hipercajas siempre y cuando\n" \
              "ello No eleve el ECM más allá de la tolerancia."
    elif id == 50:  # inicializacion - check unir
        res = "Unir efectuará un segundo ciclo en el que\n" \
              "une a las hipercajas que puedan hacerlo, de\n" \
              "este modo evita crear mas de las necesarias\n" \
              "(ralentiza)."
    elif id == 51:  # postentreno - ejecutar red
        res = "Ingresa a la red neuronal las entradas digitadas\n" \
              "en la line de abajo, lo que arrojará como\n" \
              "resultado una clase ganadora, la cual se\n" \
              "mostrará abajo junto a su color asociado."
    elif id == 52:  # postentreno - unir dendritas
        res = "Recorrerá a todas las dendritas (hipercajas)\n" \
              "verificando si pueden unirse a otra de su\n" \
              "misma clase, lo que reducirá el número de\n" \
              "estas, siempre que el error No sobrepase\n" \
              "la tolerancia."
    elif id == 53:  # postentreno - quitar dendritas
        res = "Recorrerá a todas las dendritas (hipercajas)\n" \
              "verificando si puede quitarlas (inhibirlas)\n" \
              "sin que el error sobrepase la tolerancia."
    elif id == 54:  # postentreno - deshinibir dendritas
        res = "Convierte a las dendritas quitadas (inhibidas)\n" \
              "en activas, esto puede servir para re-entrenar\n" \
              "teniendo más dendritas que mover, o como\n" \
              "revertimiento del algoritmo de Quitar."
    elif id == 55:  # postentreno - eliminar inhibidas
        res = "Para hacer a la red más ligera en memoria,\n" \
              "entreno y archivo final, elimina\n" \
              "permanentemente a las dendritas\n" \
              "quitadas (inhibidas)."
    elif id == 56:  # postentreno - escr entradas
        res = "Digite valores de entrada separados por comas,\n" \
              "tantos como entradas tenga la red, estos\n" \
              "pueden ser enteros o flotantes, con o sin\n" \
              "espacios, ej: 34, 6, 12.8, 0"
    elif id == 57:  # postentreno - check normalizar
        res = "Normalizar aplicará a las entradas ingresadas la\n" \
              "misma transformación hecha al normalizar el set:\n" \
              "- Min-Max.\n" \
              "- Z-Score.\n\n" \
              "Si No los normalizó, No habrá diferencia."
    elif id == 58:  # all - ERL
        res = "Error de regresión logistica, las salidas\n" \
              "Softmax (Y) van de 0 a 1, si la deseada es\n" \
              "(d), la ecuación seria:\n\n" \
              "e = -log10(max(Yd, 1.E-6)) / 6"
    elif id == 59:  # analisis - exportar matrices
        res = "Guarda las matrices de confusión\n" \
              "en formato (.xml), las 5:\n" \
              "entreno, validación, testeo,\n" \
              "  No-entreno, general.\n\n" \
              "No incluye análisis estadísticos\n" \
              "ni errores ECM o ERL; incluye el\n" \
              "número de dendritas."
    elif id == 60: # analisis - txt precision
        res = "Porcetaje de datos correctamente clasificados."
    elif id == 61:  # analisis - txt exactitud
        res = "Probabilidad de que la predicción entregada por el\n" \
              "detector realmente pertenezca a dicha clase."
    elif id == 62:  # analisis - txt sensibilidad
        res = "Porcentaje de los datos de dicha clase\n" \
              "que fueron correctamente clasificados."
    elif id == 63:  # analisis - txt especificidad
        res = "Porcentaje de los datos No perteneciente\n" \
              "a dicha clase que fueron correctamente\n" \
              "clasificados como No pertenecientes."
    elif id == 64:  # analisis - txt valorF
        res = "Media armónica entre la Exactitud y la\n" \
              "Sensibilidad, su mejor valor es 100%, en\n" \
              "otras palabras, esta puntuación relacióna\n" \
              "a los dos valores mencionados."
    elif id == 65: # analisis - txt kappa
        res = "Grado de acuerdo entre dos mediciónes\n" \
              "(salidas deseadas vs resultados obtenidos),\n" \
              "toma en consideración al azar; un valor\n" \
              "mayor a 0.6 es buena concordancia y mayor\n" \
              "a 0.8 muy buena."
    elif id == 66: # analisis - archivo resultados
        res = "Guarda en formato (.txt) o (.xml) la lista de\n" \
              "valores deseados y obtenidos para todos los\n" \
              "patrones, esto es análogo a la función de\n" \
              "ejecución del GUI post-entreno, solo que\n" \
              "para muchos datos; puede ingresar como\n" \
              "patrones una lista y luego usar esto."
    elif id == 67: # problema - check zscore minmax
        res = "Seleccióne uno de los dos algoritmos:\n\n" \
              "- Min-Max:\n" \
              "v = ((v - min) / (max - min)) * 2 - 1\n\n" \
              "- Z-Score:\n" \
              "v = (v - prom) / desv_std"
    return res

# a continuacion las funciones de graficacion

def GEntrenamiento(id, cambioDendras):
    # pintar linea de mejor
    if id.motor.numPEVT[1] > 1 and id.motor.curvas.shape[0] > 1:
        linea = QLineSeries()
        linea.setColor(Colores(4, False))
        linea.append(id.motor.mejV, 0.0)
        linea.append(id.motor.mejV, id.motor.curvas[id.motor.mejV, 1])
        id.figura.chart().addSeries(linea)
    # agregar las 4 lineas
    paso = int(np.ceil(id.motor.curvas.shape[0] / 300.0))
    h = (4 if cambioDendras else 3)
    for f in range(h):
        ok = True
        if f == 1 and id.motor.numPEVT[1] == 1:
            ok = False
        elif f == 2 and id.motor.numPEVT[2] == 1:
            ok = False
        if ok:
            linea = QLineSeries()
            linea.setColor(Colores(f, False))
            for z in range(1, id.motor.curvas.shape[0], paso):
                linea.append(z - 1, id.motor.curvas[z, f])
            id.figura.chart().addSeries(linea)
    # pintar punto en 0,0
    punto = QScatterSeries()
    punto.setMarkerSize(0.001)
    punto.append(0.0, 0.0)
    id.figura.chart().addSeries(punto)

def GPesos(id, esParti):
    dim = id.motor.patrones.shape[1] - 1
    if dim > 1 and len(id.motor.agentes) > 0: # problema mayor a 1D y con agentes disponibles
        XY = np.array([id.motor.grXY[0], id.motor.grXY[1]])
        # definir dimension absoluta de patrones
        pxy = np.column_stack((id.motor.patrones[:, XY[0]], id.motor.patrones[:, XY[1]])).max(axis=0) -\
              np.column_stack((id.motor.patrones[:, XY[0]], id.motor.patrones[:, XY[1]])).min(axis=0)
        # crear punto de agente mejor
        if esParti:
            puntos = QScatterSeries()
            puntos.setMarkerSize(8)
            puntos.setPen(Colores(1, False))
            puntos.setBrush(Qt.transparent)
            puntos.append(id.motor.agentes[id.motor.mejor].besW[2 * XY[0]],
                          id.motor.agentes[id.motor.mejor].besW[2 * XY[1]])
            id.figura.chart().addSeries(puntos)
        # crear lista de puntos
        puntos = QScatterSeries()
        puntos.setMarkerSize(4)
        puntos.setPen(Qt.transparent)
        puntos.setBrush(Colores(0, False))
        # agregar los puntos
        for n in id.motor.agentes:
            puntos.append(n.pesW[2 * XY[0]], n.pesW[2 * XY[1]])
        id.figura.chart().addSeries(puntos)
        # pintar dos puntos en los extremos para dimensionar grafica
        puntos = QScatterSeries()
        puntos.setMarkerSize(0.001)
        pxy *= 0.05
        puntos.append(id.motor.laRed.pesW[2 * XY[0]] - pxy[0], id.motor.laRed.pesW[2 * XY[1]] - pxy[1])
        puntos.append(id.motor.laRed.pesW[2 * XY[0]] + pxy[0], id.motor.laRed.pesW[2 * XY[1]] + pxy[1])
        id.figura.chart().addSeries(puntos)

def GPatrones(id):
    dim = id.motor.patrones.shape[1] - 1
    if dim > 1:
        XY = np.array([id.motor.grXY[0], id.motor.grXY[1]])
        # definir dimension absoluta de patrones
        pxy = np.column_stack((id.motor.patrones[:, XY[0]], id.motor.patrones[:, XY[1]])).max(axis=0) -\
              np.column_stack((id.motor.patrones[:, XY[0]], id.motor.patrones[:, XY[1]])).min(axis=0)
        # crear listas de puntos y asociarlos segun clase
        for m in range(int(id.motor.patrones[:, dim].max() + 1)):
            # puntos de entreno
            puntosE = QScatterSeries()
            puntosE.setMarkerSize(4)
            puntosE.setPen(Qt.transparent)
            puntosE.setBrush(Colores(m, False))
            # puntos de validacion
            puntosV = QScatterSeries()
            puntosV.setMarkerSize(6)
            puntosV.setMarkerShape(1)
            puntosV.setPen(Qt.transparent)
            puntosV.setBrush(Colores(m, False))
            # puntos de testeo
            puntosT = QScatterSeries()
            puntosT.setMarkerSize(8)
            puntosT.setPen(Colores(m, False))
            puntosT.setBrush(Qt.transparent)
            # seleccionar los puntos
            for p in range(id.motor.patrones.shape[0]):
                if id.motor.patrones[p, dim] == m:
                    if p < id.motor.numPEVT[0]:
                        puntosE.append(id.motor.patrones[p, XY[0]], id.motor.patrones[p, XY[1]])
                    elif p < id.motor.numPEVT[0] + id.motor.numPEVT[1]:
                        puntosV.append(id.motor.patrones[p, XY[0]], id.motor.patrones[p, XY[1]])
                    else:
                        puntosT.append(id.motor.patrones[p, XY[0]], id.motor.patrones[p, XY[1]])
            if puntosE.count() > 0:
                id.figura.chart().addSeries(puntosE)
            if puntosV.count() > 0:
                id.figura.chart().addSeries(puntosV)
            if puntosT.count() > 0:
                id.figura.chart().addSeries(puntosT)
        # pintar dos puntos en los extremos para dimensionar grafica
        puntos = QScatterSeries()
        puntos.setMarkerSize(0.001)
        pxy *= 0.05
        puntos.append(id.motor.patrones[:, XY[0]].min() - pxy[0], id.motor.patrones[:, XY[1]].min() - pxy[1])
        puntos.append(id.motor.patrones[:, XY[0]].max() + pxy[0], id.motor.patrones[:, XY[1]].max() + pxy[1])
        id.figura.chart().addSeries(puntos)

def GRed(id, red):
    tipo = id.selector.currentIndex()
    if tipo == 1 or tipo == 3: # si dibujar cajas
        dim = id.motor.patrones.shape[1] - 1
        if dim > 1:
            XY = 2 * np.array([id.motor.grXY[0], id.motor.grXY[1]])
            a = 0
            for m in range(red.numK.size):
                for k in range(red.numK[m]):
                    if red.actK[a]:
                        aa = a * dim * 2
                        cuadro = QLineSeries()
                        cuadro.setColor(Colores(m, False))
                        cuadro.append(red.pesW[aa + XY[0] + 1], red.pesW[aa + XY[1] + 1])
                        cuadro.append(red.pesW[aa + XY[0]], red.pesW[aa + XY[1] + 1])
                        cuadro.append(red.pesW[aa + XY[0]], red.pesW[aa + XY[1]])
                        cuadro.append(red.pesW[aa + XY[0] + 1], red.pesW[aa + XY[1]])
                        cuadro.append(red.pesW[aa + XY[0] + 1], red.pesW[aa + XY[1] + 1])
                        id.figura.chart().addSeries(cuadro)
                    a += 1

def GSuperficie(id):
    tipo = id.selector.currentIndex()
    if tipo == 2 or tipo == 3:  # si dibujar superficie
        dim = id.motor.patrones.shape[1] - 1
        if dim > 1:
            total = 100
            XY = np.array([id.motor.grXY[0], id.motor.grXY[1]])
            # definir dimension absoluta de patrones al %5
            pxy = (id.motor.patrones[:, :dim].max(axis=0) - id.motor.patrones[:, :dim].min(axis=0)) * 0.05
            # crear las listas de puntos para cada clase
            lista = []
            for m in range(id.motor.laRed.numK.size):
                lista.append(QScatterSeries())
                lista[m].setMarkerSize(3)
                lista[m].setPen(Qt.transparent)
                lista[m].setBrush(Colores(m, True))
            # hacer el barrido de puntos
            Xp = np.linspace(id.motor.patrones[:, XY[0]].min() - pxy[XY[0]],
                             id.motor.patrones[:, XY[0]].max() + pxy[XY[0]], num=total)
            Yp = np.linspace(id.motor.patrones[:, XY[1]].min() - pxy[XY[1]],
                             id.motor.patrones[:, XY[1]].max() + pxy[XY[1]], num=total)
            pxy *= 0.0
            for x in Xp:
                for y in Yp:
                    pxy[XY[0]] = x
                    pxy[XY[1]] = y
                    lista[id.motor.laRed.EjecutarRed(pxy).argmax()].append(x, y)
            # agregar las listas a la grafica
            for puntos in lista:
                id.figura.chart().addSeries(puntos)

def GROC(id):
    # agregar linea gris
    puntos = QLineSeries()
    puntos.setColor(Colores(1000, True))
    puntos.append(0.0, 0.0)
    puntos.append(1.0, 1.0)
    id.figura.chart().addSeries(puntos)
    # agregar lineas coloridas ROC
    cla = id.matrix.shape[0] - 3
    for m in range(cla):
        puntos = QLineSeries()
        puntos.setColor(Colores(m, False))
        puntos.append(0.0, 0.0)
        # x: 1 - especificidad, y: sensibilidad
        puntos.append(1.0 - id.matrix[cla + 2, m],
                      id.matrix[m, cla + 1])
        puntos.append(1.0, 1.0)
        id.figura.chart().addSeries(puntos)

def GAxes(id):
    id.figura.chart().createDefaultAxes()
    id.figura.chart().legend().setVisible(False)

def Colores(clase, superficie):
    basicos = [Qt.red, Qt.blue, Qt.magenta, Qt.green, Qt.cyan, Qt.yellow, Qt.gray]
    col = []
    for c in basicos:
        col.append(QColor(c))
    if superficie:
        for c in col:
            c.setAlpha(100)
    return col[max(0, min(clase, len(col) - 1))]

def FondoPSI(id):
    bac = QPixmap("img37.png")
    bac = bac.scaled(id.size(), Qt.IgnoreAspectRatio)
    pal = id.palette()
    pal.setBrush(QPalette.Background, QBrush(bac))
    id.setPalette(pal)

# esta funcion contiene todos los estilos para las GUIs
def Estilos(esc):
    txt = "QPushButton { font-size: $2px; } " \
          "QLabel { font-size: $2px; } " \
          "QLineEdit { font-size: $2px; } " \
          "QCheckBox { font-size: $2px; } " \
          "QComboBox { font-size: $2px; } " \
          "QTableWidget { font-size: $2px; } " \
          "QGroupBox { " \
          "    font-size: $1px; " \
          "    background-color: rgba(255, 255, 255, 90); " \
          "    border: 1px solid gray; " \
          "    border-radius: 15px; } " \
          "QLabel#subtitulo { font-size: $3px; } " \
          "QLabel#titulo { font-size: $3px; } " \
          "QPushButton#m_problema { background-color: rgb(218,220,217); } " \
          "QPushButton#m_inicializacion { background-color: rgb(252,250,200); } " \
          "QPushButton#m_gradiente { background-color: rgb(255,220,215); } " \
          "QPushButton#m_evolutivo { background-color: rgb(209,255,207); } " \
          "QPushButton#m_particulas { background-color: rgb(211,207,255); } " \
          "QPushButton#m_postentreno { background-color: rgb(211,213,189); } " \
          "QPushButton#m_analisis { background-color: rgb(247,214,254); } " \
          "QProgressBar#v_gradiente:chunk { " \
          "    font-size: $2px; " \
          "    background-color: rgb(194,39,30); } " \
          "QProgressBar#v_evolutivo:chunk { " \
          "    font-size: $2px; " \
          "    background-color: rgb(60,162,45); } " \
          "QProgressBar#v_particulas:chunk { " \
          "    font-size: $2px; " \
          "    background-color: rgb(62,40,185); }"
    escala = esc / 615.0
    txt = txt.replace("$1", str(int(12 * escala)))
    txt = txt.replace("$2", str(int(16 * escala)))
    txt = txt.replace("$3", str(int(20 * escala)))
    return txt

# funcion para generar parametros de compilacion en linea de comandos
# (no usada en el software), (sin dependencias)
def Compilador(img):
    com = "pyinstaller -y -F -w -i \"icono.ico\" --add-data \"" \
          "TutorialDMNN.pdf\";\".\" --add-data \"wait.gif\";\".\"" \
          " --add-data \"pitido.mp3\";\".\""
    for i in range(img):
        com += " --add-data \"img" + str(i) + ".png\";\".\""
    com += " SoftwareDMNN.py"
    f = open("compilar.txt", "w")
    f.write(com)
    f.close()
    print(com)

# funcion basica para ejecutar DMNN, no usa Softmax ni reconoce dendritas inhibidas
# (no usada en el software), (depende de: import numpy as np)
def EjecutarRed(entradas, pesW, numK):
    X = entradas
    while X.size < pesW.size / 2:
        X = np.hstack((X, entradas))
    W = pesW.copy().reshape(-1, 2)
    WH = W[:, 0] - X
    WL = X - W[:, 1]
    Wmki = np.minimum(WH, WL)
    Wmki = Wmki.reshape(-1, entradas.size)
    Smk = Wmki.min(axis=1)
    Zm = np.zeros(numK.size)
    n = 0
    for m in range(Zm.size):
        Zm[m] = Smk[n:(n + numK[m])].max()
        n += numK[m]
    y = np.argmax(Zm)
    return y

# instanciar el software
main()

# Trabajo Futuro:
# - al entrenar, si grafica solo patrones, no esta dibujando iteraciones
# - ponert try en importacion de patrones, en la lectura de matriz numoy
# - hacer sonidos diferentes al finalizar hilos, inici, entreno, optimiz
# - proceso DyC pinta numero de dendritas creadas/comparadas (progreso)
# - proceso Kmedias pinta numero de ciclos (a modo de progreso)
# - procesos unir/quitar dendritas pinta numero de comparadas (progreso)
# - todos los procesos en hilos tienen boton de stop, no solo entreno
# - limitar patrones graficados a maximo unos 1000
# - si # clases > # colores, usar un color a modo de gradiente
# - proceso DyC limita inferiormente al hipervolumen de caja
# - poceso Kmedias limita maximo de iteraciones
# - importar y exportar red deberian ser botones globales (no 2 GUIs)
# - alguna forma de limitar el maximo del set leido (para no borrar)
# - en DE hacer que ultimo individuo siempre se reemplace por su hijo
