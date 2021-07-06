import sys  # sys нужен для передачи argv в QApplication
import os
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from PyQt5.uic.properties import QtGui
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import MADX.madx as Madx
from algorithm import Gauss_Newton
import time
#from MADX.madx import structure_calculate
#matplotlib.use('Qt5Agg')

import design2  # Это наш конвертированный файл дизайна




def save_file():
    # f = open("test.txt", "w")
    # f.write("testing")
    # f.close()
    print("sozdal")



# def paint():
#     graph = pg.PlotWidget()
#     x = np.random.random(10)
#     y = np.random.random(10)
#     graph.plot(x, y, clear=True)

class Stream(QtCore.QObject):
    """Redirects console output to text widget."""
    newText = QtCore.pyqtSignal(str)

    def write(self, text):
        self.newText.emit(str(text))



test = np.random.random(100)

singular_values = 0
u = 0
v = 0
response_matrix = 0
inverted_response_matrix = 0
areSingularValuesPicked = False
isRegionClean = True
itteration = 0
isOptimized = False
table_optimized = 0


class ExampleApp(QtWidgets.QMainWindow, design2.Ui_MainWindow):

    def __init__(self):
        # Это здесь нужно для доступа к переменным, методам
        # и т.д. в файле design.py
        super(ExampleApp, self).__init__()
        self.setupUi(self)  # Это нужно для инициализации нашего дизайна
        self.isStructureLoaded = False
        #print(self.isStructureLoaded)


        self.actionOpen.triggered.connect(self.open_structure_file)
        self.calculateCorrection.clicked.connect(self.add_lattice_correction_plots)
        #self.calculateCorrection.clicked.connect(self.add_orbit_correction_plots)
        self.collectMatrix.clicked.connect(self.collect_response_matrix)
        self.invertMatrix.clicked.connect(self.choose_singular_values)

        self.collectMatrix_10.clicked.connect(self.Gauss_Newton_optimize)



        # self.ring = madx.Structure('MADX\VEPP4M.txt')


        #self.viewBox = pg.Qt.
        # self.graphicsView2 = pg.PlotWidget()
        # self.verticalLayout.addWidget(self.graphicsView2)
        # self.graphicsView2.plot(test,test)
        # self.graphicsView2.removeItem(self.verticalLayout)
        #self.graphicsView1.getViewBox()



        #self.data = 0
        #self.textEdit = QtWidgets.QTextEdit()
        #self.setCentralWidget(self.textEdit)






        # w = pg.GraphicsLayoutWidget(self.tab_2,show=True, size=(600,400), border=True)
        # #w = pg.PlotItem(self.tab_2)
        # self.formLayout_2.setWidget(6, QtWidgets.QFormLayout.FieldRole, w)
        # w1 = w.addLayout(row=0, col=0)
        # v1a = w1.addViewBox(row=1, col=0, lockAspect=True)
        # img1a = pg.PlotCurveItem(svd)
        # axis = pg.AxisItem(orientation='left',showValues=True)
        # img2a = pg.PlotDataItem(svd1)
        #
        # v1a.addItem(img1a)
        # v1a.addItem(axis)
        # v1a.addItem(img2a)


        #print(v1a.getPlotItem())








        # reg = pg.LinearRegionItem()
        # self.graphicsView_2.addItem(reg)
        # reg.sigRegionChangeFinished.connect(lambda: self.region(reg))





        self.add_functions()





    def draw_optics(self,item, checkButton):
        #self.graphicsView.setBackground('w')
        if checkButton.isChecked() == True:
            self.graphicsView.addItem(item)
        else:
            self.graphicsView.removeItem(item)









    def browse_folder(self):
        self.listWidget.clear()  # На случай, если в списке уже есть элементы
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Выберите папку")
        # открыть диалог выбора директории и установить значение переменной
        # равной пути к выбранной директории

        if directory:  # не продолжать выполнение, если пользователь не выбрал директорию
            for file_name in os.listdir(directory):  # для каждого файла в директории
                self.listWidget.addItem(file_name)  # добавить файл в listWidget

    def add_functions(self):
        self.openButton.clicked.connect(self.browse_folder)
        self.saveButton.clicked.connect(save_file)
        #self.textBrowser.
        #self.saveButton.clicked.connect(paint)

    # def changeState(self, state1, state2):
    #
    #     state1, state2 = state2, state1
    #     return state1, state2


    def draw_data(self):
        #dataX, dataY = self.ring.read_BPMs(('MADX\measureXY.txt'))
        #data = self.ring.read_BPMs(('MADX\measureXY.txt'))
        dataS = self.ring.bad_twiss.s
        dataX, dataY = self.ring.bad_twiss.x,self.ring.twiss.x
        #print(dataY)
        self.graphicsView_2.plot(dataS,dataX)
        self.graphicsView_2.plot(dataS,dataY)


    def collect_response_matrix(self):
        global response_matrix, model_response_matrix, isRegionClean
        dataX, dataY = self.ring.read_BPMs('MADX\measureBETXY.txt')
        #elem, number = self.ring.read_elements('MADX\quads.txt')
        #elem, number = self.ring.read_elements('MADX\correctors.txt')
        elem, number = self.ring.read_elements('MADX\corrs&quads.txt')
        #tunes = np.array([[8.57081948, 7.622538629]])
        #model_tunes = np.array([[8.573128249, 7.618473106]])
        #data = np.concatenate((dataX,tunes))
        #print(data)

        response_matrix = self.ring.response_matrix_calculate(self.ring.bad_structure,self.ring.bad_structure_in_lines,elem,number,dataX[:,1],0.00001,areErrorsNeeded=False)
        #model_response_matrix = self.ring.response_matrix_calculate(self.ring.structure,elem,number,dataX[:,1],0.001)
        isRegionClean = False

        #return response_matrix, model_response_matrix
        return response_matrix

    def Gauss_Newton_optimize(self):
        global isOptimized,table_optimized
        dataX, dataY = self.ring.read_BPMs('MADX\measureBETXY.txt')
        elem, number = self.ring.read_elements('MADX\corrs&quads.txt')
        withErrors = False

        optimizer = Gauss_Newton(fileName)
        if withErrors == True:
            new_grad_parameters,new_alignment_parameters = optimizer.optimize(elem,number,dataX[:,1],withErrors=withErrors,step=0.00001,tolerance=1e-5)
        else:
            new_grad_parameters,new_alignment_parameters = optimizer.optimize(elem,number,dataX[:,1],withErrors=withErrors,step=0.00001,tolerance=1e-12),0



        # new_parameters = np.array([6.88936625e-04,6.99611785e-04,-1.25348505e-05,-2.44152118e-04,
        #                   4.49023832e-06,-8.93807005e-06,1.35438095e-03,-2.62516523e-05,
        #                   9.99917659e-06,1.34855732e-05,-1.37631506e-05,8.35790424e-06,
        #                   8.35565530e-06,-1.37632447e-05,1.34873401e-05,9.99917302e-06,
        #                   -2.62571217e-05,1.35448174e-03,-8.93919408e-06,4.49084424e-06,
        #                   -2.44234132e-04,-1.25358175e-05,-3.66839148e-06,-5.42528278e-07,
        #                   -7.11985720e-07,1.94314783e-07,-1.10797254e-05,9.06372750e-06,
        #                   -1.81964154e-05,1.97227359e-06,9.16494657e-07,-4.17483188e-06,
        #                   -2.49726413e-06,1.24072698e-05,-1.92951458e-05,-1.92950244e-05,
        #                   1.24067623e-05,-2.49710924e-06,-4.17686036e-06,9.16698712e-07,
        #                   1.97506589e-06,-1.82042774e-05,9.06796278e-06,-1.10767659e-05,
        #                   1.94356527e-07,-7.12440695e-07,-5.42881421e-07,-3.66740939e-06])
        # table_optimized,_ = self.ring.change_structure(self.ring.structure,self.ring.structure_in_lines,new_grad_parameters,np.zeros(number),new_alignment_parameters,areErrorsNeeded=False,areErrorsForOptimize=True)
        # new_grad_parameters = np.zeros_like(new_grad_parameters)
        table_optimized,_ = self.ring.change_structure(self.ring.bad_structure,self.ring.bad_structure_in_lines,-new_grad_parameters,np.zeros(number),-new_alignment_parameters,base_imperfections=True,areErrorsForSimpleSVD=False,areErrorsForOptimize=withErrors)
        isOptimized = True

        return table_optimized


    def choose_singular_values(self):
        global u, v

        u, sv, v = self.ring.invert_response_matrix(response_matrix)

        self.graphicsView_3.plot(sv)
        reg = pg.LinearRegionItem()
        self.graphicsView_3.addItem(reg)

        reg.sigRegionChangeFinished.connect(lambda: self.update_inverted_matrix(sv,reg))



    def update_inverted_matrix(self,sv,reg):
        global singular_values, inverted_response_matrix
        region = reg.getRegion()
        print(region)
        left_edge = np.maximum(int(region[0]),0)
        right_edge = np.minimum(int(region[1]),len(sv))
        singular_values = sv[left_edge:right_edge]

        if len(singular_values) != len(sv):
            zero_singulars = np.zeros(len(sv)-len(singular_values))
            singular_values = 1/singular_values
            singular_values = np.diag(np.concatenate((singular_values,zero_singulars)))
        else:
            singular_values = 1/singular_values
            singular_values = np.diag(singular_values)


        inverted_response_matrix = np.matmul(np.matmul(v,singular_values),u.T)
        self.checkBox_5.stateChanged.connect(lambda: self.check_sv_chosen(reg))


    def check_sv_chosen(self,reg):
        global areSingularValuesPicked
        if self.checkBox_5.isChecked() == True:
            reg.setMovable(m = False)
            areSingularValuesPicked = True
        else:
            reg.setMovable(m = True)
            areSingularValuesPicked = False







    def add_lattice_correction_plots(self):
        if isOptimized == False:
            global itteration, previous_structure, previous_structure_short
            # if bpy.data.objects.get("asd")
            #     print(madx.Structure.__class_getitem__())

            ## for lattice correction
            #data = self.ring.read_BPMs('MADX\measureBETXY.txt')
            #tunes = np.array([[8.57081948, 7.622538629]])
            #model_tunes = np.array([[8.573128249, 7.618473106]])
            #data = np.concatenate((data,tunes))
            #elem, number = self.ring.read_elements('MADX\quads.txt')
            #matrix = self.ring.response_matrix_calculate(elem,number,data[:,1],0.001)

            if areSingularValuesPicked == True:
                print(inverted_response_matrix)
                if itteration == 0:
                    # model_structure = np.concatenate((self.ring.twiss.betx,self.ring.twiss.bety,self.ring.summ_table.q1,self.ring.summ_table.q2))
                    # real_structure = np.concatenate((self.ring.bad_twiss.betx,self.ring.bad_twiss.Fy,self.ring.bad_summ_table.q1,self.ring.bad_summ_table.q2))
                    ## lattice
                    # model_structure = np.concatenate((self.ring.twiss_short.betx,self.ring.twiss_short.bety))
                    # real_structure = np.concatenate((self.ring.bad_twiss_short.betx,self.ring.bad_twiss_short.bety))
                    ## orbit + lattice
                    # model_structure = np.concatenate((self.ring.twiss_short.x,self.ring.twiss_short.y,self.ring.twiss_short.betx,self.ring.twiss_short.bety))
                    # real_structure = np.concatenate((self.ring.bad_twiss_short.x,self.ring.bad_twiss_short.y,self.ring.bad_twiss_short.betx,self.ring.bad_twiss_short.bety))
                    ## orbit
                    model_structure = np.concatenate((self.ring.twiss_short.x,self.ring.twiss_short.y))
                    real_structure = np.concatenate((self.ring.bad_twiss_short.x,self.ring.bad_twiss_short.y))

                    #tunes = np.array([[8.57081948, 7.622538629]])
                    #model_tunes = np.array([[8.573128249, 7.618473106]])
                else:
                    ## lattice
                    # model_structure = np.concatenate((self.ring.twiss_short.betx,self.ring.twiss_short.bety))
                    # real_structure = np.concatenate((previous_structure_short.betx,previous_structure_short.bety))

                    ## orbit and lattice
                    model_structure = np.concatenate((self.ring.twiss_short.x,self.ring.twiss_short.y,self.ring.twiss_short.betx,self.ring.twiss_short.bety))
                    real_structure = np.concatenate((previous_structure_short.x,previous_structure_short.y,previous_structure_short.betx,previous_structure_short.bety))

                corrected_optics = self.ring.correct_lattice(response_matrix,inverted_response_matrix,model_structure,real_structure,self.scaleFactor.value())
                corrected_twiss = corrected_optics.twiss
                corrected_twiss_short = corrected_optics.twiss_short
                previous_structure = corrected_twiss
                previous_structure_short = corrected_twiss_short
                error_short = self.ring.twiss_short.betx-corrected_twiss_short.betx
                error = self.ring.twiss.betx-corrected_twiss.betx
                print(np.sum(error),np.sum(error_short))
                #corrected_tunes = np.array([corrected_optics.summ.q1,corrected_optics.summ.q2])
                # print('model tunes:', model_tunes)
                # print('real tunes:', tunes)
                # print('corrected tunes', corrected_tunes)

                item1 = pg.PlotCurveItem(self.ring.twiss.s,self.ring.twiss.dx)
                item2 = pg.PlotCurveItem(self.ring.twiss.s,self.ring.bad_twiss.betx)
                item3 = pg.PlotCurveItem(self.ring.twiss.s,corrected_twiss.betx)
                item4 = pg.PlotCurveItem(self.ring.twiss.s,self.ring.bad_twiss.x)
                item5 = pg.PlotCurveItem(self.ring.twiss.s,corrected_twiss.x)

                result = np.stack((self.ring.twiss.s,self.ring.twiss.betx,self.ring.twiss.bety,self.ring.twiss.dx,self.ring.twiss.dy,self.ring.bad_twiss.betx,self.ring.bad_twiss.bety,self.ring.bad_twiss.dx,self.ring.bad_twiss.dy,corrected_twiss.betx,corrected_twiss.bety,corrected_twiss.dx,corrected_twiss.dy),axis=1)
                result = pd.DataFrame(result).to_csv("result.txt",sep="\t")


                # item3 = pg.PlotCurveItem(self.ring.twiss.s,corrected_twiss.x)
                # item4 = pg.PlotCurveItem(self.ring.twiss.s,self.ring.bad_twiss.x)

                # item1 = pg.PlotCurveItem(self.ring.twiss_short.s,self.ring.twiss_short.betx)
                # item2 = pg.PlotCurveItem(self.ring.twiss_short.s,self.ring.bad_twiss_short.betx)
                # item3 = pg.PlotCurveItem(self.ring.twiss_short.s,corrected_twiss_short.betx)
                # item4 = pg.PlotCurveItem(self.ring.twiss_short.s,self.ring.twiss_short.dy)

                self.checkBox.stateChanged.connect(lambda: self.draw_optics(item1,self.checkBox))
                self.checkBox_2.stateChanged.connect(lambda: self.draw_optics(item2,self.checkBox_2))
                self.checkBox_3.stateChanged.connect(lambda: self.draw_optics(item3,self.checkBox_3))
                self.checkBox_4.stateChanged.connect(lambda: self.draw_optics(item4,self.checkBox_4))
                self.checkBox_6.stateChanged.connect(lambda: self.draw_optics(item5,self.checkBox_6))
            else:
                print("Pick singular values!")

            itteration += 1

        else:

            corrected_twiss = table_optimized.twiss
            corrected_twiss_short = table_optimized.twiss_short
            item1 = pg.PlotCurveItem(self.ring.twiss.s,self.ring.twiss.betx)
            item2 = pg.PlotCurveItem(self.ring.twiss.s,self.ring.bad_twiss.betx)
            item3 = pg.PlotCurveItem(self.ring.twiss.s,corrected_twiss.betx)
            item4 = pg.PlotCurveItem(self.ring.twiss.s,self.ring.bad_twiss.x)
            item5 = pg.PlotCurveItem(self.ring.twiss.s,corrected_twiss.x)

            self.checkBox.stateChanged.connect(lambda: self.draw_optics(item1,self.checkBox))
            self.checkBox_2.stateChanged.connect(lambda: self.draw_optics(item2,self.checkBox_2))
            self.checkBox_3.stateChanged.connect(lambda: self.draw_optics(item3,self.checkBox_3))
            self.checkBox_4.stateChanged.connect(lambda: self.draw_optics(item4,self.checkBox_4))
            self.checkBox_6.stateChanged.connect(lambda: self.draw_optics(item5,self.checkBox_6))








    def add_orbit_correction_plots(self):
        global itteration, previous_structure
        ## for orbit correction

        if areSingularValuesPicked == True:
            print(inverted_response_matrix)
            if itteration == 0:
                # model_structure = np.concatenate((self.ring.twiss.betx,self.ring.twiss.bety,self.ring.summ_table.q1,self.ring.summ_table.q2))
                # real_structure = np.concatenate((self.ring.bad_twiss.betx,self.ring.bad_twiss.bety,self.ring.bad_summ_table.q1,self.ring.bad_summ_table.q2))
                model_structure = np.concatenate((self.ring.twiss_short.x,self.ring.twiss_short.y))
                real_structure = np.concatenate((self.ring.bad_twiss_short.x,self.ring.bad_twiss_short.y))
                #tunes = np.array([[8.57081948, 7.622538629]])
                #model_tunes = np.array([[8.573128249, 7.618473106]])
            else:
                model_structure = np.concatenate((self.ring.twiss_short.x,self.ring.twiss_short.y))
                real_structure = np.concatenate((previous_structure_short.x,previous_structure_short.y))

            # data = self.ring.read_BPMs('MADX\measureXY.txt')
            # elem, number = self.ring.read_elements('MADX\correctors.txt')
            # matrix = self.ring.response_matrix_calculate(elem,number,data[:,1],0.0001)

            # model_structure = np.concatenate((self.ring.twiss.x,self.ring.twiss.y))
            # real_structure = np.concatenate((self.ring.bad_twiss.x,self.ring.bad_twiss.y))
            # corrected_twiss = self.ring.correct_lattice(matrix,model_structure,real_structure)

            corrected_optics = self.ring.correct_lattice(response_matrix,inverted_response_matrix,model_structure,real_structure,self.scaleFactor.value())
            corrected_twiss = corrected_optics.twiss
            corrected_twiss_short = corrected_optics.twiss_short
            previous_structure = corrected_twiss
            previous_structure_short = corrected_twiss_short

            # item1 = pg.PlotCurveItem(self.ring.twiss.s,self.ring.twiss.x)
            # item2 = pg.PlotCurveItem(self.ring.twiss.s,self.ring.bad_twiss.x)
            # item3 = pg.PlotCurveItem(self.ring.twiss.s,corrected_twiss.x)
            # item4 = pg.PlotCurveItem(self.ring.twiss.s,self.ring.twiss.dy)

            item1 = pg.PlotCurveItem(self.ring.twiss_short.s,self.ring.twiss_short.x)
            item2 = pg.PlotCurveItem(self.ring.twiss_short.s,self.ring.bad_twiss_short.x)
            item3 = pg.PlotCurveItem(self.ring.twiss_short.s,corrected_twiss_short.x)
            item4 = pg.PlotCurveItem(self.ring.twiss_short.s,self.ring.twiss_short.dy)

            self.checkBox.stateChanged.connect(lambda: self.draw_optics(item1,self.checkBox))
            self.checkBox_2.stateChanged.connect(lambda: self.draw_optics(item2,self.checkBox_2))
            self.checkBox_3.stateChanged.connect(lambda: self.draw_optics(item3,self.checkBox_3))
            self.checkBox_4.stateChanged.connect(lambda: self.draw_optics(item4,self.checkBox_4))
        else:
            print("Pick singular values!")

        itteration += 1





    def open_structure_file(self):
        global fileName
        fileName = QtWidgets.QFileDialog.getOpenFileName(self,'Open file','C:\\Users\\r_mam\\IdeaProjects\\Correction\\MADX')[0]
        #print(fileName)
        # f = open(fileName, 'r')
        # with f:
        #     file = f.read()

        if self.isStructureLoaded == False:
            self.ring = Madx.Structure(fileName)
            self.draw_data()
            turn = True
            #self.add_plots()
            self.isStructureLoaded = True
        else:
            print("Structure is already loaded!")
















def main():
    app = QtWidgets.QApplication(sys.argv)  # Новый экземпляр QApplication
    window = ExampleApp()  # Создаём объект класса ExampleApp
    window.setWindowTitle("Optics Correction")
    window.setGeometry(100, 100, 800, 800)
    window.show()  # Показываем окно

    sys.exit(app.exec_())  # и запускаем приложение


if __name__ == '__main__':  # Если мы запускаем файл напрямую, а не импортируем
    main()  # то запускаем функцию main()



