from PyQt5.QtWidgets import QApplication,QWidget,QScrollArea,QPushButton,QGraphicsRectItem,QGraphicsLineItem,QGraphicsTextItem,QGraphicsItemGroup,QHBoxLayout,QGraphicsItem,QGroupBox, QDialog,QComboBox,QVBoxLayout,QFormLayout,QInputDialog,QMessageBox,QMainWindow,QFileDialog,QLabel,QProgressBar,QGraphicsScene,QGraphicsView
from PyQt5.QtGui import QPen,QFont,QPolygonF
from PyQt5.QtCore import pyqtSlot,QThread,pyqtSignal,Qt,QRectF,QPointF,QLineF,QSizeF,QEvent

import sys , os
from functools import partial

from mpl_toolkits.axes_grid1 import AxesGrid, ImageGrid
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import pandas as pd
import numpy as np
import networkx as nx

import tensorflow as tf

from keras.layers import Input,Activation,Add,Dense,Conv2D,Dropout,Flatten,MaxPool2D, BatchNormalization,Conv2DTranspose
from keras.models import Model,load_model
from keras.utils import plot_model
from keras.preprocessing import image
from keras.callbacks import Callback,TensorBoard
from keras import optimizers
from keras import losses
from keras.utils import multi_gpu_model

class KerasModel(QThread):
    log_signal= pyqtSignal(dict)
    class CustomHistory(Callback):
        def __init__(self,log_signal):
            self.log_signal=log_signal
            self.validation_data = None
            self.model = None

        def on_train_begin(self, logs=None):
            self.epoch = []
            self.history = {}

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            logs['epoch']=epoch
            self.log_signal.emit(logs)
            print(f"-----")
            self.epoch.append(epoch)
            for k, v in logs.items():
                self.history.setdefault(k, []).append(v)

    def __init__(self,train_gen,model,code,parent=None):
        super(KerasModel, self).__init__(parent)
        self.model=model
        self.code=code
        self.train_gen=train_gen
        self.model_history=None
        self.hist=self.CustomHistory(self.log_signal)
        #self.customTensBordCall=TensorBoard(log_dir='logs_tensorboard',write_graph=False,write_grads=True, write_images=True)
        print(self.model.summary())
        print(self.code)
        print(self.train_gen)
        self.graph = tf.get_default_graph()
    def run(self):
        self.running = True
        with self.graph.as_default():
            print(self.code)
            exec("self.model_history=self."+self.code)
            
class customButtonGroup(QGraphicsItemGroup):
    def __init__(self,left,top,width,height,text,my_id,scne,code):
        super().__init__()
        self.my_id=my_id
        self.scne=scne
        self.code=code
        self.code2=code
        self.scne.addItem(self)
        self.scne.idx+=1
        self.rect=QGraphicsRectItem(left,top,width,height)
        self.addToGroup(self.rect)
        self.text_item  = QGraphicsTextItem(text)
        self.text_item.setFont(QFont(None,18,QFont.Bold))
        self.addToGroup(self.text_item)
        self.arrow_itm={}
        self.setFlag(QGraphicsItem.ItemIsFocusable)
        self.setFlag(QGraphicsItem.ItemIsMovable)
        self.setFlag(QGraphicsItem.ItemIsSelectable)        
        self.setFlag(QGraphicsItem.ItemSendsScenePositionChanges)
        self.parents_connected=[]
       
    def mouseDoubleClickEvent(self,event):
        QMessageBox.about(None, "Edit","Double clicked")

    def keyPressEvent(self,event):
        if event.key()==Qt.Key_Delete:
            QMessageBox.about(None, "Alert","Delete")
            print(self.my_id,self.arrow_itm,self.scne.selectedItems())
            if len(self.arrow_itm.keys())>0: 
                for i,j in self.arrow_itm.items():
                    print('s',self.my_id,self.arrow_itm,self.scne.selectedItems())                    
                    if j.myStartItem.my_id!=self.my_id:
                        print("this")
                        print(j.myStartItem.my_id)
                        j.myStartItem.del_arrow(i)                
                    else:
                        print("that")                        
                        print(j.myEndItem.my_id)                        
                        j.myEndItem.del_arrow(i)
                    print('e',self.my_id,self.arrow_itm,self.scne.selectedItems())                    
                    
            self.scne.removeItem(self)
            self.scne.layers.pop(self.my_id,None)
    def mousePressEvent(self,event):
        if event.button() == Qt.RightButton:
            if len(self.scne.selectedItems())==1 and self!=self.scne.selectedItems()[0]:
                self.scne.add_edge_custom(self,self.scne.selectedItems()[0])
                self.parents_connected.append(self.scne.selectedItems()[0].my_id)
                #print(self.my_id,self.parents_connected)
                if len(self.parents_connected)==1:
                    self.code2=self.code+f"(x{self.parents_connected[0]})"
                else:
                    self.code2=self.code+"([x"+',x'.join(map(str,self.parents_connected))+"])"
                #print(self.code2)
    def add_arrow(self,arrow):
        self.arrow_itm[arrow.my_id]=arrow
    def del_arrow(self,arrow_id):
        self.scne.removeItem(self.arrow_itm.pop(arrow_id,None))
    def itemChange(self,change,value):
        if change == QGraphicsItem.ItemScenePositionHasChanged:
            for i,j in self.arrow_itm.items():
                j.update_Position()
        return value
class customScne(QGraphicsScene):
    def __init__(self):
        super().__init__()
        self.input_idx=1
        self.idx=1
        self.arrow_idx=1
        self.layers={}
        self.gen_code=[]
        self.input_chan=[]
        self.output_chan=[]
        self.custgraph=nx.DiGraph()        
    def add_edge_custom(self,firste,seconde):
        edge1=Arrow(firste,seconde,None,self,self.arrow_idx)        
        firste.add_arrow(edge1)
        seconde.add_arrow(edge1)
        self.arrow_idx+=1
        self.custgraph.add_edge(seconde.my_id,firste.my_id)
        
class Arrow(QGraphicsLineItem):
    def __init__(self, startItem, endItem, parent=None, scene=None,my_id=-1):
        super(Arrow, self).__init__()  
        self.my_id=my_id
        self.scne=scene
        self.scne.addItem(self)
        self.scne.arrow_idx+=1
        self.arrowHead = QPolygonF()
        self.myStartItem = startItem
        self.myEndItem = endItem
        self.setPen(QPen(Qt.black,3))
        self.update_Position()
    def update_Position(self):
        if self.mapFromItem(self.myStartItem,70,0).y()>self.mapFromItem(self.myEndItem,70,0).y():
            line = QLineF(self.mapFromItem(self.myStartItem, 70, 0), self.mapFromItem(self.myEndItem, 70, 35))
        else:
            line = QLineF(self.mapFromItem(self.myEndItem, 70, 0), self.mapFromItem(self.myStartItem, 70, 35))
        self.setLine(line)

class LinearApp(QDialog):

    def __init__(self):
        super().__init__()
        print('LA Started')

        self.title = 'NN GUI Builder'
        self.left = 100
        self.top = 100
        self.width = 1800
        self.height = 1000
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.code_gen=KerasCodeGen()      

        self.modelInputDim=0
        self.model=None
        self.train_gen=None
        
        self.no_steps=0
        self.epochs=0
        self.loss=None
        self.optim=None
        self.parallel_option=False
        self.no_gpus=1
        self.train_code=None
        self.train_gen_code=None

        self.losscombo = QComboBox(self)
        self.losscombo.setStyleSheet('font:18px')
        self.losscombo.addItem("-----Select Loss-----")
        self.losscombo.addItem("categorical_crossentropy")
        self.losscombo.addItem("binary_crossentropy")
        self.losscombo.addItem("mean_squared_error")
        self.losscombo.addItem("mean_absolute_error")
        self.losscombo.addItem("sparse_categorical_crossentropy")
        self.losscombo.addItem("mean_squared_logarithmic_error")
        self.losscombo.addItem("poisson")
        self.losscombo.addItem("cosine_proximity")
        self.losscombo.addItem("kullback_leibler_divergence")
        self.losscombo.activated[str].connect(self.setLoss)

        self.optimcombo = QComboBox(self)
        self.optimcombo.setStyleSheet('font:18px')
        self.optimcombo.addItem("-----Select Optimizer------")
        self.optimcombo.addItem("SGD")
        self.optimcombo.addItem("RMSprop")
        self.optimcombo.addItem("Adam")
        self.optimcombo.addItem("Adagrad")
        self.optimcombo.addItem("Adadelta")
        self.optimcombo.activated[str].connect(self.setOptim)

        self.button21=QPushButton('Compile',self)
        self.button21.setStyleSheet('font: 20px')
        self.button21.clicked.connect(self.modelCompile)

        self.button22=QPushButton('ShowSummary',self)
        self.button22.setStyleSheet('font: 20px')
        self.button22.clicked.connect(self.modelSummary)

        self.button23=QPushButton('ModelGraph',self)
        self.button23.setStyleSheet('font: 20px')
        self.button23.clicked.connect(self.modelGraph)

        self.button24=QPushButton('ExportCode',self)
        self.button24.setStyleSheet('font: 20px')
        self.button24.clicked.connect(self.modelExportCode)
        
        self.button25=QPushButton('TrainProperties',self)
        self.button25.setStyleSheet('font: 20px')
        self.button25.clicked.connect(self.modelSetTrain)

        self.button27=QPushButton('Visualize results',self)
        self.button27.setStyleSheet('font: 20px')
        self.button27.clicked.connect(self.resultVisualize)

        self.button26=QPushButton('Train',self)
        self.button26.setStyleSheet('font: 20px')
        self.button26.clicked.connect(self.modelTrain)

        self.button261=QPushButton('Predict',self)
        self.button261.setStyleSheet('font: 20px')
        self.button261.clicked.connect(self.modelPredict)
        
        self.button28=QPushButton('Parallelize',self)
        self.button28.setStyleSheet('font: 20px')
        self.button28.setCheckable(True)
        self.button28.clicked[bool].connect(self.modelSetParallel)

        self.button29=QPushButton('Save Model',self)
        self.button29.setStyleSheet('font: 20px')
        self.button29.clicked.connect(self.modelExportKeras)


        self.button30=QPushButton('Load Model',self)
        self.button30.setStyleSheet('font: 20px')
        self.button30.clicked.connect(self.modelImportKeras)

        self.button31=QPushButton('Dense',self)
        self.button31.setStyleSheet('font: 20px')
        self.button31.clicked.connect(self.addFC)

        self.button32=QPushButton('Dropout',self)
        self.button32.setStyleSheet('font: 20px')
        self.button32.clicked.connect(self.addDrop1d)

        self.button35=QPushButton('Flatten',self)
        self.button35.setStyleSheet('font: 20px')
        self.button35.clicked.connect(self.addFlatten)

        self.button34=QPushButton('Conv2D',self)
        self.button34.setStyleSheet('font: 20px')
        self.button34.clicked.connect(self.addConv2d)

        self.button33=QPushButton('MaxPool2D',self)
        self.button33.setStyleSheet('font: 20px')
        self.button33.clicked.connect(self.addMaxpool2d)

        self.button36=QPushButton('Conv2DTranspose',self)
        self.button36.setStyleSheet('font: 20px')
        self.button36.clicked.connect(self.addConv2dTranspose)

        self.button37=QPushButton('Add',self)
        self.button37.setStyleSheet('font: 20px')
        self.button37.clicked.connect(self.addAdd)
        
        self.button63=QPushButton('tanh',self)
        self.button63.setStyleSheet('font: 20px')
        self.button63.clicked.connect(self.addTanh)

        self.button64=QPushButton('relu',self)
        self.button64.setStyleSheet('font: 20px')
        self.button64.clicked.connect(self.addRelu)

        self.button65=QPushButton('sigmoid',self)
        self.button65.setStyleSheet('font: 20px')
        self.button65.clicked.connect(self.addSigmoid)

        self.button66=QPushButton('softmax',self)
        self.button66.setStyleSheet('font: 20px')
        self.button66.clicked.connect(self.addSoftmax)

        self.button67=QPushButton('BatchNorm',self)
        self.button67.setStyleSheet('font: 20px')
        self.button67.clicked.connect(self.addBatchNorm)

        self.button1=QPushButton('Input',self)
        self.button1.setStyleSheet('font: 20px')
        self.button1.clicked.connect(self.getModelInputDim)

        self.train_pbar = QProgressBar(self)

        self.button2=QPushButton('Input Folder',self)
        self.button2.setStyleSheet('font: 20px')
        self.button2.clicked.connect(self.getInputFolder)

        self.button3=QPushButton('Output',self)
        self.button3.setStyleSheet('font: 20px')
        self.button3.clicked.connect(self.getModelOutput)
        
        self.label2 = QLabel('Epoch:    Train Loss:     Train Accuracy:     ',self)
        self.label2.setStyleSheet('font: 22px')
        self.label2.resize(200,100)
        self.mainlayout = QVBoxLayout()

        self.group_box5=QGroupBox("Training")
        self.layout6 = QVBoxLayout()
        self.layout6.addWidget(self.train_pbar)
        self.layout6.addWidget(self.label2)
        self.group_box5.setLayout(self.layout6)

        self.group_box=QGroupBox("After Train")
        self.layout2 = QHBoxLayout()
        self.layout2.addWidget(self.button30)        
        self.layout2.addWidget(self.button22)
        self.layout2.addWidget(self.button23)
        self.layout2.addWidget(self.button24)
        self.layout2.addWidget(self.button27)
        self.layout2.addWidget(self.button29)        
        self.group_box.setLayout(self.layout2)

        self.group_box7=QGroupBox("Model Parametes")
        self.layout8 = QHBoxLayout()
        self.layout8.addWidget(self.losscombo)
        self.layout8.addWidget(self.optimcombo)
        self.layout8.addWidget(self.button21)
        self.layout8.addWidget(self.button25)
        self.layout8.addWidget(self.button28)
        self.layout8.addWidget(self.button26)
        self.layout8.addWidget(self.button261)        
        self.group_box7.setLayout(self.layout8)

        self.group_box4=QGroupBox("Input")
        self.layout5 = QHBoxLayout()
        self.layout5.addWidget(self.button2)
        self.layout5.addWidget(self.button1)
        self.layout5.addWidget(self.button3)        
        self.group_box4.setLayout(self.layout5)

        self.group_box2=QGroupBox("Layers")
        self.layout3 = QVBoxLayout()
        self.layout3.addWidget(self.button31)
        self.layout3.addWidget(self.button32)
        self.layout3.addWidget(self.button33)
        self.layout3.addWidget(self.button34)
        self.layout3.addWidget(self.button35)
        self.layout3.addWidget(self.button36)
        self.layout3.addWidget(self.button37)
        self.group_box2.setLayout(self.layout3)

        self.group_box6=QGroupBox("Activation")
        self.layout7 = QVBoxLayout()
        self.layout7.addWidget(self.button63)
        self.layout7.addWidget(self.button64)
        self.layout7.addWidget(self.button65)
        self.layout7.addWidget(self.button66)
        self.layout7.addWidget(self.button67)        
        self.group_box6.setLayout(self.layout7)

        self.scne=customScne()
        self.view=QGraphicsView(self.scne,self)
        
        self.group_box3=QGroupBox('Design Area')
        self.layout4 = QHBoxLayout()
        self.layout4.addWidget(self.view)
        self.layout4.addWidget(self.group_box2)
        self.layout4.addWidget(self.group_box6)
        self.group_box3.setLayout(self.layout4)

        self.mainlayout.addWidget(self.group_box4)
        self.mainlayout.addWidget(self.group_box7)
        self.mainlayout.addWidget(self.group_box)
        self.mainlayout.addWidget(self.group_box3)
        self.mainlayout.addWidget(self.group_box5)
        self.setLayout(self.mainlayout)
        
        self.initUI()

    def initUI(self):
        self.show()
        
    def askValue(self,ques='Input',isInteger=True):
        inputDlg=QInputDialog()
        if(isInteger):
            i, okPressed = inputDlg.getInt(self,'EnterValue',
                                           f"<html style='font-size:16pt;'>{ques}</html>", 0, 0, 1000, 1)
        else:
            i, okPressed = inputDlg.getDouble(self,'EnterValue',
                                              f"<html style='font-size:16pt;'>{ques}</html>", 0.001, 0, 1000, 5)
        return i
    
    def askChoice(self,items,ques='Input'):
        inputDlg=QInputDialog()
        i, okPressed = inputDlg.getItem(self,'EnterValue',f"<html style='font-size:16pt;'>{ques}</html>",items, 0, False)
        return i

    @pyqtSlot(str)
    def setLoss(self,loss):
        print(loss)
        self.loss=loss

    @pyqtSlot(str)
    def setOptim(self,optim):
        print(optim)
        self.optim=optim

    @pyqtSlot()
    def getInputFolder(self):
        if self.modelInputDim!=0:
            for i in reversed(range(self.scrollLayout.count())): 
                self.scrollLayout.itemAt(i).widget().setParent(None)
        self.update()
        self.initUI()
        type_folder=self.askChoice(["Folder Seperation","Csv File"],"Images labels :-")
        mode=self.askChoice(["binary","categorical"],"Class type")
        
        if type_folder=="Folder Seperation":
            folder_path = QFileDialog.getExistingDirectory(self, "Select Directory Having Different Image classes seperated by folder")
            height=self.askValue('Target Image Size height')
            width=self.askValue('Target Image Size width')
            batch_size=self.askValue('Batch Size')
            QMessageBox.about(self, "Alert",f"<html style='font-size:14pt;'>Wait</html>")                 
            exec("self.train_gen="+self.code_gen.codeImageFlowFromDir(folder_path,(height,width),batch_size,mode))
            self.train_gen_code=self.code_gen.codeImageFlowFromDir(folder_path,(height,width),batch_size,mode)

        else:
            folder_path = QFileDialog.getExistingDirectory(self, "Select Directory Having Images")
            csv_path = QFileDialog.getOpenFileName(self, "Select File having labels")
            dataframe=pd.read_csv(csv_path[0])
            print(dataframe.info())
            name_col=self.askChoice(dataframe.columns,"Select Image File Name column")
            lbl_col=self.askChoice(dataframe.columns,"Select Label Column")
            dataframe[name_col]=dataframe[name_col].astype(np.str)
            print(dataframe.info())      
            height=self.askValue('Target Image Size height')
            width=self.askValue('Target Image Size width')
            batch_size=self.askValue('Batch Size')            
            QMessageBox.about(self, "Alert",f"<html style='font-size:14pt;'>Wait</html>")            
            exec("self.train_gen="+self.code_gen.codeImageFlowFromDataFrame(dataframe, folder_path,name_col,lbl_col,(height,width),batch_size,mode))
            self.train_gen_code=(self.code_gen.codeImageFlowFromDataFrame(dataframe, folder_path,name_col,lbl_col,(height,width),batch_size,mode))

        QMessageBox.about(self, "Alert",f"<html style='font-size:14pt;'>Train_generator created</html>")     

    @pyqtSlot() 
    def getModelInputDim(self):
        dimns=self.askValue('Dimension of Input Data')
        inputDim="("
        if dimns>0:
            for i in range(dimns):
                inputDim+=str(self.askValue(f'Enter Input Size in {i+1}-D'))+','
        self.modelInputDim=inputDim
        code=(self.code_gen.codeModelInput(inputDim))
        rect=customButtonGroup(0,0,105,35,"Input",self.scne.idx,self.scne,code)
        self.scne.input_chan.append(self.scne.idx-1)        
        self.scne.layers[self.scne.idx-1]=rect
        
    @pyqtSlot() 
    def getModelOutput(self):
        code='='
        rect=customButtonGroup(0,0,105,35,"Output",self.scne.idx,self.scne,code)
        print(self.scne.layers)
        self.scne.layers[self.scne.idx-1]=rect
        self.scne.output_chan.append(self.scne.idx-1)
    @pyqtSlot()
    def addRelu(self):
        code=self.code_gen.codeActivation('relu')
        rect=customButtonGroup(0,0,105,35,"Relu",self.scne.idx,self.scne,code)        
        self.scne.layers[self.scne.idx-1]=rect
    @pyqtSlot()
    def addTanh(self):
        code=self.code_gen.codeActivation('tanh')
        rect=customButtonGroup(0,0,105,35,"Tanh",self.scne.idx,self.scne,code)        
        self.scne.layers[self.scne.idx-1]=rect
    @pyqtSlot()                       
    def addSoftmax(self):
        code=self.code_gen.codeActivation('softmax')
        rect=customButtonGroup(0,0,140,35,"Softmax",self.scne.idx,self.scne,code)       
        self.scne.layers[self.scne.idx-1]=rect            
    @pyqtSlot()                    
    def addSigmoid(self):
        code=self.code_gen.codeActivation('sigmoid')
        rect=customButtonGroup(0,0,140,35,"Sigmoid",self.scne.idx,self.scne,code)            
        self.scne.layers[self.scne.idx-1]=rect
    @pyqtSlot()
    def addFC(self):
        o_dim=self.askValue('Enter Output Dim')
        code=self.code_gen.layers.codeFC(o_dim)
        rect=customButtonGroup(0,0,130,35,f"Dense ({o_dim})",self.scne.idx,self.scne,code)                    
        self.scne.layers[self.scne.idx-1]=rect
    @pyqtSlot()
    def addDrop1d(self):
        code=self.code_gen.layers.codeDropout()
        rect=customButtonGroup(0,0,150,35,"DropOut1D",self.scne.idx,self.scne,code)                    
        self.scne.layers[self.scne.idx-1]=rect      
    @pyqtSlot()
    def addBatchNorm(self):
        code=self.code_gen.layers.codeBatchNorm()
        rect=customButtonGroup(0,0,150,35,"BatchNorm",self.scne.idx,self.scne,code)                    
        self.scne.layers[self.scne.idx-1]=rect
    @pyqtSlot()
    def addMaxpool2d(self):
        code=self.code_gen.layers.codeMaxpool2d()
        rect=customButtonGroup(0,0,150,35,"MaxPool2D",self.scne.idx,self.scne,code)                    
        self.scne.layers[self.scne.idx-1]=rect
    @pyqtSlot()     
    def addFlatten(self):
        code=self.code_gen.layers.codeFlatten()
        rect=customButtonGroup(0,0,130,35,"Flatten",self.scne.idx,self.scne,code)                            
        self.scne.layers[self.scne.idx-1]=rect
    @pyqtSlot()
    def addConv2d(self):
        n_f=self.askValue('No of Filters')
        k_s=self.askValue('Kernel Size')
        s=self.askValue('Stride')
        code=self.code_gen.layers.codeConv2d(n_f,k_s,s,'relu')
        rect=customButtonGroup(0,0,180,35,f"Conv2D ({n_f},{k_s},{s})",self.scne.idx,self.scne,code)                     
        self.scne.layers[self.scne.idx-1]=rect
    @pyqtSlot()
    def addConv2dTranspose(self):
        n_f=self.askValue('No of Filters')
        k_s=self.askValue('Kernel Size')
        s=self.askValue('Stride')
        code=self.code_gen.layers.codeConv2dTranspose(n_f,k_s,s,'relu')
        rect=customButtonGroup(0,0,180,35,f"Conv2DT({n_f},{k_s},{s})",self.scne.idx,self.scne,code)                    
        self.scne.layers[self.scne.idx-1]=rect
    @pyqtSlot()
    def addAdd(self):
        code=self.code_gen.layers.codeAdd()
        rect=customButtonGroup(0,0,105,35,"Add",self.scne.idx,self.scne,code)                               
        self.scne.layers[self.scne.idx-1]=rect
    @pyqtSlot()
    def modelCompile(self):
        lr=self.askValue('Input Learning Rate',False)
        c_n=self.askValue('Input Clip Normalization Value',False)
        for i in nx.topological_sort(self.scne.custgraph):
            print(f"x{i}"+self.scne.layers[i].code2)
            exec(f"x{i}"+self.scne.layers[i].code2)
        
        self.model_create_code=self.code_gen.codeCreateModel(self.scne.input_chan,self.scne.output_chan)
        self.model_compile_code=self.code_gen.codeModelCompile(self.loss,self.optim,lr,c_n)       
        print(self.model_create_code)
        print(self.model_compile_code)
        exec("self."+self.model_create_code)
        exec("self."+self.model_compile_code)
        QMessageBox.about(self, "Alert",f"<html style='font-size:17pt;'>Compiled</html")
        print("---compiled---")

    @pyqtSlot()
    def modelExportCode(self):
#         self.dialog=CodeWindow(self)
        filename="model_code.txt"
        with open(filename, 'a') as out:
            out.write('\n#-----------\n')
            print(self.train_gen_code)
            out.write(self.train_gen_code+'\n')
            for i in nx.topological_sort(self.scne.custgraph):
                print(f"x{i}"+self.scne.layers[i].code2)
                out.write(f"x{i}"+self.scne.layers[i].code2+'\n')
            print(self.model_create_code)
            print(self.model_compile_code)            
            print(self.train_code)
            out.write(self.model_create_code+'\n')
            out.write(self.model_compile_code+'\n')
            out.write(self.train_code+'\n')            
        QMessageBox.about(self, "Alert",f"<html style='font-size:17pt;'>Code Exported to  {filename}</html>")
        print("---CodeExported---")
    
    @pyqtSlot()
    def modelExportKeras(self):
        exec("self.model.save('model_saved1.h5')")
        QMessageBox.about(self, "Alert",f"<html style='font-size:17pt;'>Model Exported to 'model_saved1.h5' </html>")
        print("---Model Exported---")

    @pyqtSlot() 
    def modelImportKeras(self):
        model_path = QFileDialog.getOpenFileName(self, "Select Keras hd5 file")                
        exec(f"self.model=load_model('{model_path[0]}')")
        QMessageBox.about(self, "Alert",f"<html style='font-size:17pt;'>Model Imported.<br>Specify train properties and train</html>")
        print("---Model Imported---")

    @pyqtSlot()
    def modelSummary(self):
        QMessageBox.about(self, "Alert",f"<html style='font-size:17pt;'>See terminal for summary</html>")         
        exec('self.model.summary()')
        print("---SummaryProduced---")

    @pyqtSlot()
    def modelSetTrain(self):
        self.no_steps=self.askValue('Steps Per Epoch')
        self.epochs=self.askValue('No of Epochs')
        self.train_pbar.setMaximum(self.epochs)
        self.train_code=self.code_gen.codeModelTrain(self.no_steps,self.epochs)
        QMessageBox.about(self, "Alert",f"<html style='font-size:17pt;'>Train Setup Done</html")
        print("---TrainSetup---")

    @pyqtSlot()
    def modelGraph(self):
        QMessageBox.about(self, "Alert",f"<html style='font-size:17pt;'>See generated model.png</html>")
        exec("plot_model(self.model, to_file='model.png',show_shapes=True)")
        print("---GraphProduced---")

    @pyqtSlot(bool)
    def modelSetParallel(self,state):
        if state:
            self.no_gpus=self.askValue('No of Gpu to use for training')
            self.parallel_option=True
        else:

            self.parallel_option=False
            self.no_gpus=1

    @pyqtSlot()
    def modelTrain(self):
        print(self.parallel_option)
        self.label2.setText(f"Wait till Train Ends")
        self.parallel_train_code=f"""try:
            parallel_model = multi_gpu_model(self.model, gpus={self.no_gpus}, cpu_relocation=True)
            print("Training using {self.no_gpus} GPUs...")
except ValueError:
            parallel_model = model
            print("Training using single GPU or CPU...")
parallel_"""+self.train_code
        print(self.train_code)
        print(self.parallel_train_code+"\n\n\n")
        if not self.parallel_option:
            self.train_thread = KerasModel(self.train_gen,self.model,self.train_code)
        else:
            self.train_thread = KerasModel(self.train_gen,self.model,self.parallel_train_code)
        self.train_thread.start()
        self.train_thread.log_signal.connect(self.updateTrainStats)
        self.train_thread.finished.connect(self.postTrainWork)

    @pyqtSlot(dict)                
    def updateTrainStats(self,log):
        self.label2.setText(f"Epoch:{log['epoch']+1}       Loss:{log['loss']}       Accuracy:{log['acc']}")
        self.train_pbar.setValue(log['epoch']+1)
    @pyqtSlot()                
    def postTrainWork(self):
        QMessageBox.about(self, "Alert",f"<html style='font-size:17pt;'>Training Done</html>")    
#         print(self.train_thread.model_history)
        self.plot=self.train_thread.model_history.history
        print(self.plot)
        print("---Training Done---")

    @pyqtSlot()
    def resultVisualize(self):
        exec("plt.subplot(2, 1, 1)")             
        exec("plt.plot(np.arange(len(self.plot['loss'])),self.plot['loss'],c='b',label='loss')")
        exec("plt.xlabel('Epochs')")
        exec("plt.ylabel('Train_loss')")

        # exec("plt.subplot(2, 1, 2)")
        # exec("plt.plot(np.arange(len(self.plot['val_loss'])),self.plot['val_loss'], c='r',label='validation loss')")
        # exec("plt.xlabel('epochs')")
        # exec("plt.ylabel('accuracy')")

        exec("plt.subplot(2, 1, 2)")    
        exec("plt.plot(np.arange(len(self.plot['acc'])), self.plot['acc'],c='r',label='accuracy')")
        exec("plt.xlabel('Epochs')")
        exec("plt.ylabel('Accuracy')")

        # exec("plt.subplot(2, 1, 3)")
        # exec("plt.plot(np.arange(len(self.plot['val_acc'])),self.plot['val_acc'], c='r',label='validation accuracy')")
        # exec("plt.xlabel('epochs')")
        # exec("plt.ylabel('accuracy')")
        exec("plt.show()")

    @pyqtSlot()
    def modelPredict(self):
        img_path = QFileDialog.getOpenFileName(self, "Select File having labels")        
        img2=image.load_img(img_path[0])
        self.vis_window=VisualizeModule(self.model,img2,)#self.train_gen.class_indices)

class VisualizeModule(QDialog):
    def __init__(self,model,img2,class_indices=None):
        super().__init__()
        self.title = 'Visualizer'
        self.left = 300
        self.top = 300
        self.width = 400
        self.height = 800
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        
        figure = Figure()
        canvas = FigureCanvas(figure)
        
        ax = figure.add_subplot(211)
        #ax.clear()
        ax.imshow(img2,interpolation="nearest")        

        img2=image.img_to_array(img2)        
        img2 = np.expand_dims(img2, axis=0)
        
        self.output_result=model.predict(img2)        
        print(self.output_result.shape)
                
        print(class_indices)
        
        ax = figure.add_subplot(212)
        ax.set_xticks(np.arange(len(self.output_result[0])))
        ax.bar(np.arange(len(self.output_result[0])),self.output_result[0],2,0.5)
        canvas.draw()
        
        print(self.output_result)
        print(self.output_result.shape)
        outputs_layers = [layer.output for layer in model.layers]        
        self.intermed_model = Model(inputs=model.input,outputs=[i for i in outputs_layers[1:]])        
        layer_names = [layer.name for layer in self.intermed_model.layers]
        output_arrs=self.intermed_model.predict(img2)
        
        self.o_images={}
        for name,out in zip(layer_names,output_arrs):
            if out.ndim==4:
                self.o_images[name]=[out[0][...,j] for j in range(out.shape[3])]
                print(out.shape[3])
        self.button_list=[]
        for i in self.o_images.keys():
            button2=QPushButton(i,self)
            button2.setStyleSheet('font: 22px')            
            button2.clicked.connect(partial(self.plot_figure,i))
            self.button_list.append(button2)
        
        self.scrollLayout = QFormLayout()
        self.scrollWidget = QWidget()
        self.scrollWidget.setLayout(self.scrollLayout)
        self.scrollArea = QScrollArea()
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setWidget(self.scrollWidget)
        
        self.layout = QVBoxLayout()
        self.layout.addWidget(canvas)
        self.layout.addWidget(self.scrollArea)
        self.initUI()
        
    def initUI(self):
        
        for i in self.button_list:
            self.scrollLayout.addWidget(i)
      
        self.setLayout(self.layout)
        self.show()
        
    @pyqtSlot()
    def plot_figure(self,name):
        fig = plt.figure(1,dpi=141)
        fig.tight_layout()
        def cust_fig(layer,num,name):
            num_layer=len(layer)
            col_num=2                        
            if num_layer%2==0:
                row_num=num_layer//2            
            else:
                row_num=num_layer//2+1
            grid1 = ImageGrid(fig,num,nrows_ncols = (row_num,col_num),
                              share_all=True,label_mode = "L",aspect =True,axes_pad=0.2)
            for i in grid1:
                i.axis('off')
            for i in range(num_layer):
                im = grid1[i].imshow(layer[i],interpolation="nearest",aspect='auto')
                if i==0:
                    grid1[i].set_title(name+f" {layer[0].shape}",loc='left')
            grid1.axes_all
        cust_fig(self.o_images[name],231,name)
        plt.show()
        
class Button(QPushButton):
    def __init__(self,title,ID,parent=None):
        super().__init__(parent)
#         self.resize(300,50)
        self.setText(title)
        self.ID=ID

class KerasCodeGen():
    class Layers():
        def codeFC(self,o_dim,actv='relu'):
            return f"=Dense({o_dim}, activation=None)"
        def codeAdd(self,p=0.5):
            return f"=Add()"
        def codeDropout(self,p=0.5):
            return f"=Dropout({p})"        
        def codeBatchNorm(self,p=0.5):
            return f"=BatchNormalization()"
        def codeMaxpool2d(self):
            return f"=MaxPool2D()"
        def codeConv2d(self,filter_num,kernel_size=3,stride=1,actv='None'):
            return f"=Conv2D({filter_num},kernel_size={kernel_size},strides={stride}, padding='same',activation=None)"
        def codeConv2dTranspose(self,num_prev_layers, filter_num,kernel_size=3,stride=1,actv='None'):
            return f"=Conv2DTranspose({filter_num},kernel_size={kernel_size},strides={stride},padding='same',activation=None)"
        def codeFlatten(self):
            return f"=Flatten()"

    def __init__(self):
        self.layers=self.Layers()

    def codeModelInput(self,i_dim):
        return f"=Input(shape="+i_dim+"))"

    def codeActivation(self,actv):
        return f"=Activation('"+actv+"')"
    
    def codeCreateModel(self,inputs,outputs):
        print(inputs,outputs)
        if len(inputs)>1: 
            input_code="[x"+",x".join(map(str,inputs))+"]"
        else:
            input_code="x"+str(inputs[0])        
        if len(outputs)>1: 
            output_code="[x"+",x".join(map(str,outputs))+"]"
        else:
            output_code="x"+str(outputs[0])
        return f"model = Model(inputs="+input_code+", outputs="+output_code+")"

    def codeImageFlowFromDir(self,direc,target_size,batch_size,mode='categorical'):
        return f"train_generator = image.ImageDataGenerator(rescale=1./255,validation_split=0.2).flow_from_directory('{direc}',target_size={target_size},batch_size={batch_size},class_mode='{mode}')"

    def codeImageFlowFromDataFrame(self,dataframe,direc,name_col, lbl_col,target_size,batch_size,mode='categorical'):
        return f"train_generator = image.ImageDataGenerator(rescale=1./255,validation_split=0.0).flow_from_dataframe(dataframe=dataframe, directory='{direc}', x_col='{name_col}', y_col='{lbl_col}', class_mode='{mode}', target_size={target_size}, batch_size={batch_size})"

    def codeModelTrain(self,stps,epochs):
        return f"model.fit_generator(self.train_gen, steps_per_epoch={stps},epochs ={epochs},callbacks=[self.hist])"

    def codeModelCompile(self,loss,optim,lr,clip):
        return f"model.compile(loss='{loss}', optimizer={self.codeOptimizers(optim,lr,clip)}, metrics=['accuracy'])"
    def codeOptimizers(self,optim,lr,clipvalue):
        return f"optimizers.{optim}(lr={lr}, clipvalue={clipvalue})"

if __name__ == '__main__':
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    app = QApplication(sys.argv)
    app.setAttribute(Qt.AA_EnableHighDpiScaling,True)
    ex = LinearApp()
    sys.exit(app.exec_())