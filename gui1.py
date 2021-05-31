from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox
from ai_stuff import *
import matplotlib.pyplot as plt


class Ui_MainWindow(object):
    #creates the ui and all the widgets
    def setupUi(self, MainWindow, Dialog):
        #main window
        self.dialog = Dialog
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        MainWindow.setUnifiedTitleAndToolBarOnMac(False)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        #button to run the train function
        self.train_btn = QtWidgets.QPushButton(self.centralwidget)
        self.train_btn.setGeometry(QtCore.QRect(620, 130, 141, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.train_btn.setFont(font)
        self.train_btn.setObjectName("train_btn")
        self.train_btn.clicked.connect(self.clicked_train)

        #button to test the network
        self.test_btn = QtWidgets.QPushButton(self.centralwidget)
        self.test_btn.setGeometry(QtCore.QRect(620, 200, 141, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.test_btn.setFont(font)
        self.test_btn.setObjectName("test_btn")
        self.test_btn.setEnabled(False)
        self.test_btn.clicked.connect(self.clicked_test)

        #button to start the predictions
        self.pre_btn = QtWidgets.QPushButton(self.centralwidget)
        self.pre_btn.setGeometry(QtCore.QRect(620, 340, 141, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pre_btn.setFont(font)
        self.pre_btn.setObjectName("pre_btn")
        self.pre_btn.setEnabled(False)
        self.pre_btn.clicked.connect(self.clicked_t)

        #button for the predictions of letter P
        self.p_btn = QtWidgets.QPushButton(self.centralwidget)
        self.p_btn.setGeometry(QtCore.QRect(620, 480, 141, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.p_btn.setFont(font)
        self.p_btn.setObjectName("p_btn")
        self.p_btn.setEnabled(False)
        self.p_btn.clicked.connect(self.clicked_p)

        #button for the predictions of letter S
        self.s_btn = QtWidgets.QPushButton(self.centralwidget)
        self.s_btn.setGeometry(QtCore.QRect(620, 410, 141, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.s_btn.setFont(font)
        self.s_btn.setObjectName("s_btn")
        self.s_btn.setEnabled(False)
        self.s_btn.clicked.connect(self.clicked_s)

        #code for the label
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(610, 80, 161, 16))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")

        #code for the blank space where function outputs are written
        self.output_canvas = QtWidgets.QTextBrowser(self.centralwidget)
        self.output_canvas.setGeometry(QtCore.QRect(40, 130, 561, 411))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.output_canvas.setFont(font)
        self.output_canvas.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.output_canvas.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.output_canvas.setObjectName("output_canvas")

        #code for the label
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(20, 35, 761, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")

        #status bar (not visible)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


    # gives text to the QtWidgets
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Letter Classification"))
        self.train_btn.setText(_translate("MainWindow", "Train the Model"))
        self.test_btn.setText(_translate("MainWindow", "Test the Model "))
        self.pre_btn.setText(_translate("MainWindow", "Predict from Test"))
        self.p_btn.setText(_translate("MainWindow", "Test for Letter P"))
        self.s_btn.setText(_translate("MainWindow", "Test for Letter S"))
        self.label.setText(_translate("MainWindow", "Control Buttons"))
        self.label_3.setText(_translate("MainWindow", "Letter Classification Using Feed Forwarding Back Propagating Neural Network"))





    #button events

    #prediction buttons


    def clicked_t(self):
        inp = 'test'
        while True:
            while True:
                num, done = QtWidgets.QInputDialog.getInt(self.dialog , 'Input Dialog', 'Write a number between 0 and 1000') #user inputs number
                if done : break

            if 0 <= num <= 1000:
                break
            else:
                #creates a message to notify the user of the error
                msg = QMessageBox()
                msg.setWindowTitle("Input Error")
                msg.setText("Write a number between 0 and 1000")
                x = msg.exec_()  # this will show our messagebox

        predicted_class , real_class = predict_mode(model, data_test, labels_test, data_p, data_s, inp, num)
        self.output_canvas.clear()
        text = 'Prediction is: ' + str(predicted_class) + '\nLetter is: ' + str(real_class)
        self.output_canvas.append(text)



    def clicked_p(self):
        inp = 'p'
        while True:
            while True:
                num, done = QtWidgets.QInputDialog.getInt(self.dialog , 'Input Dialog', 'Write a number between 0 and 20')
                if done : break

            if 0 <= num <= 20:
                break
            else:
                msg = QMessageBox()
                msg.setWindowTitle("Input Error")
                msg.setText("Write a number between 0 and 20")
                x = msg.exec_()

        predicted_class , real_class = predict_mode(model, data_test, labels_test, data_p, data_s, inp, num)
        self.output_canvas.clear()
        text = 'Prediction is: ' + str(predicted_class) + '\nLetter is: ' + str(real_class)
        self.output_canvas.append(text)


    def clicked_s(self):
        inp = 's'
        while True:
            while True:
                num, done = QtWidgets.QInputDialog.getInt(self.dialog , 'Input Dialog', 'Write a number between 0 and 20')
                if done : break

            if 0 <= num <= 20:
                break
            else:
                msg = QMessageBox()
                msg.setWindowTitle("Input Error")
                msg.setText("Write a number between 0 and 20")
                x = msg.exec_()

        predicted_class , real_class = predict_mode(model, data_test, labels_test, data_p, data_s, inp, num)
        self.output_canvas.clear()
        text = 'Prediction is: ' + str(predicted_class) + '\nLetter is: ' + str(real_class)
        self.output_canvas.append(text)



    
    def clicked_train(self):
        mode = False
        hist = neural_network_train(model, data_train, labels_train, mode)
        self.train_btn.setEnabled(False)
        self.test_btn.setEnabled(True)
        text = 'Training Finished \n \n \n'
        self.output_canvas.append(text)
        for i in range(5):
            stats = 'Epoch' + str(i+1) + ':\n            Loss: ' + str(hist.history['loss'][i]) + '\n           Accuracy: ' + str(hist.history['accuracy'][i]) + ' \n'
            self.output_canvas.append(stats)



    def clicked_test(self):
        mode = True
        test_loss, test_acc = neural_network_train(model, data_test, labels_test, mode)
        self.test_btn.setEnabled(False)
        self.s_btn.setEnabled(True)
        self.p_btn.setEnabled(True)
        self.pre_btn.setEnabled(True)
        self.output_canvas.clear()
        text = 'Testing has Finished \n \n \n \n'+ 'Test accuracy:  ' + str(test_acc) + '\n \n' + 'Loss: ' + str(test_loss)
        self.output_canvas.append(text)







if __name__ == "__main__":
    import sys
    #creates and loads the gui
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    Dialog = QtWidgets.QDialog()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow, Dialog)
    MainWindow.show()

    #calls all the functions to proccess the datasets and then the function to create the network. this run when the programma starts
    train, test, s_ds, p_ds = data_from_CSV()
    train_data, train_labels, test_data, test_labels, s_data, p_data = panda_dataframe_processing(train, test, s_ds, p_ds)
    data_train, data_test, data_s, data_p, labels_test, labels_train = numpy_array_processing(train_data, train_labels, test_data, test_labels, s_data, p_data)
    data_train, data_test, data_s, data_p, labels_test, labels_train = image_processing(data_train, data_test, data_s, data_p, labels_test, labels_train)
    model = create_model(data_train, labels_train, data_test, labels_test)




    sys.exit(app.exec_())
