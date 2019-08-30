import numpy as np
from PyQt5.QtWidgets import QMainWindow, QWidget, QPushButton, QVBoxLayout, QApplication, QSlider
from PyQt5 import uic

class StartWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("./ui/mainwindow.ui", self)
        self.save_pdf.clicked.connect(self.btnClicked)

    def btnClicked(self):
        print("Text")


if __name__ == '__main__':
    app = QApplication([])
    window = StartWindow()
    window.show()
    app.exit(app.exec_())
