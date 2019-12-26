import sys, csv
from PyQt5.QtWidgets import QMainWindow, QApplication, QDialog, QAction, QTableWidget,QTableWidgetItem,QVBoxLayout, QPushButton, QFileDialog
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
import os

class TableView(QDialog):
    def __init__(self, rows, columns, parent=None):
        QDialog.__init__(self, parent)
        self.table = QTableWidget(rows, columns, self)
        self.buttonSave = QPushButton('Save', self)
        self.buttonSave.clicked.connect(self.handleSave)
        layout = QVBoxLayout(self)
        layout.addWidget(self.table)
        layout.addWidget(self.buttonSave)

    def set_headers(self, headers):
        for i in range(len(headers)):
            self.set_item(headers[i], 0, i+1)

    def set_item(self, item, row, column):
        item = QTableWidgetItem(item)
        self.table.setItem(row, column, item)

    def handleSave(self):
        path, ext = QFileDialog.getSaveFileName(
                self, 'Save File', '', 'CSV(*.csv)')
        if path:
            root, ext = os.path.splitext(str(path))
            with open(root+".csv", 'w') as stream:
                writer = csv.writer(stream, delimiter=";")
                for row in range(self.table.rowCount()):
                    rowdata = []
                    for column in range(self.table.columnCount()):
                        item = self.table.item(row, column)
                        if item is not None:
                            rowdata.append(item.text())
                        else:
                            rowdata.append('')
                    writer.writerow(rowdata)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TableView(10, 5)
    window.resize(640, 480)
    window.show()
    sys.exit(app.exec_())
