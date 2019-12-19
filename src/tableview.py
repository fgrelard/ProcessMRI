import sys, csv
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QAction, QTableWidget,QTableWidgetItem,QVBoxLayout, QPushButton, QFileDialog
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot

class TableView(QWidget):
    def __init__(self, rows, columns):
        QWidget.__init__(self)
        self.table = QTableWidget(rows, columns, self)
        for column in range(columns - 1):
            for row in range(rows - 1):
                item = QTableWidgetItem('Text%d' % row)
                self.table.setItem(row, column, item)
        self.buttonSave = QPushButton('Save', self)
        self.buttonSave.clicked.connect(self.handleSave)
        layout = QVBoxLayout(self)
        layout.addWidget(self.table)
        layout.addWidget(self.buttonSave)

    def handleSave(self):
        path, ext = QFileDialog.getSaveFileName(
                self, 'Save File', '', 'CSV(*.csv)')
        if path:
            with open(str(path), 'w') as stream:
                writer = csv.writer(stream)
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
