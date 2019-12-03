from PyQt5.QtCore import QObject, pyqtSignal

class Signal(QObject):
    compute_signal = pyqtSignal()
