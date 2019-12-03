from PyQt5.QtWidgets import QPushButton

class HoverButton(QPushButton):
    def __init__(self, parent=None):
        super(Button, self).__init__(parent)
        # other initializations...

    def enterEvent(self, QEvent):
        # here the code for mouse hover
        pass

    def leaveEvent(self, QEvent):
        # here the code for mouse leave
        pass
