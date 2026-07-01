
from PySide6.QtWidgets import QApplication
from ui.main_window import MainWindow
import sys
app=QApplication(sys.argv)
w=MainWindow(); w.show()



sys.exit(app.exec())
