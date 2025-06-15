import os
import sys
import webbrowser

import numpy as np
import qtawesome as qta

import src.Image as Imeta

from PyQt5 import QtWidgets
from PyQt5.Qt import QVBoxLayout
from PyQt5.QtWidgets import QApplication, QTableWidget, QTableWidgetItem
from PyQt5.QtCore import QObject, QThread, pyqtSignal

from src.signal import Signal
from src.tableview import TableView

from src.expfitcontroller import ExpFitController, WorkerExpFit
from src.nlmeanscontroller import NLMeansController, WorkerNLMeans
from src.cavitycontroller import CavityController, WorkerCavity
from src.tpccontroller import TPCController, WorkerTPC
from src.houghcontroller import HoughController, WorkerHough
from src.largestcomponentcontroller import WorkerLargestComponent
from src.measurementcontroller import MeasurementController, WorkerMeasurement
from src.manualsegmentationcontroller import ManualSegmentationController, WorkerManualSegmentation
from src.manualcomponentcontroller import ManualComponentController, WorkerManualComponent

import src.imageio as io
import src.exponentialfit as expfit

from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

class WidgetPlot(QtWidgets.QDialog):
    def __init__(self, *args, **kwargs):
        QtWidgets.QWidget.__init__(self, *args, **kwargs)

        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.canvas)
        self.layout.addWidget(self.toolbar)
        self.setLayout(self.layout)
        self.move(0,0)

    def clear(self):
        self.ax.cla()

    def set_xlabel(self, xlabel):
        self.ax.set_xlabel(xlabel, fontweight="bold")

    def set_ylabel(self, ylabel):
        self.ax.set_ylabel(ylabel, fontweight="bold")

    def set_text(self, text):
        self.ax.text(0.95,
                     0.95,
                     text,
                     verticalalignment='top',
                     horizontalalignment='right',
                     transform=self.ax.transAxes)

    def plot(self, *args, **kwargs):
        self.ax.plot(*args, **kwargs)

    def show(self):
        self.fig.canvas.draw_idle()
        self.canvas.draw()
        super().show()



class MainController:
    """
    Main controller connecting all controllers and events

    Attributes
    ----------
    app: QApplication
        application
    mainview: AppWindow
        QMainWindow
    config: configparser.ConfigParser
        configuration file
    img_data: np.ndarray
        current displayed image
    echotimes: np.ndarray
        sequence of echo times
    images: dict
        dictionary mapping filename to open images
    threads: list
        thread pool
    expfitcontroller: ExpFitController
        controller for expfit dialog
    nlmeanscontroller: NLMeansController
        controller for nlmeans dialog
    tpccontroller: TPCController
        controller for tpc dialog
    """
    def __init__(self, app, mainview, config):

        self.mainview = mainview.ui
        self.mainview.parent = mainview
        self.app = app

        self.sig_abort_workers = Signal()

        self.expfitcontroller = ExpFitController(mainview.centralWidget())
        self.expfitcontroller.trigger.signal.connect(self.exp_fit_estimation)

        self.nlmeanscontroller = NLMeansController(mainview.centralWidget())
        self.nlmeanscontroller.trigger.signal.connect(self.nl_means_denoising)

        self.tpccontroller = TPCController(mainview.centralWidget())
        self.tpccontroller.trigger.signal.connect(self.tpc_denoising)

        self.cavitycontroller = CavityController(mainview.centralWidget())
        self.cavitycontroller.trigger.signal.connect(self.segment_cavity)
        self.cavitycontroller.view.buttonBox.rejected.connect(self.cancel_preview)
        self.cavitycontroller.view.horizontalSlider.valueChanged.connect(lambda: self.segment_cavity(preview=True))
        self.cavitycontroller.view.horizontalSlider_2.valueChanged.connect(lambda: self.segment_cavity(preview=True))

        self.houghcontroller = HoughController(mainview.centralWidget())
        self.houghcontroller.trigger.signal.connect(self.hough_transform)
        self.houghcontroller.view.buttonBox.rejected.connect(self.cancel_preview)
        self.houghcontroller.view.pushButton_3.clicked.connect(lambda: self.hough_transform(preview=True))



        self.manualsegmentationcontroller = ManualSegmentationController(mainview.centralWidget())
        self.manualsegmentationcontroller.trigger.signal.connect(self.manual_segmentation)
        self.manualsegmentationcontroller.view.buttonBox.rejected.connect(self.cancel_manual_segmentation)
        self.manualsegmentationcontroller.view.horizontalSlider.valueChanged.connect(self.update_pen_size)


        self.manualcomponentcontroller = ManualComponentController(mainview.centralWidget())
        self.manualcomponentcontroller.trigger.signal.connect(lambda: self.mainview.imageview.setClickable(True))
        self.manualcomponentcontroller.view.buttonBox.accepted.connect(self.end_manual_component)
        self.manualcomponentcontroller.view.buttonBox.rejected.connect(self.cancel_manual_component)
        self.manualcomponentcontroller.view.horizontalSlider.valueChanged.connect(lambda: self.manual_component(evt=None, click=False))

        self.mainview.imageview.scene.sigMouseClicked.connect(self.manual_component)

        self.measurementcontroller = MeasurementController(mainview.centralWidget())
        self.measurementcontroller.trigger.signal.connect(self.measurements)

        self.mainview.actionExit.triggered.connect(self.exit_app)
        self.mainview.actionBruker_directory.triggered.connect(self.open_bruker)
        self.mainview.actionNifti.triggered.connect(self.open_nifti)
        self.mainview.actionSave.triggered.connect(self.save_nifti)
        self.mainview.actionExponential_fitting.triggered.connect(self.expfitcontroller.show)
        self.mainview.actionDenoising_NL_means.triggered.connect(self.nlmeanscontroller.show)
        self.mainview.actionDenoising_TPC.triggered.connect(self.tpccontroller.show)
        self.mainview.actionHoughTransform.triggered.connect(self.houghcontroller.show)
        self.mainview.actionSegmentGrain.triggered.connect(self.largest_component)
        self.mainview.actionSegmentCavity.triggered.connect(self.cavitycontroller.show)
        self.mainview.actionManualSegmentation.triggered.connect(self.manualsegmentationcontroller.show)
        self.mainview.actionManualComponent.triggered.connect(self.manualcomponentcontroller.show)
        self.mainview.actionMeasurements.triggered.connect(lambda : self.measurementcontroller.show(self.images.keys()))


        self.mainview.actionUser_manual_FR.triggered.connect(lambda event : webbrowser.open_new('file://' + os.path.realpath('docs/manual.pdf')))
        self.mainview.stopButton.clicked.connect(self.abort_computation)
        self.mainview.combobox.activated[str].connect(self.choose_image)
        self.mainview.trashButton.clicked.connect(lambda : self.remove_image(self.current_name(self.img_data), manual=True))

        self.mainview.editButton.clicked.connect(self.edit_name)

        self.mainview.imageview.scene.sigMouseClicked.connect(self.on_click_image)

        self.mainview.imageview.signal_progress_export.connect(self.update_progressbar)
        self.mainview.imageview.signal_start_export.connect(self.mainview.show_run)
        self.mainview.imageview.signal_end_export.connect(self.mainview.hide_run)
        self.mainview.imageview.signal_image_change.connect(self.change_image_combobox)

        #Exp fit plot
        self.widgetPlot = WidgetPlot(parent=self.mainview.parent.centralWidget())

        self.is_edit = False

        self.app.aboutToQuit.connect(self.exit_app)
        self.config = config
        self.images = OrderedDict()
        self.metadata = OrderedDict()
        self.mainview.hide_run()
        self.threads = []
        self.img_data = None
        self.echotime = None

        self.mouse_x = 0
        self.mouse_y = 0
        self.z = 0


    def open_bruker(self):
        """
        Opens Bruker directory
        """
        dirname = QtWidgets.QFileDialog.getExistingDirectory(self.mainview.centralwidget, "Select Bruker directory", self.config['default']['NifTiDir'])
        if not dirname:
            return
        try:
            list_filenames = io.open_generic_image(dirname)
        except Exception as e:
            print(e)
        else:
            if list_filenames:
                dirname = os.path.dirname(list_filenames[0])
                self.config['default']['NifTiDir'] = dirname
                self.open_nifti()



    def open_image(self, filename):
        try:
            img = io.open_generic_image(filename)
        except Exception as e:
            print(e)
        else:
            root, name = os.path.split(filename)
            name = name.replace('.nii', '')
            name = name.replace('.gz', '')
            self.img_data = img.get_fdata()
            self.add_image(img.get_fdata(), name)
            self.mainview.combobox.setCurrentIndex(self.mainview.combobox.findText(name))
            self.choose_image(name)
            return root, name
        return None, None

    def open_nifti(self):
        """
        Opens nifti file and reads metadata
        """
        filename, ext = QtWidgets.QFileDialog.getOpenFileName(self.mainview.centralwidget, "Select Nifti", self.config['default']['NifTiDir'])
        if not filename:
            return

        self.config['default']['NifTiDir'] = os.path.dirname(filename)
        root, name = self.open_image(filename)
        self.filename = name

        try:
            metadata = io.open_metadata(root + os.path.sep + self.filename + "_visu_pars.npy")
        except Exception as e:
            print("No metadata or echotimes")
            answer, _ = QtWidgets.QInputDialog.getText(None, "No echotimes found", "Echotimes separated by a semi-colon", QtWidgets.QLineEdit.Normal, "")
            echostring = answer.split(";")
            echostring = filter(None, echostring)
            echotime = [float(i) for i in echostring]
            self.echotime = echotime
            self.metadata[self.filename] = None
        else:
            echotime = io.extract_metadata(metadata, 'VisuAcqEchoTime')
            self.echotime = echotime
            self.metadata[self.filename] = metadata


    def save_nifti(self):
        """
        Saves as Nifti file
        """
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self.mainview.centralwidget, "Save Nifti", self.config['default']['NifTiDir'])
        im = self.mainview.imageview.imageDisp
        im = im.transpose((2, 1, 0))
        im = im.reshape(self.get_image().shape, order="F")
        if not filename:
            return
        if im is not None:
            img_data_name = self.current_name(self.img_data)
            io.save_nifti_with_metadata(im, self.metadata[img_data_name], filename)


    def get_image(self):
        if self.img_data.contains_plot_info:
            return self.img_data[..., 0]
        return self.img_data

    def exit_app(self):
        """
        Exits the app and save configuration
        preferences
        """
        with open('config.ini', 'w') as configfile:
            self.config.write(configfile)
        self.app.quit()


    def exp_fit_estimation(self):
        """
        Density and T2 estimation from
        exponential fitting
        see expfit.exponentialfit_image
        """
        fit_method = self.expfitcontroller.fit_method
        threshold = self.expfitcontroller.threshold
        threshold_error = self.expfitcontroller.threshold_error
        threshold_expfactor = self.expfitcontroller.threshold_expfactor
        outname = self.config['default']['NifTiDir']

        same_length = len(self.echotime) == self.img_data.shape[-1]

        if self.get_image() is not None and same_length:
            try:
                threshold = int(threshold)
            except:
                print("Automatic threshold with gaussian mixture")
                threshold = None
            try:
                threshold_error = float(threshold_error)
                threshold_expfactor = float(threshold_expfactor)
            except:
                print("Defaulting thresholds")
                threshold_error = 1.0
                threshold_expfactor = 0.0
            finally:
                self.update_progressbar(0)
                lreg = True
                piecewise_lreg = False
                n=1
                bi_exponential = False
                if "Linear regression" in fit_method:
                    if "bi-exponential" in fit_method:
                        bi_exponential = True
                else:
                    lreg = False
                    if fit_method == "Piecewise linear regression":
                        piecewise_lreg = True
                        n=2
                    elif fit_method == "NNLS bi-exponential":
                        n=2
                    elif fit_method == "NNLS tri-exponential":
                        n=3
                worker = WorkerExpFit(img_data=self.get_image(), echotime=self.echotime, threshold=threshold, lreg=lreg, biexp=bi_exponential, piecewise_lreg=piecewise_lreg, n=n, threshold_error=threshold_error, threshold_expfactor=threshold_expfactor)
                thread = QThread()
                worker.moveToThread(thread)
                worker.signal_start.connect(self.mainview.show_run)
                worker.signal_end.connect(self.end_expfit)
                worker.signal_progress.connect(self.update_progressbar)
                self.sig_abort_workers.signal.connect(worker.abort)
                thread.started.connect(worker.work)
                thread.start()
                self.threads.append((thread, worker))

    def nl_means_denoising(self):
        """
        Image denoising through nl means
        see expfit.denoise_image
        """
        size = self.nlmeanscontroller.patch_size
        distance = self.nlmeanscontroller.patch_distance
        spread = self.nlmeanscontroller.noise_spread
        outname = self.config['default']['NifTiDir']

        if self.get_image() is not None:
            try:
                size = int(size)
                distance = int(distance)
                spread = float(spread)
            except:
                print("Defaulting.")
                size = 5
                distance = 6
                spread = 1.5
            finally:
                self.update_progressbar(0)
                worker = WorkerNLMeans(img_data=self.get_image(), patch_size=size, patch_distance=distance, noise_spread=spread)
                thread = QThread()
                worker.moveToThread(thread)
                worker.signal_start.connect(self.mainview.show_run)
                worker.signal_end.connect(self.end_denoise)
                worker.signal_progress.connect(self.update_progressbar)
                self.sig_abort_workers.signal.connect(worker.abort)
                thread.started.connect(worker.work)
                thread.start()
                self.threads.append((thread, worker))

    def tpc_denoising(self):
        """
        Temporal phase correction
        see tpc.correct_phase_temporally
        """
        order = self.tpccontroller.polynomial_order
        threshold = self.tpccontroller.threshold
        outname = self.config['default']['NifTiDir']
        if self.get_image() is not None:
            try:
                order = int(order)
                threshold = int(threshold)
            except:
                print("Defaulting to order=4 and noise=0")
                order = 4
                threshold = 0
            finally:
                self.update_progressbar(0)
                worker = WorkerTPC(img_data=self.get_image(), echotime=self.echotime, order=order, threshold=threshold)
                thread = QThread()
                worker.moveToThread(thread)
                worker.signal_start.connect(self.mainview.show_run)
                worker.signal_end.connect(self.end_tpc)
                worker.signal_progress.connect(self.update_progressbar)
                self.sig_abort_workers.signal.connect(worker.abort)
                thread.started.connect(worker.work)
                thread.start()
                self.threads.append((thread, worker))

    def segment_cavity(self, preview=False):
        if not preview:
            self.remove_image("Preview")
        if preview:
            self.cavitycontroller.update_parameters(preview)

        multiplier = self.cavitycontroller.multiplier
        size_se = self.cavitycontroller.size_se
        if self.get_image() is not None:
            if preview:
                self.abort_computation()
            else:
                self.update_progressbar(0)

            worker = WorkerCavity(img_data=self.get_image(), multiplier=multiplier, size_se=size_se, preview=preview)
            thread = QThread()
            worker.moveToThread(thread)

            if not preview:
                worker.signal_start.connect(self.mainview.show_run)
                worker.signal_end.connect(self.end_segment_cavity)
                worker.signal_progress.connect(self.update_progressbar)
            else:
                worker.signal_end.connect(self.end_preview)
            self.sig_abort_workers.signal.connect(worker.abort)
            thread.started.connect(worker.work)
            thread.start()
            self.threads.append((thread, worker))

    def largest_component(self):
        if self.get_image() is not None:
            self.update_progressbar(0)

            worker = WorkerLargestComponent(img_data=self.get_image())
            thread = QThread()
            worker.moveToThread(thread)

            worker.signal_start.connect(self.mainview.show_run)
            worker.signal_end.connect(self.end_largest_component)
            worker.signal_progress.connect(self.update_progressbar)

            self.sig_abort_workers.signal.connect(worker.abort)
            thread.started.connect(worker.work)
            thread.start()
            self.threads.append((thread, worker))

    def hough_transform(self, preview=False):
        if preview:
            self.houghcontroller.update_parameters(preview)
        min_radius = self.houghcontroller.min_radius
        max_radius = self.houghcontroller.max_radius
        if self.get_image() is not None:
            try:
                min_radius = int(min_radius)
                max_radius = int(max_radius) + 1
            except:
                print("Defaulting")
                min_radius = 7
                max_radius =  20
            finally:
                if preview:
                    self.abort_computation()
                else:
                    self.update_progressbar(0)

                worker = WorkerHough(img_data=self.get_image(), min_radius=min_radius, max_radius=max_radius, preview=preview)
                thread = QThread()
                worker.moveToThread(thread)

                if not preview:
                    worker.signal_start.connect(self.mainview.show_run)
                    worker.signal_end.connect(self.end_hough_transform)
                    worker.signal_progress.connect(self.update_progressbar)
                else:
                    worker.signal_end.connect(self.end_preview)
                self.sig_abort_workers.signal.connect(worker.abort)
                thread.started.connect(worker.work)
                thread.start()
                self.threads.append((thread, worker))

    def manual_segmentation(self):
        manual_seg = self.get_image().copy()
        self.mainview.imageview.is_drawable = True
        self.end_preview(manual_seg, 1)
        self.mainview.imageview.setDrawable(True, self.mainview.imageview.pen_size)

        worker = WorkerManualSegmentation(img_data=self.mainview.imageview.imageDisp, original=self.get_image(), shape=self.get_image().shape)
        thread = QThread()
        worker.moveToThread(thread)
        worker.signal_end.connect(self.end_manual_seg)
        self.manualsegmentationcontroller.view.buttonBox.accepted.connect(worker.abort)
        thread.started.connect(worker.work)
        thread.start()
        self.threads.append((thread, worker))


    def register_position_on_click(self, evt):
        pos = evt.pos()
        mousePoint = self.mainview.imageview.view.mapSceneToView(pos)
        self.mouse_x = int(mousePoint.x())
        self.mouse_y = int(mousePoint.y())
        self.z = self.mainview.imageview.currentIndex

    def manual_component(self, evt, click=True):
        if not self.mainview.imageview.is_clickable:
            return

        if click:
            self.register_position_on_click(evt)
        self.manualcomponentcontroller.update_parameters()
        multiplier = self.manualcomponentcontroller.multiplier
        is_3D = self.manualcomponentcontroller.is_3D
        try:
            multiplier = float(multiplier)
        except Exception as e:
            multiplier = 1.0

        worker = WorkerManualComponent(self.get_image().copy(), seed=(self.mouse_x, self.mouse_y, self.z), multiplier=multiplier, is_3D=is_3D)
        thread = QThread()
        worker.moveToThread(thread)
        worker.signal_end.connect(self.end_preview)
        thread.started.connect(worker.work)
        thread.start()
        self.threads.append((thread, worker))


    def measurements(self):
        names = self.measurementcontroller.image
        slice_range = self.measurementcontroller.slice_range
        if names == "All":
            names = list(self.images.keys())
        else:
            names = [names]
        try:
            slice_range = [list(map(int, x.split(":"))) for x in slice_range.split(",")]
            slice_range = np.concatenate([np.arange(x[0], x[1]) for x in slice_range])
        except Exception as e:
            slice_range = -1
        worker = WorkerMeasurement(images=self.images, slice_range=slice_range, parent=self.mainview.parent.centralWidget(), names=names, metadata=self.metadata)
        thread = QThread()
        worker.moveToThread(thread)
        worker.signal_start.connect(self.mainview.show_run)
        worker.signal_end.connect(self.end_measurements)
        worker.signal_progress.connect(self.update_progressbar)
        self.sig_abort_workers.signal.connect(worker.abort)
        thread.started.connect(worker.work)
        thread.start()
        self.threads.append((thread, worker))


    def update_progressbar(self, progress):
        """
        Updates the progress bar each time
        An iteration in a controller has passed

        Parameters
        ----------
        progress: int
            progress value (/100)

        """
        self.mainview.progressBar.setValue(progress)

    def change_image_combobox(self, value):
        current_index = self.mainview.combobox.currentIndex()
        count = self.mainview.combobox.count() - 1
        new_index = max(0, min(current_index + value, count))
        name = self.mainview.combobox.itemText(new_index)
        self.choose_image(name)


    def end_expfit(self, density, t2, number):
        """
        Callback function called when expfit end signal
        is emitted
        Adds two new images to the view :
        density and T2*

        Parameters
        ----------
        density: np.ndarray
            density image
        t2: np.ndarray
            t2 image
        number: int
            image number
        """
        self.mainview.hide_run()
        density_name = "density_" + str(number)
        t2_name = "t2_" + str(number)
        residual = density[..., -1]
        print("Average residuals=", np.mean(residual))
        self.add_image(density, density_name, True)
        self.add_image(t2, t2_name, True)
        self.choose_image(density_name)

    def end_denoise(self, denoised, number):
        """
        Callback function called when denoise end signal
        is emitted
        Adds a new image: denoised

        Parameters
        ----------
        denoised: np.ndarray
            denoised image
        number: int
            image number
        """
        self.mainview.hide_run()
        out_name = "denoised_"+ str(number)
        self.add_image(denoised, out_name)
        self.choose_image(out_name)

    def end_tpc(self, real, imaginary, magnitude, phase, number):
        """
        Callback function called when tpc end signal
        is emitted

        Adds four new images: real, imaginary,
        phase and magnitude tpc corrected images

        Parameters
        ----------
        real: np.ndarray
            real image
        imaginary: np.ndarray
            imaginary image
        magnitude: np.ndarray
            magnitude image
        phase: np.ndarray
            phase image
        number: int
            image number
        """
        self.mainview.hide_run()
        real_name = "real_" + str(number)
        imaginary_name = "imaginary_" + str(number)
        magnitude_name = "magnitude_" + str(number)
        phase_name = "phase_" + str(number)
        self.add_image(real, real_name)
        self.add_image(imaginary, imaginary_name)
        self.add_image(magnitude, magnitude_name)
        self.add_image(phase, phase_name)
        self.choose_image(real_name)

    def end_segment_cavity(self, cavity, number):
        self.mainview.hide_run()
        cavity_name = "cavity_" + str(number)
        self.add_image(cavity, cavity_name)
        self.choose_image(cavity_name)

    def end_largest_component(self, threshold, number):
        self.mainview.hide_run()
        largest_name = "large_component_" + str(number)
        self.add_image(threshold, largest_name)
        self.choose_image(largest_name)

    def end_hough_transform(self, circle, number):
        self.mainview.hide_run()
        self.remove_image("Preview")
        hough_name = "hough_" + str(number)
        self.add_image(circle, hough_name)
        self.choose_image(hough_name)

    def update_pen_size(self):
        self.manualsegmentationcontroller.update_parameters()
        self.mainview.imageview.update_pen(pen_size=self.manualsegmentationcontroller.pencil_size)

    def cancel_preview(self):
        self.remove_image("Preview")
        self.choose_image(self.current_name(self.img_data))

    def cancel_manual_segmentation(self):
        self.cancel_preview()
        self.mainview.imageview.setDrawable(False)

    def cancel_manual_component(self):
        self.cancel_preview()
        self.mainview.imageview.setClickable(False)

    def end_manual_seg(self, image, number):
        self.remove_image("Preview")
        manual_name = "manual_" + str(number)
        self.add_image(image, manual_name)
        self.choose_image(manual_name)
        self.mainview.imageview.setDrawable(False)

    def end_manual_component(self):
        number = 1
        manual_name = "component_" + str(number)
        while manual_name in self.images:
            number += 1
            manual_name = "component_" + str(number)
        if "Preview" in self.images:
            image = self.images["Preview"]
            self.add_image(image, manual_name)
            self.choose_image(manual_name, preview=False)
            self.remove_image("Preview")
        self.mainview.imageview.setClickable(False)

    def end_measurements(self, names, units, ranges, array):
        self.mainview.hide_run()
        table = TableView(len(names)+1, 8, parent=self.mainview.parent.centralWidget())
        table.resize(640,320)
        table.set_headers(["Area (pixels)", "Area (unit)", "Average intensity", "Min intensity", "Max intensity", "Unit", "Slice range"])
        for i in range(len(names)):
            name = names[i]
            unit = units[i]
            r = ranges[i]
            table.set_item(name, i+1, 0)
            table.set_item(str(array[i, 0]), i+1, 1)
            table.set_item(str(array[i, 1]), i+1, 2)
            table.set_item(str(array[i, 2]), i+1, 3)
            table.set_item(str(array[i, 3]), i+1, 4)
            table.set_item(str(array[i, 4]), i+1, 5)
            table.set_item(str(unit), i+1, 6)
            table.set_item(str(r), i+1, 7)
        table.show()

    def end_preview(self, image, number):
        name = "Preview"
        if name in self.images:
            self.images[name] = Imeta.Image(image)
        else:
            self.add_image(image, name)
        self.choose_image(name, preview=True, autoLevels=False)


    def abort_computation(self):
        """
        Stops any computation in progress
        Hides the progress bar and stop button
        """
        self.sig_abort_workers.signal.emit()
        self.mainview.imageview.signal_abort.emit()
        for thread, worker in self.threads:
            thread.quit()
            thread.wait()
        for thread, worker in self.mainview.imageview.threads:
            thread.quit()
            thread.wait()
        self.mainview.hide_run()

    def add_image(self, image, name, plot_info=False):
        """
        Adds an image to the combobox
        and to the self.images dictionary

        Parameters
        ----------
        image: np.ndarray
            the image
        name: str
            combobox name
        """
        self.mainview.combobox.addItem(name)
        image_with_metadata = Imeta.Image(image, contains_plot_info=plot_info)
        self.images[name] = image_with_metadata
        img_data_name = self.current_name(self.img_data)
        self.metadata[name] = self.metadata[img_data_name] if img_data_name in self.metadata else None

    def current_name(self, image):
        list_keys = list(self.images.keys())
        list_values = list(self.images.values())
        try:
            key = [np.all(image == array) for array in list_values].index(True)
        except Exception as e:
            key = -1
        if len(list_keys) > 0:
            img_data_name = list_keys[key]
        else:
            img_data_name = "No image"
        return img_data_name

    def edit_name(self):
        self.is_edit = not self.is_edit
        if self.is_edit:
            fa_check = qta.icon('fa.check', color="green")
            self.mainview.editButton.setIcon(fa_check)
        else:
            self.mainview.combobox.update()
            old_name = self.current_name(self.img_data)
            new_name = self.mainview.combobox.currentText()
            if old_name != "No image":
                self.change_name(old_name, new_name)
                self.mainview.combobox.clear()
                self.mainview.combobox.addItems(list(self.images.keys()))
                index = self.mainview.combobox.findText(new_name)
                self.mainview.combobox.setCurrentIndex(index)
            fa_edit = qta.icon('fa.edit')
            self.mainview.editButton.setIcon(fa_edit)
        self.mainview.combobox.setEditable(self.is_edit)


    def change_name(self, old_name, new_name):
        if old_name in self.images:
            self.images = OrderedDict([(new_name, v) if k == old_name else (k, v) for k, v in self.images.items()])
            self.metadata = OrderedDict([(new_name, v) if k == old_name else (k, v) for k, v in self.metadata.items()])


    def remove_image(self,  name, manual=False):
        if name in self.metadata:
            del self.metadata[name]
        if name in self.images:
            del self.images[name]
            index = self.mainview.combobox.findText(name)
            self.mainview.combobox.removeItem(index)
            if len(self.images.keys()) > 0:
                if manual:
                    self.choose_image(list(self.images.keys())[index-1])
                else:
                    self.choose_image(list(self.images.keys())[-1])
            else:
                self.choose_image("No image")

    def choose_image(self, name, preview=False, autoLevels=True):
        """
        Choose an image among available image
        The name must be in self.images

        Parameters
        ----------
        name: str
            name of the image, must be in self.images.keys()
        """
        if name == "No image":
            return
        if name not in self.images:
            return
        if not preview:
            self.img_data = self.images[name]
        self.mainview.combobox.setCurrentIndex(self.mainview.combobox.findText(name))
        is_plot = self.images[name].contains_plot_info
        vis = self.image_to_visualization(self.images[name], is_plot)
        self.mainview.imageview.setImage(vis, contains_plot_info=is_plot)
        echotime = io.extract_metadata(self.metadata[name], 'VisuAcqEchoTime')
        if echotime is None:
            self.echotime = [i for i in range(self.images[name].shape[-1])]
        else:
            self.echotime = echotime

    def image_to_visualization(self, img, info_plot=False):
        """
        Modifies the image so it can be rendered
        Converts n-D image to 3D

        Parameters
        ----------
        img: np.ndarray
            n-D image loaded by the imageio module
        """
        if info_plot:
            img2 = np.reshape(img, img.shape, order='F')
        else:
            img2 = np.reshape(img, (img.shape[0], img.shape[1]) + (-1,), order='F')
        img2 = img2.transpose()
        return img2


    def on_click_image(self, evt):
        pos = evt
        ive = self.mainview.imageview
        image = ive.imageDisp
        if image is None:
            return
        if ive.imageCopy.contains_plot_info:
            number_echoes = len(self.echotime)
            pixel_values = ive.imageCopy[1:number_echoes+1, ive.currentIndex, ive.mouse_y, ive.mouse_x]
            fit = ive.imageCopy[number_echoes+1:-1, ive.currentIndex, ive.mouse_y, ive.mouse_x]
            residual = ive.imageCopy[-1:, ive.currentIndex, ive.mouse_y, ive.mouse_x]
            x = np.linspace(0, number_echoes, 50)
            y2 = expfit.n_exponential_function(x, *fit)
            self.widgetPlot.clear()
            self.widgetPlot.set_xlabel("Echotimes (ms)")
            self.widgetPlot.set_ylabel("Intensities")
            self.widgetPlot.set_text("residual=" + str(residual) + "\nexp. factor=" + str(round(float(np.sum(fit[1::2])), 4)))
            self.widgetPlot.plot(self.echotime, pixel_values, "o", label="Pixel values")
            self.widgetPlot.plot(x, y2, label="Exponential fit")
            self.widgetPlot.show()
