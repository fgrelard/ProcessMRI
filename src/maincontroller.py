import os
import numpy as np

from PyQt5 import QtWidgets
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from src.signal import Signal

from src.expfitcontroller import ExpFitController, WorkerExpFit
from src.nlmeanscontroller import NLMeansController, WorkerNLMeans
from src.cavitycontroller import CavityController, WorkerCavity
from src.tpccontroller import TPCController, WorkerTPC
from src.houghcontroller import HoughController, WorkerHough
from src.largestcomponentcontroller import WorkerLargestComponent
from src.measurementcontroller import MeasurementController, WorkerMeasurement

import src.imageio as io
import src.exponentialfit as expfit

import webbrowser

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
        self.houghcontroller = HoughController(mainview.centralWidget())
        self.houghcontroller.trigger.signal.connect(self.hough_transform)

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
        self.mainview.actionMeasurements.triggered.connect(lambda : self.measurementcontroller.show(self.images.keys()))

        self.cavitycontroller.view.horizontalSlider.valueChanged.connect(lambda: self.segment_cavity(preview=True))

        self.houghcontroller.view.pushButton_3.clicked.connect(lambda: self.hough_transform(preview=True))

        self.mainview.actionUser_manual_FR.triggered.connect(lambda event : webbrowser.open_new('file://' + os.path.realpath('docs/manual.pdf')))
        self.mainview.stopButton.clicked.connect(self.abort_computation)
        self.mainview.combobox.activated[str].connect(self.choose_image)

        self.mainview.imageview.signal_progress_export.connect(self.update_progressbar)
        self.mainview.imageview.signal_start_export.connect(self.mainview.show_run)
        self.mainview.imageview.signal_end_export.connect(self.mainview.hide_run)

        self.app.aboutToQuit.connect(self.exit_app)
        self.config = config
        self.images = {}
        self.mainview.hide_run()
        self.threads = []
        self.img_data = None
        self.echotime = None

        self.open_image("/mnt/d/IRM/raw/BLE/250/50/nifti/50_subscan_1.nii.gz")
        self.measurementcontroller.show()

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
            root, self.filename = os.path.split(filename)
            self.filename = self.filename.replace('.nii', '')
            self.filename = self.filename.replace('.gz', '')
            self.img_data = img.get_fdata()
            self.add_image(img.get_fdata(), self.filename)
            self.mainview.combobox.setCurrentIndex(self.mainview.combobox.findText(self.filename))
            self.choose_image(self.filename)

    def open_nifti(self):
        """
        Opens nifti file and reads metadata
        """
        filename, ext = QtWidgets.QFileDialog.getOpenFileName(self.mainview.centralwidget, "Select Nifti", self.config['default']['NifTiDir'])
        if not filename:
            return

        self.config['default']['NifTiDir'] = os.path.dirname(filename)
        self.open_image(filename)

        try:
            metadata = io.open_metadata(root + os.path.sep + self.filename + "_visu_pars.npy")
        except Exception as e:
            print("No metadata or echotimes")
            answer, _ = QtWidgets.QInputDialog.getText(None, "No echotimes found", "Echotimes separated by a semi-colon", QtWidgets.QLineEdit.Normal, "")
            echostring = answer.split(";")
            echostring = filter(None, echostring)
            echotime = [int(i) for i in echostring]
            self.echotime = echotime
        else:
            echotime = io.extract_metadata(metadata, 'VisuAcqEchoTime')
            self.echotime = echotime


    def save_nifti(self):
        """
        Saves as Nifti file
        """
        filename = QtWidgets.QFileDialog.getSaveFileName(self.mainview.centralwidget, "Save Nifti", self.config['default']['NifTiDir'])
        if not filename:
            return
        io.save_nifti_with_metadata(self.img_data, self.echotime, filename)

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
        outname = self.config['default']['NifTiDir']
        if self.img_data is not None:
            try:
                threshold = int(threshold)
            except:
                print("Automatic threshold with gaussian mixture")
                threshold = None
            finally:
                self.update_progressbar(0)
                lreg = True
                n=1
                if fit_method != "Linear regression":
                    lreg = False
                    if fit_method == "Mono-exponential":
                        n=1
                    elif fit_method == "Bi-exponential":
                        n=2
                    else:
                        n=3
                worker = WorkerExpFit(img_data=self.img_data, echotime=self.echotime, threshold=threshold, lreg=lreg, n=n)
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

        if self.img_data is not None:
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
                worker = WorkerNLMeans(img_data=self.img_data, patch_size=size, patch_distance=distance, noise_spread=spread)
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
        if self.img_data is not None:
            try:
                order = int(order)
                threshold = int(threshold)
            except:
                print("Defaulting to order=4 and noise=0")
                order = 4
                threshold = 0
            finally:
                self.update_progressbar(0)
                worker = WorkerTPC(img_data=self.img_data, echotime=self.echotime, order=order, threshold=threshold)
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
            key = "Preview"
            if key in self.images:
                del self.images[key]
            index = self.mainview.combobox.findText(key)
            self.mainview.combobox.removeItem(index)
        if preview:
            self.cavitycontroller.update_parameters(preview)

        multiplier = self.cavitycontroller.multiplier
        start_slice = self.cavitycontroller.start_slice
        end_slice = self.cavitycontroller.end_slice
        if self.img_data is not None:
            try:
                start_slice = int(start_slice)
                end_slice = int(end_slice) + 1
            except:
                print("Defaulting")
                start_slice = 1
                end_slice =  0
            finally:
                if preview:
                    self.abort_computation()
                else:
                    self.update_progressbar(0)

                worker = WorkerCavity(img_data=self.img_data, multiplier=multiplier, start=start_slice, end=end_slice, preview=preview)
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
        if self.img_data is not None:
            self.update_progressbar(0)

            worker = WorkerLargestComponent(img_data=self.img_data)
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
        if not preview:
            key = "Preview"
            if key in self.images:
                del self.images[key]
            index = self.mainview.combobox.findText(key)
            self.mainview.combobox.removeItem(index)
        if preview:
            self.houghcontroller.update_parameters(preview)

        min_radius = self.houghcontroller.min_radius
        max_radius = self.houghcontroller.max_radius
        if self.img_data is not None:
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

                worker = WorkerHough(img_data=self.img_data, min_radius=min_radius, max_radius=max_radius, preview=preview)
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

    def measurements(self):
        image = self.measurementcontroller.image
        slice_range = self.measurementcontroller.slice_range
        try:
            image = self.images[image]
        except:
            image = self.img_data
        try:
            slice_range = [list(map(int, x.split(":"))) for x in slice_range.split(",")]
            slice_range = np.concatenate([np.arange(x[0], x[1]) for x in slice_range])
        except Exception as e:
            slice_range = -1

        worker = WorkerMeasurement(img_data=image, slice_range=slice_range)
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
        self.add_image(density, density_name)
        self.add_image(t2, t2_name)
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
        hough_name = "hough_" + str(number)
        self.add_image(circle, hough_name)
        self.choose_image(hough_name)

    def end_measurements(self, area_pix, area_unit, average, min, max):
        print(area_pix, " ", area_unit, " ", average, " ", min, " ", max)

    def end_preview(self, image, number):
        name = "Preview"
        if name in self.images:
            self.images[name] = image
        else:
            self.add_image(image, name)
        self.choose_image(name, preview=True)


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

    def add_image(self, image, name):
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
        self.images[name] = image

    def choose_image(self, name, preview=False):
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
        vis = self.image_to_visualization(self.images[name])
        self.mainview.imageview.setImage(vis)

    def image_to_visualization(self, img):
        """
        Modifies the image so it can be rendered
        Converts n-D image to 3D

        Parameters
        ----------
        img: np.ndarray
            n-D image loaded by the imageio module
        """
        img2 = np.reshape(img, (img.shape[0], img.shape[1]) + (-1,), order='F')
        img2 = img2.transpose()
        return img2
