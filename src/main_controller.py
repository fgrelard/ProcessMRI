import json
import src.exponentialfit as expfit
import src.imageio as io
import src.temporalphasecorrection as tpc
import src.compleximage as ci

import tkinter.filedialog as filedialog

import os
import configparser
import nibabel as nib
import numpy as np
import threading
import queue
import time

class MainController:
    def __init__(self, mainview):
        self.mainview = mainview

        self.init_states()
        self.init_callbacks()


        #Model
        self.img_data = None
        self.echotime = None
        mainview.protocol('WM_DELETE_WINDOW', self.exit_app)

    def init_states(self):
        self.mainview.process_menu.entryconfig(0, state="normal")
        self.mainview.process_menu.entryconfig(1, state="normal")

    def init_callbacks(self):
        self.mainview.open_menu.entryconfig(0, command=self.open_nifti)
        self.mainview.open_menu.entryconfig(1, command=self.open_bruker)
        self.mainview.process_menu.entryconfig(0, command=lambda : self.mainview.show_frame("ExponentialFitView"))
        self.mainview.process_menu.entryconfig(1, command=lambda : self.mainview.show_frame("TemporalPhaseCorrectionView"))
        self.mainview.process_menu.entryconfig(2, command=lambda : self.mainview.show_frame("DenoiseView"))

        self.mainview.file_menu.entryconfig(2, command=self.exit_app)
        self.mainview.back_button.config(command=lambda : self.mainview.show_frame("MainView"))
        self.mainview.expframe.compute_button.config(command=self.thread_density_estimation)
        self.mainview.tpcframe.compute_button.config(command=self.thread_phase_correction)
        self.mainview.denoiseframe.compute_button.config(command=self.thread_image_denoising)


    def exit_app(self):
        with open('config.ini', 'w') as configfile:
            self.mainview.config.write(configfile)
        self.mainview.quit()

    def open_nifti(self):
        filename =  filedialog.askopenfilename(initialdir = self.mainview.config['default']['NifTiDir'],title = "Select NifTi image",filetypes = (("nii files","*.nii.gz"),("all files","*.*")))
        try:
            self.mainview.config['default']['NifTiDir'] = os.path.dirname(filename)
            img = io.open_generic_image(filename)
            metadata = io.open_metadata(filename)
        except Exception as e:
            print(e)
        else:
            echotime = io.extract_metadata(metadata, 'VisuAcqEchoTime')
            self.filename = os.path.split(filename)[1]
            self.img_data = img.get_fdata()
            self.echotime = echotime
            self.mainview.description.config(text="Image \"" + os.path.join(os.path.split(os.path.dirname(filename))[1], self.filename) + "\" loaded")
            self.mainview.process_menu.entryconfig(0, state="normal")
            self.mainview.process_menu.entryconfig(1, state="normal")


    def open_bruker(self):
        dirname =  filedialog.askdirectory(initialdir = self.mainview.config['default']['NifTiDir'], title = "Select Bruker directory")
        try:
            list_filenames = io.open_generic_image(dirname)
        except Exception as e:
            print(e)
        else:
            if list_filenames:
                dirname = os.path.dirname(list_filenames[0])
                self.mainview.config['default']['NifTiDir'] = dirname
                self.open_nifti()

    def thread_image_denoising(self):
        self.queue = queue.Queue()
        self.mainview.show_bar()
        self.mainview.progbar.start()
        ThreadedTask(self.queue, self.image_denoising).start()
        self.mainview.after(100, self.process_queue)


    def thread_density_estimation(self):
        self.queue = queue.Queue()
        self.mainview.show_bar()
        self.mainview.progbar.start()
        ThreadedTask(self.queue, self.density_estimation).start()
        self.mainview.after(100, self.process_queue)


    def thread_phase_correction(self):
        self.queue = queue.Queue()
        self.mainview.show_bar()
        self.mainview.progbar.start()
        ThreadedTask(self.queue, self.phase_correction).start()
        self.mainview.after(100, self.process_queue)

    def process_queue(self):
        try:
            self.mainview.update()
            msg = self.queue.get(0)
            self.mainview.progbar.stop()
            self.mainview.hide_bar()
        except queue.Empty:
            self.mainview.after(100, self.process_queue)

    def image_denoising(self):
        size = self.mainview.denoiseframe.size.get()
        distance = self.mainview.denoiseframe.distance.get()
        spread = self.mainview.denoiseframe.spread.get()
        outname = self.mainview.denoiseframe.path.get()

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
                img = expfit.denoise_image(self.img_data, size, distance, spread)
                denoised_img = nib.Nifti1Image(img, np.eye(4))
                denoised_img.to_filename(os.path.join(outname,  self.filename+"_denoised.nii"))


    def phase_correction(self):
        order = self.mainview.tpcframe.order.get()
        outname = self.mainview.tpcframe.path.get()
        if self.img_data is not None:
            try:
                order = int(order)
            except:
                print("Defaulting to order=4")
                order = 4
            finally:
                self.echotime = np.array(self.echotime).tolist()
                order = 3
                temporally_corrected = tpc.correct_phase_temporally(self.echotime, self.img_data[:,:,4,:], order)
                magnitude = ci.complex_to_magnitude(temporally_corrected)
                phase = ci.complex_to_phase(temporally_corrected)

                magnitude_img = nib.Nifti1Image(magnitude, np.eye(4))
                magnitude_img.to_filename(os.path.join(outname, self.filename+"_magnitude_tpc.nii"))

                phase_img = nib.Nifti1Image(phase, np.eye(4))
                phase_img.to_filename(os.path.join(outname, self.filename+"_phase_tpc.nii"))

    def density_estimation(self):
        fit_method = self.mainview.expframe.choice_method.get()
        threshold = self.mainview.expframe.threshold.get()
        outname = self.mainview.expframe.path.get()

        if self.img_data is not None:
            try:
                threshold = int(threshold)
            except:
                print("Please enter a correct value for threshold")
                threshold = None
            finally:
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
                density, t2 = expfit.exponentialfit_image(self.echotime, self.img_data, threshold, lreg, n)
                density_img = nib.Nifti1Image(density, np.eye(4))
                density_img.to_filename(os.path.join(outname,  self.filename+"_density.nii"))

                t2_img = nib.Nifti1Image(t2, np.eye(4))
                t2_img.to_filename(os.path.join(outname,  self.filename+"_t2_star.nii"))

class ThreadedTask(threading.Thread):
    def __init__(self, queue, function):
        threading.Thread.__init__(self)
        self.queue = queue
        self.function = function
    def run(self):
        self.function()
        self.queue.put("Task finished")
