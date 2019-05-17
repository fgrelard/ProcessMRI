import json
from src.exponentialfit import estimation_density_image
import src.imageio as io
import tkinter.filedialog as filedialog
import os
import configparser
import src.exponentialfitview as expview

class MainController:
    def __init__(self, window, lbl, open_menu, file_menu, process_menu):
        #Storage preferences initialization
        self.init_configuration()

        #View
        self.window = window
        self.label = lbl
        self.open_menu = open_menu
        self.file_menu = file_menu
        self.process_menu = process_menu

        self.expframe = expview.ExponentialFitView(window, self.config)

        self.init_states()
        self.init_callbacks()


        #Model
        self.img_data = None
        self.echotime = None
        window.protocol('WM_DELETE_WINDOW', self.exit_app)

    def init_states(self):
        self.process_menu.entryconfig(0, state="normal")
        self.process_menu.entryconfig(1, state="disabled")

    def init_callbacks(self):
        self.open_menu.entryconfig(0, command=self.open_nifti)
        self.open_menu.entryconfig(1, command=self.open_bruker)
        self.process_menu.entryconfig(0, command=self.expframe.show)
        self.file_menu.entryconfig(2, command=self.exit_app)
        self.expframe.compute_button.config(command=self.estimation_density)

    def init_configuration(self):
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')
        if 'default' not in self.config:
            self.config['default'] = {}
            if 'NifTiDir' not in self.config['default']:
                self.config['default']['NifTiDir'] = os.getcwd()

    def exit_app(self):
        with open('config.ini', 'w') as configfile:
            self.config.write(configfile)
        self.window.quit()

    def open_nifti(self):
        filename =  filedialog.askopenfilename(initialdir = self.config['default']['NifTiDir'],title = "Select NifTi image",filetypes = (("nii files","*.nii.gz"),("all files","*.*")))
        try:
            img = io.open_generic_image(filename)
            metadata = io.open_metadata(filename)
        except Exception as e:
            print(e)
        else:
            echotime = io.extract_metadata(metadata, 'VisuAcqEchoTime')
            self.img_data = img.get_fdata()
            self.echotime = echotime
            self.label.config(text="Image \"" + os.path.join(os.path.split(os.path.dirname(filename))[1], os.path.split(filename)[1]) + "\" loaded")
            self.process_menu.entryconfig(0, state="normal")
            self.process_menu.entryconfig(1, state="normal")


    def open_bruker(self):
        dirname =  filedialog.askdirectory (initialdir = self.config['default']['NifTiDir'],title = "Select Bruker directory")
        try:
            list_filenames = io.open_generic_image(dirname)
        except Exception as e:
            print(e)
        else:
            if list_filenames:
                dirname = os.path.dirname(list_filenames[0])
                self.config['default']['NifTiDir'] = dirname
                self.open_nifti()


    def estimation_density(self):
        fit_method = self.expframe.choice_method.get()
        threshold = self.expframe.threshold.get()
        outname = self.expframe.path.get()

        if self.img_data:
            try:
                int(threshold)
            except:
                print("Please enter a correct value for threshold")
            else:
                threshold = None
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

                estimation_density_image(self.echotime, self.img_data, threshold, lreg, n)
                out_img = nib.Nifti1Image(outname, np.eye(4))
                out_img.to_filename(output)
        else:
            print("No image opened")
