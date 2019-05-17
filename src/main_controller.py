import json
from src.exponentialfit import estimation_density_image
import src.imageio as io
import tkinter.filedialog as filedialog
import os
import configparser

class MainController:
    def __init__(self, window, lbl, open_menu, file_menu, process_menu):
        #View
        self.window = window
        self.label = lbl
        self.open_menu = open_menu
        self.file_menu = file_menu
        self.process_menu = process_menu

        self.init_states()
        self.init_callbacks()
        self.init_configuration()

        #Model
        self.img_data = None
        self.echotime = None

    def init_states(self):
        self.process_menu.entryconfig(0, state="disabled")
        self.process_menu.entryconfig(1, state="disabled")

    def init_callbacks(self):
        self.open_menu.entryconfig(0, command=self.open_nifti)
        self.open_menu.entryconfig(1, command=self.open_bruker)
        self.file_menu.entryconfig(2, command=self.exit_app)

    def init_configuration(self):
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')
        if 'default' not in self.config or 'NifTiDir' not in self.config['default']:
            self.config['default'] = {}
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


    def estimation_density(self, threshold):
        dim = len(img_data.shape)
        x = self.img_data.shape[0]
        y = self.img_data.shape[1]
        z = self.img_data.shape[2]
        out_img_data = np.zeros(shape=(x, y, z))

        for k in progressbar.progressbar(range(z)):
            s = self.img_data[:, :, k, :]
            out_data = estimation_density_image(self.echotime, s, threshold)
            out_img_data[:,:,k] = out_data

        out_img = nib.Nifti1Image(out_img_data, np.eye(4))
        out_img.to_filename(output)
