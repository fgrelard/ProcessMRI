import nibabel as nib
import json
from src.exponentialfit import estimation_density_image
import tkinter.filedialog as filedialog
import os


class MainController:
    def __init__(self, lbl, open_button, tpc_button, density_button, menu):
        #View
        self.label = lbl
        self.open_button = open_button
        self.tpc_button = tpc_button
        self.density_button = density_button
        self.menu = menu

        self.init_states()
        self.init_callbacks()

        #Model
        self.img_data = None
        self.echotime = None

    def init_states(self):
        self.tpc_button.config(state="disabled")
        self.density_button.config(state="disabled")

    def init_callbacks(self):
        self.open_button.config(command=self.open_image)


    def open_image(self):
        filename =  filedialog.askopenfilename(initialdir = "/mnt/d/IRM/nifti/7/BLE RECITAL/1_BLE 250DJ",title = "Select file",filetypes = (("nii files","*.nii"),("all files","*.*")))
        img = nib.load(filename)
        filename_stripped = os.path.splitext(filename)[0]
        dirname = os.path.dirname(filename)
        with open(filename_stripped+'.json') as f:
            data = json.load(f)
        self.img_data = img.get_fdata()
        self.echotime = [item for sublist in data['EchoTime'] for item in sublist]

        self.density_button.config(state="normal")
        self.tpc_button.config(state="normal")
        self.label.config(text="Image \"" + os.path.join(os.path.split(dirname)[1], os.path.split(filename)[1]) + "\" loaded")

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
