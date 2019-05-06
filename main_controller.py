import nibabel as nib
import json
from exponentialfit import estimation_density_image

def open_image(filename):
    img = nib.load(filename)
    img_data = img.get_fdata()
    img  = nib.load(input)
    filename = os.path.splitext(input)[0]
    with open(filename+'.json') as f:
        data = json.load(f)
    echotime = [item for sublist in data['EchoTime'] for item in sublist]
    return echotime, img_data



def correct_phase(echotime, img, threshold):
    dim = len(img.shape)
    x = img.shape[0]
    y = img.shape[1]
    z = img.shape[2]
    out_img_data = np.zeros(shape=(x, y, z))

    for k in progressbar.progressbar(range(z)):
        s = img_data[:, :, k, :]
        out_data = estimation_density_image(echotime, s, threshold)
        out_img_data[:,:,k] = out_data

    out_img = nib.Nifti1Image(out_img_data, np.eye(4))
    out_img.to_filename(output)
