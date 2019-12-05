from lib.bruker2nifti.bruker2nifti.converter import Bruker2Nifti
import os
import nibabel as nib
import numpy as np

def open_generic_image(input_name):
    """
    Generic function to open NifTi images or Bruker
    images

    Parameters
    ----------
    input_name: string
        image name

    Returns
    ----------
    list or numpy.ndarray
        the image or list of images
    """
    if os.path.isdir(input_name):
        return open_bruker(input_name)
    elif input_name.endswith(".nii") or input_name.endswith(".nii.gz"):
        return open_nifti(input_name)
    else:
        raise RuntimeError("Unknown file format. Please open a Bruker NMR data directory or NifTi images.")

def open_bruker(input_name):
    """
    Opens a Bruker directory containing NMR data
    thanks to bruker2nifti

    Parameters
    ----------
    input_name: string
        image name

    Returns
    ----------
    list
        list of filenames (.nii)

    """
    bruker2nifti(input_name)
    l = []
    for subdir, dirs, files in os.walk(os.path.join(input_name, "nifti")):
        for f in files:
            if f.endswith(".nii.gz"):
                l.append(os.path.join(subdir,f))
    return l


def open_nifti(input_name):
    """
    Opens a NifTi image thanks to nibabel

    Parameters
    ----------
    input_name: string
        image filename

    Returns
    ----------
    numpy.ndarray
        image
    """
    return nib.load(input_name)

def open_metadata(input_name):
    """
    Loads metadata from a .npy file
    associated to a NifTi image
    normally produced from bruker2nifti

    Parameters
    ----------
    input_name: string
        image (nii) filename

    Returns
    ----------
    dict
        dictionary (key-val) metadata

    """
    dic =  np.load(input_name, allow_pickle=True)
    return dic.item()

def extract_metadata(metadata, key):
    """
    Extracts a given value from the metadata
    dict with given key

    Parameters
    ----------
    metadata: dict
        dictionary
    key: string
        the key to search for

    Returns
    ----------
    any
        metadata associated with key

    """
    return metadata[key]

def save_metadata_by_keyval(output_name, key, val):
    """
    Saves metadata to file

    Parameters
    ----------
    output_name: string
        name of output file
    key: string
        key in dictionary
    val: any
        value
    """
    metadata = {}
    metadata[key] = val
    save_metadata(output_name, metadata)

def save_metadata(output_name, metadata):
    """
    Saves metadata to file

    Parameters
    ----------
    output_name: string
        name of output file
    metadata: any
        metadata to save
    """
    np.save(output_name, metadata)

def bruker2nifti(input_name, output_name=None):
    """
    Converts Bruker data to NifTi
    Calls the bruker2nifti submodule
    (in lib/bruker2nifti)
    with appropriate parameters

    Parameters
    ----------
    input_name: string
        image filename

    """
    if output_name is None:
        output_name = os.path.join(input_name, "nifti")
        if not os.path.exists(output_name):
            os.makedirs(output_name)
    # Instantiate a converter:
    bruconv = Bruker2Nifti(os.path.dirname(input_name),
                           output_name)

    # # Basics
    bruconv.nifti_version       = 1
    bruconv.qform_code          = 1
    bruconv.sform_code          = 2
    bruconv.save_human_readable = True
    bruconv.correct_slope       = True
    bruconv.correct_offset      = False
    bruconv.verbose             = False
    # Sample position
    bruconv.sample_upside_down       = True
    bruconv.frame_body_as_frame_head = True

    print("Converting to nifti")
    bruconv.convert_scan(input_name,
                         output_name,
                         nifti_file_name=os.path.basename(input_name),
                         create_output_folder_if_not_exists=True)
    print("Nifti saved to "+  output_name)


def save_nifti(img, filename):
    nibimg = nib.Nifti1Image(img, np.eye(4))
    nibimg.to_filename(filename)

def save_nifti_with_metadata(img, echotime, filename):
    root, ext = os.path.splitext(filename)
    image_name = root + ".nii"
    metadata_name = root + "_visu_pars.npy"
    save_nifti(img, image_name)
    save_metadata_by_keyval(metadata_name, "VisuAcqEchoTime", echotime)
