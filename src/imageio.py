from lib.bruker2nifti.bruker2nifti.converter import Bruker2Nifti
import os
import nibabel as nib
import numpy as np

def open_generic_image(input_name):
    if os.path.isdir(input_name):
        return open_bruker(input_name)
    elif input_name.endswith(".nii") or input_name.endswith(".nii.gz"):
        return open_nifti(input_name)
    else:
        raise RuntimeError("Unknown file format. Please open a Bruker NMR data directory or NifTi images.")

def open_bruker(input_name):
    bruker2nifti(input_name)
    l = []
    for subdir, dirs, files in os.walk(os.path.join(input_name, "nifti")):
        for f in files:
            if f.endswith(".nii.gz"):
                l.append(os.path.join(subdir,f))
    return l


def open_nifti(input_name):
    return nib.load(input_name)

def open_metadata(input_name):
    filename_stripped = os.path.splitext(input_name)[0]
    if input_name.endswith(".nii.gz"):
        filename_stripped = os.path.splitext(filename_stripped)[0]
    dic =  np.load(filename_stripped + "_visu_pars.npy", allow_pickle=True)
    return dic.item()

def extract_metadata(metadata, key):
    return metadata[key]


def bruker2nifti(input_name):
    output_name = os.path.join(input_name, "nifti")
    # Instantiate a converter:
    bruconv = Bruker2Nifti(os.path.dirname(input_name),
                           output_name)

    # # if args.scans_list is not None:
    # #     bruconv.scans_list = args.scans_list
    # # if args.list_new_name_each_scan is not None:
    # #     bruconv.list_new_name_each_scan = args.list_new_name_each_scan

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
