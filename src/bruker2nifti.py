# from __future__ import absolute_import
# import sys
# sys.path.append('../lib/bruker2nifti/')

from lib.bruker2nifti.bruker2nifti.converter import Bruker2Nifti
import os

def convert(input_name):

    output_name = os.path.join(input_name, "nifti")
    print(output_name)
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
