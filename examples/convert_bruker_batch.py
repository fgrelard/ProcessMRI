import src.imageio as io
import os
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Input MALDI (.imzML)")
    parser.add_argument("-o", "--output", help="Output")
    args = parser.parse_args()

    inputdir = args.input
    outputdir = args.output

    for dirname in next(os.walk(inputdir))[1]:
        input = os.path.join(inputdir, dirname)
        output = os.path.join(outputdir, dirname)
        io.bruker2nifti(input, output)
