# ProcessMRI
Process of MRI images including:
- Rician denoising by non-local means ([Wiest-Daesslé et al., 2008](#Wiest-Daesslé2008))
- Temporal phase correction of complex data with multiple echoes ([Bjarnason et al., 2013](#Bjarnason2013)) 
- n-exponential fitting on multiple echoes for density and T2* estimation

For methods involving multi-echo data, it is assumed the echoes are represented along the **last** dimension of the image of size n.

For temporal phase correction, the real and imaginary parts are assumed to be mixed in the **last** dimension of the image, that is to say the "echo"-dimension of size 2\*n. The first n images are the real images, and the last n images are the complex images for the n echoes.

The mono-exponential fitting can be performed by linear regression on the logarithm of the data, or through non-negative least squares of a n-exponential function (see [scipy.optimize.curve_fit](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html)).

## Installation
```
cd ProcessMRI
git submodule init
git submodule update
pip install -r requirements.txt
```

## Usage
```
python main.py
```
A graphical user interface should open.
1. **File/Open**: Open a Bruker directory or a NifTi MRI image. If this operation was successful, this 
2. **Process/..."**: choose the method to apply. For each method, various parameters can be chosen, as well as the output directory. The filename is determined as : `input_name + method_name.nii`.

## References

<a name="Wiest-Daesslé2008"> Wiest-Daesslé, N., Prima, S., Coupé, P., Morrissey, S.P., Barillot, C., 2008. Rician Noise Removal by Non-Local Means Filtering for Low Signal-to-Noise Ratio MRI: Applications to DT-MRI, in: Metaxas, D., Axel, L., Fichtinger, G., Székely, G. (Eds.), Medical Image Computing and Computer-Assisted Intervention – MICCAI 2008. Springer Berlin Heidelberg, Berlin, Heidelberg, pp. 171–179. https://doi.org/10.1007/978-3-540-85990-1_21</a>


<a name="Bjarnason2013">Bjarnason, T.A., Laule, C., Bluman, J., Kozlowski, P., 2013. Temporal phase correction of multiple echo T2 magnetic resonance images. Journal of Magnetic Resonance 231, 22–31. https://doi.org/10.1016/j.jmr.2013.02.019</a>
