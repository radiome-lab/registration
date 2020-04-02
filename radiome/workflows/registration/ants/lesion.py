import ntpath
import os
import shutil

import nibabel as nib
from nipype.interfaces import afni
from radiome.core.jobs import NipypeJob, PythonJob
from radiome.core.utils.s3 import S3Resource

from .nifti import more_zeros_than_ones, inverse_nifti_values


def inverse_lesion(lesion_path):
    """
    Check if the image contains more zeros than non-zeros, if so,
    replaces non-zeros by zeros and zeros by ones.

    Parameters
    ----------
    lesion_path: str
        path to the nifti file to be checked and inverted if needed

    Returns
    -------
    lesion_out: str
        path to the output file, if the lesion does not require to be inverted
        it returns the unchanged lesion_path input
    """

    lesion_out = lesion_path

    if more_zeros_than_ones(image=lesion_path):
        lesion_out = os.path.join(os.getcwd(), ntpath.basename(lesion_path))
        shutil.copyfile(lesion_path, lesion_out)
        nii = inverse_nifti_values(image=lesion_path)
        nib.save(nii, lesion_out)
        return {
            'lesion_out': lesion_out
        }
    else:
        return {
            'lesion_out': lesion_out
        }


def lesion_preproc(lesion_mask: str):
    lesion_deoblique = NipypeJob(interface=afni.Refit(deoblique=True), reference='lesion_deoblique')
    lesion_inverted = PythonJob(function=inverse_lesion, reference='inverse_lesion')
    lesion_reorient = NipypeJob(interface=afni.Resample(orientation='RPI', outputtype='NIFTI_GZ'),
                                reference='lesion_reorient')

    lesion_inverted.lesion_path = S3Resource(lesion_mask) if lesion_mask.lower().startswith('s3://') else lesion_mask
    lesion_deoblique.in_file = lesion_inverted.lesion_out
    lesion_reorient.in_file = lesion_deoblique.out_file
    return lesion_reorient.out_file
