from pathlib import Path

from radiome.core import workflow, AttrDict
from radiome.core.context import Context
from radiome.core.jobs import PythonJob
from radiome.core.resource_pool import ResourcePool, ResourceKey as R

from .lesion import lesion_preproc


def hardcoded_reg(moving_brain, reference_brain, moving_skull,
                  reference_skull, ants_para, fixed_image_mask=None, interp=None):
    # TODO: expand transforms to cover all in ANTs para
    import os
    import subprocess
    from pathlib import Path
    regcmd = ["antsRegistration"]
    for para_index in range(len(ants_para)):
        for para_type in ants_para[para_index]:
            if para_type == 'dimensionality':
                if ants_para[para_index][para_type] not in [2, 3, 4]:
                    err_msg = 'Dimensionality specified in ANTs parameters: %d, is not supported. ' \
                              'Change to 2, 3, or 4 and try again' % ants_para[para_index][para_type]
                    raise Exception(err_msg)
                else:
                    regcmd.append("--dimensionality")
                    regcmd.append(str(ants_para[para_index][para_type]))

            elif para_type == 'collapse-output-transforms':
                if ants_para[para_index][para_type] not in [0, 1]:
                    err_msg = 'collapse-output-transforms specified in ANTs parameters: %d, is not supported. ' \
                              'Change to 0 or 1 and try again' % ants_para[para_index][para_type]
                    raise Exception(err_msg)
                else:
                    regcmd.append("--collapse-output-transforms")
                    regcmd.append(str(ants_para[para_index][para_type]))
            elif para_type == 'initial-moving-transform':
                if ants_para[para_index][para_type]['initializationFeature'] is None:
                    err_msg = 'Please specifiy initializationFeature of ANTs parameters in pipeline config. '
                    raise Exception(err_msg)
                else:
                    regcmd.append("--initial-moving-transform")
                    regcmd.append("[{0},{1},{2}]".format(
                        reference_brain, moving_brain, ants_para[para_index][para_type]['initializationFeature']))

            elif para_type == 'transforms':
                for trans_index in range(len(ants_para[para_index][para_type])):
                    for trans_type in ants_para[para_index][para_type][trans_index]:
                        regcmd.append("--transform")
                        if trans_type == 'Rigid' or trans_type == 'Affine':
                            if ants_para[para_index][para_type][trans_index][trans_type]['gradientStep'] is None:
                                err_msg = 'Please specifiy % s Gradient Step of ANTs parameters in pipeline config. ' % trans_type
                                raise Exception(err_msg)
                            else:
                                regcmd.append("{0}[{1}]".format(
                                    trans_type,
                                    ants_para[para_index][para_type][trans_index][trans_type]['gradientStep']))

                        if trans_type == 'SyN':
                            if ants_para[para_index][para_type][trans_index][trans_type]['gradientStep'] is None:
                                err_msg = 'Please specifiy % s Gradient Step of ANTs parameters in pipeline config. ' % trans_type
                                raise Exception(err_msg)
                            else:
                                SyN_para = []
                                SyN_para.append("{0}".format(
                                    ants_para[para_index][para_type][trans_index][trans_type]['gradientStep']))
                                if ants_para[para_index][para_type][trans_index][trans_type][
                                    'updateFieldVarianceInVoxelSpace'] is not None:
                                    SyN_para.append("{0}".format(
                                        ants_para[para_index][para_type][trans_index][trans_type][
                                            'updateFieldVarianceInVoxelSpace']))
                                if ants_para[para_index][para_type][trans_index][trans_type][
                                    'totalFieldVarianceInVoxelSpace'] is not None:
                                    SyN_para.append("{0}".format(
                                        ants_para[para_index][para_type][trans_index][trans_type][
                                            'totalFieldVarianceInVoxelSpace']))
                                SyN_para = ','.join([str(elem)
                                                     for elem in SyN_para])
                                regcmd.append("{0}[{1}]".format(
                                    trans_type, SyN_para))

                        if ants_para[para_index][para_type][trans_index][trans_type]['metric']['type'] == 'MI':
                            if ants_para[para_index][para_type][trans_index][trans_type]['metric'][
                                'metricWeight'] is None or \
                                    ants_para[para_index][para_type][trans_index][trans_type]['metric'][
                                        'numberOfBins'] is None:
                                err_msg = 'Please specifiy metricWeight and numberOfBins for metric MI of ANTs parameters in pipeline config.'
                                raise Exception(err_msg)
                            else:
                                MI_para = []
                                MI_para.append(
                                    "{0},{1}".format(ants_para[para_index][para_type][trans_index][trans_type]['metric']
                                                     ['metricWeight'],
                                                     ants_para[para_index][para_type][trans_index][trans_type][
                                                         'metric']['numberOfBins']))
                                if 'samplingStrategy' in ants_para[para_index][para_type][trans_index][trans_type][
                                    'metric'] and ants_para[para_index][para_type][trans_index][trans_type]['metric'][
                                    'samplingStrategy'] in ['None', 'Regular', 'Random']:
                                    MI_para.append("{0}".format(
                                        ants_para[para_index][para_type][trans_index][trans_type]['metric'][
                                            'samplingStrategy']))
                                if 'samplingPercentage' in ants_para[para_index][para_type][trans_index][trans_type][
                                    'metric'] and ants_para[para_index][para_type][trans_index][trans_type]['metric'][
                                    'samplingPercentage'] is not None:
                                    MI_para.append("{0}".format(
                                        ants_para[para_index][para_type][trans_index][trans_type]['metric'][
                                            'samplingPercentage']))
                                MI_para = ','.join([str(elem) for elem in MI_para])
                                regcmd.append("--metric")
                                regcmd.append("MI[{0},{1},{2}]".format(
                                    reference_brain, moving_brain, MI_para))

                        if ants_para[para_index][para_type][trans_index][trans_type]['metric']['type'] == 'CC':
                            if ants_para[para_index][para_type][trans_index][trans_type]['metric'][
                                'metricWeight'] is None or \
                                    ants_para[para_index][para_type][trans_index][trans_type]['metric'][
                                        'radius'] is None:
                                err_msg = 'Please specifiy metricWeight and radius for metric CC of ANTs parameters in pipeline config.'
                                raise Exception(err_msg)
                            else:
                                CC_para = []
                                CC_para.append(
                                    "{0},{1}".format(ants_para[para_index][para_type][trans_index][trans_type]['metric']
                                                     ['metricWeight'],
                                                     ants_para[para_index][para_type][trans_index][trans_type][
                                                         'metric']['radius']))
                                if 'samplingStrategy' in ants_para[para_index][para_type][trans_index][trans_type][
                                    'metric'] and ants_para[para_index][para_type][trans_index][trans_type]['metric'][
                                    'samplingStrategy'] in ['None', 'Regular', 'Random']:
                                    CC_para.append("{0}".format(
                                        ants_para[para_index][para_type][trans_index][trans_type]['metric'][
                                            'samplingStrategy']))
                                if 'samplingPercentage' in ants_para[para_index][para_type][trans_index][trans_type][
                                    'metric'] and ants_para[para_index][para_type][trans_index][trans_type]['metric'][
                                    'samplingPercentage'] is not None:
                                    CC_para.append("{0}".format(
                                        ants_para[para_index][para_type][trans_index][trans_type]['metric'][
                                            'samplingPercentage']))
                                CC_para = ','.join([str(elem) for elem in CC_para])
                                regcmd.append("--metric")
                                regcmd.append("CC[{0},{1},{2}]".format(
                                    reference_skull, moving_skull, CC_para))

                        if 'convergence' in ants_para[para_index][para_type][trans_index][trans_type]:
                            convergence_para = []
                            if ants_para[para_index][para_type][trans_index][trans_type]['convergence'][
                                'iteration'] is None:
                                err_msg = 'Please specifiy convergence iteration of ANTs parameters in pipeline config.'
                                raise Exception(err_msg)
                            else:
                                convergence_para.append("{0}".format(
                                    ants_para[para_index][para_type][trans_index][trans_type]['convergence'][
                                        'iteration']))
                                if 'convergenceThreshold' in ants_para[para_index][para_type][trans_index][trans_type][
                                    'convergence'] and \
                                        ants_para[para_index][para_type][trans_index][trans_type]['convergence'][
                                            'convergenceThreshold'] is not None:
                                    convergence_para.append("{0}".format(
                                        ants_para[para_index][para_type][trans_index][trans_type]['convergence'][
                                            'convergenceThreshold']))
                                if 'convergenceWindowSize' in ants_para[para_index][para_type][trans_index][trans_type][
                                    'convergence'] and \
                                        ants_para[para_index][para_type][trans_index][trans_type]['convergence'][
                                            'convergenceWindowSize'] is not None:
                                    convergence_para.append("{0}".format(
                                        ants_para[para_index][para_type][trans_index][trans_type]['convergence'][
                                            'convergenceWindowSize']))
                                convergence_para = ','.join(
                                    [str(elem) for elem in convergence_para])
                                regcmd.append("--convergence")
                                regcmd.append("[{0}]".format(convergence_para))

                        if 'smoothing-sigmas' in ants_para[para_index][para_type][trans_index][trans_type] and \
                                ants_para[para_index][para_type][trans_index][trans_type][
                                    'smoothing-sigmas'] is not None:
                            regcmd.append("--smoothing-sigmas")
                            regcmd.append("{0}".format(
                                ants_para[para_index][para_type][trans_index][trans_type]['smoothing-sigmas']))

                        if 'shrink-factors' in ants_para[para_index][para_type][trans_index][trans_type] and \
                                ants_para[para_index][para_type][trans_index][trans_type]['shrink-factors'] is not None:
                            regcmd.append("--shrink-factors")
                            regcmd.append("{0}".format(
                                ants_para[para_index][para_type][trans_index][trans_type]['shrink-factors']))

                        if 'use-histogram-matching' in ants_para[para_index][para_type][trans_index][trans_type]:
                            if ants_para[para_index][para_type][trans_index][trans_type]['use-histogram-matching']:
                                regcmd.append("--use-histogram-matching")
                                regcmd.append("1")
                            else:
                                continue

                        if 'winsorize-image-intensities' in ants_para[para_index][para_type][trans_index][
                            trans_type] and ants_para[para_index][para_type][trans_index][trans_type][
                            'winsorize-image-intensities']['lowerQuantile'] is not None and \
                                ants_para[para_index][para_type][trans_index][trans_type][
                                    'winsorize-image-intensities']['upperQuantile'] is not None:
                            regcmd.append("--winsorize-image-intensities")
                            regcmd.append("[{0},{1}]".format(
                                ants_para[para_index][para_type][trans_index][trans_type]['winsorize-image-intensities']
                                ['lowerQuantile'], ants_para[para_index][para_type][trans_index][trans_type][
                                    'winsorize-image-intensities']['upperQuantile']))

    if interp is not None:
        regcmd.append("--interpolation")
        regcmd.append("{0}".format(interp))

    regcmd.append("--output")
    regcmd.append("[transform,transform_Warped.nii.gz]")

    if fixed_image_mask is not None:
        regcmd.append("-x")
        regcmd.append(str(fixed_image_mask))

    # write out the actual command-line entry for testing/validation later
    command_file = os.path.join(os.getcwd(), 'command.txt')
    with open(command_file, 'wt') as f:
        f.write(' '.join(regcmd))

    try:
        retcode = subprocess.check_output(regcmd)
    except Exception as e:
        raise Exception('[!] ANTS registration did not complete successfully.'
                        '\n\nError details:\n{0}\n{1}\n'.format(e, e.output))

    warp_list = []
    warped_image = None

    files = [f for f in os.listdir('.') if os.path.isfile(f)]

    for f in files:
        if ("transform" in f) and ("Warped" not in f):
            warp_list.append(os.getcwd() + "/" + f)
        if "Warped" in f:
            warped_image = os.getcwd() + "/" + f

    if not warped_image:
        raise Exception("\n\n[!] No registration output file found. ANTS "
                        "registration may not have completed "
                        "successfully.\n\n")

    return {
        'warp_list': warp_list,
        'warped_image': Path(warped_image)
    }


def seperate_warps_list(warp_list, selection):
    from pathlib import Path

    selected_warp = None
    for warp in warp_list:
        if selection == 'Warp':
            if '3Warp' in warp or '2Warp' in warp or '1Warp' in warp:
                selected_warp = warp
        else:
            if selection in warp:
                selected_warp = warp
    return {
        'selected_warp': Path(selected_warp)
    }


@workflow()
def create_workflow(config: AttrDict, resource_pool: ResourcePool, context: Context):
    for _, rp in resource_pool[['label-reorient_T1w', 'brain']]:
        calculate_ants_warp = PythonJob(function=hardcoded_reg, reference='calc_ants_warp')
        # calculate_ants_warp.interface.num_threads = num_threads
        select_forward_initial = PythonJob(function=seperate_warps_list, reference='select_forward_initial')
        select_forward_initial.selection = "Initial"

        select_forward_rigid = PythonJob(function=seperate_warps_list, reference='select_forward_rigid')
        select_forward_rigid.selection = "Rigid"

        select_forward_affine = PythonJob(function=seperate_warps_list, reference='select_forward_affine')
        select_forward_affine.selection = "Affine"

        select_forward_warp = PythonJob(function=seperate_warps_list, reference='select_forward_warp')
        select_forward_warp.selection = "Warp"

        select_inverse_warp = PythonJob(function=seperate_warps_list, reference='select_inverse_warp')
        select_inverse_warp.selection = "Inverse"

        calculate_ants_warp.interp = config.interpolation
        calculate_ants_warp.ants_para = config.params

        if config.use_lesion_mask:
            calculate_ants_warp.fixed_image_mask = lesion_preproc(config.lesion_mask_path)
        else:
            calculate_ants_warp.fixed_image_mask = None

        calculate_ants_warp.moving_brain = rp[R('brain')].content
        calculate_ants_warp.reference_brain = config.template_brain

        if config.reg_with_skull:
            calculate_ants_warp.reference_skull = config.template_skull
            calculate_ants_warp.moving_skull = rp[R('T1w', label='reorient')]
        else:
            calculate_ants_warp.moving_skull = rp[R('brain')].content
            calculate_ants_warp.reference_skull = config.template_brain

        # inter-workflow connections
        select_forward_initial.warp_list = calculate_ants_warp.warp_list
        select_forward_rigid.warp_list = calculate_ants_warp.warp_list
        select_forward_affine.warp_list = calculate_ants_warp.warp_list
        select_forward_warp.warp_list = calculate_ants_warp.warp_list
        select_inverse_warp.warp_list = calculate_ants_warp.warp_list

        # connections to outputspec
        #
        # outputspec.ants_initial_xfm = select_forward_initial.selected_warp
        #
        # outputspec.ants_rigid_xfm = select_forward_rigid.selected_warp
        #
        # outputspec.ants_affine_xfm = select_forward_affine.selected_warp
        #
        # outputspec.warp_field = select_forward_warp.selected_warp
        #
        # outputspec.inverse_warp_field = select_inverse_warp.selected_warp

        rp[R('brain', space='MNI')] = calculate_ants_warp.warped_image
