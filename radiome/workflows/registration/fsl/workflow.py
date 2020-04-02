from nipype.interfaces import fsl
from radiome.core import workflow, AttrDict
from radiome.core.context import Context
from radiome.core.jobs import NipypeJob
from radiome.core.resource_pool import ResourcePool, ResourceKey as R
# todo bug fix 1) refer to resource more than once 2. move files

@workflow()
def create_workflow(config: AttrDict, resource_pool: ResourcePool, context: Context):
    for _, rp in resource_pool[['brain', 'label-reorient_T1w']]:
        # TODO: disable skullstrip
        linear_reg = NipypeJob(interface=fsl.FLIRT(cost='corratio'), reference='linear_reg_0')
        inv_flirt_xfm = NipypeJob(interface=fsl.utils.ConvertXFM(invert_xfm=True), reference='inv_linear_reg0_xfm')

        linear_reg.in_file = rp[R('brain')]
        linear_reg.reference = config.template_brain
        linear_reg.interp = config.interpolation

        inv_flirt_xfm.in_file = linear_reg.out_matrix_file

        if config.linear_only:
            rp[R('brain', space='MNI')] = linear_reg.out_file
            # other xfm
            return
        else:
            nonlinear_reg = NipypeJob(interface=fsl.FNIRT(fieldcoeff_file=True, jacobian_file=True),
                                      reference='nonlinear_reg_1')
            brain_warp = NipypeJob(interface=fsl.ApplyWarp(), reference='brain_warp')

            nonlinear_reg.in_file = rp[R('T1w', label='reorient')]
            nonlinear_reg.ref_file = config.template_skull
            nonlinear_reg.refmask_file = config.ref_mask
            nonlinear_reg.config_file = config.fnirt_config
            nonlinear_reg.affine_file = linear_reg.out_matrix_file

            brain_warp.interp = config.interpolation
            brain_warp.in_file = rp[R('brain')]
            brain_warp.field_file = nonlinear_reg.fieldcoeff_file
            brain_warp.ref_file = config.template_brain
            rp[R('brain', space='MNI')] = brain_warp.out_file
