# Modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py

from . import gaussian_diffusion as gd
from . import gaussian_diffusion_cfg as gdcfg
from .respace import SpacedDiffusion, SpacedDiffusionCFG, space_timesteps


def create_diffusion(
    timestep_respacing,
    noise_schedule="linear", 
    use_kl=False,
    sigma_small=False,
    predict_xstart=False,
    learn_sigma=True,
    rescale_learned_sigmas=False,
    diffusion_steps=1000,
    classifier_free_guidance=False,
    p_uncondition=0.1,
    guidance_strength=3,
    t_extra_steps=10
):
    
    if not classifier_free_guidance:
        betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)
        if use_kl:
            loss_type = gd.LossType.RESCALED_KL
        elif rescale_learned_sigmas:
            loss_type = gd.LossType.RESCALED_MSE
        else:
            loss_type = gd.LossType.MSE
    else:
        betas = gdcfg.get_named_beta_schedule(noise_schedule, diffusion_steps)
        if use_kl:
            loss_type = gdcfg.LossType.RESCALED_KL
        elif rescale_learned_sigmas:
            loss_type = gdcfg.LossType.RESCALED_MSE
        else:
            loss_type = gdcfg.LossType.MSE

    if timestep_respacing is None or timestep_respacing == "":
        timestep_respacing = [diffusion_steps]
    
    if not classifier_free_guidance:
        return SpacedDiffusion(
            use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
            betas=betas,
            model_mean_type=(
                gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
            ),
            model_var_type=(
                (
                    gd.ModelVarType.FIXED_LARGE
                    if not sigma_small
                    else gd.ModelVarType.FIXED_SMALL
                )
                if not learn_sigma
                else gd.ModelVarType.LEARNED_RANGE
            ),
            loss_type=loss_type
            # rescale_timesteps=rescale_timesteps,
        )
    else:
        return SpacedDiffusionCFG(
            use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
            betas=betas,
            model_mean_type=(
                gdcfg.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
            ),
            model_var_type=(
                (
                    gdcfg.ModelVarType.FIXED_LARGE
                    if not sigma_small
                    else gdcfg.ModelVarType.FIXED_SMALL
                )
                if not learn_sigma
                else gdcfg.ModelVarType.LEARNED_RANGE
            ),
            loss_type=loss_type,
            # rescale_timesteps=rescale_timesteps,
            p_uncondition=p_uncondition,
            guidance_strength=guidance_strength,
            t_extra_steps=t_extra_steps
        )