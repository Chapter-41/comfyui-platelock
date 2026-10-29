"""Local ComfyUI backends. Never downloads models or submits external jobs."""
from pathlib import Path

LTX_MODELS = {
    'diffusion_models': 'ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors',
    'vae': 'ltx-2.5-video-vae-bf16.safetensors',
    'text_encoders': 'gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors',
}
AUDIO_VAE = 'ltx-2.5-audio-vae-bf16.safetensors'


def values(out):
    if isinstance(out, dict):
        return out['result']
    return getattr(out, 'args', None) or getattr(out, 'result', None) or out


def load_rife(name='auto'):
    import folder_paths
    from comfy_extras.nodes_frame_interpolation import FrameInterpolationModelLoader
    names = folder_paths.get_filename_list('frame_interpolation')
    if name == 'auto':
        name = next((n for n in sorted(names) if 'rife' in n.lower()), None)
    if name not in names:
        raise ValueError('No local RIFE model found. Install a RIFE checkpoint in models/frame_interpolation or connect interp_model.')
    return values(FrameInterpolationModelLoader.execute(name))[0]


def preflight_ltx():
    import folder_paths
    missing = [v for k,v in LTX_MODELS.items() if not folder_paths.get_full_path(k,v)]
    if not folder_paths.get_full_path('vae', AUDIO_VAE):
        missing.append(AUDIO_VAE)
    if missing:
        raise ValueError('Integrated LTX 2.5 needs these local models: '+', '.join(missing))


class LocalLTX:
    """Equivalent to the bundled LTX fill graph, invoked for each missing window."""
    def __init__(self, prompt, seed=0):
        preflight_ltx()
        self.prompt, self.seed = prompt, seed
        self.cache = {}

    def call(self, name, **kw):
        import nodes
        cls = nodes.NODE_CLASS_MAPPINGS.get(name)
        if cls is None:
            raise ValueError(f'Integrated LTX requires ComfyUI node {name}. Update the local ComfyUI installation.')
        # Comfy's V3 classes expose execute; V1 classes declare FUNCTION.
        instance = cls()
        fn = getattr(instance, 'execute', None) or getattr(instance, cls.FUNCTION)
        return values(fn(**kw))

    def load(self):
        if self.cache:
            return self.cache
        self.cache = dict(
            model=self.call('UNETLoader',unet_name=LTX_MODELS['diffusion_models'],weight_dtype='default')[0],
            vae=self.call('VAELoader',vae_name=LTX_MODELS['vae'])[0],
            audio=self.call('VAELoader',vae_name=AUDIO_VAE)[0],
            clip=self.call('CLIPLoader',clip_name=LTX_MODELS['text_encoders'],type='ltxv',device='default')[0])
        return self.cache

    def __call__(self, context, width, height, length, fps):
        c=self.load()
        pos=self.call('CLIPTextEncode',clip=c['clip'],text=self.prompt)[0]
        neg=self.call('CLIPTextEncode',clip=c['clip'],text='cut, scene change, morphing, distorted faces')[0]
        pos,neg=self.call('LTXVConditioning',positive=pos,negative=neg,frame_rate=fps)
        latent=self.call('EmptyLTXVLatentVideo',width=width,height=height,length=length,batch_size=1)[0]
        guide=self.call('LTXVPreprocess',image=context,img_compression=18)[0]
        pos,neg,latent=self.call('LTXVAddGuide',positive=pos,negative=neg,vae=c['vae'],latent=latent,image=guide,frame_idx=0,strength=1.)
        audio=self.call('LTXVEmptyLatentAudio',audio_vae=c['audio'],frames_number=length,frame_rate=fps,batch_size=1)[0]
        av=self.call('LTXVConcatAVLatent',video_latent=latent,audio_latent=audio)[0]
        guider=self.call('CFGGuider',model=c['model'],positive=pos,negative=neg,cfg=1.)[0]
        sampler=self.call('KSamplerSelect',sampler_name='euler_ancestral')[0]
        sigmas=self.call('ManualSigmas',sigmas='1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0')[0]
        noise=self.call('RandomNoise',noise_seed=self.seed)[0]
        av=self.call('SamplerCustomAdvanced',noise=noise,guider=guider,sampler=sampler,sigmas=sigmas,latent_image=av)[0]
        latent=self.call('LTXVSeparateAVLatent',av_latent=av)[0]
        latent=self.call('LTXVCropGuides',positive=pos,negative=neg,latent=latent)[2]
        return self.call('VAEDecodeTiled',samples=latent,vae=c['vae'],tile_size=512,overlap=64,temporal_size=64,temporal_overlap=8)[0]


def compatible_interpolator(model):
    """Use an isolated CPU copy on Apple MPS, whose grid_sample border mode fails.

    Do not change the user's supplied model or global ComfyUI device settings.
    """
    if getattr(getattr(model, 'load_device', None), 'type', None) != 'mps':
        return model
    import copy
    import logging
    import torch
    from comfy.model_patcher import CoreModelPatcher
    logging.info('PlateLock: running interpolation on CPU; MPS border sampling is unsupported.')
    cpu_model = copy.deepcopy(model.model).to(device='cpu', dtype=torch.float32)
    return CoreModelPatcher(cpu_model, load_device=torch.device('cpu'), offload_device=torch.device('cpu'))
