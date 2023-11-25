from huggingface_hub import login
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, AutoencoderKL
import torch
import random
import os

login(token = os.getenv('HF_TOKEN'))

pipe = DiffusionPipeline.from_pretrained(
    "prompthero/openjourney",
    torch_dtype=torch.float16
)
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16).to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.vae = vae
pipe = pipe.to("cuda")
pipe.safety_checker = lambda images, clip_input: (images, False)
# Setting for image generation
prompt = 'Cats, in hipster style outfits, wearing sporty glasses, dynamic pose, dramatic sky, sunset, hyper-detailed, realistic rendering, ultra HD, professional photograph'
num_steps = 20
num_variations = 4
prompt_guidance = 9
dimensions = (1280, 768) # (width, height) tuple
random_seeds = [random.randint(0, 65000) for _ in range(num_variations)]
images = pipe(prompt= num_variations * [prompt],
              num_inference_steps=num_steps,
              guidance_scale=prompt_guidance,
              height = dimensions[0],
              width = dimensions[1],
              generator = [torch.Generator('cuda').manual_seed(i) for i in random_seeds]
             ).images
i = 0
for img in images:
    name = f"test-{i}.jpg"
    images[i].save(name)
    i+=1

