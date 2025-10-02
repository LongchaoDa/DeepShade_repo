import torch
import cv2
import json
import numpy as np
import sys
sys.path.append("/scratch/YOURNAME/project/plantShade/ControlNet")
from cldm.model import create_model, load_state_dict
from ldm.models.diffusion.ddim import DDIMSampler

# Model and Configurations
# resume_path = '/scratch/YOURNAME/project/plantShade/ControlNet/models/epoch_51_step_1351.ckpt'
resume_path = "/scratch/YOURNAME/project/plantShade/ControlNet/0out/ControlNet_vanilla_Plant/2025-09-30_23-57-05/periodic/epoch-epoch=99.ckpt"
model = create_model('/scratch/YOURNAME/project/plantShade/ControlNet/models/cldm_v21.yaml').to('cuda')  # Move model to GPU
model.load_state_dict(load_state_dict(resume_path, location='cuda:0'))  # Load weights on GPU

# Set model parameters as in training
model.eval()  # Set model to evaluation mode
model.learning_rate = 1e-5
model.sd_locked = True
model.only_mid_control = False

# Set control scales if needed
model.control_scales = [1.0] * 13  # Adjust these values if necessary

# Move the conditioning stage model to the same device
model.cond_stage_model.to('cuda')

# Create DDIMSampler
sampler = DDIMSampler(model)

# Load Dataset
class MyDataset:
    def __init__(self):
        self.data = []
        with open('/scratch/YOURNAME/project/plantShade/evaluation/output_10.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def get_item(self, idx):
        item = self.data[idx]
        source_filename = item['source']
        prompt = item['prompt']

        source = cv2.imread(source_filename)
        source = cv2.resize(source, (512, 512), interpolation=cv2.INTER_AREA)
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        source = source.astype(np.float32) / 255.0

        return source, prompt

dataset = MyDataset()

# Single-image Inference Function
def generate_image(idx, custom_prompt=None):
    # Retrieve the image and prompt
    source, prompt = dataset.get_item(idx)
    prompt = custom_prompt if custom_prompt else prompt

    # Ensure all components are on the same device
    device = torch.device('cuda')

    # Convert source to tensor and move to model's device
    source_tensor = torch.tensor(source).permute(2, 0, 1).unsqueeze(0).to(device)

    # Encode the prompt for cross-attention
    prompt_embedding = model.get_learned_conditioning([prompt])

    # Prepare the conditioning dictionary
    cond = {
        "c_concat": [source_tensor],
        "c_crossattn": [prompt_embedding]
    }

    # Prepare unconditional conditioning for classifier-free guidance
    unconditional_prompt = ""
    unconditional_embedding = model.get_learned_conditioning([unconditional_prompt])
    uncond = {
        "c_concat": [source_tensor],  # Keep the same source_tensor
        "c_crossattn": [unconditional_embedding]
    }

    # Prepare the shape of the latent vector
    latent_shape = (1, model.channels, source_tensor.shape[2] // 8, source_tensor.shape[3] // 8)

    # Generate a random noise latent vector
    x = torch.randn(latent_shape, device=device)

    # Set the number of inference steps and guidance scale
    ddim_steps = 50
    guidance_scale = 9.0  # Adjust as needed (commonly between 5.0 and 12.0)

    # Perform sampling
    with torch.no_grad():
        samples, _ = sampler.sample(
            S=ddim_steps,
            conditioning=cond,
            batch_size=1,
            shape=latent_shape[1:],
            verbose=False,
            unconditional_guidance_scale=guidance_scale,
            unconditional_conditioning=uncond,
            eta=0.0
        )

    # Decode the samples to get images
    generated_image = model.decode_first_stage(samples)

    # Post-process the output and convert it to image format
    generated_image = generated_image[0].permute(1, 2, 0).cpu().numpy()
    generated_image = (generated_image * 255.0).clip(0, 255).astype(np.uint8)

    return generated_image

# Example usage:
index = 1  # Image index in dataset
# case1
# custom_prompt = "solar_declincation: -20.722542433072444, angle: 0.0, time_of_day:6"

# case2
# custom_prompt = "solar_declincation: -20.722542433072444, angle: -15.0, time_of_day:7"

# case3
# custom_prompt = "solar_declincation: -20.722542433072444, angle: -30.0, time_of_day:8"

# case4
# custom_prompt = "solar_declincation: -20.722542433072444, angle: -45.0, time_of_day:9"

# # case5
# custom_prompt = "solar_declincation: -20.722542433072444, angle: -60.0, time_of_day:10"

# case6
# custom_prompt = "solar_declincation: -20.722542433072444, angle: -75.0, time_of_day:11"

# case7
# custom_prompt = "solar_declincation: -20.722542433072444, angle: -90.0, time_of_day:12"


# case8
# custom_prompt = "solar_declincation: -20.722542433072444, angle: -105.0, time_of_day:13"


# case9
# custom_prompt = "solar_declincation: -20.722542433072444, angle: -120.0, time_of_day:14"

# case10
# custom_prompt = "solar_declincation: -20.722542433072444, angle: -135.0, time_of_day:15"

# case11
# custom_prompt = "solar_declincation: -20.722542433072444, angle: -150.0, time_of_day:16"

# case12
# custom_prompt = "solar_declincation: -20.722542433072444, angle: -165.0, time_of_day:17"


# case13
# custom_prompt = "solar_declincation: -20.722542433072444, angle: -180.0, time_of_day:18"



result_image = generate_image(index, custom_prompt)


folder_save = "/scratch/YOURNAME/project/plantShade/evaluation/plantTopV0/epoch100/"
cv2.imwrite(folder_save + "generated_result_case10.png", cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))




