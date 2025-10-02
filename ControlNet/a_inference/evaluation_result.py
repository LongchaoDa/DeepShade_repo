import torch
import cv2
import json
import numpy as np
import os
import sys
import math
from tqdm import tqdm
from lpips import LPIPS
sys.path.append("/scratch/YOURNAME/project/ControlNet/")
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error as mse
from cldm.model import create_model, load_state_dict
from ldm.models.diffusion.ddim import DDIMSampler
from utils import get_statistics

# LPIPS 初始化
lpips_fn = LPIPS(net='alex').to('cuda')

# Model & config
resume_path = '/scratch/YOURNAME MODEL PATH'
config_path = '/scratch/YOURNAME/project/ControlNet/models/cldm_v21.yaml'
model = create_model(config_path).to('cuda')
model.load_state_dict(load_state_dict(resume_path, location='cuda:0'))
model.eval()
model.learning_rate    = 1e-4
model.sd_locked        = True
model.only_mid_control = False
model.control_scales   = [1.0] * 13
model.cond_stage_model.to('cuda')
sampler = DDIMSampler(model)

# 城市列表
city_list = [
    "Aswan",
    "Beijing",
    "Brasilia",
    "Cairo",
    "Jaipur",
    "Madrid",
    "Mumbai",
    "Phoenix",
    "SauPaulo",
    "Seville",
    "Tempe",
    "XiAn",
]

def folder_contains_png(folder):
    return any(f.endswith('.png') for f in os.listdir(folder))

def compute_miou(mask1, mask2):
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return inter/union if union>0 else 0.0

def compute_boundary_iou(mask1, mask2, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    b1 = cv2.dilate(mask1.astype(np.uint8), kernel) - cv2.erode(mask1.astype(np.uint8), kernel)
    b2 = cv2.dilate(mask2.astype(np.uint8), kernel) - cv2.erode(mask2.astype(np.uint8), kernel)
    inter = np.logical_and(b1, b2).sum()
    union = np.logical_or(b1, b2).sum()
    return inter/union if union>0 else 0.0

def generate_image_from_path(image_path, prompt, device='cuda', ddim_steps=50, guidance_scale=9.0):
    src = cv2.imread(image_path)
    if src is None:
        raise FileNotFoundError(f"{image_path} not found")
    src = cv2.resize(src,(512,512)); src = cv2.cvtColor(src,cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    gray = cv2.cvtColor((src*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    canny = cv2.Canny(gray,100,200).astype(np.float32)/255.0
    inp = np.concatenate([src, canny[...,None]], axis=2)
    x_in = torch.from_numpy(inp).permute(2,0,1).unsqueeze(0).to(device)

    emb = model.get_learned_conditioning([prompt])
    cond = {"c_concat":[x_in], "c_crossattn":[emb]}
    unemb = model.get_learned_conditioning([""])
    uncond = {"c_concat":[x_in], "c_crossattn":[unemb]}

    latent_shape = (1, model.channels, 512//8, 512//8)
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
    img = model.decode_first_stage(samples)[0].permute(1,2,0).cpu().numpy()
    return (img*255).clip(0,255).astype(np.uint8)

for city in city_list:
    folder_save = f"/scratch/YOURNAME/project/ControlNet_official_data/IJCAI_release/ControlNet_vanilla_Tempe/result/{city}"
    test_file   = f"/scratch/YOURNAME/project/ControlNet_official_data/dataset/{city}/test.json"
    os.makedirs(folder_save, exist_ok=True)

    # 结果收集
    results = []
    skip_generation = folder_contains_png(folder_save)
    print(f"[{city}] Skip generation: {skip_generation}")

    # 用列表收集、忽略 nan
    ssim_list = []; mse_list = []; miou_list = []; biou_list = []; lpips_list = []

    # 读取 test.json
    with open(test_file,'r') as f:
        data = json.load(f)

    for item in tqdm(data, desc=f"{city} items", mininterval=0.2):
        src_path = item["source"]
        tgt_path = item["target"]
        prompt   = item["prompt"]
        out_fn   = os.path.basename(tgt_path).replace("target","generated")
        out_fp   = os.path.join(folder_save, out_fn)

        # 生成或加载
        if skip_generation and os.path.exists(out_fp):
            gen = cv2.imread(out_fp); gen = cv2.cvtColor(gen,cv2.COLOR_BGR2RGB)
        else:
            gen = generate_image_from_path(src_path, prompt)
            cv2.imwrite(out_fp, cv2.cvtColor(gen,cv2.COLOR_RGB2BGR))

        # 读取 GT
        gt = cv2.imread(tgt_path)
        gt = cv2.resize(gt,(512,512)); gt = cv2.cvtColor(gt,cv2.COLOR_BGR2RGB)

        # 灰度 & 阴影 mask
        ggray = cv2.cvtColor(gen, cv2.COLOR_RGB2GRAY)
        tgray = cv2.cvtColor(gt,  cv2.COLOR_RGB2GRAY)
        mask  = (ggray<40)&(tgray<40)

        if mask.any():
            # full masked 2D
            mgen = np.zeros_like(ggray); mgen[mask]=ggray[mask]
            mgt  = np.zeros_like(tgray); mgt[mask]=tgray[mask]

            # SSIM
            try:
                val_ssim = ssim(
                    mgen, mgt,
                    win_size=3,
                    data_range=mgen.max()-mgen.min()
                )
            except Exception:
                val_ssim = float('nan')

            # MSE
            val_mse  = float(mse(ggray[mask], tgray[mask]))

            # mIoU / B-IoU
            bin_g = ggray<40; bin_t = tgray<40
            val_miou = compute_miou(bin_g, bin_t)
            val_biou = compute_boundary_iou(bin_g, bin_t)

            # LPIPS
            gt_t = torch.from_numpy(gt).permute(2,0,1).unsqueeze(0).float()/255.0
            gen_t= torch.from_numpy(gen).permute(2,0,1).unsqueeze(0).float()/255.0
            val_lpips = lpips_fn(gen_t.to('cuda'), gt_t.to('cuda')).item()

            # 仅在非 nan 时加入各自列表
            if not math.isnan(val_ssim): ssim_list.append(val_ssim)
            if not math.isnan(val_mse):  mse_list.append(val_mse)
            if not math.isnan(val_miou): miou_list.append(val_miou)
            if not math.isnan(val_biou): biou_list.append(val_biou)
            if not math.isnan(val_lpips):lpips_list.append(val_lpips)

            results.append({
                "generated_path": out_fp,
                "target_gt_path": tgt_path,
                "SSIM": val_ssim,
                "MSE": val_mse,
                "mIoU": val_miou,
                "B-IoU": val_biou,
                "LPIPS": val_lpips
            })
        else:
            print(f"No shade regions in {out_fn}")

    # 逐文件结果
    with open(os.path.join(folder_save, "result.json"), 'w') as f:
        json.dump(results, f, indent=4)
    print(f"[{city}] result.json saved")

    # 平均结果（忽略 nan）
    ave = {
        "Average SSIM":  np.mean(ssim_list)  if len(ssim_list)>0 else 0.0,
        "Average MSE":   np.mean(mse_list)   if len(mse_list)>0  else 0.0,
        "Average mIoU":  np.mean(miou_list)  if len(miou_list)>0 else 0.0,
        "Average B-IoU": np.mean(biou_list)  if len(biou_list)>0 else 0.0,
        "Average LPIPS": np.mean(lpips_list) if len(lpips_list)>0 else 0.0,
    }
    with open(os.path.join(folder_save, "ave_result.json"), 'w') as f:
        json.dump(ave, f, indent=4)
    print(f"[{city}] ave_result.json saved")

    # 调用分析脚本
    stats_dir = os.path.join(folder_save, "analysis_path")
    os.makedirs(stats_dir, exist_ok=True)
    get_statistics(
        file_path = os.path.join(folder_save, "result.json"),
        output_plot_dir = stats_dir
    )


# /scratch/YOURNAME/project/ControlNet_official_data/IJCAI_release/ControlNet_vanilla_Tempe/2025-07-14_04-04-27/periodic/epoch-epoch=29.ckpt

# nohup python /scratch/YOURNAME/project/ControlNet/a_inference/evaluate_grey.py     > /scratch/YOURNAME/project/ControlNet_official_data/testingLogs/vanillaControl.log 2>&1 &

# [1] 822688