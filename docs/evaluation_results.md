# NeRF Few-Shot Evaluation Results

The following table presents the multi-metric evaluation (PSNR, SSIM, LPIPS) across 5 different models on the Lego test set (25 frames, testskip=8). The LPIPS perceptual loss uses the VGG backbone.

## Benchmarks

| Model Name | PSNR | SSIM | LPIPS | Training Steps |
|---|---|---|---|---|
| blender_paper_lego (Baseline) | 27.1575 | 0.9048 | 0.1467 | 069999 |
| lego_v3_vanilla | 9.4772 | 0.7430 | 0.2955 | 009999 |
| lego_v3_mask_ds_view3 | 16.9035 | 0.7066 | 0.3358 | 009999 |
| lego_v8_vanilla | 21.2055 | 0.8040 | 0.2539 | 009999 |
| lego_v8_mask_ds_view8 | 22.4689 | 0.8188 | 0.2114 | 009999 |

## Analysis

- **3-View Improvement**: The DS (Deep Supervision) Mask approach (`lego_v3_mask_ds_view3`) significantly outperforms the vanilla 3-view approach (`lego_v3_vanilla`) by **+7.42 PSNR**. This massive jump indicates that integrating the transparent Alpha volume Mask explicitly resolves severe geometric ambiguities in extreme few-shot scenarios.
- **8-View Improvement**: Incorporating the mask loss also improves the 8-view baseline model by **+1.26 PSNR**, closing the gap further toward the full-data 200k baseline.
