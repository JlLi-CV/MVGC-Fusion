## Multi-View Gradient Consistency and Spectral Backtracking for Hyperspectral Image Fusion

![Figure_1](README/Figure_1.png)

#### Abstract

Hyperspectral and multispectral image fusion aims to integrate the rich spectral information of hyperspectral images with the high spatial resolution of multispectral images to reconstruct high-resolution hyperspectral data. However, existing methods often struggle to simultaneously enhance spatial structures while preserving spectral consistency, particularly in scenarios with numerous spectral bands and significant local spectral heterogeneity, which can lead to spatial detail degradation or spectral distortion.
To address these challenges, this paper proposes a fusion method collaboratively driven by multi-view gradient consistency and spectral backtracking. A multi-view gradient consistency network is constructed to learn the nonlinear gradient mapping between the fused image and the high-resolution multispectral image under both global and local sub-view perspectives, guiding spatial structure reconstruction and improving detail fidelity. In addition, a spectral backtracking unit is designed to dynamically evaluate spectral states at different fusion stages based on spectral angle mapping, and selectively applies backtracking constraints when spectral degradation is detected, effectively mitigating spectral distortion caused by spatial gradient enhancement.
Experimental results on multiple public datasets demonstrate that the proposed method outperforms several state-of-the-art approaches in both qualitative and quantitative metrics, while maintaining high efficiency in terms of parameter size and computational complexity.

`Currently, only a portion of the code has been released. The full codebase will be uploaded after the paper is formally accepted.`