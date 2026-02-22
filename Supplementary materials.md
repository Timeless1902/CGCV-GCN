# Supplementary Materials

The following table summarizes the layer configurations and tensor transformations of CGCV-GCN.

|         Layer          |                   Input Shape                   |   Output Shape   |
| ---------------------- | ----------------------------------------------- | ---------------- |
| Complex Mapping (PEM)  | [N, 3, 48, 16]                                  | [N, 3, 48, 16]   |
| Complex Mapping (KEM)  | [N, 8, 48, 16]                                  | [N, 8, 48, 16]   |
| CST-GCN (PEM_1)        | [N, 3, 48, 16]                                  | [N, 64, 48, 16]  |
| CST-GCN (KEM_1)        | [N, 8, 48, 16]                                  | [N, 64, 48, 16]  |
| RTCN (AEM_1)           | [N, 31, 48]                                     | [N, 64, 48]      |
| CST-GCN (PEM_2)        | [N, 64, 48, 16]                                 | [N, 64, 48, 16]  |
| CST-GCN (PEM_3)        | [N, 64, 48, 16]                                 | [N, 64, 48, 16]  |
| CST-GCN (PEM_4)        | [N, 64, 48, 16]                                 | [N, 64, 48, 16]  |
| CST-GCN (KEM_2)        | [N, 64, 48, 16]                                 | [N, 64, 48, 16]  |
| CST-GCN (KEM_3)        | [N, 64, 48, 16]                                 | [N, 64, 48, 16]  |
| CST-GCN (KEM_4)        | [N, 64, 48, 16]                                 | [N, 64, 48, 16]  |
| RTCN (AEM_2)           | [N, 64, 48]                                     | [N, 64, 48]      |
| CGFM_pa_1              | [N, 64, 48, 16] & [N, 64, 48] & None            | [N, 64, 48]      |
| CGFM_ka_1              | [N, 64, 48, 16] & [N, 64, 48] & None            | [N, 64, 48]      |
| CST-GCN (PEM_5)        | [N, 64, 48, 16]                                 | [N, 128, 24, 16] |
| CST-GCN (PEM_6)        | [N, 128, 24, 16]                                | [N, 128, 24, 16] |
| CST-GCN (PEM_7)        | [N, 128, 24, 16]                                | [N, 128, 24, 16] |
| CST-GCN (KEM_5)        | [N, 64, 48, 16]                                 | [N, 128, 24, 16] |
| CST-GCN (KEM_6)        | [N, 128, 24, 16]                                | [N, 128, 24, 16] |
| CST-GCN (KEM_7)        | [N, 128, 24, 16]                                | [N, 128, 24, 16] |
| RTCN (AEM_3)           | [N, 64, 48]                                     | [N, 128, 24]     |
| CGFM_pa_2              | [N, 128, 24, 16] & [N, 128, 24] & [N, 64, 48]   | [N, 128, 24]     |
| CGFM_ka_2              | [N, 128, 24, 16] & [N, 128, 24] & [N, 64, 48]   | [N, 128, 24]     |
| CST-GCN (PEM_8)        | [N, 128, 24, 16]                                | [N, 256, 12, 16] |
| CST-GCN (PEM_9)        | [N, 256, 12, 16]                                | [N, 256, 12, 16] |
| CST-GCN (PEM_10)       | [N, 256, 12, 16]                                | [N, 256, 12, 16] |
| CST-GCN (KEM_8)        | [N, 128, 24, 16]                                | [N, 256, 12, 16] |
| CST-GCN (KEM_9)        | [N, 256, 12, 16]                                | [N, 256, 12, 16] |
| CST-GCN (KEM_10)       | [N, 256, 12, 16]                                | [N, 256, 12, 16] |
| Real Mapping (PEM)     | [N, 256, 12, 16]                                | [N, 256, 12, 16] |
| Real Mapping (KEM)     | [N, 256, 12, 16]                                | [N, 256, 12, 16] |
| CGFM_pa_3              | [N, 256, 12, 16] & [N, 256, 12] & [N, 128, 24]  | [N, 256, 12]     |
| CGFM_ka_3              | [N, 256, 12, 16] & [N, 256, 12] & [N, 128, 24]  | [N, 256, 12]     |

> **Note:** For exhaustive specifications regarding kernel sizes, strides, and specific hyperparameters, please examine the [Source Code] directly.
