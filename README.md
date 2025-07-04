# PclCDR: Prototype-based Contrastive Learning for Cross-Domain Recommendation



Official codebase for the paper Preference Prototype-Aware Learning for Universal Cross-Domain Recommendation.



## Overview

**Abstract:** Cross-domain recommendation (CDR) leverages data from both source and target domains to extract user preference features, thereby mitigating the cold-start problem. Existing CDR methods predominantly employ contrastive learningto align user and item features across domains and employ shared embedding spaces to facilitate cross-domain knowledge transfer. However, these approaches often encounter negative transfer, where knowledge acquired from the source domain adversely affects recommendations in the target domain. Two primary causes of negative transfer have been identified: (1) \textbf{Noise Interference}, in which irrelevant information disrupts the learning of user preference features; (2) \textbf{Preference Feature Bias}, where the extracted features fail to accurately capture users' true preferences. To overcome these challenges, we propose PclCDR, a cross-domain recommendation model that introduces two novel strategies within the contrastive learning framework: (1) \textbf{Global Denoising}, which removes irrelevant information by comparing enhanced and denoised views to ensure the accurate extraction of preference features; (2) \textbf{Prototype-guided Feature Selection}, which partitions preference features into representative prototypes, classifies them as positive or negative based on their similarity to users, and employs contrastive learning to align users with more relevant prototype features. Extensive experiments conducted on four benchmark CDR datasets demonstrate that PclCDR significantly outperforms state-of-the-art baselines, underscoring the effectiveness of the proposed model and its components.

## Datasets

We use the datasets provided by [UniCDR](https://github.com/cjx96/UniCDR)



## Usage

Running example:


# sport_cloth
CUDA_VISIBLE_DEVICES=0  python -u train_rec.py --cuda -domains sport_cloth --aggregator Transformer --lambda_pp 0.2


# game_video
CUDA_VISIBLE_DEVICES=0  python -u train_rec.py --cuda -domains game_video --aggregator Transformer --lambda_pp 0.2  --ssl_reg_game 100



