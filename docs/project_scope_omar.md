# Project Scope (Supervisor: Omar)

## Title
AI-Driven Automated Treatment Planning for MRgRT Pancreas Radiotherapy Using DoseGNN and DoseGAN

## Clinical Context & Problem Statement
Pancreatic tumors pose a challenge in MR-guided radiotherapy (MRgRT) due to organ motion, anatomical variability, and proximity of critical structures. Manual plan adaptation for each fraction is labor-intensive and limits throughput. AI models like DoseGNN and DoseGAN can offer automated, patient-specific dose predictions directly from MR anatomy, enabling efficient daily adaptive planning in MRgRT for pancreatic cancer.

## Methodology & Approach
Two deep learning pipelines are developed and compared:
- **DoseGNN**: geometric-aware dose predictor — constructs bipartite graphs linking anatomical regions from MR scans to dose prediction nodes using geometric features
- **DoseGAN**: heterogeneity-aware GAN — uses adversarial training to produce realistic dose maps from anatomical inputs

Both models are trained on annotated MRgRT prostate data and evaluated on dose accuracy and plan deliverability.

## Dataset
- **Cohort**: 200 pancreatic cancer patients treated with MRgRT between 2019–2024
- **Data**: daily MR scans, segmented targets and OARs (duodenum, stomach, bowel), clinical dose plans
- **Preprocessing**: motion compensation, MR normalisation, DVH metric extraction

## Objectives
1. Train DoseGNN and DoseGAN on MRgRT pancreas datasets
2. Evaluate spatial dose accuracy and clinical constraint satisfaction
3. Demonstrate plan deliverability and time-to-plan on daily anatomy
4. Assess model robustness to anatomical variations (bowel filling, target shifts)

## Expected Outcome
An automated system for generating clinically viable pancreatic MRgRT plans from daily MR inputs. Validates graph and generative models under challenging abdominal conditions, supporting workflow automation and improving access to adaptive radiotherapy.
