import { useState, useEffect, useRef } from 'react'
import { useSearchParams, useNavigate } from 'react-router-dom'
import { Brain, Cpu, Sparkles, ArrowRight, Play, ChevronDown, RotateCcw, MessageSquare, Server, Cloud, CheckCircle, X, Loader2, StopCircle } from 'lucide-react'

// ─── Training Types ───────────────────────────────────────────────────────────
export type TrainingType = 'classification' | 'detection' | 'segmentation' | 'vlm-finetune' | 'self-supervised' | 'llm-text'
export type DataType    = 'rgb' | 'thermal' | 'xray' | 'microscopy' | 'lidar' | 'general'
export type TargetType  = 'finetune' | 'export'

export interface TrainOption {
  value: string
  label: string
  engine: string
  model: string
  hardware: string
  description: string
  compatible: boolean
  zeroShot?: boolean
  // Optional metadata for the Deploy-Only catalog (see DEPLOY_CATALOG)
  dataTypes?: DataType[]
  source?: string
  paper?: string
}

interface TrainingConfig {
  trainingType: TrainingType
  dataType: DataType
  target: TargetType
  datasetId: number | null
  epochs: number
  imgsz: number
  batchSize: number
  learningRate: number
  optimizer: string
  engine: string
  model: string
  notes: string
  // LLM/VLM-specific
  loraRank: number
  quantization: string
  maxSeqLen: number
  chatTemplate: string
  gradAccum: number
  textDatasetId: string
  cluster: 'ray' | 'modal'
}

// ─── Training Matrix ───────────────────────────────────────────────────────────
export const TRAINING_MATRIX: Record<string, TrainOption[]> = {
  // ── Classification ──────────────────────────────────────────────────────────
  classification: [
    { value: 'eff-b2',     label: 'EfficientNet-B2',              engine: 'PyTorch',     model: 'efficientnet-b2',               hardware: 'GPU 8GB+',     description: 'Balanced accuracy & speed — defect grading, product quality classification',   compatible: true  },
    { value: 'eff-b4',     label: 'EfficientNet-B4',              engine: 'PyTorch',     model: 'efficientnet-b4',               hardware: 'GPU 12GB+',    description: 'Higher accuracy — fine-grained surface defect or part classification',        compatible: false },
    { value: 'mobilenet',  label: 'MobileNetV3-Small',            engine: 'PyTorch+TIMM', model: 'mobilenetv3_small_100',        hardware: 'CPU / Edge',   description: 'Lightweight — deploy บน Raspberry Pi, Jetson Nano, industrial edge device',   compatible: true  },
    { value: 'res50',      label: 'ResNet-50',                    engine: 'TorchVision', model: 'resnet50',                      hardware: 'GPU 6GB+',     description: 'Classic baseline — เสถียร, ปรับง่าย, รองรับ transfer learning ทุก domain',   compatible: true  },
    { value: 'convnext',   label: 'ConvNeXt-Tiny',               engine: 'PyTorch',     model: 'convnext_tiny',                 hardware: 'GPU 8GB+',     description: 'Modern CNN — strong baseline สำหรับ visual inspection, PCB, textile',        compatible: true  },
    { value: 'regnet',     label: 'RegNet-Y400M',                 engine: 'TorchVision', model: 'regnet_y_400mf',                hardware: 'GPU 6GB+',     description: 'Efficient for large-scale classification — automotive parts, warehouse SKU',  compatible: true  },
    { value: 'swin-t',     label: 'Swin-Tiny',                   engine: 'TIMM',        model: 'swin_tiny_patch4_window7_224',  hardware: 'GPU 8GB+',     description: 'Transformer-based — ดีสำหรับ texture-rich images เช่น fabric, metal surface', compatible: true  },
    { value: 'vit-b',      label: 'ViT-Base (Vision Transformer)', engine: 'TIMM',      model: 'vit_base_patch16_224',          hardware: 'GPU 12GB+',    description: 'Pure transformer — เหมาะกับ dataset ขนาดใหญ่, multi-class defect',          compatible: false },
    { value: 'dino-s',     label: 'DINOv2-Small',                engine: 'TIMM',        model: 'vit_small_patch14_dinov2',      hardware: 'GPU 8GB+',     description: 'Self-supervised feature backbone — few-shot classification, anomaly detection', compatible: true },
    { value: 'efficientvit', label: 'EfficientViT-M5',           engine: 'TIMM',        model: 'efficientvit_m5',               hardware: 'GPU 6GB+',     description: 'Ultra-fast edge transformer — real-time line inspection, <1ms latency',       compatible: true  },
    // ── Medical pre-trained ─────────────────────────────────────────────────────
    { value: 'rad-dino',   label: 'RAD-DINO (Radiology)',        engine: 'HuggingFace', model: 'microsoft/rad-dino',            hardware: 'GPU 8GB+',     description: 'Pre-trained บน 1M+ chest X-ray (CheXpert, MIMIC) — fine-tune ได้สำหรับ X-ray classification', compatible: true  },
    { value: 'txrv-densenet', label: 'CheXNet DenseNet121',      engine: 'PyTorch',     model: 'densenet121-res224-all',        hardware: 'GPU 6GB+',     description: 'Pre-trained บน 800k+ chest X-ray (TorchXRayVision) — pneumonia, effusion, nodule', compatible: true  },
    { value: 'uni-pathology', label: 'UNI (Pathology ViT-L)',    engine: 'HuggingFace', model: 'MahmoodLab/UNI',               hardware: 'GPU 16GB+',    description: 'Pre-trained บน 100k+ pathology slides (TCGA) — tissue classification, cancer grading', compatible: false },
    { value: 'monai-densenet', label: 'MONAI DenseNet121',        engine: 'MONAI',       model: 'monai-densenet121',             hardware: 'GPU 8GB+',     description: 'MONAI medical classification — CT/MRI/X-ray, auto-handle DICOM & NIfTI format',  compatible: true  },
  ],

  // ── Object Detection ─────────────────────────────────────────────────────────
  detection: [
    { value: 'yolov8-s',   label: 'YOLOv8-Small',               engine: 'Ultralytics', model: 'yolov8s.pt',                    hardware: 'GPU 6GB+',     description: 'Real-time detection — assembly line inspection, package detection',           compatible: true  },
    { value: 'yolov8-m',   label: 'YOLOv8-Medium',              engine: 'Ultralytics', model: 'yolov8m.pt',                    hardware: 'GPU 10GB+',    description: 'Balanced — surface scratch, weld defect, component presence detection',       compatible: true  },
    { value: 'yolov8-l',   label: 'YOLOv8-Large',               engine: 'Ultralytics', model: 'yolov8l.pt',                    hardware: 'GPU 16GB+',    description: 'High recall — critical defect detection ที่ต้องการ miss rate ต่ำมาก',       compatible: false },
    { value: 'yolonano',   label: 'YOLOv8-Nano',                engine: 'Ultralytics', model: 'yolov8n.pt',                    hardware: 'CPU / Edge',   description: 'Edge-optimized — Jetson Orin, Hailo, industrial smart camera',               compatible: true  },
    { value: 'yolov9-c',   label: 'YOLOv9-C',                   engine: 'Ultralytics', model: 'yolov9c.pt',                    hardware: 'GPU 12GB+',    description: 'Programmable gradient — better small defect detection vs YOLOv8',            compatible: false },
    { value: 'rt-detr',    label: 'RT-DETR-L',                  engine: 'Ultralytics', model: 'rtdetr-l.pt',                   hardware: 'GPU 12GB+',    description: 'Transformer detection — anchor-free, ดีสำหรับ dense object counting',        compatible: false },
    { value: 'detr',       label: 'DETR-ResNet50',              engine: 'HuggingFace', model: 'facebook/detr-resnet-50',         hardware: 'GPU 10GB+',    description: 'End-to-end transformer detection — robotic picking, warehouse automation',    compatible: false },
  ],

  // ── Segmentation ─────────────────────────────────────────────────────────────
  segmentation: [
    { value: 'unet',       label: 'UNet ResNet34',              engine: 'Segmentation Models PyTorch', model: 'resnet34-unet', hardware: 'GPU 8GB+', description: 'Classic — surface defect, corrosion, weld bead segmentation',               compatible: true  },
    { value: 'unetpp',     label: 'UNet++ (ResNet34)',          engine: 'Segmentation Models PyTorch', model: 'resnet34-unetplusplus', hardware: 'GPU 10GB+', description: 'Nested UNet — ดีสำหรับ boundary ซับซ้อน เช่น crack, delamination',      compatible: true  },
    { value: 'deeplabv3',  label: 'DeepLabV3+ (ResNet101)',     engine: 'TorchVision', model: 'resnet101-deeplabv3',            hardware: 'GPU 10GB+',    description: 'Atrous conv — semantic segmentation บน aerial, industrial floor plan',       compatible: true  },
    { value: 'maskrcnn',   label: 'Mask R-CNN',                 engine: 'TorchVision', model: 'maskrcnn_resnet50_fpn',          hardware: 'GPU 12GB+',    description: 'Instance segmentation — individual part isolation, robotic grasping',         compatible: false },
    { value: 'yoloseg',    label: 'YOLOv8-Seg',                engine: 'Ultralytics', model: 'yolov8m-seg.pt',                hardware: 'GPU 10GB+',    description: 'Fast instance seg — real-time defect segmentation on production line',        compatible: true  },
    { value: 'monai-unet', label: 'MONAI UNet (Medical)',        engine: 'MONAI',       model: 'monai-unet',                    hardware: 'GPU 8GB+',     description: 'MONAI UNet — organ/tumor segmentation, CT/MRI, handles DICOM & NIfTI automatically', compatible: true  },
    { value: 'medsam',     label: 'MedSAM (Fine-tune)',          engine: 'MedSAM',      model: 'medsam_vit_b',                  hardware: 'GPU 12GB+',    description: 'SAM fine-tuned บน medical images — segment อวัยวะ/tumor ด้วย bounding box prompt', compatible: false },
    { value: 'nnunet-2d',  label: 'nnU-Net 2D Auto',            engine: 'nnU-Net',     model: 'nnunet-2d',                     hardware: 'GPU 16GB+',    description: 'Auto-configure segmentation pipeline — MICCAI standard, organ/tumor, CT/MRI', compatible: false },
  ],

  // ── LLM Text Fine-tuning ─────────────────────────────────────────────────────
  'llm-text': [
    { value: 'llama31-8b',  label: 'LLaMA-3.1-8B',             engine: 'Unsloth', model: 'unsloth/llama-3.1-8b-bnb-4bit',            hardware: 'GPU 16GB+',  description: 'Meta LLaMA 3.1 8B — strong general LLM, ดีสำหรับ technical doc Q&A, SOP',     compatible: false },
    { value: 'llama32-3b',  label: 'LLaMA-3.2-3B-Instruct',   engine: 'Unsloth', model: 'unsloth/Llama-3.2-3B-Instruct-bnb-4bit',   hardware: 'GPU 8GB+',   description: 'Compact LLaMA 3.2 — เร็ว, เหมาะกับ domain chatbot, maintenance assistant',     compatible: true  },
    { value: 'mistral-7b',  label: 'Mistral-7B-Instruct',     engine: 'Unsloth', model: 'unsloth/mistral-7b-instruct-v0.3-bnb-4bit', hardware: 'GPU 14GB+',  description: 'Strong instruction following — structured output, report generation, API agent', compatible: false },
    { value: 'qwen25-7b',   label: 'Qwen2.5-7B-Instruct',    engine: 'Unsloth', model: 'unsloth/Qwen2.5-7B-Instruct-bnb-4bit',     hardware: 'GPU 14GB+',  description: 'Excellent multilingual (EN/TH/ZH) — ดีสำหรับ Thai industrial documentation',   compatible: false },
    { value: 'phi35-mini',  label: 'Phi-3.5-Mini-Instruct',   engine: 'Unsloth', model: 'unsloth/Phi-3.5-mini-instruct-bnb-4bit',   hardware: 'GPU 6GB+',   description: 'Microsoft Phi-3.5 3.8B — lightweight แต่แรง, เหมาะ edge inference, IoT gateway', compatible: true  },
    { value: 'gemma2-2b',   label: 'Gemma-2-2B-Instruct',    engine: 'Unsloth', model: 'unsloth/gemma-2-2b-it-bnb-4bit',           hardware: 'GPU 6GB+',   description: 'Google Gemma 2 2B — ขนาดเล็กมาก, ดีสำหรับ classification text, FAQ bot',       compatible: true  },
    { value: 'deepseek-r1', label: 'DeepSeek-R1-8B',          engine: 'Unsloth', model: 'unsloth/DeepSeek-R1-0528-Qwen3-8B-bnb-4bit', hardware: 'GPU 20GB+', description: 'Reasoning model — RCA (root cause analysis), troubleshooting chain-of-thought',  compatible: false },
    { value: 'qwen3-14b',   label: 'Qwen3-14B',               engine: 'Unsloth', model: 'unsloth/Qwen3-14B-bnb-4bit',               hardware: 'GPU 24GB+',  description: 'Top-tier reasoning — complex process optimization, multi-step planning',         compatible: false },
    { value: 'medgemma-27b', label: 'MedGemma-27B-IT',        engine: 'Unsloth', model: 'google/medgemma-27b-it',                   hardware: 'GPU 48GB+',  description: 'Google medical LLM — pre-trained บน medical text, ดีสำหรับ clinical NLP, radiology report',  compatible: false },
    { value: 'meditron-7b',  label: 'Meditron-7B',            engine: 'Unsloth', model: 'epfl-llm/meditron-7b',                    hardware: 'GPU 14GB+',  description: 'EPFL+Yale — fine-tuned จาก Llama 2 บน medical corpus, ดีสำหรับ clinical reasoning, medical QA', compatible: false },
    { value: 'biomistral-7b', label: 'BioMistral-7B',         engine: 'Unsloth', model: 'BioMistral/BioMistral-7B',                hardware: 'GPU 14GB+',  description: 'Fine-tune จาก Mistral บน PubMed — medical Q&A, summarization, literature review',              compatible: false },
    { value: 'openbiollm-8b', label: 'OpenBioLLM-8B',         engine: 'Unsloth', model: 'aaditya/Llama3-OpenBioLLM-8B',           hardware: 'GPU 16GB+',  description: 'Saama — Llama 3 fine-tuned บน biomedical corpus, ดีสำหรับ drug research, biomedical NLP',       compatible: true  },
  ],

  // ── VLM Fine-tuning ───────────────────────────────────────────────────────────
  // All VLM models go through the Unsloth path on the backend. Unsloth pins
  // compatible transformers/peft/bitsandbytes versions, so we avoid the
  // bare-HF version-conflict class of bugs and the numpy/pandas ABI mismatch
  // on the Ray cluster. Unsloth supports Qwen2-VL, LLaVA, Pixtral, and
  // Llama-3.2-Vision; for models unsloth doesn't have a pre-built
  // bnb-4bit version of, we still pass the original model id and unsloth
  // tries to load it (will raise a clear error if unsupported).
  //
  // Medical VLMs (Med-R1, Hulu-Med) are flagged zeroShot=true because they
  // are SOTA off-the-shelf for diagnosis / VQA / report generation across
  // 8-14 imaging modalities, so users can deploy them as-is before any
  // LoRA fine-tune.
  'vlm-finetune': [
    { value: 'llava16-7b',  label: 'LLaVA-1.6-7B',            engine: 'Unsloth',    model: 'unsloth/llava-v1.6-mistral-7b-bnb-4bit',  hardware: 'GPU 16GB+',  description: 'General VLM — visual inspection report, defect description generation',       compatible: true  },
    { value: 'qwen2vl-7b',  label: 'Qwen2-VL-7B-Instruct',    engine: 'Unsloth',    model: 'unsloth/Qwen2-VL-7B-Instruct-bnb-4bit',   hardware: 'GPU 16GB+',  description: 'Strong multimodal — Thai/EN, ดีสำหรับ industrial doc + image Q&A',         compatible: true  },
    { value: 'internvl2',   label: 'InternVL2-8B',            engine: 'Unsloth',    model: 'unsloth/InternVL2-8B-bnb-4bit',           hardware: 'GPU 16GB+',  description: 'Top-ranked open VLM — OCR, diagram understanding, part recognition',         compatible: true  },
    { value: 'paligemma',   label: 'PaliGemma-2-3B',          engine: 'Unsloth',    model: 'unsloth/paligemma2-3b-pt-448-bnb-4bit',  hardware: 'GPU 8GB+',   description: 'Google compact VLM — ดีสำหรับ captioning, VQA, grounding on industrial images', compatible: true  },
    { value: 'medgemma-4b', label: 'MedGemma-4B-IT',          engine: 'Unsloth',    model: 'unsloth/medgemma-4b-it-bnb-4bit',         hardware: 'GPU 12GB+',  description: 'Google medical VLM — pre-trained บน medical images & text, ดีสำหรับ radiology, pathology, dermatology', compatible: true  },
    { value: 'smolvlm',     label: 'SmolVLM-500M',            engine: 'Unsloth',    model: 'HuggingFaceTB/SmolVLM-500M-Instruct',     hardware: 'GPU 6GB+',   description: 'Ultra-lightweight VLM — edge deployment, Jetson Orin, smart camera',         compatible: true  },
    { value: 'medr1-3b',    label: 'Med-R1-3B (GRPO Medical VLM)', engine: 'Unsloth', model: 'yuxianglai117/Med-R1',                hardware: 'GPU 14GB+',  description: 'Qwen2-VL fine-tuned ด้วย GRPO reinforcement learning — 8 imaging modalities (CT/MRI/X-Ray/Fundus/Dermoscopy/Microscopy/OCT/US), 5 tasks (anatomy ID, disease diagnosis, lesion grading, modality recognition, bio-attribute) — chain-of-thought reasoning แข็งแกร่ง, เหมาะ zero-shot diagnostic VQA', compatible: true, zeroShot: true },
    { value: 'hulumed-7b',  label: 'Hulu-Med-7B (Medical Generalist VLM)', engine: 'Unsloth', model: 'ZJU-AI4H/Hulu-Med-7B',          hardware: 'GPU 16GB+',  description: 'ZJU-AI4H transparent generalist VLM — 14 imaging modalities (CT/MRI/X-Ray/US/PET/OCT/Endoscopy/Histopathology/Fundus/Dermoscopy/Angiography/...), รองรับ 2D/3D NIfTI/video, SOTA บน 30 medical benchmarks — generalist diagnostic + report generation', compatible: true, zeroShot: true },
    { value: 'huatuogpt-v-7b', label: 'HuatuoGPT-Vision-7B (PubMedVision)', engine: 'Unsloth', model: 'FreedomIntelligence/HuatuoGPT-Vision-7B', hardware: 'GPU 16GB+', description: 'FreedomIntelligence medical MLLM (Qwen2-7B + LLaVA-style vision adapter) — train บน PubMedVision 1.3M medical VQA, SOTA บน VQA-RAD 63.7 / SLAKE 76.2 / PathVQA 57.9 / PMC-VQA 54.3 / OmniMedVQA 74.0 — แข่งกับ LLaVA-v1.6-34B ที่ขนาดเล็กกว่า', compatible: true, zeroShot: true },
    { value: 'bimedix2-8b', label: 'BiMediX2-8B (Bilingual EN/AR Medical LMM)', engine: 'Unsloth', model: 'MBZUAI/BiMediX2',                   hardware: 'GPU 16GB+', description: 'MBZUAI bilingual medical LMM (Llama 3.1 + LLaVA-pp) — 1.6M Arabic-English multimodal instruction (BiMed-V), SOTA บน BiMed-MBench (+9% EN, +20% AR), USMLE +8% over GPT-4, medical VQA + report generation + summarization — bilingual medical assistant', compatible: true, zeroShot: true },
  ],

  // ── Self-supervised / SSL ────────────────────────────────────────────────────
  'self-supervised': [
    { value: 'mae',        label: 'MAE (Masked AutoEncoder)',   engine: 'PyTorch', model: 'mae_vit_base',                   hardware: 'GPU 12GB+',   description: 'Self-supervised pretraining — ดีเมื่อ label น้อย แต่ unlabeled data เยอะ', compatible: false },
    { value: 'dino',       label: 'DINOv2 (ViT-B/14)',         engine: 'TIMM',    model: 'vit_base_patch14_dinov2',        hardware: 'GPU 10GB+',   description: 'Self-distillation — universal feature extractor สำหรับ downstream tasks',   compatible: false },
    { value: 'simCLR',     label: 'SimCLR (ResNet50)',         engine: 'PyTorch', model: 'resnet50-simclr',                hardware: 'GPU 8GB+',    description: 'Contrastive learning — pretrain ก่อน fine-tune เมื่อ label data ขาดแคลน',  compatible: true  },
    { value: 'byol',       label: 'BYOL (ResNet50)',           engine: 'PyTorch', model: 'resnet50-byol',                  hardware: 'GPU 8GB+',    description: 'Bootstrap your own latent — no negative samples, stable training',           compatible: true  },
    { value: 'padim',      label: 'PaDiM (Anomaly Detection)', engine: 'Anomalib', model: 'padim_resnet18',               hardware: 'GPU 6GB+',    description: 'Unsupervised anomaly detection — zero defect sample needed for training',    compatible: true  },
    { value: 'patchcore',  label: 'PatchCore',                engine: 'Anomalib', model: 'patchcore_wide_resnet50',       hardware: 'GPU 8GB+',    description: 'Memory bank anomaly detection — MVTec-style industrial inspection',          compatible: true  },
  ],

  // ── Export ────────────────────────────────────────────────────────────────────
}

// ─── Deploy-Only Catalog ───────────────────────────────────────────────────
//
// SOTA pre-trained models that should be **deployed as-is** (not fine-tuned
// via the Unsloth path in this UI). Each one needs a custom training
// pipeline because its vision encoder / architecture is not in the set
// that Unsloth knows how to LoRA-wrap:
//
//   - RaDialog_v2: LLaVA + BioViL (chest X-ray) + CheXbert + flash-attn
//   - Med3DVLM:    DCFormer (3D decomposed conv) + Qwen-2.5-7B
//
// The Deploy page reads from this list and offers Ray Serve / Modal
// endpoints. Adding a model here keeps the Train menu uncluttered and
// avoids the false expectation that "select → train" will work — these
// are inference-only out of the box.
//
// The Deploy page ALSO surfaces any model in TRAINING_MATRIX with
// `zeroShot: true` set (see DeployModels.tsx for the union) — those
// are SOTA off-the-shelf models that can be deployed without fine-tune
// even though they are also fine-tunable via Unsloth.
export const DEPLOY_CATALOG: TrainOption[] = [
  {
    value:        'radialog-v2',
    label:        'RaDialog-v2 (LLaVA Radiology)',
    engine:       'LLaVA + BioViL',
    model:        'ChantalPellegrini/RaDialog-interactive-radiology-report-generation',
    hardware:     'GPU 16GB+',
    dataTypes:    ['xray'],
    description:  'MIDL 2025 — LLaVA-based radiology VLM with BioViL image encoder, CheXbert classifier, fine-tuned on MIMIC-CXR + RaDialog-Instruct — supports chest X-ray report generation, view classification, impression generation, and interactive dialog correction',
    compatible:   true,
    source:       'https://github.com/ChantalMP/RaDialog_v2',
    paper:        'https://openreview.net/pdf?id=trUvr1gSNI',
  },
  {
    value:        'med3dvlm',
    label:        'Med3DVLM (3D Medical VLM)',
    engine:       'DCFormer + Qwen-2.5-7B',
    model:        'MagicXin/Med3DVLM-Qwen-2.5-7B',
    hardware:     'GPU 24GB+',
    dataTypes:    ['xray'],   // 3D CT/MRI volumes — surfaced as 'xray' data type
    description:  'IEEE JBHI 2025 — 3D medical VLM with DCFormer (decomposed 3D conv encoder) + SigLIP contrastive pretraining + Qwen-2.5-7B LLM, trained on M3D-Cap (120K 3D image-text) and M3D-VQA (96K 3D VQA) — supports volumetric CT/MRI report generation, image-text retrieval (R@1 61%), open-ended VQA (METEOR 36.76), closed-ended VQA (acc 79.95%)',
    compatible:   true,
    source:       'https://github.com/mirthai/med3dvlm',
    paper:        'https://ieeexplore.ieee.org/document/11145341/',
  },
]

// ─── Data-type compatibility per model ───────────────────────────────────────────────
const _A: DataType[] = ['rgb', 'thermal', 'xray', 'microscopy', 'lidar', 'general'] // all
const _V: DataType[] = ['rgb', 'thermal', 'xray', 'microscopy', 'general']           // visual (no lidar)
const _C: DataType[] = ['rgb', 'thermal', 'general']                                 // camera-based
const _X: DataType[] = ['xray', 'microscopy', 'rgb', 'general']                      // precision detail
const _I: DataType[] = ['rgb', 'thermal', 'xray', 'general']                         // inspection/anomaly

export const MODEL_DATA_TYPES: Record<string, DataType[]> = {
  // Classification
  'eff-b2':            _A,
  'eff-b4':            _V,
  'mobilenet':         _C,
  'res50':             _A,
  'convnext':          ['rgb', 'thermal', 'microscopy', 'general'],
  'regnet':            ['rgb', 'general'],
  'swin-t':            ['rgb', 'thermal', 'microscopy', 'general'],
  'vit-b':             ['rgb', 'general'],
  'dino-s':            ['rgb', 'thermal', 'microscopy', 'general'],
  'efficientvit':      ['rgb', 'general'],
  'rad-dino':          ['xray', 'microscopy', 'rgb'],
  'txrv-densenet':     ['xray'],
  'uni-pathology':     ['microscopy', 'xray'],
  // MedGemma
  'medgemma-4b':       ['xray', 'microscopy', 'rgb', 'general'],
  'medgemma-27b':      _A,
  'meditron-7b':       _A,
  'biomistral-7b':     _A,
  'openbiollm-8b':     _A,
  // MONAI / MedSAM / nnU-Net
  'monai-densenet':    ['xray', 'microscopy', 'rgb'],
  'monai-unet':        ['xray', 'microscopy', 'rgb'],
  'medsam':            ['xray', 'microscopy', 'rgb'],
  'nnunet-2d':         ['xray', 'microscopy', 'rgb'],
  // Detection
  'yolov8-s':          _C,
  'yolov8-m':          _C,
  'yolov8-l':          _C,
  'yolonano':          ['rgb', 'general'],
  'yolov9-c':          _C,
  'rt-detr':           _C,
  'detr':              ['rgb', 'general'],
  // Segmentation
  'unet':              _V,
  'unetpp':            _X,
  'deeplabv3':         _C,
  'maskrcnn':          _C,
  'yoloseg':           _C,
  // LLM text (data type irrelevant — text model)
  'llama31-8b':        _A,
  'llama32-3b':        _A,
  'mistral-7b':        _A,
  'qwen25-7b':         _A,
  'phi35-mini':        _A,
  'gemma2-2b':         _A,
  'deepseek-r1':       _A,
  'qwen3-14b':         _A,
  // VLM fine-tune
  'llava16-7b':        ['rgb', 'xray', 'microscopy', 'general'],
  'qwen2vl-7b':        _V,
  'internvl2':         _V,
  'paligemma':         ['rgb', 'general'],
  'smolvlm':           ['rgb', 'general'],
  'medr1-3b':          ['xray', 'microscopy', 'rgb', 'general'],
  'hulumed-7b':        ['xray', 'microscopy', 'rgb', 'general'],
  'huatuogpt-v-7b':    ['xray', 'microscopy', 'rgb', 'general'],
  'bimedix2-8b':       ['xray', 'microscopy', 'rgb', 'general'],
  // Self-supervised / anomaly
  'mae':               ['rgb', 'general'],
  'dino':              ['rgb', 'microscopy', 'general'],
  'simCLR':            _V,
  'byol':              _V,
  'padim':             _I,
  'patchcore':         _I,
  // Export formats (data-type agnostic)
  'tflite':            _A,
  'onnx':              _A,
  'tensorrt':          _A,
  'openvino':          _A,
  'coreml':            _A,
}

// ─── Image Domain Labels ──────────────────────────────────────────────────────
const DATA_TYPE_LABELS: Record<DataType, string> = {
  rgb:        'Standard RGB Camera',
  thermal:    'Thermal / IR Camera',
  xray:       'X-Ray / CT Scan',
  microscopy: 'Microscopy / Macro',
  lidar:      'LiDAR / Depth / Point Cloud',
  general:    'General Images',
}

// ─── Step Wizard ─────────────────────────────────────────────────────────────
const STEPS = ['Training Type', 'Data & Target', 'Model', 'Config & Launch']

// Parse hardware string like "GPU 16GB+" → minimum GB, or 0 for CPU/Edge
function parseRequiredGb(hardware: string): number {
  const m = hardware.match(/(\d+)\s*GB/i)
  return m ? parseInt(m[1]) : 0
}

export default function TrainModel({ types }: { types?: TrainingType[] } = {}) {
  // If a `types` prop is passed, only those training-type cards show
  // in step 0. Used by the Deep-Learning / LLM wrapper pages so the
  // sidebar submenus can scope their content without duplicating the
  // 1480-line form. Default = all six training types (legacy /train).
  const _allowed: TrainingType[] | null = types && types.length ? types : null
  const showType = (t: TrainingType) => _allowed === null || _allowed.includes(t)
  const [searchParams] = useSearchParams()
  const navigate = useNavigate()
  const [retrainSource, setRetrainSource] = useState<{ id: string; name: string } | null>(null)
  const [gpuFreeGb, setGpuFreeGb] = useState<number | null>(null)  // max free GPU memory on Ray
  const [step, setStep] = useState(0)
  const [config, setConfig] = useState<TrainingConfig>({
    trainingType: 'classification',
    dataType: 'rgb',
    target: 'finetune',
    datasetId: null,
    epochs: 30,
    imgsz: 640,
    batchSize: 16,
    learningRate: 0.001,
    optimizer: 'adamw',
    engine: '',
    model: '',
    notes: '',
    loraRank: 16,
    quantization: '4bit',
    maxSeqLen: 2048,
    chatTemplate: 'alpaca',
    gradAccum: 4,
    textDatasetId: '',
    cluster: 'ray',
  })
  const [toast, setToast] = useState<{msg: string; type: 'success' | 'error'} | null>(null)
  const [launching, setLaunching] = useState(false)
  const [jobId, setJobId] = useState('')
  const [showAllModels, setShowAllModels] = useState(false)
  // Existing trained models — used when target='export' to let the
  // user pick a previously-trained model to continue/export from.
  const [existingModels, setExistingModels] = useState<Array<{
    id: string; name: string; model_name?: string; engine?: string;
    training_type?: string; epochs?: number; finished_at?: string;
  }>>([])
  const [existingModelsLoading, setExistingModelsLoading] = useState(false)
  const [clusterStatus, setClusterStatus] = useState<{
    ray:   { available: boolean; url: string; info: string }
    modal: { available: boolean; status: string; ray_url: string | null; creds_saved?: boolean; gpu_type?: string; num_workers?: number }
    minio_for_ray?:            string
    minio_for_ray_reachable?:  boolean
  } | null>(null)

  // Fetch cluster availability when on step 3
  useEffect(() => {
    if (step !== 3) return
    fetch('/api/train/cluster-status', { credentials: 'include' })
      .then(r => r.json())
      .then(setClusterStatus)
      .catch(() => {})
  }, [step])

  // Fetch existing trained models when the user picks target='export' —
  // step 2 then renders a list of these for the user to choose one
  // to continue/export from. Triggered by target change, not step
  // change, so the list is ready when the user advances to step 2.
  useEffect(() => {
    if (config.target !== 'export') {
      setExistingModels([])
      return
    }
    setExistingModelsLoading(true)
    fetch('/api/jobs?view=models', { credentials: 'include' })
      .then(r => r.json())
      .then(d => {
        const jobs = Array.isArray(d) ? d : (d?.jobs ?? [])
        setExistingModels(jobs.map((j: any) => ({
          id: j.id,
          name: j.name || j.id?.slice(0, 8) || 'unnamed',
          model_name: j.model_name,
          engine: j.engine,
          training_type: j.training_type,
          epochs: j.epochs,
          finished_at: j.finished_at || j.updated_at,
        })))
      })
      .catch(() => setExistingModels([]))
      .finally(() => setExistingModelsLoading(false))
  }, [config.target])

  // ── Modal Start-Cluster popup (shown when the user picks Modal but the
  //    cluster isn't running yet).  Lets them start inline rather than
  //    bouncing to the Modal Config page.
  const [modalStartOpen, setModalStartOpen]     = useState(false)
  const [modalGpu, setModalGpu]                 = useState('T4')
  const [modalWorkers, setModalWorkers]         = useState(1)
  const [modalStarting, setModalStarting]       = useState(false)
  const [modalStartError, setModalStartError]   = useState('')
  const [modalClusterLive, setModalClusterLive] = useState<{
    status: string; ray_url: string | null; gpu_type?: string; num_workers?: number; logs?: string[]
  } | null>(null)
  const modalPollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // Poll /api/modal/status while the popup is open so the user sees when
  // the cluster flips to 'running' (Modal cold-start is 1–3 min).
  useEffect(() => {
    if (!modalStartOpen) {
      if (modalPollRef.current) { clearInterval(modalPollRef.current); modalPollRef.current = null }
      setModalClusterLive(null)
      return
    }
    const tick = async () => {
      try {
        const r = await fetch('/api/modal/status')
        if (r.ok) setModalClusterLive(await r.json())
      } catch { /* ignore */ }
    }
    tick()
    modalPollRef.current = setInterval(tick, 4000)
    return () => { if (modalPollRef.current) clearInterval(modalPollRef.current) }
  }, [modalStartOpen])

  // If the cluster becomes running while the popup is open, auto-close it
  // and select Modal.
  useEffect(() => {
    if (modalClusterLive?.status === 'running') {
      setModalStartOpen(false)
      set('cluster', 'modal')
      setClusterStatus(s => s ? { ...s, modal: { ...s.modal, available: true, status: 'running' } } : s)
    }
  }, [modalClusterLive?.status])

  // Sync the gpu/workers selectors to whatever the cluster is actually
  // running with (so the user doesn't have to re-pick on every open).
  useEffect(() => {
    if (modalClusterLive?.gpu_type)    setModalGpu(modalClusterLive.gpu_type)
    if (modalClusterLive?.num_workers) setModalWorkers(modalClusterLive.num_workers)
  }, [modalClusterLive?.gpu_type, modalClusterLive?.num_workers])

  async function startModalFromPopup() {
    setModalStarting(true)
    setModalStartError('')
    try {
      const r = await fetch('/api/modal/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ gpu_type: modalGpu, num_workers: modalWorkers }),
      })
      if (!r.ok) {
        const t = await r.text()
        setModalStartError(t)
      }
    } catch (e) {
      setModalStartError((e as Error).message)
    } finally { setModalStarting(false) }
  }

  async function stopModalFromPopup() {
    setModalStarting(true)
    try {
      await fetch('/api/modal/stop', { method: 'POST' })
    } finally { setModalStarting(false) }
  }

  // Reset showAllModels when training type changes
  useEffect(() => { setShowAllModels(false) }, [config.trainingType])

  // Re-train: detect ?retrain=<jobId>, fetch job, pre-fill config
  useEffect(() => {
    const retrainId = searchParams.get('retrain')
    if (!retrainId) return
    fetch(`/api/jobs/${retrainId}`)
      .then(r => r.json())
      .then(job => {
        setRetrainSource({ id: job.id, name: job.name })
        setConfig(prev => ({
          ...prev,
          trainingType: (job.training_type as TrainingType) || prev.trainingType,
          datasetId:    job.project_id || prev.datasetId,
          model:        job.model_name || prev.model,
          engine:       job.engine || prev.engine,
          epochs:       job.epochs || prev.epochs,
          batchSize:    job.batch_size || prev.batchSize,
          learningRate: job.learning_rate || prev.learningRate,
          optimizer:    job.optimizer || prev.optimizer,
          imgsz:        job.imgsz || prev.imgsz,
          notes:        job.notes ? `Re-train of: ${job.name}\n${job.notes}` : `Re-train of: ${job.name}`,
        }))
        setStep(3)
      })
      .catch(() => {})
  }, [])

  // Fetch projects for dataset dropdown
  const [projects, setProjects] = useState<Array<{id: number; name: string}>>([])
  const [textDatasets, setTextDatasets] = useState<Array<{id: string; name: string; format: string; row_count: number; size_bytes: number}>>([])

  const isLlmType = (t: TrainingType) => t === 'llm-text' || t === 'vlm-finetune'
  const needsTextDataset = (t: TrainingType) => t === 'llm-text'

  // Fetch text datasets when in LLM/VLM mode
  useEffect(() => {
    if (!isLlmType(config.trainingType)) return
    fetch('/api/text-datasets')
      .then(r => r.json())
      .then(d => setTextDatasets(d.datasets ?? []))
      .catch(() => {})
  }, [config.trainingType])

  // Query Ray cluster for actual GPU memory
  useEffect(() => {
    fetch('/api/ray/nodes?view=summary')
      .then(r => r.json())
      .then(data => {
        const nodes: any[] = data?.data?.summary ?? []
        let maxFreeGb = 0
        for (const node of nodes) {
          for (const gpu of (node.gpus ?? [])) {
            const freeGb = (gpu.memoryTotal - gpu.memoryUsed) / 1024
            if (freeGb > maxFreeGb) maxFreeGb = freeGb
          }
        }
        if (maxFreeGb > 0) setGpuFreeGb(Math.floor(maxFreeGb))
      })
      .catch(() => {})
  }, [])

  useEffect(() => {
    fetch('/api/ls/projects/?page_size=1000', {
      headers: { Authorization: 'Token medimage-ls-token-2026' },
    })
      .then(r => r.json())
      .then(data => {
        const list = Array.isArray(data) ? data : (data?.results ?? [])
        setProjects(list.map((p: { id: number; title: string }) => ({ id: p.id, name: p.title })))
      })
      .catch(() => {})
  }, [])

  const trainingOptions = TRAINING_MATRIX[config.trainingType] || []

  // Dynamic compatibility: if we know real GPU memory, use it; else fall back to hardcoded
  const isCompatible = (opt: TrainOption) => {
    if (gpuFreeGb !== null) return parseRequiredGb(opt.hardware) <= gpuFreeGb
    return opt.compatible
  }

  // Data-type filtering
  const getModelDT   = (opt: TrainOption) => MODEL_DATA_TYPES[opt.value] ?? _A
  const isDataMatch  = (opt: TrainOption) => getModelDT(opt).includes(config.dataType)
  const filteredByDomain  = showAllModels ? trainingOptions : trainingOptions.filter(isDataMatch)
  const hiddenCount       = showAllModels ? 0 : trainingOptions.filter(o => !isDataMatch(o)).length
  const compatibleOptions   = filteredByDomain.filter(o => isCompatible(o))
  const incompatibleOptions = filteredByDomain.filter(o => !isCompatible(o))

  // Per-card counts for step 0 — each training type card shows ITS OWN count
  const countForType = (tt: TrainingType) => {
    const opts = TRAINING_MATRIX[tt] || []
    const dataFiltered = opts.filter(o => (MODEL_DATA_TYPES[o.value] ?? _A).includes(config.dataType))
    return dataFiltered.filter(o => isCompatible(o)).length
  }

  const set = <K extends keyof TrainingConfig>(key: K, val: TrainingConfig[K]) => {
    setConfig(prev => ({ ...prev, [key]: val }))
  }

  const showToast = (msg: string, type: 'success' | 'error') => {
    setToast({ msg, type })
    setTimeout(() => setToast(null), 4000)
  }

  const handleModelSelect = (opt: TrainOption) => {
    set('engine', opt.engine)
    set('model', opt.model)
    setStep(3)
  }

  // When target='export', the model selection step shows previously-
  // trained models from the existingModels list. Picking one
  // continues with that model's engine / model_name (or a
  // generic 'export-<id>' marker if the original model_name
  // isn't available) and jumps to step 3 to configure the export.
  const handleExistingModelSelect = (m: {
    id: string; name: string; model_name?: string; engine?: string; training_type?: string;
  }) => {
    set('model', m.model_name || m.id)        // HF model id or our job id
    set('engine', m.engine || 'Export')
    if (m.training_type) set('trainingType', m.training_type as TrainingType)
    setStep(3)
  }

  const handleLaunch = async () => {
    if (!config.engine || !config.model) {
      showToast('กรุณาเลือก model ก่อน', 'error')
      return
    }
    const needsText = needsTextDataset(config.trainingType)
    if (!needsText && !config.datasetId) {
      showToast('กรุณาเลือก dataset', 'error')
      return
    }
    if (needsText && !config.textDatasetId) {
      showToast('กรุณาเลือก text dataset', 'error')
      return
    }

    // Block zero-shot / open-vocabulary models that cannot be fine-tuned
    const ZERO_SHOT_MODELS = ['grounding-dino', 'groundingdino', 'owl-vit', 'owlvit', 'owlv2', 'sam']
    const modelLower = config.model.toLowerCase()
    if (ZERO_SHOT_MODELS.some(z => modelLower.includes(z))) {
      showToast(
        `❌ ${config.model} เป็น zero-shot model — ไม่รองรับการ fine-tune\nใช้ DETR หรือ YOLOS แทน เช่น facebook/detr-resnet-50`,
        'error'
      )
      return
    }
    setLaunching(true)
    showToast('🚀 Launching training job...', 'success')

    const projectId = needsText ? 0 : config.datasetId

    try {
      const res = await fetch(`/api/train/${projectId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({
          training_type: config.trainingType,
          model_name:    config.model,
          engine:        config.engine,
          epochs:        config.epochs,
          imgsz:         config.imgsz || 640,
          batch_size:    config.batchSize,
          learning_rate: config.learningRate,
          optimizer:     config.optimizer,
          notes:         config.notes,
          cluster:       config.cluster,
          // LLM fields
          lora_rank:     config.loraRank,
          quantization:  config.quantization,
          max_seq_len:   config.maxSeqLen,
          chat_template: config.chatTemplate,
          grad_accum:    config.gradAccum,
          text_dataset:  config.textDatasetId,
        }),
      })

      const ct = res.headers.get('content-type') ?? ''
      if (!ct.includes('application/json')) {
        showToast(`❌ Training backend ไม่พร้อม (HTTP ${res.status}) — กรุณาตรวจสอบ backend service`, 'error')
        return
      }

      const data = await res.json()

      if (!res.ok) {
        showToast(`❌ ${data.detail || data.error || `HTTP ${res.status}`}`, 'error')
      } else {
        setJobId(data.job_id || data.training_id || `job-${Date.now()}`)
        showToast(`✅ Training started — ${data.message || config.model}`, 'success')
      }
    } catch (err) {
      showToast(`❌ เชื่อมต่อ training backend ล้มเหลว`, 'error')
    } finally {
      setLaunching(false)
    }
  }

  const nextStep = () => setStep(s => Math.min(s + 1, 3))
  const prevStep = () => setStep(s => Math.max(s - 1, 0))

  return (
    <div className="max-w-5xl mx-auto">
      {toast && (
        <div className={`toast ${toast.type === 'error' ? 'toast-error' : 'toast-success'} mb-4`}>
          {toast.msg}
        </div>
      )}

      {/* ── Modal Start-Cluster popup ──────────────────────────────────────── */}
      {modalStartOpen && (
        <div
          onClick={() => setModalStartOpen(false)}
          style={{
            position: 'fixed', inset: 0, zIndex: 100,
            background: 'rgba(0,0,0,0.55)', display: 'flex',
            alignItems: 'center', justifyContent: 'center', padding: 16,
          }}
        >
          <div
            onClick={e => e.stopPropagation()}
            style={{
              background: 'var(--bg-card)', borderRadius: 14, padding: 24,
              width: '100%', maxWidth: 460,
              border: '1px solid rgba(139,92,246,0.35)',
              boxShadow: '0 20px 50px rgba(0,0,0,0.45)',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 4 }}>
              <Cloud size={18} color="#8b5cf6" />
              <h2 style={{ margin: 0, fontSize: 16, fontWeight: 700, color: 'var(--text-primary)' }}>
                Start Modal Ray Cluster
              </h2>
              <button
                onClick={() => setModalStartOpen(false)}
                style={{ marginLeft: 'auto', background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)', display: 'flex', padding: 4 }}
                aria-label="Close"
              ><X size={16} /></button>
            </div>
            <p style={{ fontSize: 12, color: 'var(--text-muted)', margin: '0 0 16px' }}>
              ใช้ credentials ที่บันทึกไว้ — ไม่ต้องกรอก secret ใหม่
            </p>

            {modalClusterLive && (
              <div style={{
                display: 'flex', alignItems: 'center', gap: 8,
                padding: '8px 12px', borderRadius: 8,
                background: 'var(--bg-base)', border: '1px solid var(--border-subtle)',
                marginBottom: 14, fontSize: 12,
              }}>
                <span style={{
                  width: 8, height: 8, borderRadius: '50%',
                  background: modalClusterLive.status === 'running' ? '#22c55e' :
                               modalClusterLive.status === 'error'    ? '#ef4444' : '#f59e0b',
                  boxShadow: `0 0 6px ${modalClusterLive.status === 'running' ? '#22c55e' : '#f59e0b'}`,
                  flexShrink: 0,
                }} />
                <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>
                  Status: {modalClusterLive.status}
                </span>
                {modalClusterLive.status === 'deploying' && (
                  <span style={{ color: 'var(--text-muted)' }}>— Modal cold-start ใช้เวลา 1–3 นาที</span>
                )}
                {modalClusterLive.ray_url && (
                  <code style={{ marginLeft: 'auto', fontFamily: 'var(--font-mono)', color: '#8b5cf6', fontSize: 11 }}>
                    {modalClusterLive.ray_url}
                  </code>
                )}
              </div>
            )}

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, marginBottom: 14 }}>
              <label style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                <span style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>GPU</span>
                <select
                  value={modalGpu}
                  onChange={e => setModalGpu(e.target.value)}
                  disabled={modalClusterLive?.status === 'running' || modalClusterLive?.status === 'deploying'}
                  style={{ padding: '8px 10px', fontSize: 13, background: 'var(--bg-base)', color: 'var(--text-primary)', border: '1px solid var(--border)', borderRadius: 8 }}
                >
                  <option value="T4">T4</option>
                  <option value="L4">L4</option>
                  <option value="A10G">A10G</option>
                  <option value="L40S">L40S</option>
                  <option value="A100">A100 (40GB)</option>
                  <option value="A100-80GB">A100 (80GB)</option>
                  <option value="H100">H100</option>
                  <option value="cpu">CPU only</option>
                </select>
              </label>
              <label style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                <span style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>Workers</span>
                <select
                  value={modalWorkers}
                  onChange={e => setModalWorkers(parseInt(e.target.value))}
                  disabled={modalClusterLive?.status === 'running' || modalClusterLive?.status === 'deploying'}
                  style={{ padding: '8px 10px', fontSize: 13, background: 'var(--bg-base)', color: 'var(--text-primary)', border: '1px solid var(--border)', borderRadius: 8 }}
                >
                  {[1, 2, 4, 8].map(n => <option key={n} value={n}>{n}</option>)}
                </select>
              </label>
            </div>

            {modalStartError && (
              <div style={{
                padding: '8px 12px', borderRadius: 8, fontSize: 11, fontFamily: 'var(--font-mono)',
                background: '#ef444410', border: '1px solid #ef444430', color: '#ef4444', marginBottom: 12,
                whiteSpace: 'pre-wrap', maxHeight: 120, overflow: 'auto',
              }}>{modalStartError}</div>
            )}

            {/* Live logs from the API — most recent at the bottom. Useful
                when Start/Stop hangs so the user can see what's happening. */}
            {modalClusterLive?.logs && modalClusterLive.logs.length > 0 && (
              <div style={{ marginBottom: 12 }}>
                <div style={{ fontSize: 10, fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 4 }}>
                  Logs
                </div>
                <div style={{
                  fontFamily: 'var(--font-mono)', fontSize: 10, lineHeight: 1.45,
                  background: '#0a0a0a', color: '#a3e635', borderRadius: 6,
                  padding: '8px 10px', maxHeight: 160, overflowY: 'auto',
                  border: '1px solid #1f1f1f',
                }}>
                  {modalClusterLive.logs.slice(-20).map((l, i) => (
                    <div key={i} style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>{l}</div>
                  ))}
                </div>
              </div>
            )}

            <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
              <button
                onClick={() => setModalStartOpen(false)}
                style={{ padding: '8px 16px', fontSize: 12, fontWeight: 600, background: 'transparent', color: 'var(--text-muted)', border: '1px solid var(--border)', borderRadius: 8, cursor: 'pointer' }}
              >Close</button>
              {modalClusterLive?.status === 'running' ? (
                <button
                  onClick={stopModalFromPopup}
                  disabled={modalStarting}
                  style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '8px 16px', fontSize: 12, fontWeight: 600, background: '#ef444415', color: '#ef4444', border: '1px solid #ef444430', borderRadius: 8, cursor: 'pointer' }}
                >
                  {modalStarting ? <Loader2 size={12} className="animate-spin" /> : <StopCircle size={12} />}
                  Stop Cluster
                </button>
              ) : modalClusterLive?.status === 'deploying' ? (
                <button disabled style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '8px 16px', fontSize: 12, fontWeight: 600, background: 'var(--bg-elevated)', color: 'var(--text-muted)', border: '1px solid var(--border)', borderRadius: 8, cursor: 'not-allowed' }}>
                  <Loader2 size={12} className="animate-spin" /> Starting…
                </button>
              ) : (
                <button
                  onClick={startModalFromPopup}
                  disabled={modalStarting}
                  style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '8px 16px', fontSize: 12, fontWeight: 600, background: '#8b5cf6', color: '#fff', border: 'none', borderRadius: 8, cursor: 'pointer', opacity: modalStarting ? 0.6 : 1 }}
                >
                  {modalStarting ? <Loader2 size={12} className="animate-spin" /> : <Play size={12} />}
                  Start {modalWorkers}× {modalGpu}
                </button>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ── Step Indicator ── */}
      <div className="flex items-center gap-2 mb-8 overflow-x-auto pb-2">
        {STEPS.map((s, i) => (
          <div key={i} className="flex items-center gap-2 flex-shrink-0">
            <button
              onClick={() => i < 3 && setStep(i)}
              style={{
                width: 28, height: 28, borderRadius: '50%',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: 12, fontWeight: 600,
                transition: 'all 0.12s ease',
                cursor: i < step ? 'pointer' : 'default',
                background: i === step ? 'var(--primary)' : i < step ? 'var(--primary-dim)' : 'var(--bg-elevated)',
                color: i === step ? '#fff' : i < step ? 'var(--primary-hover)' : 'var(--text-muted)',
                border: 'none',
              }}
            >
              {i < step ? '✓' : i + 1}
            </button>
            <span style={{ fontSize: 13, whiteSpace: 'nowrap', color: i === step ? 'var(--text-primary)' : 'var(--text-muted)' }}>{s}</span>
            {i < 3 && <ArrowRight size={13} style={{ color: 'var(--border-subtle)', margin: '0 4px', flexShrink: 0 }} />}
          </div>
        ))}
      </div>

      {/* ── STEP 0: Training Type ── */}
      {step === 0 && (
        <div>
          <h2 style={{ fontSize: 16, fontWeight: 600, color: 'var(--text-primary)', marginBottom: 4 }}>เลือก Training Type</h2>
          <p style={{ fontSize: 13, color: 'var(--text-muted)', marginBottom: 24 }}>เลือกประเภทงานที่ต้องการ — ระบบจะแนะนำ engine และ model ที่เหมาะสม</p>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
            {showType('classification') && (<button
              onClick={() => { set('trainingType', 'classification'); nextStep() }}
              className="card text-left"
              style={config.trainingType === 'classification' ? { borderColor: 'var(--primary)', background: 'var(--primary-dim)' } : {}}
            >
              <div className="flex items-center gap-3 mb-3">
                <div style={{ width: 36, height: 36, borderRadius: 8, background: 'var(--primary-dim)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <Brain size={18} style={{ color: 'var(--primary-hover)' }} />
                </div>
                <div>
                  <h3 style={{ fontWeight: 600, fontSize: 14, color: 'var(--text-primary)' }}>Classification</h3>
                  <p style={{ fontSize: 12, color: 'var(--text-muted)' }}>Image classification</p>
                </div>
              </div>
              <p style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                แบ่งประเภทภาพ เช่น Normal/Pneumonia/COVID, Fundus disease grading, CT slice triage
              </p>
              <div style={{ marginTop: 12, display: 'flex', alignItems: 'center', gap: 4, color: 'var(--primary-hover)', fontSize: 12 }}>
                <span>{countForType('classification')} models</span>
                <ChevronDown size={12} />
              </div>
            </button>)}

            {showType('detection') && (<button
              onClick={() => { set('trainingType', 'detection'); nextStep() }}
              className="card text-left"
              style={config.trainingType === 'detection' ? { borderColor: 'var(--warning)', background: 'var(--warning-dim)' } : {}}
            >
              <div className="flex items-center gap-3 mb-3">
                <div style={{ width: 36, height: 36, borderRadius: 8, background: 'var(--warning-dim)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <Cpu size={18} style={{ color: 'var(--warning)' }} />
                </div>
                <div>
                  <h3 style={{ fontWeight: 600, fontSize: 14, color: 'var(--text-primary)' }}>Object Detection</h3>
                  <p style={{ fontSize: 12, color: 'var(--text-muted)' }}>Detect & localize</p>
                </div>
              </div>
              <p style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                หา bounding box ของ defect, component, object บน production line / aerial / industrial camera
              </p>
              <div style={{ marginTop: 12, display: 'flex', alignItems: 'center', gap: 4, color: 'var(--warning)', fontSize: 12 }}>
                <span>{countForType('detection')} models</span>
                <ChevronDown size={12} />
              </div>
            </button>)}

            {showType('segmentation') && (<button
              onClick={() => { set('trainingType', 'segmentation'); nextStep() }}
              className="card text-left"
              style={config.trainingType === 'segmentation' ? { borderColor: 'var(--success)', background: 'var(--success-dim)' } : {}}
            >
              <div className="flex items-center gap-3 mb-3">
                <div style={{ width: 36, height: 36, borderRadius: 8, background: 'var(--success-dim)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <Sparkles size={18} style={{ color: 'var(--success)' }} />
                </div>
                <div>
                  <h3 style={{ fontWeight: 600, fontSize: 14, color: 'var(--text-primary)' }}>Segmentation</h3>
                  <p style={{ fontSize: 12, color: 'var(--text-muted)' }}>Semantic / Instance</p>
                </div>
              </div>
              <p style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                แบ่งส่วน defect area, surface crack, part boundary ด้วย pixel-level precision
              </p>
              <div style={{ marginTop: 12, display: 'flex', alignItems: 'center', gap: 4, color: 'var(--success)', fontSize: 12 }}>
                <span>{countForType('segmentation')} models</span>
                <ChevronDown size={12} />
              </div>
            </button>)}

            {showType('vlm-finetune') && (<button
              onClick={() => { set('trainingType', 'vlm-finetune'); nextStep() }}
              className="card text-left"
              style={config.trainingType === 'vlm-finetune' ? { borderColor: 'var(--info)', background: 'var(--info-dim)' } : {}}
            >
              <div className="flex items-center gap-3 mb-3">
                <div style={{ width: 36, height: 36, borderRadius: 8, background: 'var(--info-dim)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <Sparkles size={18} style={{ color: 'var(--info)' }} />
                </div>
                <div>
                  <h3 style={{ fontWeight: 600, fontSize: 14, color: 'var(--text-primary)' }}>VLM Fine-tune</h3>
                  <p style={{ fontSize: 12, color: 'var(--text-muted)' }}>Vision-Language Model</p>
                </div>
              </div>
              <p style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                Fine-tune LLaVA, Qwen2-VL, InternVL2 สำหรับ visual inspection report, defect Q&A, part recognition
              </p>
              <div style={{ marginTop: 12, display: 'flex', alignItems: 'center', gap: 4, color: 'var(--info)', fontSize: 12 }}>
                <span>GPU 16GB+ required</span>
                <ChevronDown size={12} />
              </div>
            </button>)}

            {showType('self-supervised') && (<button
              onClick={() => { set('trainingType', 'self-supervised'); nextStep() }}
              className="card text-left"
              style={config.trainingType === 'self-supervised' ? { borderColor: 'var(--primary)', background: 'var(--primary-dim)' } : {}}
            >
              <div className="flex items-center gap-3 mb-3">
                <div style={{ width: 36, height: 36, borderRadius: 8, background: 'var(--primary-dim)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <Brain size={18} style={{ color: 'var(--primary-hover)' }} />
                </div>
                <div>
                  <h3 style={{ fontWeight: 600, fontSize: 14, color: 'var(--text-primary)' }}>Self-Supervised</h3>
                  <p style={{ fontSize: 12, color: 'var(--text-muted)' }}>SSL / Pre-training</p>
                </div>
              </div>
              <p style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                Pre-train ด้วย MAE, DINO, SimCLR ก่อน fine-tune — ดีสำหรับ data ใหม่ที่ยังไม่มี label
              </p>
              <div style={{ marginTop: 12, display: 'flex', alignItems: 'center', gap: 4, color: 'var(--primary-hover)', fontSize: 12 }}>
                <span>{countForType('self-supervised')} models</span>
                <ChevronDown size={12} />
              </div>
            </button>)}

            {showType('llm-text') && (<button
              onClick={() => { set('trainingType', 'llm-text'); nextStep() }}
              className="card text-left"
              style={config.trainingType === 'llm-text' ? { borderColor: '#8b5cf6', background: 'rgba(139,92,246,0.08)' } : {}}
            >
              <div className="flex items-center gap-3 mb-3">
                <div style={{ width: 36, height: 36, borderRadius: 8, background: 'rgba(139,92,246,0.12)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <MessageSquare size={18} style={{ color: '#8b5cf6' }} />
                </div>
                <div>
                  <h3 style={{ fontWeight: 600, fontSize: 14, color: 'var(--text-primary)' }}>LLM Fine-tune</h3>
                  <p style={{ fontSize: 12, color: 'var(--text-muted)' }}>Text · Unsloth + QLoRA</p>
                </div>
              </div>
              <p style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                Fine-tune LLaMA-3.1, Mistral, Qwen2.5, Phi-3.5 สำหรับ industrial Q&A, SOP bot, Thai/EN assistant
              </p>
              <div style={{ marginTop: 12, display: 'flex', alignItems: 'center', gap: 6, fontSize: 12 }}>
                <span style={{ background: 'rgba(139,92,246,0.15)', color: '#8b5cf6', borderRadius: 4, padding: '2px 6px', fontWeight: 600 }}>QLoRA 4-bit</span>
                <span style={{ background: 'rgba(139,92,246,0.15)', color: '#8b5cf6', borderRadius: 4, padding: '2px 6px', fontWeight: 600 }}>Unsloth 2x faster</span>
              </div>
            </button>)}
          </div>
        </div>
      )}

      {/* ── STEP 1: Data & Target ── */}
      {step === 1 && (
        <div>
          <h2 style={{ fontSize: 16, fontWeight: 600, color: 'var(--text-primary)', marginBottom: 4 }}>Data & Target</h2>
          <p style={{ fontSize: 13, color: 'var(--text-muted)', marginBottom: 24 }}>เลือกประเภทข้อมูลและเป้าหมายการ training</p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
            {/* Data Type */}
            <div className="card">
              <label>Data Type (ประเภทข้อมูล)</label>
              <select
                value={config.dataType}
                onChange={e => set('dataType', e.target.value as DataType)}
                className="mt-1"
              >
                {Object.entries(DATA_TYPE_LABELS).map(([k, v]) => (
                  <option key={k} value={k}>{v}</option>
                ))}
              </select>
              <p style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 8 }}>
                ประเภทข้อมูลจะช่วยกรอง model ที่เหมาะสมกับ domain
              </p>
            </div>

            {/* Target */}
            <div className="card">
              <label>Target (เป้าหมาย)</label>
              <select
                value={config.target}
                onChange={e => set('target', e.target.value as TargetType)}
                className="mt-1"
              >
                <option value="finetune">Fine-tune (ปรับแต่งจาก pretrained)</option>
                <option value="export">Export (export model ที่มีอยู่แล้ว)</option>
              </select>
              <p style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 8 }}>
                {config.target === 'finetune' && 'Fine-tune จาก pretrained model เช่น ImageNet, MedCLIP'}
                {config.target === 'export' && 'Export checkpoint ที่มีอยู่ไปเป็น format ที่ต้องการ — step ถัดไปจะให้เลือก existing model ที่จะ export'}
              </p>
            </div>

            {/* Dataset — only pure LLM uses text datasets; VLM uses LS image projects */}
            {needsTextDataset(config.trainingType) ? (
              <div className="card md:col-span-2">
                <label>Text Dataset (.jsonl)</label>
                {textDatasets.length === 0 ? (
                  <div style={{ marginTop: 8, padding: '16px', borderRadius: 8, background: 'var(--bg-elevated)', textAlign: 'center' }}>
                    <p style={{ fontSize: 13, color: 'var(--text-muted)', marginBottom: 8 }}>ยังไม่มี text dataset — ไปที่ Datasets เพื่ออัปโหลด .jsonl</p>
                    <a href="/datasets" className="btn btn-secondary btn-sm">ไปที่ Datasets →</a>
                  </div>
                ) : (
                  <select
                    value={config.textDatasetId}
                    onChange={e => set('textDatasetId', e.target.value)}
                    className="mt-1"
                  >
                    <option value="">-- เลือก text dataset --</option>
                    {textDatasets.map(d => (
                      <option key={d.id} value={d.id}>
                        {d.name} ({d.row_count.toLocaleString()} rows · {d.format})
                      </option>
                    ))}
                  </select>
                )}
              </div>
            ) : (
              <div className="card md:col-span-2">
                <label>Dataset (เลือก project ที่จะใช้ train)</label>
                <select
                  value={config.datasetId ?? ''}
                  onChange={e => set('datasetId', e.target.value ? Number(e.target.value) : null)}
                  className="mt-1"
                >
                  <option value="">-- เลือก project --</option>
                  {projects.map(p => (
                    <option key={p.id} value={p.id}>{p.name} (id={p.id})</option>
                  ))}
                </select>
              </div>
            )}
          </div>

          <div className="flex justify-between mt-6">
            <button className="btn btn-secondary" onClick={prevStep}>← Back</button>
            <button className="btn btn-primary" onClick={nextStep}>Next: Model →</button>
          </div>
        </div>
      )}

      {/* ── STEP 2: Model Selection ── */}
      {step === 2 && (
        <div>
          {config.target === 'export' ? (
            <div>
              <h2 style={{ fontSize: 16, fontWeight: 600, color: 'var(--text-primary)', marginBottom: 4 }}>เลือก Existing Model</h2>
              <p style={{ fontSize: 13, color: 'var(--text-muted)', marginBottom: 24 }}>
                เลือก model ที่ train เสร็จแล้ว — จะถูก export เป็น format ที่เลือกในขั้นถัดไป ({existingModels.length} model ที่ train เสร็จ)
              </p>
              {existingModelsLoading ? (
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '20px', color: 'var(--text-muted)' }}>
                  <Loader2 size={16} className="animate-spin" />
                  กำลังโหลด existing models ...
                </div>
              ) : existingModels.length === 0 ? (
                <div style={{ padding: '20px', borderRadius: 8, background: 'var(--bg-elevated)', textAlign: 'center' }}>
                  <p style={{ fontSize: 13, color: 'var(--text-muted)', marginBottom: 8 }}>
                    ยังไม่มี model ที่ train เสร็จ — ไปที่ <a href="/models" style={{ color: 'var(--primary)' }}>Models</a> ดูผลงานที่ train เสร็จแล้ว
                  </p>
                  <p style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                    หรือกลับไปเลือก target = "Fine-tune" เพื่อ train model ใหม่จาก pretrained
                  </p>
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {existingModels.map(m => (
                    <button
                      key={m.id}
                      onClick={() => handleExistingModelSelect(m)}
                      className="card text-left"
                    >
                      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 8 }}>
                        <h3 style={{ fontWeight: 600, fontSize: 14, color: 'var(--text-primary)' }}>{m.name}</h3>
                        {m.training_type && (
                          <span className="badge badge-primary" style={{ fontSize: 10 }}>{m.training_type}</span>
                        )}
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', marginBottom: 6 }}>
                        <span style={{ color: 'var(--primary-hover)' }}>{m.engine || '—'}</span>
                        <span>·</span>
                        <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{m.model_name || m.id}</span>
                      </div>
                      {m.epochs !== undefined && (
                        <p style={{ fontSize: 11, color: 'var(--text-muted)' }}>epochs: {m.epochs}{m.finished_at ? ` · finished ${new Date(m.finished_at).toLocaleDateString()}` : ''}</p>
                      )}
                      <div style={{ marginTop: 8, fontSize: 12, color: 'var(--primary-hover)', fontWeight: 500 }}>
                        เลือก model นี้ →
                      </div>
                    </button>
                  ))}
                </div>
              )}
              <div className="flex justify-between mt-6">
                <button className="btn btn-secondary" onClick={prevStep}>← Back</button>
              </div>
            </div>
          ) : (
            <div>
              <h2 style={{ fontSize: 16, fontWeight: 600, color: 'var(--text-primary)', marginBottom: 4 }}>เลือก Model</h2>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 24, flexWrap: 'wrap', gap: 8 }}>
            <p style={{ fontSize: 13, color: 'var(--text-muted)', margin: 0 }}>
              {showAllModels
                ? <span>All models <strong style={{ color: 'var(--text-primary)' }}>({trainingOptions.length})</strong></span>
                : <span>{filteredByDomain.length} models for <strong style={{ color: 'var(--text-primary)' }}>{DATA_TYPE_LABELS[config.dataType]}</strong></span>
              }
              {gpuFreeGb !== null && <span style={{ marginLeft: 8, color: 'var(--text-muted)' }}>· GPU ว่าง {gpuFreeGb} GB</span>}
            </p>
            <label style={{ display: 'flex', alignItems: 'center', gap: 7, cursor: 'pointer', fontSize: 12, color: 'var(--text-secondary)', userSelect: 'none' as const }}>
              <input
                type="checkbox"
                checked={showAllModels}
                onChange={e => setShowAllModels(e.target.checked)}
                style={{ width: 14, height: 14, cursor: 'pointer', accentColor: 'var(--primary)' }}
              />
              Show all models ({trainingOptions.length} total)
            </label>
          </div>

          {/* Compatible */}
          <div className="mb-6">
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
              <div style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--success)' }} />
              <span style={{ fontSize: 13, fontWeight: 500, color: 'var(--success)' }}>
                Available on your hardware{gpuFreeGb !== null ? ` (${gpuFreeGb} GB GPU free)` : ''}
              </span>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {compatibleOptions.map(opt => (
                <button
                  key={opt.value}
                  onClick={() => handleModelSelect(opt)}
                  className="card text-left"
                >
                  <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 8 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      <h3 style={{ fontWeight: 600, fontSize: 14, color: 'var(--text-primary)' }}>{opt.label}</h3>
                      {opt.zeroShot && <span style={{ fontSize: 10, padding: '1px 6px', borderRadius: 4, background: 'var(--warning, #f59e0b)20', color: 'var(--warning, #f59e0b)', border: '1px solid var(--warning, #f59e0b)', fontWeight: 600 }}>Zero-Shot</span>}
                      {['TF-Lite','ONNX','NVIDIA','Intel','Apple'].includes(opt.engine) && <span style={{ fontSize: 10, padding: '1px 6px', borderRadius: 4, background: '#6366f120', color: '#6366f1', border: '1px solid #6366f1', fontWeight: 600 }}>Export-Only</span>}
                    </div>
                    <span className="badge badge-success" style={{ fontSize: 11 }}>{opt.hardware}</span>
                  </div>
                  <p style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 8, lineHeight: 1.5 }}>{opt.description}</p>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                    <span style={{ color: 'var(--primary-hover)' }}>{opt.engine}</span>
                    <span>·</span>
                    <span>{opt.model}</span>
                  </div>
                  {showAllModels && (
                    <div style={{ marginTop: 8, display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                      {getModelDT(opt).map(dt => (
                        <span key={dt} style={{
                          fontSize: 9, padding: '1px 5px', borderRadius: 5, fontWeight: 600,
                          background: dt === config.dataType ? 'var(--primary)' : 'var(--bg-elevated)',
                          color: dt === config.dataType ? '#fff' : 'var(--text-muted)',
                          border: '1px solid',
                          borderColor: dt === config.dataType ? 'var(--primary)' : 'var(--border-default)',
                        }}>
                          {dt}
                        </span>
                      ))}
                    </div>
                  )}
                  <div style={{ marginTop: 8, fontSize: 12, color: 'var(--primary-hover)', fontWeight: 500 }}>
                    เลือก model นี้ →
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Hidden models (other data types) */}
          {hiddenCount > 0 && (
            <div
              style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '10px 16px', borderRadius: 8, background: 'var(--bg-elevated)', marginBottom: 16, cursor: 'pointer', border: '1px dashed var(--border-default)' }}
              onClick={() => setShowAllModels(true)}
            >
              <span style={{ fontSize: 12, color: 'var(--text-muted)', flex: 1 }}>
                +{hiddenCount} more model{hiddenCount > 1 ? 's' : ''} available for other data types
              </span>
              <span style={{ fontSize: 12, color: 'var(--primary)', fontWeight: 600 }}>Show all →</span>
            </div>
          )}

          {/* Incompatible (dimmed but clickable) */}
          {incompatibleOptions.length > 0 && (
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
                <div style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--border-subtle)' }} />
                <span style={{ fontSize: 13, fontWeight: 500, color: 'var(--text-muted)' }}>Requires more hardware — click to select anyway</span>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3" style={{ opacity: 0.5 }}>
                {incompatibleOptions.map(opt => (
                  <button key={opt.value} className="card text-left" onClick={() => handleModelSelect(opt)} style={{ cursor: 'pointer', width: '100%' }}>
                    <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 8 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                        <h3 style={{ fontWeight: 600, fontSize: 14, color: 'var(--text-muted)' }}>{opt.label}</h3>
                        {opt.zeroShot && <span style={{ fontSize: 10, padding: '1px 6px', borderRadius: 4, background: '#f59e0b20', color: '#f59e0b', border: '1px solid #f59e0b', fontWeight: 600 }}>Zero-Shot</span>}
                        {['TF-Lite','ONNX','NVIDIA','Intel','Apple'].includes(opt.engine) && <span style={{ fontSize: 10, padding: '1px 6px', borderRadius: 4, background: '#6366f120', color: '#6366f1', border: '1px solid #6366f1', fontWeight: 600 }}>Export-Only</span>}
                      </div>
                      <span className="badge badge-warning" style={{ fontSize: 11 }}>{opt.hardware}</span>
                    </div>
                    <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 8 }}>{opt.description}</p>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                      <span>{opt.engine}</span>
                      <span>·</span>
                      <span>{opt.model}</span>
                    </div>
                    {showAllModels && (
                      <div style={{ marginTop: 8, display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                        {getModelDT(opt).map(dt => (
                          <span key={dt} style={{
                            fontSize: 9, padding: '1px 5px', borderRadius: 5, fontWeight: 600,
                            background: dt === config.dataType ? 'var(--primary)' : 'var(--bg-elevated)',
                            color: dt === config.dataType ? '#fff' : 'var(--text-muted)',
                            border: '1px solid',
                            borderColor: dt === config.dataType ? 'var(--primary)' : 'var(--border-default)',
                          }}>
                            {dt}
                          </span>
                        ))}
                      </div>
                    )}
                  </button>
                ))}
              </div>
            </div>
          )}

          <div className="flex justify-between mt-6">
            <button className="btn btn-secondary" onClick={prevStep}>← Back</button>
          </div>
            </div>
          )}
        </div>
      )}

      {/* ── STEP 3: Config & Launch ── */}
      {step === 3 && (
        <div>
          <h2 style={{ fontSize: 16, fontWeight: 600, color: 'var(--text-primary)', marginBottom: 4 }}>Config & Launch</h2>
          <p style={{ fontSize: 13, color: 'var(--text-muted)', marginBottom: 24 }}>ตั้งค่า hyperparameters แล้ว launch training ไปที่ cluster ที่เลือก</p>

          {/* ── Cluster Selector ── */}
          <div className="card mb-6">
            <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)', marginBottom: 12 }}>
              Training Cluster
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {/* Ray Cluster */}
              <button
                onClick={() => set('cluster', 'ray')}
                style={{
                  display: 'flex', alignItems: 'center', gap: 12,
                  padding: '12px 16px', borderRadius: 10, cursor: 'pointer',
                  background: config.cluster === 'ray' ? 'var(--primary-dim)' : 'var(--bg-secondary)',
                  border: `2px solid ${config.cluster === 'ray' ? 'var(--primary)' : 'var(--border)'}`,
                  textAlign: 'left', width: '100%',
                }}
              >
                <Server size={20} color={config.cluster === 'ray' ? 'var(--primary)' : 'var(--text-muted)'} style={{ flexShrink: 0 }} />
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)' }}>Ray Cluster (On-Prem)</div>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>
                    {clusterStatus
                      ? (clusterStatus.ray.available
                          ? <span style={{ color: 'var(--success)' }}>● Connected — {clusterStatus.ray.url}</span>
                          : <span style={{ color: 'var(--error)' }}>● Unavailable — {clusterStatus.ray.info}</span>)
                      : <span style={{ color: 'var(--text-muted)' }}>Checking...</span>}
                  </div>
                </div>
                {config.cluster === 'ray' && <CheckCircle size={16} color="var(--primary)" style={{ flexShrink: 0 }} />}
              </button>

              {/* Modal Cluster */}
              <button
                onClick={() => {
                  if (!clusterStatus) return  // still loading
                  if (!clusterStatus.modal.creds_saved) {
                    // No creds yet — can't do anything here, just direct them.
                    showToast('Set Modal token & secret on the Modal Config page first', 'error')
                    return
                  }
                  if (!clusterStatus.modal.available) {
                    // Creds saved, cluster not running — open the inline start popup.
                    setModalStartOpen(true)
                    return
                  }
                  set('cluster', 'modal')
                }}
                style={{
                  display: 'flex', alignItems: 'center', gap: 12,
                  padding: '12px 16px', borderRadius: 10, cursor: 'pointer',
                  background: config.cluster === 'modal' ? 'rgba(139,92,246,0.08)' : 'var(--bg-secondary)',
                  border: `2px solid ${config.cluster === 'modal' ? 'rgba(139,92,246,0.6)' : 'var(--border)'}`,
                  textAlign: 'left', width: '100%',
                  opacity: (clusterStatus && !clusterStatus.modal.available) ? 0.6 : 1,
                }}
              >
                <Cloud size={20} color={config.cluster === 'modal' ? '#8b5cf6' : 'var(--text-muted)'} style={{ flexShrink: 0 }} />
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)' }}>Modal Config (Cloud GPU)</div>
                  <div style={{ fontSize: 11, marginTop: 2 }}>
                    {clusterStatus
                      ? (clusterStatus.modal.available
                          ? <span style={{ color: 'var(--success)' }}>● Running — ready for training</span>
                          : clusterStatus.modal.creds_saved
                            ? <span style={{ color: 'var(--warning)' }}>● {clusterStatus.modal.status} — click to start cluster</span>
                            : <span style={{ color: 'var(--error)' }}>● {clusterStatus.modal.status} — set token & secret in <a href="/modal-cluster" onClick={e => e.stopPropagation()} style={{ color: 'var(--primary-hover)' }}>Modal Config</a></span>)
                      : <span style={{ color: 'var(--text-muted)' }}>Checking...</span>}
                  </div>
                </div>
                {config.cluster === 'modal' && <CheckCircle size={16} color="#8b5cf6" style={{ flexShrink: 0 }} />}
              </button>
            </div>
          </div>

          {/* Re-train banner */}
          {retrainSource && (
            <div style={{
              display: 'flex', alignItems: 'flex-start', gap: 12, padding: '12px 16px',
              borderRadius: 10, background: 'var(--warning-dim)', border: '1px solid var(--warning)',
              marginBottom: 20,
            }}>
              <RotateCcw size={16} color="var(--warning)" style={{ flexShrink: 0, marginTop: 1 }} />
              <div style={{ flex: 1 }}>
                <p style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)', marginBottom: 2 }}>Re-train mode</p>
                <p style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
                  Config pre-filled จาก <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--warning)' }}>{retrainSource.name}</span>
                  {' '}— ปรับ hyperparameters ตามต้องการแล้วกด Launch
                </p>
              </div>
              <button
                className="btn btn-secondary btn-sm"
                style={{ fontSize: 11, flexShrink: 0 }}
                onClick={() => { setRetrainSource(null); navigate('/train') }}
              >
                ✕ Clear
              </button>
            </div>
          )}

          {/* Summary */}
          <div className="card mb-6" style={{ background: 'var(--primary-dim)', borderColor: 'var(--primary-border)' }}>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>Training Type</div>
                <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--primary-hover)', textTransform: 'capitalize' }}>{config.trainingType}</div>
              </div>
              <div>
                <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>Data Type</div>
                <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--primary-hover)' }}>{DATA_TYPE_LABELS[config.dataType]}</div>
              </div>
              <div>
                <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>Model</div>
                <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--primary-hover)', fontFamily: 'var(--font-mono)' }}>{config.model || '-'}</div>
              </div>
              <div>
                <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>Engine</div>
                <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--primary-hover)', fontFamily: 'var(--font-mono)' }}>{config.engine || '-'}</div>
              </div>
            </div>
          </div>

          {/* Hyperparameters */}
          {isLlmType(config.trainingType) ? (
            /* ── LLM / VLM Config ── */
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
              {/* VRAM estimate */}
              <div className="card md:col-span-2" style={{ background: 'rgba(139,92,246,0.06)', borderColor: 'rgba(139,92,246,0.3)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
                  <span style={{ fontSize: 13, fontWeight: 600, color: '#8b5cf6' }}>Estimated VRAM</span>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: 13, color: 'var(--text-primary)' }}>
                    {config.quantization === '4bit' ? '≈ 4–10 GB' : config.quantization === '8bit' ? '≈ 8–16 GB' : '≈ 14–32 GB'}
                  </span>
                  <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                    ({config.quantization === '4bit' ? 'QLoRA 4-bit' : config.quantization === '8bit' ? 'QLoRA 8-bit' : 'Full fine-tune'})
                  </span>
                  <span style={{ marginLeft: 'auto', fontSize: 12, color: 'var(--text-muted)' }}>
                    Unsloth 2× faster training + gradient checkpointing
                  </span>
                </div>
              </div>

              <div className="card">
                <label>LoRA Rank (r)</label>
                <select value={config.loraRank} onChange={e => set('loraRank', Number(e.target.value))}>
                  <option value={8}>8 — ประหยัด VRAM สุด</option>
                  <option value={16}>16 — แนะนำ (balance)</option>
                  <option value={32}>32 — คุณภาพสูงขึ้น</option>
                  <option value={64}>64 — เกือบเทียบเท่า full fine-tune</option>
                </select>
                <p style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>alpha = {config.loraRank * 2} (2×rank)</p>
              </div>
              <div className="card">
                <label>Quantization</label>
                <select value={config.quantization} onChange={e => set('quantization', e.target.value)}>
                  <option value="4bit">4-bit QLoRA (แนะนำ — ประหยัด VRAM สุด)</option>
                  <option value="8bit">8-bit QLoRA (คุณภาพดีกว่า 4-bit)</option>
                  <option value="full">Full Precision (ต้องการ VRAM มาก)</option>
                </select>
              </div>
              <div className="card">
                <label>Max Sequence Length</label>
                <select value={config.maxSeqLen} onChange={e => set('maxSeqLen', Number(e.target.value))}>
                  <option value={512}>512 tokens (เร็ว, ประหยัด)</option>
                  <option value={1024}>1,024 tokens</option>
                  <option value={2048}>2,048 tokens (แนะนำ)</option>
                  <option value={4096}>4,096 tokens (ต้องการ VRAM เพิ่ม)</option>
                </select>
              </div>
              <div className="card">
                <label>Chat Template</label>
                <select value={config.chatTemplate} onChange={e => set('chatTemplate', e.target.value)}>
                  <option value="alpaca">Alpaca (instruction/response)</option>
                  <option value="chatml">ChatML (system/user/assistant)</option>
                  <option value="llama3">LLaMA-3 (เหมาะกับ LLaMA-3.x)</option>
                  <option value="gemma">Gemma (เหมาะกับ Gemma-2)</option>
                </select>
              </div>
              <div className="card">
                <label>Epochs</label>
                <input type="number" min="1" max="10" value={config.epochs} onChange={e => set('epochs', Number(e.target.value))} />
                <p style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>LLM: 1–3 epochs มักเพียงพอ</p>
              </div>
              <div className="card">
                <label>Gradient Accumulation Steps</label>
                <select value={config.gradAccum} onChange={e => set('gradAccum', Number(e.target.value))}>
                  <option value={2}>2 steps</option>
                  <option value={4}>4 steps (แนะนำ)</option>
                  <option value={8}>8 steps (effective batch ใหญ่ขึ้น)</option>
                  <option value={16}>16 steps</option>
                </select>
              </div>
              <div className="card">
                <label>Learning Rate</label>
                <input type="text" placeholder="e.g. 2e-4" value={config.learningRate}
                  onChange={e => set('learningRate', Number(e.target.value))} />
                <p style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>LLM: 2e-4 ถึง 5e-5 แนะนำ</p>
              </div>
            </div>
          ) : (
            /* ── Image / Vision Config ── */
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
              <div className="card">
                <label>Epochs</label>
                <input
                  type="number"
                  min="1"
                  max="500"
                  value={config.epochs}
                  onChange={e => set('epochs', Number(e.target.value))}
                />
              </div>
              <div className="card">
                <label>Batch Size</label>
                <input
                  type="number"
                  min="1"
                  max="256"
                  value={config.batchSize}
                  onChange={e => set('batchSize', Number(e.target.value))}
                />
              </div>
              <div className="card">
                <label>Learning Rate</label>
                <input
                  type="text"
                  placeholder="e.g. 0.001"
                  value={config.learningRate}
                  onChange={e => set('learningRate', Number(e.target.value))}
                />
              </div>
              <div className="card">
                <label>Optimizer</label>
                <select
                  value={config.optimizer}
                  onChange={e => set('optimizer', e.target.value)}
                >
                  <option value="adamw">AdamW (แนะนำ)</option>
                  <option value="adam">Adam</option>
                  <option value="sgd">SGD with momentum</option>
                  <option value="lion">Lion (modern, lightweight)</option>
                </select>
              </div>
            </div>
          )}

          {/* Notes */}
          <div className="card mb-6">
            <label>Notes (optional)</label>
            <textarea
              rows={3}
              placeholder="เพิ่ม note สำหรับ job นี้ เช่น ลำดับการทดสอบ, หมายเหตุ"
              value={config.notes}
              onChange={e => set('notes', e.target.value)}
              className="mt-1"
            />
          </div>

          {/* Ray Cluster Info */}
          <div className="card mb-6" style={{ background: 'var(--bg-elevated)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
              <div style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--success)' }} />
              <span style={{ fontWeight: 600, fontSize: 14, color: 'var(--text-primary)' }}>Ray Cluster Ready</span>
            </div>
            <div className="grid grid-cols-3 gap-4">
              <div>
                <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 4 }}>Head Node</div>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: 13, color: 'var(--primary-hover)' }}>100.68.53.118</div>
              </div>
              <div>
                <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 4 }}>Dashboard</div>
                <a href="http://100.68.53.118:8265" target="_blank" rel="noopener noreferrer"
                  style={{ fontFamily: 'var(--font-mono)', fontSize: 13, color: 'var(--primary-hover)' }}>:8265</a>
              </div>
              <div>
                <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 4 }}>MinIO</div>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: 13, color: 'var(--text-secondary)' }}>100.68.221.236:9000</div>
              </div>
            </div>
          </div>

          {/* Job ID result */}
          {jobId && (
            <div className="card mb-6" style={{ background: 'var(--success-dim)', borderColor: 'var(--success)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                <div style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--success)' }} />
                <span style={{ fontWeight: 600, fontSize: 14, color: 'var(--success)' }}>Training Job Launched!</span>
              </div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: 13, color: 'var(--success)', marginBottom: 12 }}>{jobId}</div>
              <div style={{ display: 'flex', gap: 12 }}>
                <a
                  href="http://100.68.53.118:8265"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="btn btn-secondary btn-sm"
                >
                  View Ray Dashboard
                </a>
                <button
                  className="btn btn-secondary btn-sm"
                  onClick={() => window.location.href = '/jobs'}
                >
                  View Jobs
                </button>
              </div>
            </div>
          )}

          {/* MinIO reachability warning — shown when the Ray worker can't
              reach the host the API will sign the dataset URL against. */}
          {clusterStatus && clusterStatus.minio_for_ray_reachable === false && (
            <div className="card mb-4" style={{ background: '#ef444408', border: '1px solid #ef444430', padding: '12px 14px' }}>
              <div style={{ display: 'flex', alignItems: 'flex-start', gap: 10 }}>
                <div style={{ width: 8, height: 8, borderRadius: '50%', background: '#ef4444', marginTop: 6, flexShrink: 0 }} />
                <div style={{ flex: 1, fontSize: 12, lineHeight: 1.5 }}>
                  <div style={{ fontWeight: 700, color: '#ef4444', marginBottom: 4 }}>
                    ⚠ Ray worker can't reach MinIO
                  </div>
                  <div style={{ color: 'var(--text-secondary)' }}>
                    The API will sign the dataset URL against{' '}
                    <code style={{ background: 'var(--bg-base)', padding: '1px 5px', borderRadius: 3, color: '#ef4444' }}>{clusterStatus.minio_for_ray}</code>{' '}
                    but the host is unreachable from the Ray cluster.
                    Training will fail with <code style={{ background: 'var(--bg-base)', padding: '1px 5px', borderRadius: 3 }}>Connection refused</code>.
                  </div>
                  <div style={{ marginTop: 6, color: 'var(--text-muted)' }}>
                    Fix: on the <code style={{ background: 'var(--bg-base)', padding: '1px 5px', borderRadius: 3 }}>medimage-api</code> container, set{' '}
                    <code style={{ background: 'var(--bg-base)', padding: '1px 5px', borderRadius: 3, color: '#8b5cf6' }}>MINIO_HOST_IP=&lt;host-running-docker-compose&gt;</code>{' '}
                    (or <code style={{ background: 'var(--bg-base)', padding: '1px 5px', borderRadius: 3, color: '#8b5cf6' }}>MINIO_PUBLIC_URL=http://that-host:9000</code>),
                    then restart the API.
                  </div>
                </div>
              </div>
            </div>
          )}

          <div className="flex justify-between">
            <button className="btn btn-secondary" onClick={prevStep} disabled={launching}>← Back</button>
            <button
              className="btn btn-primary flex items-center gap-2"
              onClick={handleLaunch}
              disabled={launching || !config.engine || (clusterStatus?.minio_for_ray_reachable === false)}
            >
              {launching ? (
                <>
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  Launching...
                </>
              ) : (
                <>
                  <Play size={16} />
                  Launch Training
                </>
              )}
            </button>
          </div>
        </div>
      )}
    </div>
  )
}