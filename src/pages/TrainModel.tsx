import { useState, useEffect } from 'react'
import { useSearchParams, useNavigate } from 'react-router-dom'
import { Brain, Cpu, Sparkles, Database, ArrowRight, Play, ChevronDown, RotateCcw, MessageSquare } from 'lucide-react'

// ─── Training Types ───────────────────────────────────────────────────────────
type TrainingType = 'classification' | 'detection' | 'segmentation' | 'vlm-finetune' | 'export-edge' | 'self-supervised' | 'llm-text'
type DataType    = 'cxr' | 'fundus' | 'ct' | 'pathology' | 'ultrasound' | 'general'
type TargetType  = 'prelabel' | 'finetune' | 'export'

interface TrainOption {
  value: string
  label: string
  engine: string
  model: string
  hardware: string
  description: string
  compatible: boolean
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
}

// ─── Training Matrix ───────────────────────────────────────────────────────────
const TRAINING_MATRIX: Record<string, TrainOption[]> = {
  // ── Classification ──────────────────────────────────────────────────────────
  classification: [
    { value: 'eff-b2',     label: 'EfficientNet-B2',       engine: 'PyTorch',    model: 'efficientnet-b2',             hardware: 'GPU 8GB+',     description: 'Balanced accuracy & speed สำหรับ CXR, Fundus, CT slices',         compatible: true  },
    { value: 'eff-b4',     label: 'EfficientNet-B4',       engine: 'PyTorch',    model: 'efficientnet-b4',             hardware: 'GPU 12GB+',    description: 'Higher accuracy สำหรับงานที่ต้องการความละเอียด',         compatible: false },
    { value: 'mobilenet',  label: 'MobileNetV3-Small',    engine: 'PyTorch+TIMM', model: 'mobilenetv3_small_100',     hardware: 'CPU / Edge',    description: 'Lightweight — รันบน Raspberry Pi, Jetson Nano, Edge GPU',        compatible: true  },
    { value: 'res50',      label: 'ResNet-50',             engine: 'TorchVision', model: 'resnet50',                   hardware: 'GPU 6GB+',      description: 'Classic baseline — เสถียร, เร็ว, ง่ายต่อ fine-tune',          compatible: true  },
    { value: 'convnext',   label: 'ConvNeXt-Tiny',        engine: 'PyTorch',    model: 'convnext_tiny',              hardware: 'GPU 8GB+',      description: 'Modern CNN architecture — ดีกว่า ResNet มาก',                  compatible: true  },
    { value: 'regnet',     label: 'RegNet-Y400M',          engine: 'TorchVision', model: 'regnet_y_400mf',           hardware: 'GPU 6GB+',      description: 'Efficient for large datasets, good for pathology',             compatible: true  },
    { value: 'swin-t',     label: 'Swin-Tiny',            engine: 'TIMM',       model: 'swin_tiny_patch4_window7_224', hardware: 'GPU 8GB+', description: 'Transformer-based — ดีสำหรับ CT/MRI slice classification',     compatible: true  },
    { value: 'vit-b',      label: 'ViT-Base (Vision Transformer)', engine: 'TIMM', model: 'vit_base_patch16_224',     hardware: 'GPU 12GB+',    description: 'Pure transformer — ต้อง data เยอะ, ดีสำหรับ CXR ขนาดใหญ่',  compatible: false },
    { value: 'medclip',    label: 'MedCLIP',              engine: 'HuggingFace', model: 'microsoft/BiomedCLIP',     hardware: 'GPU 10GB+',     description: 'Domain-pretrained CLIP บน biomedical images — zero-shot',         compatible: false },
    { value: 'chextnet',   label: 'CheXNet (DenseNet-121)', engine: 'PyTorch',   model: 'densenet121',               hardware: 'GPU 8GB+',      description: 'Specialized for chest X-ray pathology classification',         compatible: true  },
  ],

  // ── Object Detection ─────────────────────────────────────────────────────────
  detection: [
    { value: 'yolov8-s',   label: 'YOLOv8-Small',        engine: 'Ultralytics', model: 'yolov8s.pt',               hardware: 'GPU 6GB+',    description: 'เร็วมาก — real-time inference, รองรับทุก imaging modality',       compatible: true  },
    { value: 'yolov8-m',   label: 'YOLOv8-Medium',        engine: 'Ultralytics', model: 'yolov8m.pt',               hardware: 'GPU 10GB+',   description: 'Balanced speed/accuracy สำหรับ nodule, fracture detection',    compatible: true  },
    { value: 'yolov8-l',   label: 'YOLOv8-Large',         engine: 'Ultralytics', model: 'yolov8l.pt',               hardware: 'GPU 16GB+',   description: 'High accuracy — สำหรับงาน detection ที่ต้องการ recall สูง',   compatible: false },
    { value: 'yolonano',   label: 'YOLOv8-Nano',          engine: 'Ultralytics', model: 'yolov8n.pt',               hardware: 'CPU / Edge',  description: 'Edge-optimized — deploy บน Jetson, Orin, edge devices',        compatible: true  },
    { value: 'rt-detr',    label: 'RT-DETR-Hybrid',      engine: 'Ultralytics', model: 'yolov8l-rt-detr',          hardware: 'GPU 12GB+',   description: 'Real-time DE-TR, transformer-based detection, ดีสำหรับ CT',   compatible: false },
    { value: 'detr',       label: 'DETR-ResNet50',        engine: 'HuggingFace', model: 'facebook/detr-resnet50',   hardware: 'GPU 10GB+',   description: 'Transformer-based detection — ดีสำหรับ pathology slides',      compatible: false },
    { value: 'sam-b',      label: 'SAM-Base (zero-shot detection)', engine: 'Meta SAM', model: 'sam_b.pt',      hardware: 'GPU 12GB+',   description: 'Zero-shot segmentation — ใช้ prompt หา ROI ได้เลย',        compatible: true  },
    { value: 'nnunet',     label: 'nnU-Net (3D Detection)', engine: 'MONAI',     model: 'nnunet_res3d',            hardware: 'GPU 16GB+',   description: '3D volumes — หา tumor บน CT/MRI แบบ volumetric detection',     compatible: false },
  ],

  // ── Segmentation ─────────────────────────────────────────────────────────────
  segmentation: [
    { value: 'sam-b',      label: 'SAM-Base',            engine: 'Meta SAM',    model: 'sam_b.pt',              hardware: 'GPU 12GB+',   description: 'Zero-shot segmentation — prompt ด้วย box/point',                  compatible: true  },
    { value: 'sam-l',      label: 'SAM-Large',           engine: 'Meta SAM',    model: 'sam_l.pt',              hardware: 'GPU 20GB+',   description: 'แม่นยำสูงสุด — สำหรับ fine boundary segmentation',             compatible: false },
    { value: 'unet',       label: 'UNet ResNet34',        engine: 'Segmentation Models PyTorch', model: 'resnet34-unet', hardware: 'GPU 8GB+',   description: 'Classic segmentation — organ, tumor, lesion segmentation',        compatible: true  },
    { value: 'unetpp',     label: 'UNet++ (ResNet34)',   engine: 'Segmentation Models PyTorch', model: 'resnet34-unetplusplus', hardware: 'GPU 10GB+', description: 'Nested UNet — ดีสำหรับ boundary ที่ซับซ้อน',              compatible: true  },
    { value: 'deeplabv3',  label: 'DeepLabV3+ (ResNet101)', engine: 'TorchVision', model: 'resnet101-deeplabv3', hardware: 'GPU 10GB+', description: 'Atrous conv — fine boundary บน retinal, pathology',             compatible: true  },
    { value: 'segany',     label: 'SAM-Med2D (MedSAM)',   engine: 'MedSAM',     model: 'medsam_vit_b',           hardware: 'GPU 16GB+',   description: 'Medical image-specific SAM — zero-shot บน CXR, CT, MRI',         compatible: false },
    { value: '3d-unet',    label: '3D UNet (ResNet50)',   engine: 'MONAI',      model: 'unet_resnet50_3d',      hardware: 'GPU 16GB+',   description: 'Volumetric segmentation สำหรับ CT/MRI 3D volumes',             compatible: false },
    { value: 'swin-unet',  label: 'Swin-UNet (TransUNet)', engine: 'MONAI',     model: 'swin_unet',             hardware: 'GPU 12GB+',   description: 'Transformer-based segmentation — ดีสำหรับ histology',           compatible: false },
    { value: 'nnunet',     label: 'nnU-Net (3D Full)',     engine: 'MONAI',      model: 'nnunet',                hardware: 'GPU 20GB+',   description: 'State-of-the-art 3D medical image segmentation — self-configures', compatible: false },
  ],

  // ── LLM Text Fine-tuning ─────────────────────────────────────────────────────
  'llm-text': [
    { value: 'llama31-8b',   label: 'LLaMA-3.1-8B',          engine: 'Unsloth', model: 'unsloth/llama-3.1-8b-bnb-4bit',         hardware: 'GPU 16GB+',  description: 'Meta LLaMA 3.1 8B — state-of-the-art open LLM, QLoRA fine-tune สำหรับ medical text', compatible: false },
    { value: 'llama32-3b',   label: 'LLaMA-3.2-3B-Instruct',  engine: 'Unsloth', model: 'unsloth/Llama-3.2-3B-Instruct-bnb-4bit', hardware: 'GPU 8GB+',   description: 'Compact LLaMA 3.2 3B — เร็ว, ดีสำหรับ domain-specific clinical Q&A',              compatible: true  },
    { value: 'mistral-7b',   label: 'Mistral-7B-Instruct',    engine: 'Unsloth', model: 'unsloth/mistral-7b-instruct-v0.3-bnb-4bit', hardware: 'GPU 14GB+', description: 'Mistral 7B — strong instruction following, ดีสำหรับ structured medical output',  compatible: false },
    { value: 'qwen25-7b',    label: 'Qwen2.5-7B-Instruct',   engine: 'Unsloth', model: 'unsloth/Qwen2.5-7B-Instruct-bnb-4bit',  hardware: 'GPU 14GB+',  description: 'Alibaba Qwen2.5 7B — excellent multilingual, ดีสำหรับ Thai medical text',        compatible: false },
    { value: 'phi35-mini',   label: 'Phi-3.5-Mini-Instruct',  engine: 'Unsloth', model: 'unsloth/Phi-3.5-mini-instruct-bnb-4bit', hardware: 'GPU 6GB+',   description: 'Microsoft Phi-3.5 3.8B — lightweight แต่แรง, เหมาะ edge inference',             compatible: true  },
    { value: 'gemma2-2b',    label: 'Gemma-2-2B-Instruct',   engine: 'Unsloth', model: 'unsloth/gemma-2-2b-it-bnb-4bit',        hardware: 'GPU 6GB+',   description: 'Google Gemma 2 2B — ขนาดเล็กมาก รันได้บน GPU 6GB, ดีสำหรับ summarization',    compatible: true  },
    { value: 'deepseek-r1',  label: 'DeepSeek-R1-8B',         engine: 'Unsloth', model: 'unsloth/DeepSeek-R1-0528-Qwen3-8B-bnb-4bit', hardware: 'GPU 20GB+', description: 'DeepSeek R1 reasoning model — ดีสำหรับ clinical reasoning, chain-of-thought',  compatible: false },
    { value: 'qwen3-14b',    label: 'Qwen3-14B',              engine: 'Unsloth', model: 'unsloth/Qwen3-14B-bnb-4bit',           hardware: 'GPU 24GB+',  description: 'Qwen3 14B — top-tier reasoning, ดีสำหรับ medical report generation',             compatible: false },
  ],

  // ── VLM Fine-tuning ───────────────────────────────────────────────────────────
  'vlm-finetune': [
    { value: 'llava1.6',   label: 'LLaVA-1.6-7B',         engine: 'LLaVA',      model: 'llava-v1.6-7b',          hardware: 'GPU 16GB+',   description: 'VLM สำหรับ medical report generation, clinical Q&A',           compatible: false },
    { value: 'phi4v',      label: 'Phi-4-Multimodal',     engine: 'Microsoft',  model: 'phi-4-mini-megvii',       hardware: 'GPU 12GB+',   description: 'Compact VLM — fast inference บน H100, medical vision-language',   compatible: false },
    { value: 'medvlm',     label: 'MedLVLM (MedLLaVA)',  engine: 'LLaVA',      model: 'medllava_v1.5_7b',       hardware: 'GPU 16GB+',   description: 'Domain-specific VLM fine-tuned บน medical images & reports',    compatible: false },
    { value: 'biomedclip', label: 'BiomedCLIP',          engine: 'HuggingFace', model: 'BiomedCLIP',             hardware: 'GPU 8GB+',    description: 'Domain-pretrained CLIP — zero-shot classification & retrieval', compatible: false },
    { value: 'med Flamingo', label: 'MedFlamingo',       engine: 'Flamingo',   model: 'medflamingo_7b',         hardware: 'GPU 20GB+',   description: 'Few-shot VLM สำหรับ medical visual reasoning',               compatible: false },
  ],

  // ── Self-supervised / SSL ────────────────────────────────────────────────────
  'self-supervised': [
    { value: 'mae',        label: 'MAE (Masked AutoEncoder)', engine: 'PyTorch', model: 'mae_vit_base',         hardware: 'GPU 12GB+',   description: 'Self-supervised pretraining — ดีสำหรับ data ขนาดใหญ่',          compatible: false },
    { value: 'dino',       label: 'DINOv2 (ViT-B/14)',      engine: 'TIMM',     model: 'vit_base_patch14_dinov2', hardware: 'GPU 10GB+', description: 'Self-distillation — feature extractor สำหรับ downstream tasks',  compatible: false },
    { value: 'simCLR',     label: 'SimCLR (ResNet50)',     engine: 'PyTorch',   model: 'resnet50-simclr',        hardware: 'GPU 8GB+',    description: 'Contrastive learning — ใช้ pretrain ก่อน fine-tune',             compatible: true  },
    { value: 'byol',       label: 'BYOL (ResNet50)',       engine: 'PyTorch',   model: 'resnet50-byol',          hardware: 'GPU 8GB+',    description: 'Bootstrap your own latent — no negative samples needed',            compatible: true  },
    { value: 'medcone',    label: 'MedCoNe (Contrastive SSL)', engine: 'PyTorch', model: 'medcone_resnet50',    hardware: 'GPU 8GB+',    description: 'Medical-specific contrastive — pretrained บน CXR, CT, MRI',        compatible: false },
  ],

  // ── Export ────────────────────────────────────────────────────────────────────
  'export-edge': [
    { value: 'tflite',     label: 'TensorFlow Lite',     engine: 'TF-Lite',    model: 'model.tflite',         hardware: 'CPU / Mobile', description: 'iOS, Android, Edge TPU, Raspberry Pi — int8 quantization',        compatible: true  },
    { value: 'onnx',       label: 'ONNX Runtime',        engine: 'ONNX',       model: 'model.onnx',            hardware: 'CPU / GPU',    description: 'Universal format — รันได้ทุก platform, GPU acceleration',     compatible: true  },
    { value: 'tensorrt',   label: 'TensorRT',            engine: 'NVIDIA',     model: 'model.plan',            hardware: 'NVIDIA GPU',  description: 'Optimized for NVIDIA edge — Jetson, Orin, T4, A100',            compatible: true  },
    { value: 'coreml',     label: 'CoreML',              engine: 'Apple',      model: 'model.mlmodel',         hardware: 'Apple Neural', description: 'iOS/iPad native — รันบน Neural Engine, ANE acceleration',      compatible: true  },
  ],
}

// ─── Medical Modality Labels ──────────────────────────────────────────────────
const DATA_TYPE_LABELS: Record<DataType, string> = {
  cxr:       'Chest X-Ray (CXR)',
  fundus:    'Retinal Fundus',
  ct:        'CT Scan / MRI (3D Volumes)',
  pathology: 'Pathology (H&E Stained)',
  ultrasound:'Ultrasound (US)',
  general:   'General Images',
}

// ─── Step Wizard ─────────────────────────────────────────────────────────────
const STEPS = ['Training Type', 'Data & Target', 'Model', 'Config & Launch']

// Parse hardware string like "GPU 16GB+" → minimum GB, or 0 for CPU/Edge
function parseRequiredGb(hardware: string): number {
  const m = hardware.match(/(\d+)\s*GB/i)
  return m ? parseInt(m[1]) : 0
}

export default function TrainModel() {
  const [searchParams] = useSearchParams()
  const navigate = useNavigate()
  const [retrainSource, setRetrainSource] = useState<{ id: string; name: string } | null>(null)
  const [gpuFreeGb, setGpuFreeGb] = useState<number | null>(null)  // max free GPU memory on Ray
  const [step, setStep] = useState(0)
  const [config, setConfig] = useState<TrainingConfig>({
    trainingType: 'classification',
    dataType: 'cxr',
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
  })
  const [toast, setToast] = useState<{msg: string; type: 'success' | 'error'} | null>(null)
  const [launching, setLaunching] = useState(false)
  const [jobId, setJobId] = useState('')

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
  const compatibleOptions   = trainingOptions.filter(o => isCompatible(o))
  const incompatibleOptions = trainingOptions.filter(o => !isCompatible(o))

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

  const handleLaunch = async () => {
    if (!config.engine || !config.model) {
      showToast('กรุณาเลือก model ก่อน', 'error')
      return
    }
    const llm = isLlmType(config.trainingType)
    if (!llm && !config.datasetId) {
      showToast('กรุณาเลือก dataset', 'error')
      return
    }
    if (llm && !config.textDatasetId) {
      showToast('กรุณาเลือก text dataset', 'error')
      return
    }
    setLaunching(true)
    showToast('🚀 Launching training job...', 'success')

    const projectId = llm ? 0 : config.datasetId

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
            {/* Classification */}
            <button
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
                <span>{compatibleOptions.length} models</span>
                <ChevronDown size={12} />
              </div>
            </button>

            {/* Object Detection */}
            <button
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
                หา bounding box ของ nodule, tumor, lesion บน CXR/CT/Pathology
              </p>
              <div style={{ marginTop: 12, display: 'flex', alignItems: 'center', gap: 4, color: 'var(--warning)', fontSize: 12 }}>
                <span>{compatibleOptions.length} models</span>
                <ChevronDown size={12} />
              </div>
            </button>

            {/* Segmentation */}
            <button
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
                แบ่งส่วนเนื้อเยื่อ, ก้อนเนื้อ, อวัยวะ ด้วย pixel-level precision
              </p>
              <div style={{ marginTop: 12, display: 'flex', alignItems: 'center', gap: 4, color: 'var(--success)', fontSize: 12 }}>
                <span>{compatibleOptions.length} models</span>
                <ChevronDown size={12} />
              </div>
            </button>

            {/* VLM Fine-tune */}
            <button
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
                Fine-tune LLaVA, Phi-4-Vision สำหรับ medical report generation, clinical Q&A
              </p>
              <div style={{ marginTop: 12, display: 'flex', alignItems: 'center', gap: 4, color: 'var(--info)', fontSize: 12 }}>
                <span>GPU 16GB+ required</span>
                <ChevronDown size={12} />
              </div>
            </button>

            {/* Self-Supervised */}
            <button
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
                <span>{trainingOptions.filter(o => o.compatible).length} models available</span>
                <ChevronDown size={12} />
              </div>
            </button>

            {/* Export Edge */}
            <button
              onClick={() => { set('trainingType', 'export-edge'); nextStep() }}
              className="card text-left"
              style={config.trainingType === 'export-edge' ? { borderColor: 'var(--primary)', background: 'var(--primary-dim)' } : {}}
            >
              <div className="flex items-center gap-3 mb-3">
                <div style={{ width: 36, height: 36, borderRadius: 8, background: 'var(--primary-dim)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <Database size={18} style={{ color: 'var(--primary-hover)' }} />
                </div>
                <div>
                  <h3 style={{ fontWeight: 600, fontSize: 14, color: 'var(--text-primary)' }}>Export Edge</h3>
                  <p style={{ fontSize: 12, color: 'var(--text-muted)' }}>TF-Lite / ONNX / TensorRT</p>
                </div>
              </div>
              <p style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                Export trained model ไปเป็น TF-Lite, ONNX, TensorRT สำหรับ deploy บน edge board
              </p>
              <div style={{ marginTop: 12, display: 'flex', alignItems: 'center', gap: 4, color: 'var(--primary-hover)', fontSize: 12 }}>
                <span>{compatibleOptions.length} formats</span>
                <ChevronDown size={12} />
              </div>
            </button>

            {/* LLM Text Fine-tuning */}
            <button
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
                Fine-tune LLaMA-3.1, Mistral, Qwen2.5, Phi-3.5 สำหรับ clinical NLP, Thai medical chatbot, report generation
              </p>
              <div style={{ marginTop: 12, display: 'flex', alignItems: 'center', gap: 6, fontSize: 12 }}>
                <span style={{ background: 'rgba(139,92,246,0.15)', color: '#8b5cf6', borderRadius: 4, padding: '2px 6px', fontWeight: 600 }}>QLoRA 4-bit</span>
                <span style={{ background: 'rgba(139,92,246,0.15)', color: '#8b5cf6', borderRadius: 4, padding: '2px 6px', fontWeight: 600 }}>Unsloth 2x faster</span>
              </div>
            </button>
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
                <option value="prelabel">Pre-label (เตรียมข้อมูลก่อน label)</option>
                <option value="finetune">Fine-tune (ปรับแต่งจาก pretrained)</option>
                <option value="export">Export (export model ที่มีอยู่แล้ว)</option>
              </select>
              <p style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 8 }}>
                {config.target === 'prelabel' && 'ใช้ AI ช่วย pre-label ก่อน เพื่อลดเวลาการ label ของ expert'}
                {config.target === 'finetune' && 'Fine-tune จาก pretrained model เช่น ImageNet, MedCLIP'}
                {config.target === 'export' && 'Export checkpoint ที่มีอยู่ไปเป็น format ที่ต้องการ'}
              </p>
            </div>

            {/* Dataset — image types use LS projects; LLM/VLM use text datasets */}
            {isLlmType(config.trainingType) ? (
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
          <h2 style={{ fontSize: 16, fontWeight: 600, color: 'var(--text-primary)', marginBottom: 4 }}>เลือก Model</h2>
          <p style={{ fontSize: 13, color: 'var(--text-muted)', marginBottom: 24 }}>
            แนะนำ {config.trainingType} สำหรับ {DATA_TYPE_LABELS[config.dataType]}{gpuFreeGb !== null ? ` — GPU ว่างอยู่ ${gpuFreeGb} GB (Ray cluster)` : ''}
          </p>

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
                    <h3 style={{ fontWeight: 600, fontSize: 14, color: 'var(--text-primary)' }}>{opt.label}</h3>
                    <span className="badge badge-success" style={{ fontSize: 11 }}>{opt.hardware}</span>
                  </div>
                  <p style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 8, lineHeight: 1.5 }}>{opt.description}</p>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                    <span style={{ color: 'var(--primary-hover)' }}>{opt.engine}</span>
                    <span>·</span>
                    <span>{opt.model}</span>
                  </div>
                  <div style={{ marginTop: 12, fontSize: 12, color: 'var(--primary-hover)', fontWeight: 500 }}>
                    เลือก model นี้ →
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Incompatible (dimmed) */}
          {incompatibleOptions.length > 0 && (
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
                <div style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--border-subtle)' }} />
                <span style={{ fontSize: 13, fontWeight: 500, color: 'var(--text-muted)' }}>Requires more powerful hardware</span>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3" style={{ opacity: 0.45 }}>
                {incompatibleOptions.map(opt => (
                  <div key={opt.value} className="card text-left">
                    <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 8 }}>
                      <h3 style={{ fontWeight: 600, fontSize: 14, color: 'var(--text-muted)' }}>{opt.label}</h3>
                      <span className="badge badge-warning" style={{ fontSize: 11 }}>{opt.hardware}</span>
                    </div>
                    <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 8 }}>{opt.description}</p>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                      <span>{opt.engine}</span>
                      <span>·</span>
                      <span>{opt.model}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="flex justify-between mt-6">
            <button className="btn btn-secondary" onClick={prevStep}>← Back</button>
          </div>
        </div>
      )}

      {/* ── STEP 3: Config & Launch ── */}
      {step === 3 && (
        <div>
          <h2 style={{ fontSize: 16, fontWeight: 600, color: 'var(--text-primary)', marginBottom: 4 }}>Config & Launch</h2>
          <p style={{ fontSize: 13, color: 'var(--text-muted)', marginBottom: 24 }}>ตั้งค่า hyperparameters แล้ว launch training ไปที่ Ray cluster</p>

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

          <div className="flex justify-between">
            <button className="btn btn-secondary" onClick={prevStep} disabled={launching}>← Back</button>
            <button
              className="btn btn-primary flex items-center gap-2"
              onClick={handleLaunch}
              disabled={launching || !config.engine}
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