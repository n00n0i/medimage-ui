import TrainModel from './TrainModel'

const LLM_TYPES = ['llm-text', 'vlm-finetune'] as const

export default function TrainLLM() {
  return (
    <TrainModel types={[...LLM_TYPES]} />
  )
}
