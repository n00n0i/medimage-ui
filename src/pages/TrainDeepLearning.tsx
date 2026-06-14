import TrainModel from './TrainModel'

const DL_TYPES = ['classification', 'detection', 'segmentation', 'self-supervised'] as const

export default function TrainDeepLearning() {
  return (
    <TrainModel types={[...DL_TYPES]} />
  )
}
