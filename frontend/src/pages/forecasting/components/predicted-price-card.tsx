import { Card, CardContent } from '@/components/ui/card'
import { formatCurrency } from '@/lib/utils'

export default function PredictedPriceCard({
  currentPrice,
  predictedPrice,
  lowerBound,
  upperBound,
}: {
  currentPrice: number
  predictedPrice: number
  lowerBound: number
  upperBound: number
}) {
  const range = upperBound - lowerBound
  const currentProgressPercentage = Math.max(0, ((currentPrice - lowerBound) / range) * 100)
  
  const predictedProgressPercentage =
    ((predictedPrice - lowerBound) / range) * 100

  const isPriceLower = currentPrice < predictedPrice
  const progressBarColor = isPriceLower ? 'bg-red-500' : 'bg-green-500'
  return (
    <Card className='w-full h-full'>
      <CardContent className='flex flex-col justify-center h-full py-1 sm:py-2 md:py-3'>
        <div className='text-center'>
          <span className='text-2xl font-bold sm:text-3xl md:text-4xl'>
            {formatCurrency(predictedPrice, false)}
          </span>
          <p className='sm:text-md text-sm text-muted-foreground'>
            Predicted Price
          </p>
          <div className='mt-2 space-y-1 sm:mt-4 sm:space-y-2 md:mt-4'>
            <div className='relative'>
              <div className='mb-1 flex h-1.5 overflow-hidden rounded bg-secondary text-xs sm:h-2'>
                <div
                  style={{ width: `${currentProgressPercentage}%` }}
                  className={`flex flex-col justify-center whitespace-nowrap text-center text-white shadow-none ${progressBarColor}`}
                ></div>
              </div>
              <div
                className={`absolute top-0 -ml-1.5 -mt-1.5 px-0.5 py-0.5 sm:-ml-2 sm:-mt-2 sm:px-1 ${progressBarColor} text-xxs rounded-sm text-white sm:text-xs`}
                style={{ left: `${currentProgressPercentage}%` }}
              >
                {formatCurrency(currentPrice, false)}
              </div>
              <div
                className='absolute bottom-0 top-0 w-0.5 bg-black'
                style={{ left: `${predictedProgressPercentage}%` }}
              ></div>
            </div>
            <div className='text-xxs flex justify-between text-muted-foreground sm:text-xs md:text-sm'>
              <span>{formatCurrency(lowerBound, false)}</span>
              <span>{formatCurrency(upperBound, false)}</span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
