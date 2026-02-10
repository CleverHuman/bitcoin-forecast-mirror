import { Card, CardContent } from '@/components/ui/card'
import { formatCurrency } from '@/lib/utils'

export default function CurrentPriceCard(
  { currentPrice, predictedPrice }: { currentPrice: number, predictedPrice: number }
) {
  const percentageDiff = ((predictedPrice - currentPrice) / currentPrice) * 100
  const isPriceLower = currentPrice < predictedPrice

  const currentPriceColor = isPriceLower ? 'text-red-500' : 'text-green-500'
  const formattedPercentageDiff = Math.abs(percentageDiff).toFixed(2)

  return (
    <Card className='w-full h-full'>
      <CardContent className='flex flex-col justify-center h-full py-1 sm:py-2 md:py-3'>
        <div className='text-center'>
          <span className={`text-2xl sm:text-3xl md:text-4xl font-bold ${currentPriceColor}`}>
            {formatCurrency(currentPrice, false)}
          </span>
          <p className='text-sm sm:text-md text-muted-foreground'>Current Price</p>
          <div className='mt-2 flex items-center justify-center'>
            <span className='mr-1 text-sm sm:text-base md:text-md font-semibold text-gray-500'>
              {formattedPercentageDiff}%
            </span>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
