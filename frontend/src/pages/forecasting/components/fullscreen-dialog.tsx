import { Button } from '@/registry/new-york/ui/button'
import { ExpandIcon, X } from 'lucide-react'
import ForecastChart from './forecast-chart'

import { Dialog, DialogContent, DialogTitle, DialogTrigger } from '@/components/ui/dialog'
import ShareForecastDropdown from './share-forecast-dropdown'

interface FullscreenDialogProps {
  isFullscreenOpen: boolean
  setIsFullscreenOpen: (open: boolean) => void
  handleShare: (
    platform: 'twitter' | 'facebook' | 'linkedin' | 'download'
  ) => Promise<void>
}

export default function FullscreenDialog({
  isFullscreenOpen,
  setIsFullscreenOpen,
  handleShare,
}: FullscreenDialogProps) {
  return (
    <Dialog
      open={isFullscreenOpen}
      onOpenChange={setIsFullscreenOpen}
    >
      <DialogTrigger asChild>
        <Button
          variant='ghost'
          size='icon'
          onClick={() => setIsFullscreenOpen(true)}
        >
          <ExpandIcon className='h-4 w-4' />
        </Button>
      </DialogTrigger>
      <DialogContent
        className='h-[95vh] max-h-[95vh] w-[95vw] max-w-[95vw] pt-8'
        withoutCloseButton
      >
        <DialogTitle className='mb-4'>
          <div className='flex items-center justify-between'>
            <div className='flex items-center'>
              <img
                src='https://bitcoin.org/img/icons/opengraph.png'
                alt='Bitcoin logo'
                width={24}
                height={24}
                className='mr-1'
              />
              <div>
                <h2 className='text-base font-bold md:text-lg'>
                  BTC / USD Forecast
                </h2>
                <p className='text-xs text-gray-400'>BTCUSD • BITSTAMP</p>
              </div>
            </div>
            <div className='flex items-center space-x-2'>
              <ShareForecastDropdown handleShare={handleShare} />
              <Button
                variant='ghost'
                size='icon'
                onClick={() => setIsFullscreenOpen(false)}
              >
                <X className='h-4 w-4' />
              </Button>
            </div>
          </div>
        </DialogTitle>
        <div className='flex h-full flex-col'>
          <div className='flex-grow'>
            <ForecastChart className='h-full w-full' isFullscreen={true} />
          </div>
          <div className='mt-4 flex items-center justify-center space-x-4'>
            <div className='flex items-center'>
              <div className='mr-2 h-4 w-4 bg-[rgba(47,119,127,1)]'></div>
              <span>Actual Price</span>
            </div>
            <div className='flex items-center'>
              <div className='mr-2 h-4 w-4 bg-[rgba(105,65,198,1)]'></div>
              <span>Forecast</span>
            </div>
            <div className='flex items-center'>
              <div className='mr-2 h-4 w-4 border border-[rgba(105,65,198,0.5)] bg-[rgba(105,65,198,0.1)]'></div>
              <span>Uncertainty Interval</span>
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}
