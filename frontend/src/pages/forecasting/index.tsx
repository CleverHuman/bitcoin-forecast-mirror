import React, { useEffect, useState } from 'react'
import BasePageView from '@/components/base-page-view'
import PredictionPrice from './components/prediction-price'
import PriceDisplay from './components/price-display'
import { Card } from '@/components/ui/card'
import CurrentPriceCard from './components/current-price-card'
import PredictedPriceCard from './components/predicted-price-card'
import { useDispatch, useSelector } from 'react-redux'
import { RootState } from '@/stores/store'
import { isSameDay, parseISO, startOfDay } from 'date-fns'
import ForecastChart from './components/forecast-chart'
import { exportElementAsImage, shareImage } from '@/lib/utils/image-export'
import { useTheme } from '@/components/theme-provider'
import {
  fetchForecastData,
  ForecastData,
} from '@/stores/slices/coingecko/coins-market'
import { FormForecastDialog } from './components/form-forecast-dialog'
import type { ForecastFormData } from './components/form-forecast-dialog'
import { ForecastHistoryItem } from './data/dummy'
import ShareForecastDropdown from './components/share-forecast-dropdown'
import FullscreenDialog from './components/fullscreen-dialog'
import BlockchainSwitcher from '@/components/blockchain-switcher'
import SidebarHistory from './components/sidebar-history'
import { Button } from '@/registry/new-york/ui/button'
import { PlayIcon } from 'lucide-react'
import UploadCSVDialog from './components/upload-csv-dialog'

const ForecastingPage: React.FC = () => {
  const { data, currentCoinData, forecastData } = useSelector(
    (state: RootState) => state.coinsMarket
  )
  const [todayForecast, setTodayForecast] = useState<ForecastData | undefined>(
    undefined
  )
  const currentPrice = currentCoinData?.market_data.current_price.usd || 0

  useEffect(() => {
    const today = startOfDay(new Date())

    setTodayForecast(
      forecastData.find((item) => isSameDay(parseISO(item.ds), today))
    )
  }, [data, forecastData])

  const predictedPrice = todayForecast?.yhat || 0
  const lowerBound = todayForecast?.yhat_lower || 0
  const upperBound = todayForecast?.yhat_upper || 0

  const [isFullscreenOpen, setIsFullscreenOpen] = useState(false)
  const { theme } = useTheme()
  const dispatch = useDispatch()

  const handleShare = async (
    platform: 'twitter' | 'facebook' | 'linkedin' | 'download'
  ) => {
    const imageUrl = await exportElementAsImage({
      elementId: 'prediction-container',
      fileName: 'crypto-forecast.png',
      theme: theme as 'dark' | 'light',
    })

    if (imageUrl) {
      await shareImage(imageUrl, platform, 'crypto-forecast.png')
    }
  }

  const [selectedForecast, setSelectedForecast] =
    useState<ForecastHistoryItem | null>(null)

  const handleRunForecast = (forecast: ForecastHistoryItem) => {
    if (forecast.id === 1) {
      dispatch<any>(fetchForecastData('/sample/forecast.csv'))
    } else if (forecast.id === 2) {
      dispatch<any>(fetchForecastData('/sample/forecast2.csv'))
    }
    setIsCollapsed(false)
  }

  const [isCollapsed, setIsCollapsed] = useState(true)
  const [formForecastOpen, setFormForecastOpen] = useState(false)

  const handleCreateForecast = (_data: ForecastFormData) => {
    setSelectedForecast(null)
    // Handle the forecast logic here
  }

  const handleSelectForecast = (forecast: ForecastHistoryItem) => {
    setSelectedForecast(forecast)
    setIsCollapsed(false)
    setFormForecastOpen(true)
  }

  const handleOpenCreateForecast = (open: boolean) => {
    setSelectedForecast(null)
    setFormForecastOpen(open)
  }

  const handleUploadSuccess = () => {
    setIsCollapsed(false)
  }

  return (
    <BasePageView
      title='Forecasting'
      description='Analyze and predict cryptocurrency trends'
    >
      <main
        className={`overflow-x-hidden  transition-[margin] md:overflow-y-hidden md:pt-0 ${isCollapsed ? 'mr-0' : 'md:mr-72 xl:mr-96'} h-full pb-4`}
      >
        <div className='flex flex-col space-y-4  '>
          <div className='flex flex-col justify-between gap-4 sm:flex-row'>
            <div className='flex flex-col gap-4 sm:flex-row'>
            <div className='w-full sm:w-auto'>
                <BlockchainSwitcher />
              </div>
              <div className='w-full sm:w-auto'>
                <Button
                  variant='outline'
                  onClick={() => setIsCollapsed(false)}
                  className='w-full sm:w-auto'
                >
                  <PlayIcon className='mr-2 h-4 w-4' /> Run History
                </Button>
              </div>
              <div className='w-full sm:w-auto'>
                <UploadCSVDialog onUploadSuccess={handleUploadSuccess} />
              </div>
              <div className='w-full sm:w-auto'>
                <FormForecastDialog
                  forecast={selectedForecast}
                  open={formForecastOpen}
                  onOpenChange={handleOpenCreateForecast}
                  onSubmit={handleCreateForecast}
                />
              </div>
            </div>
          </div>
          <div id='prediction-container' className='flex flex-col space-y-4 w-full mx-auto'>
            <Card className='flex flex-col py-2'>
              <div className='flex-shrink-0 px-2 pb-1 pt-1 sm:pb-2 sm:pt-1'>
                <div className='flex h-full w-full items-center justify-between'>
                  <PriceDisplay />
                  <FullscreenDialog
                    isFullscreenOpen={isFullscreenOpen}
                    setIsFullscreenOpen={setIsFullscreenOpen}
                    handleShare={handleShare}
                  />
                  <ShareForecastDropdown handleShare={handleShare} />
                </div>
              </div>
              <ForecastChart className='pb-4 h-[200px] sm:h-[300px] md:h-[400px] xl:h-[500px]' />
            </Card>
            <div className=' flex-none space-y-4'>
              <div className='grid h-full grid-cols-1 gap-4 md:grid-cols-2'>
                <CurrentPriceCard
                  currentPrice={currentPrice}
                  predictedPrice={predictedPrice}
                />
                <PredictedPriceCard
                  currentPrice={currentPrice}
                  predictedPrice={predictedPrice}
                  lowerBound={lowerBound}
                  upperBound={upperBound}
                />
              </div>
            </div>
            <PredictionPrice />
          </div>
        </div>
      </main>
      <SidebarHistory
        isCollapsed={isCollapsed}
        setIsCollapsed={setIsCollapsed}
        onRunForecast={handleRunForecast}
        onOpenForecast={handleSelectForecast}
      />
    </BasePageView>
  )
}

export default ForecastingPage
