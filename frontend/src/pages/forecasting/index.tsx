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
import { runForecast } from '@/api/app/forecast'
import type { ForecastRequest, ForecastProgressEvent, ForecastResponse } from '@/api/app/forecast'
import { useToast } from '@/components/ui/use-toast'

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

  const predictedPrice = todayForecast?.yhat_ensemble || 0
  const lowerBound = todayForecast?.yhat_ensemble_lower || 0
  const upperBound = todayForecast?.yhat_ensemble_upper || 0

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
    if (forecast.csv_path) {
      console.log('Loading forecast from history:', forecast.csv_path)
      toast({
        title: 'Loading forecast...',
        description: 'Fetching forecast data from history',
      })

      dispatch<any>(fetchForecastData(forecast.csv_path))
        .then(() => {
          console.log('Historical forecast loaded successfully')
          toast({
            title: 'Forecast Loaded',
            description: 'Chart updated with historical forecast',
          })
        })
        .catch((error: any) => {
          console.error('Failed to load historical forecast:', error)
          toast({
            variant: 'destructive',
            title: 'Failed to load forecast',
            description: 'Could not load the selected forecast',
          })
        })
    }
    setIsCollapsed(false)
  }

  const [isCollapsed, setIsCollapsed] = useState(true)
  const [formForecastOpen, setFormForecastOpen] = useState(false)
  const [historyRefreshTrigger, setHistoryRefreshTrigger] = useState(0)
  const [currentProgress, setCurrentProgress] = useState<ForecastProgressEvent | null>(null)
  const { toast } = useToast()

  const handleCreateForecast = (data: ForecastFormData) => {
    setSelectedForecast(null)
    setCurrentProgress({ step: 0, total: 6, message: 'Initializing...' })

    console.log('Starting forecast generation...')

    toast({
      title: 'Forecast Generation Started',
      description: 'Watch the progress bar below for updates.',
    })

    // Calculate days between start_date and end_date
    const startDate = new Date(data.start_date)
    const endDate = new Date(data.end_date)
    const daysDiff = Math.ceil((endDate.getTime() - startDate.getTime()) / (1000 * 60 * 60 * 24))

    const request: ForecastRequest = {
      from_date: data.start_date,
      days: daysDiff > 0 ? daysDiff : undefined,
      no_signals: false, // Always generate signals for now
    }

    console.log('Forecast request:', request)

    runForecast(request, {
      onProgress: (progress: ForecastProgressEvent) => {
        console.log('Progress update:', progress)
        setCurrentProgress(progress)
      },
      onComplete: (result: ForecastResponse) => {
        console.log('Forecast complete:', result)
        setCurrentProgress(null)
        toast({
          title: 'Forecast Completed!',
          description: `Successfully created ${result.forecast_file}. Loading results...`,
        })

        // Load the generated forecast CSV
        const baseURL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000/api'
        const csvUrl = `${baseURL}/forecast/file/${result.forecast_file}`
        console.log('Loading forecast CSV from:', csvUrl)

        dispatch<any>(fetchForecastData(csvUrl))
          .then(() => {
            console.log('Forecast data loaded successfully')
            toast({
              title: 'Forecast Loaded',
              description: 'Chart updated with new forecast data',
            })
          })
          .catch((error: any) => {
            console.error('Failed to load forecast data:', error)
            toast({
              variant: 'destructive',
              title: 'Failed to load forecast',
              description: 'The forecast was created but could not be loaded for display',
            })
          })

        // Trigger history refresh in sidebar
        setHistoryRefreshTrigger(prev => prev + 1)
      },
      onError: (error: string) => {
        console.error('Forecast error:', error)
        setCurrentProgress(null)
        toast({
          variant: 'destructive',
          title: 'Forecast Failed',
          description: error,
        })
      },
    })
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
    toast({
      title: 'CSV Loaded',
      description: 'Chart updated with uploaded forecast data',
    })
    setIsCollapsed(false)
  }

  const handleOpenRunHistory = () => {
    setHistoryRefreshTrigger(prev => prev + 1)
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
                  onClick={handleOpenRunHistory}
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
          {currentProgress && (
            <div className='bg-primary/10 border-2 border-primary p-4 rounded-lg shadow-lg'>
              <div className='flex items-center justify-between mb-2'>
                <span className='text-base font-semibold text-primary'>
                  🔄 Creating Forecast: Step {currentProgress.step} of {currentProgress.total}
                </span>
                <span className='text-sm font-bold text-primary'>
                  {Math.round((currentProgress.step / currentProgress.total) * 100)}%
                </span>
              </div>
              <div className='w-full bg-secondary rounded-full h-3 mb-2 overflow-hidden'>
                <div
                  className='bg-primary h-3 rounded-full transition-all duration-500 ease-out animate-pulse'
                  style={{ width: `${(currentProgress.step / currentProgress.total) * 100}%` }}
                />
              </div>
              <p className='text-sm font-medium'>{currentProgress.message}</p>
            </div>
          )}
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
        refreshTrigger={historyRefreshTrigger}
      />
    </BasePageView>
  )
}

export default ForecastingPage
