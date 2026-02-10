import { useEffect, useState } from 'react'
import { IconX } from '@tabler/icons-react'
import { cn, formatDateVariants } from '@/lib/utils'
import { Layout } from '@/components/custom/layout'
import { Button } from '@/components/custom/button'
import { ForecastHistoryItem } from '../data/dummy'
import { isXLargeScreen } from '@/lib/utils/screen-size'
import { getForecastHistory } from '@/api/app/forecast'
import type { ForecastHistoryItem as ApiForecastHistoryItem } from '@/api/app/forecast'

interface SidebarHistoryProps extends React.HTMLAttributes<HTMLElement> {
  isCollapsed: boolean
  setIsCollapsed: React.Dispatch<React.SetStateAction<boolean>>
  onRunForecast: (forecast: ForecastHistoryItem) => void
  onOpenForecast: (forecast: ForecastHistoryItem) => void
  refreshTrigger?: number
}

export default function SidebarHistory({
  className,
  isCollapsed,
  setIsCollapsed,
  onRunForecast,
  refreshTrigger,
}: SidebarHistoryProps) {
  const [navOpened, setNavOpened] = useState(false)
  const [selectedForecast, setSelectedForecast] =
    useState<ApiForecastHistoryItem | null>(null)
  const [forecastHistory, setForecastHistory] = useState<ApiForecastHistoryItem[]>([])

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const response = await getForecastHistory()
        setForecastHistory(response.forecasts)
      } catch (error) {
        console.error('Failed to fetch forecast history:', error)
      }
    }
    fetchHistory()
  }, [refreshTrigger])

  const sortedHistory = [...forecastHistory].sort(
    (a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
  )

  useEffect(() => {
    if (navOpened) {
      document.body.classList.add('overflow-hidden')
    } else {
      document.body.classList.remove('overflow-hidden')
    }
  }, [navOpened])

  useEffect(() => {
    setNavOpened(isCollapsed)
  }, [isCollapsed])

  return (
    <aside
      className={cn(
        'fixed bottom-0 right-0 z-30 h-svh border-l-2 border-l-muted bg-background pt-28 transition-all duration-300 md:pt-16',
        isCollapsed
          ? 'pointer-events-none -right-full md:right-0 md:w-0'
          : 'right-0 w-full md:w-72 xl:w-96',
        className
      )}
    >
      <Layout fixed className='h-full'>
        <Layout.Header
          sticky
          className='flex flex-col gap-2 px-4 py-3 shadow-sm md:px-4'
        >
          <div className='flex w-full items-start justify-between'>
            <h1>History</h1>
            <Button
              variant='ghost'
              size='icon'
              onClick={() => setIsCollapsed(true)}
              className='h-8 w-8'
            >
              <IconX className='h-4 w-4' />
            </Button>
          </div>
          <input
            type='search'
            placeholder='Search history...'
            className='w-full rounded-md border border-input bg-background px-3 py-1 text-sm shadow-sm transition-colors placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring'
          />
        </Layout.Header>
        <div className='flex flex-col'>
          {sortedHistory.length === 0 ? (
            <div className='flex items-center justify-center py-8 text-sm text-muted-foreground'>
              No forecast history yet
            </div>
          ) : (
            sortedHistory.map((item, index) => (
              <div
                key={item.filename}
                className={cn(
                  'flex flex-col gap-1 border-b py-2 px-3 transition-colors w-full',
                  selectedForecast === item
                    ? 'bg-primary/10 text-primary'
                    : 'hover:bg-muted cursor-pointer'
                )}
                onClick={(e) => {
                  e.stopPropagation()
                  // Convert API forecast to ForecastHistoryItem for compatibility
                  const baseUrl = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000/api'
                  const legacyForecast: ForecastHistoryItem = {
                    id: index,
                    csv_path: `${baseUrl}/forecast/file/${item.filename}`,
                    created_at: item.timestamp,
                    start_date: item.from_date || '',
                    end_date: '',
                    trading_symbol: 'BTCUSDT',
                    side: 'buy',
                    dates: [],
                    events: [],
                    growth: 'linear',
                    seasonality_mode: 'multiplicative',
                    interval_width: 0.95,
                    changepoint_prior_scale: 0.1,
                    changepoint_range: 0.8,
                    n_changepoints: 300,
                    seasonality_prior_scale: 10,
                    daily_seasonality: false,
                    weekly_seasonality: true,
                    yearly_seasonality: true,
                    holidays: null,
                    cap: 0,
                    floor: 0,
                    trading_volume: null,
                  }
                  onRunForecast(legacyForecast)
                  setSelectedForecast(item)
                }}
              >
                <div className='flex flex-row justify-between w-full'>
                  <span className='text-sm md:text-base'>
                    {new Date(item.timestamp).toLocaleDateString()}
                  </span>
                </div>
                <div className='flex flex-col md:flex-row gap-1 md:gap-2'>
                  <span className='text-xs md:text-sm'>Points: {item.forecast_points} {isXLargeScreen(window.innerWidth) && '•'}</span>
                  {item.from_date && (
                    <span className='text-xs md:text-sm'>
                      From: {item.from_date} {isXLargeScreen(window.innerWidth) && '•'}
                    </span>
                  )}
                  <span className='text-xs md:text-sm'>
                    {(item.file_size / 1024).toFixed(1)} KB
                  </span>
                </div>
              </div>
            ))
          )}
        </div>
      </Layout>
    </aside>
  )
}
