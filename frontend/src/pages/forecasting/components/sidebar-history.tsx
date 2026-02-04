import { useEffect, useState } from 'react'
import { IconX } from '@tabler/icons-react'
import { cn, formatDateVariants } from '@/lib/utils'
import { Layout } from '@/components/custom/layout'
import { Button } from '@/components/custom/button'
import { forecastDummyHistory, ForecastHistoryItem } from '../data/dummy'
import { isXLargeScreen } from '@/lib/utils/screen-size'

interface SidebarHistoryProps extends React.HTMLAttributes<HTMLElement> {
  isCollapsed: boolean
  setIsCollapsed: React.Dispatch<React.SetStateAction<boolean>>
  onRunForecast: (forecast: ForecastHistoryItem) => void
  onOpenForecast: (forecast: ForecastHistoryItem) => void
}

export default function SidebarHistory({
                                         className,
                                         isCollapsed,
                                         setIsCollapsed,
                                         onRunForecast,
                                       }: SidebarHistoryProps) {
  const [navOpened, setNavOpened] = useState(false)
  const [selectedForecast, setSelectedForecast] =
    useState<ForecastHistoryItem | null>(forecastDummyHistory[0])

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
          {forecastDummyHistory.map((item) => (
            <div
              key={item.id}
              className={cn(
                'flex flex-col gap-1 border-b py-2 px-3 transition-colors w-full',
                selectedForecast?.id === item.id
                  ? 'bg-primary/10 text-primary'
                  : 'hover:bg-muted cursor-pointer'
              )}
              onClick={(e) => {
                e.stopPropagation()
                onRunForecast(item)
                setSelectedForecast(item)
              }}
            >
              <div className='flex flex-row justify-between w-full'>
                <span className='text-sm md:text-base'>
                  {formatDateVariants.dateAgoShort(new Date(item.created_at))}
                </span>
              </div>
              <div className='flex flex-col md:flex-row gap-1 md:gap-2'>
                <span className='text-xs md:text-sm'>Growth: {item.growth} {isXLargeScreen(window.innerWidth) && '•'}</span>
                <span className='text-xs md:text-sm'>
                  Interval: {item.interval_width} {isXLargeScreen(window.innerWidth) && '•'}
                </span>
                <span className='text-xs md:text-sm'>
                  Seasonality: {item.yearly_seasonality ? 'yearly' : 'none'}
                </span>
              </div>
            </div>
          ))}
        </div>
      </Layout>
    </aside>
  )
}
