'use client'

import React, { useEffect, useRef } from 'react'
import {
  createChart,
  ColorType,
  ISeriesApi,
  IChartApi,
} from 'lightweight-charts'
import { useDispatch, useSelector } from 'react-redux'
import { RootState } from '@/stores/store'
import { UpperLowerIndicator, CustomBandData } from './upper-lower-indicator'
import { useTheme } from '@/components/theme-provider'
import styles from '@/styles/forecast-chart.module.css'
import { fetchCryptoData, fetchForecastData } from '@/stores/slices/coingecko/coins-market'
import { cn, formatNumber } from '@/lib/utils'
import { isSmallScreen } from '@/lib/utils/screen-size'

interface ForecastChartProps {
  id?: string
  className?: string
  isFullscreen?: boolean
}

const ForecastChart: React.FC<ForecastChartProps> = ({
  id,
  className,
  isFullscreen = false,
}) => {
  const chartContainerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const forecastSeriesRef = useRef<ISeriesApi<'Line'> | null>(null)
  const cryptoSeriesRef = useRef<ISeriesApi<'Area'> | null>(null)
  const bandIndicatorRef = useRef<UpperLowerIndicator | null>(null)
  const dispatch = useDispatch()

  const {
    data: cryptoData,
    status,
    forecastData,
  } = useSelector((state: RootState) => state.coinsMarket)


  useEffect(() => {
    if (status === 'idle') {
      dispatch<any>(fetchCryptoData({ coinId: 'bitcoin', days: 365 }))
      dispatch<any>(fetchForecastData())
    }
  }, [status, dispatch])

  const theme = useTheme()

  useEffect(() => {

    if (!chartContainerRef.current || !cryptoData?.length ) return

    const handleResize = () => {
      const container = chartContainerRef.current
      if (!container || !chartRef.current) return

      const parentWidth = container.parentElement?.clientWidth || container.clientWidth
      const parentHeight = container.parentElement?.clientHeight || container.clientHeight

      chartRef.current.applyOptions({
        width: parentWidth,
        height: parentHeight,
      })

      chartRef.current.applyOptions({
        rightPriceScale: {
          scaleMargins: {
            top: 0.1,
            bottom: isSmallScreen(parentWidth) ? 0.3 : 0.2,
          },
        },
      })


      chartRef.current.applyOptions({
        handleScroll: {
          mouseWheel: isFullscreen || !isSmallScreen(parentWidth),
          pressedMouseMove: isFullscreen || !isSmallScreen(parentWidth),
        },
        handleScale: {
          axisPressedMouseMove: {
            time: isFullscreen || !isSmallScreen(parentWidth),
            price: isFullscreen || !isSmallScreen(parentWidth),
          },
          mouseWheel: isFullscreen || !isSmallScreen(parentWidth),
          pinch: isFullscreen || !isSmallScreen(parentWidth),
        },
      })
    }

    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: theme.theme === 'dark' ? 'white' : 'black',
      },
      rightPriceScale: {
        visible: true,
      },
      leftPriceScale: {
        visible: false,
      },
    })
    chartRef.current = chart

    let forecastSeries
    if (forecastData) {
      forecastSeries = chart.addLineSeries({
        color: 'rgba(105, 65, 198, 1)',
        lineWidth: 2,
        priceScaleId: 'right',
        priceFormat: {
          type: 'custom',
          formatter: (price: number) => isSmallScreen(window.innerWidth) ? formatNumber.formatWithSubscript(price) : formatNumber.currency(price),
        },
      })
      forecastSeriesRef.current = forecastSeries
    }

    const cryptoSeries = chart.addAreaSeries({
      lineColor: 'rgba(47, 119, 127, 1)',
      topColor: `rgba(47, 119, 127, 1.2)`,
      bottomColor: `rgba(47, 119, 127, 0.1)`,
      priceScaleId: 'right',
      priceFormat: {
        type: 'custom',
        formatter: (price: number) => isSmallScreen(window.innerWidth) ? formatNumber.formatWithSubscript(price) : formatNumber.currency(price),
      },
    })
    cryptoSeriesRef.current = cryptoSeries

    if (forecastData && forecastSeries) {
      const bandIndicator = new UpperLowerIndicator({
        lineColor: 'rgba(105, 65, 198, 0.5)',
        fillColor: 'rgba(105, 65, 198, 0.1)',
        lineWidth: 1,
      })
      bandIndicatorRef.current = bandIndicator

      forecastSeries.attachPrimitive(bandIndicator)
    }

    if (
      chartRef.current &&
      cryptoSeriesRef.current &&
      bandIndicatorRef.current
    ) {
      chartRef.current.applyOptions({
        layout: {
          textColor: theme.theme === 'dark' ? 'white' : 'black',
        },
      })

      const currentDate = new Date()
      currentDate.setMonth(currentDate.getMonth() - 6)
      const startDate =
        cryptoData.length > 0 && cryptoData[0].date
          ? new Date(cryptoData[0].date)
          : currentDate
      const endDate = new Date()
      endDate.setMonth(endDate.getMonth() + 6)

      const chartData = cryptoData.map((item) => ({
        time: item.date,
        value: item.price,
      }))

      let filteredForecastData
      if (forecastSeriesRef.current) {
        filteredForecastData = forecastData.filter((d) => {
          const forecastDate = new Date(d.ds)
          return d.ds && forecastDate >= startDate && forecastDate <= endDate
        })

        const forecastChartData = filteredForecastData.map((item) => ({
          time: new Date(item.ds).toISOString().split('T')[0],
          value: item.yhat_ensemble,
        }))

        forecastSeriesRef.current.setData(forecastChartData)
      }

      cryptoSeriesRef.current.setData(chartData)

      if (bandIndicatorRef.current && filteredForecastData) {
        const confidenceIntervalData: CustomBandData[] =
          filteredForecastData.map((item) => ({
            time: new Date(item.ds).toISOString().split('T')[0],
            upper: Number(item.yhat_ensemble_upper),
            lower: Number(item.yhat_ensemble_lower),
          }))
        bandIndicatorRef.current.setData(confidenceIntervalData)
      }
    }

    if (chartRef.current) {
      if (isFullscreen) {
        chartRef.current.applyOptions({
          rightPriceScale: {
            visible: true,
            borderVisible: false,
          },
          leftPriceScale: {
            visible: false,
          },
          timeScale: {
            borderVisible: false,
          },
        })
      }

      // Ensure content fits after resize
      chartRef.current.timeScale().fitContent()
    }

    const resizeObserver = new ResizeObserver(() => {
      handleResize()
    })

    if (chartContainerRef.current) {
      resizeObserver.observe(chartContainerRef.current)
    }

    window.addEventListener('resize', handleResize)
    handleResize()

    return () => {
      window.removeEventListener('resize', handleResize)
      resizeObserver.disconnect()
      chart.remove()
    }
  }, [cryptoData, forecastData, status, theme, isFullscreen])

  if (status === 'loading') {
    return (
      <div className={`${styles.chartContainer} ${className} relative`}>
        <div className='flex h-full w-full items-center justify-center'>
          <p>Loading...</p>
        </div>
      </div>
    )
  }

  if (status === 'failed') {
    return (
      <div className={`${styles.chartContainer} relative`}>
        <div className='flex h-full w-full items-center justify-center'>
          <p>Error loading crypto data. Please try again later.</p>
        </div>
      </div>
    )
  }

  return (
   <div id={id} className={cn('w-full relative', className)}>
     <div ref={chartContainerRef} className="w-full h-full">
      {/* Chart will be rendered here */}
    </div>
   </div>
  )
}

export default ForecastChart
