'use client'

import { useSelector } from 'react-redux'
import { useEffect, useState } from 'react'
import { RootState } from '@/stores/store'
import { formatCurrency } from '@/lib/utils'
import { parseISO, format, startOfDay, isAfter } from 'date-fns'
import { Card, CardHeader, CardContent, CardDescription, CardTitle } from '@/components/ui/card'
import { CryptoData } from '@/stores/slices/coingecko/coins-market'
import { ForecastData } from '@/stores/slices/coingecko/coins-market'

const apiBaseUrl = import.meta.env.VITE_API_BASE_URL;

const calculatePercentageChange = (current: number, previous: number) => {
  const percentageChange = ((current - previous) / previous) * 100
  return percentageChange.toFixed(2)
}

export default function PredictionPrice() {
  const { data, status, forecastData } = useSelector(
    (state: RootState) => state.coinsMarket
  )
  const [forecastRunResult, setForecastRunResult] = useState([]);

  useEffect(() => {
    const fetchForecastRunResult = async () => {
      try {
        const token = localStorage.getItem("accessToken");
        const response = await fetch(
          `${apiBaseUrl}/forecasting/api/forecast-results/`,
          {
            method: "GET",
            headers: {
              "Content-Type": "application/json",
              Authorization: `Bearer ${token}`,
            },
          },
        );
        const data = await response.json();

        const res = data.results;
        setForecastRunResult(res);
      } catch (error) {
        console.error("Error fetching forecast run result:", error);
      }
    };

    fetchForecastRunResult();
  }, []);

  function processForecastData(data: ForecastData[]): CryptoData[] {
    return data.map((item) => ({
      date: item.ds,
      price: item.yhat,
      uncertaintyLower: item.yhat_lower,
      uncertaintyUpper: item.yhat_upper,
      ma7: 0,
      ma14: 0,
      ma28: 0,
      ma90: 0,
      volume: 0,
    }))
  }
  const [filteredCombinedData, setFilteredCombinedData] = useState<CryptoData[]>([])

  useEffect(() => {
    const today = startOfDay(new Date())
    // const tomorrow = startOfDay(new Date(today.getTime() + 24 * 60 * 60 * 1000))

    const combinedData = [...data, ...processForecastData(forecastRunResult)]
    
    const sortedData = combinedData
    .filter(item => 
      isAfter(parseISO(item.date), today)
    )
      // .sort((a, b) => parseISO(b.date).getTime() - parseISO(a.date).getTime())
      .slice(0, 30)

    const filtered = [...sortedData]

    setFilteredCombinedData(filtered)
  }, [data, forecastData])

  if (status === 'loading') {
    return <div>Loading...</div>
  }

  if (status === 'failed') {
    return <div>Error loading data. Please try again later.</div>
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Prediction Price</CardTitle>
        <CardDescription>
          This table shows the predicted price for the next 30 days.
        </CardDescription>
      </CardHeader>
     <CardContent>
     <table className='w-full text-sm'>
    <thead className="sticky top-0">
      <tr className='border-b'>
        <th className='p-2 text-left'>Date</th>
        <th className='p-2 text-right'>Price</th>
        <th className='p-2 text-right'>Change</th>
      </tr>
    </thead>
    <tbody>
      {filteredCombinedData.map((data, index) => {
        const nextData = index > 0 ? filteredCombinedData[index - 1] : null
        const changeValue = nextData ? nextData.price - data.price : 0
        const changePercentage = nextData
          ? calculatePercentageChange(nextData.price, data.price)
          : '0.00'
        return (
          <tr key={index} className='border-b last:border-b-0'>
            <td className='p-2'>
              {format(parseISO(data.date), 'MMM dd')}
            </td>
            <td className='p-2 text-right'>
              {formatCurrency(data.price, false)}
            </td>
            <td className='p-2 text-right'>
              <span
                className={
                  changeValue >= 0 ? 'text-red-500' : 'text-green-500'
                }
              >
                {changeValue >= 0 ? '-' : '+'}$
                {Math.abs(changeValue).toFixed(2)} (
                {Math.abs(parseFloat(changePercentage))}%)
              </span>
            </td>
          </tr>
        )
      })}
    </tbody>
  </table>
     </CardContent>
    </Card>
  )
}
