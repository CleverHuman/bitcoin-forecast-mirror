export const generateHistoricalData = () => {
    const data = []
    const basePrice = 65000
    const volatility = 0.02
    let currentPrice = basePrice
  
    for (let i = 180; i >= 0; i--) {
      const date = new Date()
      date.setDate(date.getDate() - i)
      
      currentPrice = currentPrice * (1 + (Math.random() - 0.5) * volatility)
      
      data.push({
        time: date.toISOString().split('T')[0],
        value: currentPrice
      })
    }
    return data
  }
  
export const generateForecastData = (lastHistoricalPrice: number) => {
    const data = []
    const volatility = 0.015
    let currentPrice = lastHistoricalPrice
  
    for (let i = 1; i <= 90; i++) {
      const date = new Date()
      date.setDate(date.getDate() + i)
      
      currentPrice = currentPrice * (1 + (Math.random() * volatility))
      
      data.push({
        time: date.toISOString().split('T')[0],
        value: currentPrice
      })
    }
    return data
  }
  
export const generateConfidenceBands = (forecastData: any[]) => {
    return forecastData.map(item => ({
      time: item.time,
      upper: item.value * (1 + Math.random() * 0.1),
      lower: item.value * (1 - Math.random() * 0.1)
    }))
  }
  