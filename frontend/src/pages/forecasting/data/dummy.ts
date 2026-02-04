export type ForecastHistoryItem = {
  id: number
  start_date: string
  end_date: string
  trading_symbol: string
  side: string
  dates: string[]
  events: string[]
  growth: 'linear' | 'logistic'
  seasonality_mode: 'additive' | 'multiplicative'
  interval_width: number
  changepoint_prior_scale: number
  changepoint_range: number
  n_changepoints: number
  seasonality_prior_scale: number
  daily_seasonality: boolean
  weekly_seasonality: boolean
  yearly_seasonality: boolean
  holidays: any
  cap: number
  floor: number
  trading_volume: any
  created_at: string
}

export const forecastDummyHistory: ForecastHistoryItem[] = [
  {
    id: 1,
    start_date: new Date(2024, 2, 15).toISOString().split('T')[0],
    end_date: new Date(2024, 2, 15).toISOString().split('T')[0],
    trading_symbol: 'BTCUSDT',
    side: 'buy',
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
    cap: 100000,
    floor: 20000,
    trading_volume: null,
    dates: ['2024-04-19'],
    events: ['4th Halving'],
    created_at: new Date(2024, 10, 15).toISOString().split('T')[0],
  },
  {
    id: 2,
    start_date: new Date(2024, 2, 14).toISOString().split('T')[0],
    end_date: new Date(2024, 2, 14).toISOString().split('T')[0],
    trading_symbol: 'BTCUSDT',
    side: 'sell',
    growth: 'logistic',
    seasonality_mode: 'additive',
    interval_width: 0.90,
    changepoint_prior_scale: 0.2,
    changepoint_range: 0.7,
    n_changepoints: 200,
    seasonality_prior_scale: 8,
    daily_seasonality: true,
    weekly_seasonality: true,
    yearly_seasonality: false,
    holidays: null,
    cap: 80000,
    floor: 15000,
    trading_volume: null,
    dates: ['2024-04-19'],
    events: ['4th Halving'],
    created_at: new Date(2024, 10, 10).toISOString().split('T')[0],
  },
  {
    id: 4,
    start_date: new Date(2024, 2, 14).toISOString().split('T')[0],
    end_date: new Date(2024, 2, 14).toISOString().split('T')[0],
    trading_symbol: 'BTCUSDT',
    side: 'sell',
    growth: 'logistic',
    seasonality_mode: 'additive',
    interval_width: 0.90,
    changepoint_prior_scale: 0.2,
    changepoint_range: 0.7,
    n_changepoints: 200,
    seasonality_prior_scale: 8,
    daily_seasonality: true,
    weekly_seasonality: true,
    yearly_seasonality: false,
    holidays: null,
    cap: 80000,
    floor: 15000,
    trading_volume: null,
    dates: ['2024-04-19'],
    events: ['4th Halving'],
    created_at: new Date(2024, 10, 10).toISOString().split('T')[0],
  },
]


export const symbols = {
  ethereum: [
    { value: 'ETHUSDT', label: 'Ethereum (ETHUSDT)' },
      { value: "UNIUSDT", label: "Uniswap (UNIUSDT)" },
      { value: "LINKUSDT", label: "Chainlink (LINKUSDT)" },
      { value: "AAVEUSDT", label: "Aave (AAVEUSDT)" },
      { value: "MKRUSDT", label: "Maker (MKRUSDT)" }
    ],
    bitcoin: [
      { value: "BTCUSDT", label: "Bitcoin (BTCUSDT)" },
      { value: "ORDUSDT", label: "Ordinals (ORDUSDT)" },
      { value: "SATSUSDT", label: "Sats (SATSUSDT)" }
    ],
    solana: [
      { value: "SOLUSDT", label: "Solana (SOLUSDT)" },
      { value: "BONKUSDT", label: "Bonk (BONKUSDT)" },
      { value: "JUPUSDT", label: "Jupiter (JUPUSDT)" },
      { value: "RAYUSDT", label: "Raydium (RAYUSDT)" }
    ],
    binance: [
      { value: "BNBUSDT", label: "BNB (BNBUSDT)" },
      { value: "CAKEUSDT", label: "PancakeSwap (CAKEUSDT)" },
      { value: "BSCUSDT", label: "BSC Token (BSCUSDT)" }
    ],
    cardano: [
      { value: "ADAUSDT", label: "Cardano (ADAUSDT)" },
      { value: "DJEDUSDT", label: "Djed (DJEDUSDT)" },
      { value: "IAGYUSDT", label: "IAGY (IAGYUSDT)" }
  ],
}
