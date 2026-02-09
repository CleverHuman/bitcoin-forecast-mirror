import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import Papa from 'papaparse';
import { format } from 'date-fns';
import { getCoinData, getCoinMarketChart, getCoinMarketChartRange } from '@/api/coingecko/coin';
import { CoinMarketChartRangeParams, CoinResponse, MarketChartData } from '@/types/coingecko/coin';

export type MarketChartParams = {
  coinId: string;
  days?: number | 365;
  vsCurrency?: string | 'usd';
  interval?: string | 'daily' | 'hourly' | '5m';
}

export interface CryptoData {
  date: string;
  price: number;
  uncertaintyLower: number;
  uncertaintyUpper: number;
  [key: string]: number | string;
}

interface CryptoState {
    data: CryptoData[];
    forecastData: ForecastData[];
    currentCoinData: CoinResponse | null;
    simplePrice: Record<string, any> | null;
    status: 'idle' | 'loading' | 'succeeded' | 'failed';
    error: string | null;
}

const initialState: CryptoState = {
    data: [],
    forecastData: [],
    currentCoinData: null,
    simplePrice: null,
    status: 'idle',
    error: null,
};

export const fetchCryptoData = createAsyncThunk(
  'crypto/fetchData',
  async (params: CoinMarketChartRangeParams) => {
    let data: MarketChartData;
    if (params.from && params.to) {
      data = await getCoinMarketChartRange(params);
    } else {
      data = await getCoinMarketChart(params);
    }
    
    const processedData = processData(data.prices, data.total_volumes);
    return { blockchain: params.coinId, data: processedData };
  }
);


export const fetchForecastData = createAsyncThunk(
  'crypto/fetchForecastData',
  async (dataUrl?: string) => {
    const response = await fetch(dataUrl ?? '/sample/forecast.csv');
    const csvText = await response.text();
    const result = Papa.parse<ForecastData>(csvText, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
    });
    
    const formattedData = result.data.map(item => ({
      ...item,
      ds: format(new Date(item.ds), "yyyy-MM-dd'T'HH:mm:ss'Z'")
    }));
    
    return formattedData;
  }
);

export const fetchCurrentCoinData = createAsyncThunk(
  'crypto/fetchCurrentCoinData',
  async (coinId: string) => {
    const data = await getCoinData(coinId, {
      community_data: false,
      developer_data: false
    });
    return data;
  }
);


const cryptoSlice = createSlice({
  name: 'crypto',
  initialState,
  reducers: {},
  extraReducers: (builder) => {
    builder
      .addCase(fetchCryptoData.pending, (state) => {
        state.data = [];
        state.status = 'loading';
      })
      .addCase(fetchCryptoData.fulfilled, (state, action) => {
        const { data } = action.payload;
        state.data = data;
        state.status = 'succeeded';
      })
      .addCase(fetchCryptoData.rejected, (state, action) => {
        state.status = 'failed';
        state.error =  action.error?.message || 'Failed to fetch data';
      })
      .addCase(fetchForecastData.pending, (state) => {
        state.status = 'loading';
      })
      .addCase(fetchForecastData.fulfilled, (state, action) => {
        state.forecastData = action.payload;
        state.status = 'succeeded';
      })
      .addCase(fetchForecastData.rejected, (state, action) => {
        state.status = 'failed';
        state.error = action.error.message || 'Failed to fetch forecast data';
      })
      .addCase(fetchCurrentCoinData.pending, (state) => {
        state.currentCoinData = null;
        state.status = 'loading';
      })
      .addCase(fetchCurrentCoinData.fulfilled, (state, action) => {
        state.currentCoinData = action.payload;
        state.status = 'succeeded';
      })
      .addCase(fetchCurrentCoinData.rejected, (state, action) => {
        state.status = 'failed';
        state.error = action.error.message || 'Failed to fetch current coin data';
      });
  },
});

function processData(prices: [number, number][], volumes: [number, number][]): CryptoData[] {
  prices.sort((a, b) => a[0] - b[0]);

  const uniqueDataMap = new Map<string, CryptoData>();

  prices.forEach(([timestamp, price], index) => {
    const date = new Date(timestamp).toISOString().split('T')[0]; // Use YYYY-MM-DD format
    uniqueDataMap.set(date, {
      date,
      price,
      ma7: 0,
      ma14: 0,
      ma28: 0,
      ma90: 0,
      volume: volumes[index][1],
      uncertaintyLower: price * 0.9,
      uncertaintyUpper: price * 1.1,
    });
  });

  // Convert Map to array
  const data = Array.from(uniqueDataMap.values());
  return data;
}


export interface ForecastData {
  ds: string
  yhat_ensemble: number
  yhat_ensemble_lower: number
  yhat_ensemble_upper: number
  trend: number
  trend_lower: number
  trend_upper: number
  cycle_phase?: string
  days_since_halving?: number
  days_until_halving?: number
  cycle_progress?: number
  [key: string]: any  // Allow additional columns from the forecast
}

export default cryptoSlice.reducer;