import { createSlice, createAsyncThunk } from '@reduxjs/toolkit'
import {
  FetchTokensParams,
  TokenData,
  TokenParams,
} from '@/types/coingecko/token'
import axios from 'axios'

interface TokenState {
  tokens: TokenData[]
  status: 'idle' | 'loading' | 'succeeded' | 'failed'
  error: string | null
}

const initialState: TokenState = {
  tokens: [],
  status: 'idle',
  error: null,
}

export const fetchTokensAsync = createAsyncThunk(
  'tokens/fetchTokens',
  async ({
    sector = 'all',
    timeRange,
    query,
    page = 1,
    perPage = 100,
    order = 'market_cap_desc',
    sparkline = false,
  }: FetchTokensParams): Promise<TokenData[]> => {
    const params: TokenParams = {
      vs_currency: 'usd',
      order: order,
      per_page: perPage,
      page: page,
      sparkline: sparkline,
      price_change_percentage: timeRange.toString().toLowerCase(),
      ids: query,
    }

    if (sector !== 'all') {
      params.category = sector
    }

    const response = await axios.get(
      `${import.meta.env.VITE_COINGECKO_BASE_URL}/coins/markets`,
      {
        headers: {
          'Content-Type': 'application/json',
          'x-cg-demo-api-key': import.meta.env.VITE_COINGECKO_API_KEY,
        },
        params: params,
      }
    )
    return response.data
  }
)

const tokenSlice = createSlice({
  name: 'tokens',
  initialState,
  reducers: {},
  extraReducers: (builder) => {
    builder
      .addCase(fetchTokensAsync.pending, (state) => {
        state.status = 'loading'
      })
      .addCase(fetchTokensAsync.fulfilled, (state, action) => {
        state.status = 'succeeded'
        state.tokens = action.payload
      })
      .addCase(fetchTokensAsync.rejected, (state, action) => {
        state.status = 'failed'
        state.error = action.error.message || null
      })
  },
})

export default tokenSlice.reducer
