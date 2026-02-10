import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import { getSimplePrice } from '@/api/coingecko/coin';
import { SimplePrice, SimplePriceParams } from '@/types/coingecko/coin';

interface SimplePriceState {
  data: SimplePrice | null;
  status: 'idle' | 'loading' | 'succeeded' | 'failed';
  error: string | null;
}

const initialState: SimplePriceState = {
  data: null,
  status: 'idle',
  error: null
};

export const fetchSimplePrice = createAsyncThunk(
  'simplePrice/fetch',
  async (params: SimplePriceParams) => {
    const data = await getSimplePrice(params);
    return data;
  }
);

const simplePriceSlice = createSlice({
  name: 'simplePrice',
  initialState,
  reducers: {},
  extraReducers: (builder) => {
    builder
      .addCase(fetchSimplePrice.pending, (state) => {
        state.status = 'loading';
      })
      .addCase(fetchSimplePrice.fulfilled, (state, action) => {
        state.data = action.payload;
        state.status = 'succeeded';
      })
      .addCase(fetchSimplePrice.rejected, (state, action) => {
        state.status = 'failed';
        state.error = action.error.message || 'Failed to fetch simple price data';
      });
  }
});

export default simplePriceSlice.reducer;
