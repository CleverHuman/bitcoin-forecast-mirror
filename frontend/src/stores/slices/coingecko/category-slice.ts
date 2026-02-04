import { createAsyncThunk, createSlice } from '@reduxjs/toolkit'
import axios from 'axios'

export interface Category {
  category_id: string
  name: string
}

interface CategoryState {
  categories: Category[]
  status: 'idle' | 'loading' | 'succeeded' | 'failed'
  error: string | null
}

const initialState: CategoryState = {
  categories: [],
  status: 'idle',
  error: null,
}

export const fetchCategoriesAsync = createAsyncThunk(
  'tokens/fetchCategories',
  async () => {
    const response = await axios.get(`${import.meta.env.VITE_COINGECKO_BASE_URL}/coins/categories/list`, {
      headers: {
          'Content-Type': 'application/json',
          'x-cg-demo-api-key': import.meta.env.VITE_COINGECKO_API_KEY
        },
  });
  return response.data;
  }
)

const tokenSlice = createSlice({
  name: 'tokens',
  initialState,
  reducers: {},
  extraReducers: (builder) => {
    builder
      .addCase(fetchCategoriesAsync.pending, (state) => {
        state.status = 'loading'
      })
      .addCase(fetchCategoriesAsync.fulfilled, (state, action) => {
        state.status = 'succeeded'
        state.categories = action.payload
      })
      .addCase(fetchCategoriesAsync.rejected, (state, action) => {
        state.status = 'failed'
        state.error = action.error.message || null
      })
  },
})

export default tokenSlice.reducer
