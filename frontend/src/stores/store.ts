import { configureStore } from '@reduxjs/toolkit';
import { TypedUseSelectorHook, useDispatch, useSelector } from 'react-redux';
import tokenReducer from './slices/coingecko/token-slice';
import categoryReducer from './slices/coingecko/category-slice';
import authReducer from './slices/auth';
import simplePriceReducer from './slices/coingecko/simple-price';
import coinsMarketReducer from './slices/coingecko/coins-market';


export const store = configureStore({
  reducer: {
    tokens: tokenReducer,
    category: categoryReducer,
    auth: authReducer,
    simplePrice: simplePriceReducer,
    coinsMarket: coinsMarketReducer,
  },
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
export const useAppSelector: TypedUseSelectorHook<RootState> = useSelector;
export const useAppDispatch = useDispatch<AppDispatch>;