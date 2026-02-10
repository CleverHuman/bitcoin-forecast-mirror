import { createAsyncThunk } from '@reduxjs/toolkit';

import Papa from 'papaparse';
import { format } from 'date-fns';
import { ForecastData } from './coingecko/coins-market';



export const fetchForecastData = createAsyncThunk(
    'crypto/fetchForecastData',
    async () => {
      const response = await fetch('/sample/forecast.csv');
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
  