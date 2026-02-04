import axios from 'axios';

const coingeckoApi = axios.create({
  baseURL: 'https://api.coingecko.com/api/v3',
  headers: {
    'Content-Type': 'application/json',
    'x-cg-demo-api-key': import.meta.env.VITE_COINGECKO_API_KEY
  },
});

export default coingeckoApi;
