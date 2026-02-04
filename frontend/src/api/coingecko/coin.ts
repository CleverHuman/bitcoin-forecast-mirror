import { CoinParams, CoinMarketChartParams, CoinResponse, MarketChartData, SimplePrice, CoinMarketChartRangeParams, SimplePriceParams } from '@/types/coingecko/coin';
import coingeckoApi from './api-instance';



export const getCoinData = async (coinId: string, params: CoinParams = {}): Promise<CoinResponse> => {
  const response = await coingeckoApi.get(`/coins/${coinId}`, {
    params: params,
  });
  return response.data;
};


export const getCoinMarketChart = async ({
  coinId,
  days = 365,
  interval,
  vsCurrency = 'usd'
}: CoinMarketChartParams): Promise<MarketChartData> => {
  const response = await coingeckoApi.get(`/coins/${coinId}/market_chart`, {
    params: {
      vs_currency: vsCurrency,
      days,
      interval,
    },
  });
  return response.data;
};

export const getCoinMarketChartRange = async ({
    coinId,
    days = 365,
    interval,
    vsCurrency = 'usd',
    from,
    to
  }: CoinMarketChartRangeParams): Promise<MarketChartData> => {
    const response = await coingeckoApi.get(`/coins/${coinId}/market_chart/range`, {
      params: {
        vs_currency: vsCurrency,
        days,
        interval,
        from,
        to
      },
    });
    return response.data;
  };

export const getCoinsMarkets = async (
  page: number = 1,
  perPage: number = 100,
  sparkline: boolean = false,
  order: string = 'market_cap_desc',
  priceChangePercentage?: string,
  category?: string,
  ids?: string
) => {
  const response = await coingeckoApi.get('/coins/markets', {
    params: {
      vs_currency: 'usd',
      order,
      per_page: perPage,
      page,
      sparkline,
      price_change_percentage: priceChangePercentage,
      category,
      ids,
    },
  });
  return response.data;
};





export const getSimplePrice = async ({
  ids,
  vsCurrencies,
  includeMarketCap = true,
  include24hrVol = true,
  include24hrChange = true,
  includeLastUpdatedAt = false
}: SimplePriceParams): Promise<SimplePrice> => {
  const response = await coingeckoApi.get('/simple/price', {
    params: {
      ids,
      vs_currencies: vsCurrencies,
      include_market_cap: includeMarketCap,
      include_24hr_vol: include24hrVol,
      include_24hr_change: include24hrChange,
      include_last_updated_at: includeLastUpdatedAt
    },
  });
  return response.data;
};