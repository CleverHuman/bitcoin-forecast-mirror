export type ScreenerOption = 'Token Flow' | 'Price / Market Cap' | 'On-chain Metrics'
export type TimeRange = '24h' | '7d' | '30d' | '24h,7d,30d'


export interface FetchTokensParams {
    sector?: string;
    timeRange: TimeRange;
    query?: string;
    page?: number;
    perPage?: number;
    order?: string | 'market_cap_desc' | 'market_cap_asc' | 'volume_desc' | 'volume_asc' | 'price_desc' | 'price_asc' | 'name_desc' | 'name_asc';
    sparkline?: boolean;
  }
  

export interface TokenData {
    id: string | null;
    symbol: string | null;
    name: string | null;
    image: string | null;
    current_price: number | null;
    market_cap: number | null;
    market_cap_rank: number | null;
    fully_diluted_valuation: number | null;
    total_volume: number | null;
    high_24h: number | null;
    low_24h: number | null;
    price_change_24h: number | null;
    price_change_percentage_24h: number | null;
    market_cap_change_24h: number | null;
    market_cap_change_percentage_24h: number | null;
    circulating_supply: number | null;
    total_supply: number | null;
    sparkline_in_7d?: {
      price: number[];
    };
    max_supply: number | null;
    ath: number | null;
    ath_change_percentage: number | null;
    ath_date: string | null;
    atl: number | null;
    atl_change_percentage: number | null;
    atl_date: string | null;
    roi: number | null;
    last_updated: string | null;
    price_change_percentage_24h_in_currency: number | null;
    price_change_percentage_30d_in_currency: number | null;
    price_change_percentage_7d_in_currency: number | null;
  }

  
export type TokenParams = {
  vs_currency: string;
  order: string;
  per_page: number;
  page: number;
  sparkline: boolean;
  category?: string;
  price_change_percentage?: string;
  ids?: string;
};