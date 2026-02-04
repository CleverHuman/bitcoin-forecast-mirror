export interface CoinResponse {
  id: string
  symbol: string
  name: string
  web_slug: string
  asset_platform_id: string | null
  platforms: Record<string, string>
  detail_platforms: Record<string, DetailPlatform>
  block_time_in_minutes: number
  hashing_algorithm: string
  categories: string[]
  preview_listing: boolean
  public_notice: string | null
  additional_notices: string[]
  localization: Record<string, string>
  description: Record<string, string>
  links: Links
  image: {
    thumb: string
    small: string
    large: string
  }
  country_origin: string
  genesis_date: string
  sentiment_votes_up_percentage: number
  sentiment_votes_down_percentage: number
  watchlist_portfolio_users: number
  market_cap_rank: number
  market_data: MarketData
  tickers: Ticker[]
}

export interface DetailPlatform {
  decimal_place: number | null
  contract_address: string
}

export interface Links {
  homepage: string[]
  whitepaper: string
  blockchain_site: string[]
  official_forum_url: string[]
  chat_url: string[]
  announcement_url: string[]
  twitter_screen_name: string
  facebook_username: string
  bitcointalk_thread_identifier: number | null
  telegram_channel_identifier: string
  subreddit_url: string
  repos_url: {
    github: string[]
    bitbucket: string[]
  }
}

export interface MarketData {
  current_price: Record<string, number>
  total_value_locked: number | null
  mcap_to_tvl_ratio: number | null
  fdv_to_tvl_ratio: number | null
  roi: number | null
  ath: Record<string, number>
  ath_change_percentage: Record<string, number>
  ath_date: Record<string, string>
  atl: Record<string, number>
  atl_change_percentage: Record<string, number>
  atl_date: Record<string, string>
  market_cap: Record<string, number>
  market_cap_rank: number
  fully_diluted_valuation: Record<string, number>
  market_cap_fdv_ratio: number
  total_volume: Record<string, number>
  price_change_percentage_24h_in_currency: Record<string, number>
  high_24h: Record<string, number>
  low_24h: Record<string, number>
  circulating_supply: number
  total_supply: number
  max_supply: number
  price_change_24h_in_currency: Record<string, number>
  price_change_percentage_24h: number
}

interface Ticker {
  base: string
  target: string
  market: {
    name: string
    identifier: string
    has_trading_incentive: boolean
  }
  last: number
  volume: number
  converted_last: {
    btc: number
    eth: number
    usd: number
  }
  converted_volume: {
    btc: number
    eth: number
    usd: number
  }
  trust_score: string
  bid_ask_spread_percentage: number
  timestamp: string
  last_traded_at: string
  last_fetch_at: string
  is_anomaly: boolean
  is_stale: boolean
  trade_url: string
  token_info_url: string | null
  coin_id: string
  target_coin_id?: string
}

export interface SimplePrice {
  [key: string]: {
    usd: number
    usd_market_cap?: number
    usd_24h_vol?: number
    usd_24h_change?: number
    last_updated_at?: number
  }
}

export interface MarketChartData {
  prices: [number, number][]
  market_caps: [number, number][]
  total_volumes: [number, number][]
}

export interface SimplePriceParams {
  ids: string
  vsCurrencies: string
  includeMarketCap?: boolean
  include24hrVol?: boolean
  include24hrChange?: boolean
  includeLastUpdatedAt?: boolean
}

export interface CoinMarketChartParams {
  coinId: string
  days?: number | string
  interval?: string | 'daily'
  vsCurrency?: string
}

export interface CoinMarketChartRangeParams extends CoinMarketChartParams {
  from?: string
  to?: string
}

export interface CoinParams {
  localization?: boolean | false
  tickers?: boolean | false
  market_data?: boolean | false
  community_data?: boolean | false
  developer_data?: boolean | false
  sparkline?: boolean | false
}
