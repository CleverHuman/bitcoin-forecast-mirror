export interface CoinBalanceHistoryResponse {
    items: CoinBalanceHistoryItem[]
    next_page_params: NextPageParams
  }
  
  export interface CoinBalanceHistoryItem {
    transaction_hash: string
    block_number: number
    block_timestamp: string
    delta: string
    value: string
  }
  
  export interface NextPageParams {
    block_number: number
    items_count: number
  }
  