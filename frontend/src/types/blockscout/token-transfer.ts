export interface TokenTransferResponse {
    items: TokenTransfer[]
    next_page_params: NextPageParams
  }
  
  export interface TokenTransfer {
    block_hash: string
    from: From
    log_index: number
    method: string
    timestamp: string
    to: To
    token: Token
    total: Total
    transaction_hash: string
    type: string
  }
  
  export interface From {
    hash: string
    implementation_name: string
    name: string
    ens_domain_name: string
    metadata: Metadata
    is_contract: boolean
    private_tags: PrivateTag[]
    watchlist_names: WatchlistName[]
    public_tags: PublicTag[]
    is_verified: boolean
  }
  
  export interface Metadata {
    slug: string
    name: string
    tagType: string
    ordinal: number
    meta: Meta
  }
  
  export interface Meta {}
  
  export interface PrivateTag {
    address_hash: string
    display_name: string
    label: string
  }
  
  export interface WatchlistName {
    display_name: string
    label: string
  }
  
  export interface PublicTag {
    address_hash: string
    display_name: string
    label: string
  }
  
  export interface To {
    hash: string
    implementation_name: string
    name: string
    ens_domain_name: string
    metadata: Metadata2
    is_contract: boolean
    private_tags: PrivateTag2[]
    watchlist_names: WatchlistName2[]
    public_tags: PublicTag2[]
    is_verified: boolean
  }
  
  export interface Metadata2 {
    slug: string
    name: string
    tagType: string
    ordinal: number
    meta: Meta2
  }
  
  export interface Meta2 {}
  
  export interface PrivateTag2 {
    address_hash: string
    display_name: string
    label: string
  }
  
  export interface WatchlistName2 {
    display_name: string
    label: string
  }
  
  export interface PublicTag2 {
    address_hash: string
    display_name: string
    label: string
  }
  
  export interface Token {
    circulating_market_cap: string
    icon_url: string
    name: string
    decimals: string
    symbol: string
    address: string
    type: string
    holders: string
    exchange_rate: string
    total_supply: string
  }
  
  export interface Total {
    decimals: string
    value: string
  }
  
  export interface NextPageParams {
    block_number: number
    index: number
  }
  