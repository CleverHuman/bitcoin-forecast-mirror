export type BalanceHistoryResponse = BalanceHistoryItem[]

export interface BalanceHistoryItem {
  token_instance: TokenInstance
  value: string
  token_id: string
  token: Token2
}

export interface TokenInstance {
  is_unique: boolean
  id: string
  holder_address_hash: string
  image_url: string
  animation_url: string
  external_app_url: string
  metadata: Metadata
  owner: Owner
  token: Token
}

export interface Metadata {
  year: number
  tags: string[]
  name: string
  image_url: string
  home_url: string
  external_url: string
  description: string
  attributes: Attribute[]
}

export interface Attribute {
  value: string
  trait_type: string
}

export interface Owner {
  hash: string
  implementation_name: string
  name: string
  ens_domain_name: string
  metadata: Metadata2
  is_contract: boolean
  private_tags: PrivateTag[]
  watchlist_names: WatchlistName[]
  public_tags: PublicTag[]
  is_verified: boolean
}

export interface Metadata2 {
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

export interface Token2 {
  name: string
  decimals: string
  symbol: string
  address: string
  type: string
  holders: number
  exchange_rate: string
  total_supply: string
}
