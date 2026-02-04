

export interface PublicTag {
  address_hash: string
  display_name: string
  label: string
}

export interface PrivateTag {
  address_hash: string
  display_name: string
  label: string
}

export interface WatchlistName {
  display_name: string
  label: string
}

export interface Meta {}

export interface Metadata {
  slug: string
  name: string
  tagType: string
  ordinal: number
  meta: Meta
}

export interface BaseAddress {
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

export interface From extends BaseAddress {}
export interface To extends BaseAddress {}
export interface CreatedContract extends BaseAddress {}
export interface Owner extends BaseAddress {}

export interface TokenTransfer {
  block_hash: string
  from: From2
  log_index: number
  method: string
  timestamp: string
  to: To2
  token: Token
  total: Total
  transaction_hash: string
  type: string
}

export interface From2 {
  hash: string
  implementation_name: string
  name: string
  ens_domain_name: string
  metadata: Metadata3
  is_contract: boolean
  private_tags: PrivateTag3[]
  watchlist_names: WatchlistName3[]
  public_tags: PublicTag3[]
  is_verified: boolean
}

export interface Metadata3 {
  slug: string
  name: string
  tagType: string
  ordinal: number
  meta: Meta3
}

export interface Meta3 {}

export interface PrivateTag3 {
  address_hash: string
  display_name: string
  label: string
}

export interface WatchlistName3 {
  display_name: string
  label: string
}

export interface PublicTag3 {
  address_hash: string
  display_name: string
  label: string
}

export interface To2 {
  hash: string
  implementation_name: string
  name: string
  ens_domain_name: string
  metadata: Metadata4
  is_contract: boolean
  private_tags: PrivateTag4[]
  watchlist_names: WatchlistName4[]
  public_tags: PublicTag4[]
  is_verified: boolean
}

export interface Metadata4 {
  slug: string
  name: string
  tagType: string
  ordinal: number
  meta: Meta4
}

export interface Meta4 {}

export interface PrivateTag4 {
  address_hash: string
  display_name: string
  label: string
}

export interface WatchlistName4 {
  display_name: string
  label: string
}

export interface PublicTag4 {
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

// Remove this duplicate interface
// export interface CreatedContract {
//   hash: string
//   implementation_name: string
//   name: string
//   ens_domain_name: string
//   metadata: Metadata5
//   is_contract: boolean
//   private_tags: PrivateTag5[]
//   watchlist_names: WatchlistName5[]
//   public_tags: PublicTag5[]
//   is_verified: boolean
// }

// Also remove these unused interfaces since CreatedContract now uses the base interfaces
// export interface Metadata5 {...}
// export interface Meta5 {...}
// export interface PrivateTag5 {...}
// export interface WatchlistName5 {...}
// export interface PublicTag5 {...}

export interface Metadata5 {
  slug: string
  name: string
  tagType: string
  ordinal: number
  meta: Meta5
}

export interface Meta5 {}

export interface PrivateTag5 {
  address_hash: string
  display_name: string
  label: string
}

export interface WatchlistName5 {
  display_name: string
  label: string
}

export interface PublicTag5 {
  address_hash: string
  display_name: string
  label: string
}

export interface Action {
  data: Data
  protocol: string
  type: string
}

export interface Data {
  debt_amount?: string
  debt_symbol?: string
  debt_address?: string
  collateral_amount?: string
  collateral_symbol?: string
  collateral_address?: string
  block_number?: number
  amount?: string
  symbol?: string
  address?: string
  name?: string
  to?: string
  ids?: string[]
  address0?: string
  address1?: string
  amount0?: string
  amount1?: string
  symbol0?: string
  symbol1?: string
}

export interface DecodedInput {
  method_call: string
  method_id: string
  parameters: Parameter[]
}

export interface Parameter {
  name: string
  type: string
  value: string
}

export interface NextPageParams {
  block_number: number
  index: number
  items_count: number
}


export interface AddressInfoResponse  {
    creator_address_hash: string
    creation_transaction_hash: string
    token: Token
    coin_balance: string
    exchange_rate: string
    implementation_address: string
    block_number_balance_updated_at: number
    hash: string
    implementation_name: string
    name: string
    ens_domain_name: string
    is_contract: boolean
    private_tags: PrivateTag[]
    watchlist_names: WatchlistName[]
    public_tags: PublicTag[]
    is_verified: boolean
    has_beacon_chain_withdrawals: boolean
    has_custom_methods_read: boolean
    has_custom_methods_write: boolean
    has_decompiled_code: boolean
    has_logs: boolean
    has_methods_read: boolean
    has_methods_write: boolean
    has_methods_read_proxy: boolean
    has_methods_write_proxy: boolean
    has_token_transfers: boolean
    has_tokens: boolean
    has_validated_blocks: boolean
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
  

  export type TokenBalancesResponse = TokenBalance[]

export interface TokenBalance {
  token_instance?: TokenInstance
  value?: string
  token_id?: string
  token?: Token
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
  metadata: Metadata  // Changed from Metadata2 to Metadata
  is_contract: boolean
  private_tags: PrivateTag[]
  watchlist_names: WatchlistName[]
  public_tags: PublicTag[]
  is_verified: boolean
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

