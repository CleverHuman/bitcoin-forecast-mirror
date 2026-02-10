export interface TransactionsResponse {
    items: TransactionItem[]
    next_page_params: NextPageParams
  }
  
  export interface TransactionItem {
    timestamp: string
    fee: Fee
    gas_limit: number
    block_number: number
    status: string
    method: string
    confirmations: number
    type: number
    exchange_rate: string
    to: To
    transaction_burnt_fee: string
    max_fee_per_gas: string
    result: string
    hash: string
    gas_price: string
    priority_fee: string
    base_fee_per_gas: string
    from: From
    token_transfers: TokenTransfer[]
    transaction_types: string[]
    gas_used: string
    created_contract: CreatedContract
    position: number
    nonce: number
    has_error_in_internal_transactions: boolean
    actions: Action[]
    decoded_input: DecodedInput
    token_transfers_overflow: boolean
    raw_input: string
    value: string
    max_priority_fee_per_gas: string
    revert_reason: string
    confirmation_duration: number[]
    transaction_tag: string
  }
  
  export interface Fee {
    type: string
    value: string
  }
  
  export interface To {
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
  
  export interface From {
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
  
  export interface CreatedContract {
    hash: string
    implementation_name: string
    name: string
    ens_domain_name: string
    metadata: Metadata5
    is_contract: boolean
    private_tags: PrivateTag5[]
    watchlist_names: WatchlistName5[]
    public_tags: PublicTag5[]
    is_verified: boolean
  }
  
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
    [key: string]: any
  }
  