'use client'

import * as React from 'react'
import { Dialog, DialogContent } from "@/components/ui/dialog"
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from "@/components/ui/command"
import { useSearch } from '@/contexts/search-context'
import { IconCoins, IconRouteSquare, IconWallet } from '@tabler/icons-react'

interface SearchResult {
  type: string
  name?: string | null
  symbol?: string | null
  value?: string | null
  hash?: string | null
}

export default function SearchModal() {
  const { isOpen, closeSearch } = useSearch()
  

  const [searchResults, setSearchResults] = React.useState<SearchResult[]>([])
  const mockResults: SearchResult[] = [
    { type: 'crypto', name: 'Bitcoin', symbol: 'BTC' },
    { type: 'crypto', name: 'Ethereum', symbol: 'ETH' },
    { type: 'crypto', name: 'Ripple', symbol: 'XRP' },
    { type: 'crypto', name: 'Litecoin', symbol: 'LTC' },
    { type: 'crypto', name: 'Cardano', symbol: 'ADA' },
    { type: 'crypto', name: 'Polkadot', symbol: 'DOT' },
    { type: 'crypto', name: 'Chainlink', symbol: 'LINK' },
    { type: 'crypto', name: 'Stellar', symbol: 'XLM' },
    { type: 'crypto', name: 'Dogecoin', symbol: 'DOGE' },
    { type: 'crypto', name: 'Uniswap', symbol: 'UNI' },
    { type: 'address', value: '0x742d35Cc6634C0532925a3b844Bc454e4438f44e' },
    { type: 'address', value: '0x1f9090aaE28b8a3dCeaDf281B0F12828e676c326' },
    { type: 'address', value: '0x3fC91A3afd70395Cd496C647d5a6CC9D4B2b7FAD' },
    { type: 'address', value: '0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D' },
    { type: 'address', value: '0x6B175474E89094C44Da98b954EedeAC495271d0F' },
    { type: 'transaction', hash: '0x123f681646d4a755815f9cb19e1acc8565a0c2ac' },
    { type: 'transaction', hash: '0x2a5c2ce7b5bdb8a5ef3f4c7c3c7f6c6d5e4d3c2b' },
    { type: 'transaction', hash: '0x3b6d3de8f4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s' },
    { type: 'transaction', hash: '0x4c7e4df9g0h1i2j3k4l5m6n7o8p9q0r1s2t3u4v' },
    { type: 'transaction', hash: '0x5d8f5eg0h1i2j3k4l5m6n7o8p9q0r1s2t3u4v5w' },
    { type: 'transaction', hash: '0x6e9f6fh1i2j3k4l5m6n7o8p9q0r1s2t3u4v5w6x' },
    { type: 'transaction', hash: '0x7f0g7gi2j3k4l5m6n7o8p9q0r1s2t3u4v5w6x7y' },
    { type: 'transaction', hash: '0x8g1h8hj3k4l5m6n7o8p9q0r1s2t3u4v5w6x7y8z' },
    { type: 'transaction', hash: '0x9h2i9ik4l5m6n7o8p9q0r1s2t3u4v5w6x7y8z9a' },
  ]

  React.useEffect(() => {
    setSearchResults(mockResults.slice(0, 5))
  }, [])

  const handleSearch = React.useCallback((query: string) => {
    const results = mockResults.filter(item => 
      item.name?.toLowerCase().includes(query.toLowerCase()) ||
      item.symbol?.toLowerCase().includes(query.toLowerCase()) ||
      item.value?.toLowerCase().includes(query.toLowerCase()) ||
      item.hash?.toLowerCase().includes(query.toLowerCase())
    )
    setSearchResults(results)
  }, [])

  const CommandItemLabel = React.memo(({ children }: { children: React.ReactNode }) => (
    <span className="ml-auto text-xs text-muted-foreground border rounded px-1 whitespace-nowrap">
      {children}
    </span>
  ));

  return (
    <Dialog open={isOpen} onOpenChange={closeSearch}>
      <DialogContent className="overflow-hidden p-0" withoutCloseButton>
        <Command className="[&_[cmdk-group-heading]]:px-2 [&_[cmdk-group-heading]]:font-medium [&_[cmdk-group-heading]]:text-muted-foreground [&_[cmdk-group]:not([hidden])_~[cmdk-group]]:pt-0 [&_[cmdk-group]]:px-2 [&_[cmdk-input-wrapper]_svg]:h-5 [&_[cmdk-input-wrapper]_svg]:w-5 [&_[cmdk-input]]:h-12 [&_[cmdk-item]]:px-2 [&_[cmdk-item]]:py-3 [&_[cmdk-item]_svg]:h-5 [&_[cmdk-item]_svg]:w-5">
          <CommandInput placeholder="Search for crypto, address, or transaction hash..." onValueChange={handleSearch} />
          <CommandList>
            <CommandEmpty>No results found.</CommandEmpty>
            <CommandGroup heading="Search Results">
              {searchResults.map((result, index) => (
                <CommandItem key={index} onSelect={() => {/* Handle selection */}}>
                  {result.type === 'crypto' && (
                    <>
                      <IconCoins size={18} />
                      <span className='mx-1'>{result.name} ({result.symbol})</span>
                      <CommandItemLabel>Tokens</CommandItemLabel>
                    </>
                  )}
                  {result.type === 'address' && (
                    <>
                      <IconWallet size={18} />
                      <span className='mx-1'>Address: {result.value}</span>
                      <CommandItemLabel>Account Profiler</CommandItemLabel>
                    </>
                  )}
                  {result.type === 'transaction' && (
                    <>
                      <IconRouteSquare size={18} />
                      <span className='mx-1'>Transaction: {result.hash}</span>
                      <CommandItemLabel>Blockchain tracer</CommandItemLabel>
                    </>
                  )}
                </CommandItem>
              ))}
            </CommandGroup>
          </CommandList>
        </Command>
      </DialogContent>
    </Dialog>
  )
}
