"use client"

import * as React from "react"
import { ChevronDown, Check } from "lucide-react"
import { IconCoinBitcoin, IconCurrencyEthereum, IconBrandBinance, IconCurrencyDogecoin, IconCurrencyLitecoin, IconCurrencyMonero } from '@tabler/icons-react'

import { cn } from "@/lib/utils"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Button } from "@/components/ui/button"
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command"
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover"
import { PopoverTriggerProps } from "@radix-ui/react-popover"

const cryptoGroups = [
  {
    label: "Popular Cryptocurrencies",
    coins: [
      {
        label: "Bitcoin",
        value: "bitcoin",
        icon: IconCoinBitcoin,
      },
      {
        label: "Ethereum",
        value: "ethereum",
        icon: IconCurrencyEthereum,
      },
      {
        label: "Binance Coin",
        value: "binancecoin",
        icon: IconBrandBinance,
      },
    ],
  },
  {
    label: "Other Cryptocurrencies",
    coins: [
      {
        label: "Dogecoin",
        value: "dogecoin",
        icon: IconCurrencyDogecoin,
      },
      {
        label: "Litecoin",
        value: "litecoin",
        icon: IconCurrencyLitecoin,
      },
      {
        label: "Monero",
        value: "monero",
        icon: IconCurrencyMonero,
      },
    ],
  },
]

type Cryptocurrency = (typeof cryptoGroups)[number]["coins"][number]

export interface CryptoSwitcherProps extends PopoverTriggerProps {}

export default function CryptoSwitcher({ 
  className
}: CryptoSwitcherProps) {
  const [open, setOpen] = React.useState(false)
  const [selectedCrypto, setSelectedCrypto] = React.useState<Cryptocurrency>(cryptoGroups[0].coins[0])

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          role="combobox"
          aria-expanded={open}
          aria-label="Select a cryptocurrency"
          className={cn("w-[200px] justify-between", className, "sm:w-[200px] w-full")}
        >
          <Avatar className="mr-2 h-5 w-5">
            <AvatarImage
              src={`https://avatar.vercel.sh/${selectedCrypto.value}.png`}
              alt={selectedCrypto.label}
            />
            <AvatarFallback>
              {React.createElement(selectedCrypto.icon, { className: "h-4 w-4" })}
            </AvatarFallback>
          </Avatar>
          {selectedCrypto.label}
          <ChevronDown className="ml-auto h-4 w-4 shrink-0 opacity-50" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-[200px] p-0">
        <Command>
          <CommandInput placeholder="Search cryptocurrency..." />
          <CommandList>
            <CommandEmpty>No cryptocurrency found.</CommandEmpty>
            {cryptoGroups.map((group) => (
              <CommandGroup key={group.label} heading={group.label}>
                {group.coins.map((coin) => (
                  <CommandItem
                    key={coin.value}
                    onSelect={() => {
                      setSelectedCrypto(coin)
                      setOpen(false)
                    }}
                    className="text-sm"
                  >
                    <Avatar className="mr-2 h-5 w-5">
                      <AvatarImage
                        src={`https://avatar.vercel.sh/${coin.value}.png`}
                        alt={coin.label}
                      />
                      <AvatarFallback>
                        {React.createElement(coin.icon, { className: "h-4 w-4" })}
                      </AvatarFallback>
                    </Avatar>
                    {coin.label}
                    <Check
                      className={cn(
                        "ml-auto h-4 w-4",
                        selectedCrypto.value === coin.value
                          ? "opacity-100"
                          : "opacity-0"
                      )}
                    />
                  </CommandItem>
                ))}
              </CommandGroup>
            ))}
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  )
}

export type { Cryptocurrency }
export { cryptoGroups }