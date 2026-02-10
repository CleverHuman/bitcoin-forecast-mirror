"use client";

import * as React from "react";
import { ChevronDown, Check, Plus } from "lucide-react";
import {
  IconCurrencyEthereum,
  IconSun,
  IconHexagon,
} from "@tabler/icons-react";

import { cn } from "@/lib/utils";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandSeparator,
} from "@/components/ui/command";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { PopoverTriggerProps } from "@radix-ui/react-popover";
import { useBlockchain } from "@/contexts/blockchain-context";
const apiBaseUrl = import.meta.env.VITE_API_BASE_URL;

const blockchainGroups = [
  {
    label: "Popular Tokens",
    chains: [
      {
        label: "Bitcoin",
        value: "bitcoin",
        icon: "BTC",
      },
      {
        label: "Ethereum",
        value: "ethereum",
        icon: "ETH",
      },
      {
        label: "Binance Smart Chain",
        value: "bsc",
        icon: "BSC",
      },
    ],
  },
  {
    label: "Other Tokens",
    chains: [
      {
        label: "Solana",
        value: "solana",
        icon: "SOL",
      },
      {
        label: "Polygon",
        value: "polygon",
        icon: "POLY",
      },
      {
        label: "Cardano",
        value: "cardano",
        icon: "CARDANO",
      },
    ],
  },
];

type Blockchain = (typeof blockchainGroups)[number]["chains"][number];

export interface BlockchainSwitcherProps extends PopoverTriggerProps {}

export default function BlockchainSwitcher({
  className,
}: BlockchainSwitcherProps) {
  const [open, setOpen] = React.useState(false);
  const [showNewBlockchainDialog, setShowNewBlockchainDialog] =
    React.useState(false);
  const { selectedBlockchain, setSelectedBlockchain } = useBlockchain();
  const [allBlockchainGroups, setAllBlockchainGroups] = React.useState([
    {
      label: "Popular Tokens",
      chains: [],
    },
  ]);
  const handleBlockchainChange = (blockchain: Blockchain) => {
    setSelectedBlockchain(blockchain);
    setOpen(false);
  };

  React.useEffect(() => {
    const fetchTokens = async () => {
      const token = localStorage.getItem("accessToken");
      try {
        const response = await fetch(`${apiBaseUrl}/forecasting/api/tokens/`, {
          method: "GET",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
        });
        const data = await response.json();
        const allTokens = data.results.map((token: any) => ({
          label: token.token_name,
          value: token.coingecko_id,
          icon: token.symbol,
        }));

        setAllBlockchainGroups([
          {
            label: "Popular Tokens",
            chains: allTokens,
          },
        ]);
      } catch (error) {
        console.error("Error fetching tokens:", error);
      }
    };

    fetchTokens();
  }, []);

  return (
    <Dialog
      open={showNewBlockchainDialog}
      onOpenChange={setShowNewBlockchainDialog}
    >
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <Button
            variant="outline"
            role="combobox"
            aria-expanded={open}
            aria-label="Select a token"
            className={cn(
              "w-[200px] justify-between",
              className,
              "sm:w-[200px] w-full",
            )}
          >
            <Avatar className="mr-2 h-5 w-5">
              <AvatarImage
                src={`https://avatar.vercel.sh/${selectedBlockchain.value}.png`}
                alt={selectedBlockchain.label}
              />
              <AvatarFallback>
                {React.createElement(selectedBlockchain.icon, {
                  className: "h-4 w-4",
                })}
              </AvatarFallback>
            </Avatar>
            {selectedBlockchain.label}
            <ChevronDown className="ml-auto h-4 w-4 shrink-0 opacity-50" />
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-[200px] p-0">
          <Command>
            <CommandInput placeholder="Search token..." />
            <CommandList>
              <CommandEmpty>No token found.</CommandEmpty>
              {allBlockchainGroups.map((group) => (
                <CommandGroup key={group.label} heading={group.label}>
                  {group.chains.map(
                    (chain: { label: string; value: string; icon: string }) => (
                      <CommandItem
                        key={chain.value}
                        onSelect={() => {
                          handleBlockchainChange(chain);
                          setOpen(false);
                        }}
                        className="text-sm"
                      >
                        <Avatar className="mr-2 h-5 w-5">
                          <AvatarImage
                            src={`https://avatar.vercel.sh/${chain.value}.png`}
                            alt={chain.label}
                          />
                          <AvatarFallback>
                            {React.createElement(chain.icon, {
                              className: "h-4 w-4",
                            })}
                          </AvatarFallback>
                        </Avatar>
                        {chain.label}
                        <Check
                          className={cn(
                            "ml-auto h-4 w-4",
                            selectedBlockchain.value === chain.value
                              ? "opacity-100"
                              : "opacity-0",
                          )}
                        />
                      </CommandItem>
                    ),
                  )}
                </CommandGroup>
              ))}
            </CommandList>
            <CommandSeparator />
            <CommandList>
              <CommandGroup>
                <DialogTrigger asChild>
                  <CommandItem
                    onSelect={() => {
                      setOpen(false);
                      setShowNewBlockchainDialog(true);
                    }}
                  >
                    <Plus className="mr-2 h-5 w-5" />
                    Add Custom Token
                  </CommandItem>
                </DialogTrigger>
              </CommandGroup>
            </CommandList>
          </Command>
        </PopoverContent>
      </Popover>
      <DialogContent>
        <DialogHeader>
          <DialogDescription>
            Add a new token to the list of available options.
          </DialogDescription>
        </DialogHeader>
        <div>
          <div className="space-y-4 py-2 pb-4">
            <div className="space-y-2">
              <Label htmlFor="name">Token name</Label>
              <Input id="name" placeholder="Custom Token" />
            </div>
            <div className="space-y-2">
              <Label htmlFor="icon">Token icon</Label>
              <Select>
                <SelectTrigger>
                  <SelectValue placeholder="Select an icon" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="chain">
                    <div className="flex items-center">
                      <IconHexagon className="mr-2 h-4 w-4" />
                      <span>Chain</span>
                    </div>
                  </SelectItem>
                  <SelectItem value="diamond">
                    <div className="flex items-center">
                      <IconCurrencyEthereum className="mr-2 h-4 w-4" />
                      <span>Diamond</span>
                    </div>
                  </SelectItem>
                  <SelectItem value="star">
                    <div className="flex items-center">
                      <IconSun className="mr-2 h-4 w-4" />
                      <span>Star</span>
                    </div>
                  </SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        </div>
        <DialogFooter>
          <Button
            variant="outline"
            onClick={() => setShowNewBlockchainDialog(false)}
          >
            Cancel
          </Button>
          <Button type="submit">Add Blockchain</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

// Export the Blockchain type and blockchainGroups for use in other components
export type { Blockchain };
export { blockchainGroups };
