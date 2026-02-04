import * as React from "react"
import { Check } from "lucide-react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
} from "@/components/ui/command"
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover"
import { FormControl, FormItem, FormLabel, FormMessage } from "@/components/ui/form"
import { ControllerRenderProps } from "react-hook-form"
import { ForecastFormData } from "./form-forecast-dialog"
import { useBlockchain } from "@/contexts/blockchain-context"
import { symbols } from "../data/dummy"


interface SymbolSelectProps {
  field: ControllerRenderProps<ForecastFormData, "trading_symbol">
}

export function SymbolSelect({ field }: SymbolSelectProps) {
  const [open, setOpen] = React.useState(false)
  const { selectedBlockchain } = useBlockchain()

  const blockchainSymbols = symbols[selectedBlockchain.value as keyof typeof symbols] || []

  return (
    <FormItem className="w-full">
      <FormLabel>Token Pair</FormLabel>
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <FormControl>
            <Button
              variant="outline"
              role="combobox"
              aria-expanded={open}
              className="w-full justify-between"
            >
              {field.value && field.value !== ""
                ? blockchainSymbols.find((symbol) => symbol.value === field.value)?.label
                : "Select token..."}
            </Button>
          </FormControl>
        </PopoverTrigger>
        <PopoverContent className="w-[--radix-popover-trigger-width] p-0" align="start">
          <Command>
            <CommandInput 
              placeholder="Search token..." 
              className="h-9"
            />
            <CommandEmpty>No token found.</CommandEmpty>
            <CommandGroup className="max-h-[200px] overflow-y-auto">
              {blockchainSymbols.map((symbol) => (
                <CommandItem
                  key={symbol.value}
                  value={symbol.value}
                  onSelect={() => {
                    field.onChange(symbol.value)
                    setOpen(false)
                  }}
                >
                  <Check
                    className={cn(
                      "mr-2 h-4 w-4",
                      field.value === symbol.value ? "opacity-100" : "opacity-0"
                    )}
                  />
                  {symbol.label}
                </CommandItem>
              ))}
            </CommandGroup>
          </Command>
        </PopoverContent>
      </Popover>
      <FormMessage />
    </FormItem>
  )
}