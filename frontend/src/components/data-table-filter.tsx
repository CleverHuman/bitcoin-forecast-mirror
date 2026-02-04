import * as React from 'react'
import { CheckIcon, PlusCircledIcon } from '@radix-ui/react-icons'
import { Column } from '@tanstack/react-table'

import { cn } from '@/lib/utils'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/custom/button'
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandSeparator,
} from '@/components/ui/command'
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover'
import { Separator } from '@/components/ui/separator'

interface DataTableFilterProps<TData, TValue> {
  column?: Column<TData, TValue>
  title?: string
  options: {
    label: string
    value: string
    icon?: React.ComponentType<{ className?: string }>
  }[]
  loading?: boolean
  multiSelect?: boolean
  showFacets?: boolean
  showSelected?: boolean
}

export function DataTableFilter<TData, TValue>({
                                                 column,
                                                 title,
                                                 options,
                                                 loading = false,
                                                 multiSelect = true,
                                                 showFacets = false,
                                                 showSelected = true,
                                               }: DataTableFilterProps<TData, TValue>) {
  const facets = column?.getFacetedUniqueValues()
  const selectedValues = new Set(column?.getFilterValue() as string[])

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button
          variant='outline'
          size='sm'
          className='h-8 border-dashed'
          disabled={loading}
        >
          {loading ? (
            <div className='mr-2 h-4 w-4 animate-spin rounded-full border-2 border-primary border-t-transparent' />
          ) : (
            <PlusCircledIcon className='mr-2 h-4 w-4' />
          )}
          {title}
          {selectedValues?.size > 0 && (
            <>
              <Separator orientation='vertical' className='mx-2 h-4' />

              <Badge
                variant='secondary'
                className='rounded-sm px-1 font-normal lg:hidden'
              >
                {selectedValues.size}
              </Badge>
              <div className='hidden space-x-1 lg:flex'>
                {selectedValues.size > 2 || !showSelected ? (
                  <Badge
                    variant='secondary'
                    className='rounded-sm px-1 font-normal'
                  >
                    {selectedValues.size} selected
                  </Badge>
                ) : (
                  options
                    .filter((option) => selectedValues.has(option.value))
                    .map((option) => (
                      <Badge
                        variant='secondary'
                        key={option.value}
                        className='rounded-sm px-1 font-normal'
                      >
                        {option.label}
                      </Badge>
                    ))
                )}
              </div>
            </>
          )}
        </Button>
      </PopoverTrigger>
      <PopoverContent className='w-[200px] p-0' align='start'>
        <Command>
          <CommandInput placeholder={title} disabled={loading} />
          {loading ? (
            <CommandGroup>
              <div className='flex items-center justify-center py-6'>
                <div className='h-4 w-4 animate-spin rounded-full border-2 border-primary border-t-transparent' />
              </div>
            </CommandGroup>
          ) : (
            <>
              <CommandList>
                <CommandEmpty>No results found.</CommandEmpty>
                <CommandGroup>
                  {options.map((option) => {
                    const isSelected = selectedValues.has(option.value)
                    return (
                      <CommandItem
                        key={option.value}
                        onSelect={() => {
                          if (isSelected) {
                            selectedValues.delete(option.value)
                          } else {
                            if (!multiSelect) {
                              // Clear previous selections for single select mode
                              selectedValues.clear()
                            }
                            selectedValues.add(option.value)
                          }
                          const filterValues = Array.from(selectedValues)
                          column?.setFilterValue(
                            filterValues.length ? filterValues : undefined
                          )
                        }}
                      >
                        {multiSelect && (
                          <div
                            className={cn(
                              'mr-2 flex h-4 w-4 items-center justify-center rounded-sm border border-primary',
                              isSelected
                                ? 'bg-primary text-primary-foreground'
                                : 'opacity-50 [&_svg]:invisible'
                            )}
                          >
                            <CheckIcon className={cn('h-4 w-4')} />
                          </div>
                        )}
                        {!multiSelect && isSelected && (
                          <CheckIcon className='mr-2 h-4 w-4 text-primary' />
                        )}
                        {!multiSelect && !isSelected && (
                          <div className='mr-2 h-4 w-4' />
                        )}
                        {option.icon && (
                          <option.icon className='mr-2 h-4 w-4 text-muted-foreground' />
                        )}
                        <span>{option.label}</span>
                        {facets?.get(option.value) && showFacets && (
                          <span className='ml-auto flex h-4 w-4 items-center justify-center font-mono text-xs'>
                            {facets.get(option.value)}
                          </span>
                        )}
                      </CommandItem>
                    )
                  })}
                </CommandGroup>
                {selectedValues.size > 0 && (
                  <>
                    <CommandSeparator />
                    <CommandGroup>
                      <CommandItem
                        onSelect={() => column?.setFilterValue(undefined)}
                        className='justify-center text-center'
                      >
                        Clear filters
                      </CommandItem>
                    </CommandGroup>
                  </>
                )}
              </CommandList>
            </>
          )}
        </Command>
      </PopoverContent>
    </Popover>
  )
}
