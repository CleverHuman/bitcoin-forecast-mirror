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
import { useApiPagination, BasePaginationParams } from '@/hooks/use-api-pagination'
import { BaseResponse } from '@/types/base-response'

interface DataTableFilterPaginatedProps<TData, TValue, TFilterOption> {
  column?: Column<TData, TValue>
  title?: string
  fetchOptions: (params: BasePaginationParams & { search?: string }) => Promise<BaseResponse<TFilterOption[]>>
  getOptionProps: (option: TFilterOption) => {
    label: string
    value: string
    icon?: React.ComponentType<{ className?: string }>
  }
  loading?: boolean
  multiSelect?: boolean
  showFacets?: boolean
  pageSize?: number
  showSelected?: boolean
  onFetchStateChange?: (state: { isLoading: boolean; isError?: boolean; data?: TFilterOption[] }) => void
}

export function DataTableFilterPaginated<TData, TValue, TFilterOption>({
  column,
  title,
  fetchOptions,
  getOptionProps,
  loading: externalLoading = false,
  multiSelect = true,
  showFacets = false,
  showSelected = true,
  pageSize = 10,
  onFetchStateChange,
}: DataTableFilterPaginatedProps<TData, TValue, TFilterOption>) {
  const facets = column?.getFacetedUniqueValues()
  const selectedValues = new Set(column?.getFilterValue() as string[])
  const [open, setOpen] = React.useState(false)
  const [search, setSearch] = React.useState('')
  
  const {
    data: options,
    isLoading,
    currentPage,
    pageCount,
    goToPage,
    updateParams,
    isInitialized
  } = useApiPagination({
    fetchFn: fetchOptions,
    pageSize,
    initialParams: { page: 1, search: '' }
  })

  const handleSearch = React.useCallback((value: string) => {
    setSearch(value)
    updateParams({ search: value, page: 1 })
  }, [updateParams])

  const loadMoreOptions = React.useCallback(() => {
    if (currentPage < pageCount) {
      goToPage(currentPage + 1)
    }
  }, [currentPage, pageCount, goToPage])

  const handleScroll = React.useCallback((e: React.UIEvent<HTMLDivElement>) => {
    const bottom = e.currentTarget.scrollHeight - e.currentTarget.scrollTop === e.currentTarget.clientHeight
    if (bottom && !isLoading && currentPage < pageCount) {
      loadMoreOptions()
    }
  }, [loadMoreOptions, isLoading, currentPage, pageCount])

  React.useEffect(() => {
    if (open && !isInitialized) {
      updateParams({ search: '' })
    }
  }, [open, isInitialized, updateParams])

  const mappedOptions = React.useMemo(() => 
    options.map(option => getOptionProps(option)), 
    [options, getOptionProps]
  )

  const isValueSelected = React.useCallback((value: string) => {
    const currentValues = (column?.getFilterValue() as (string | number)[] || []).map(String)
    return currentValues.includes(value)
  }, [column])

  React.useEffect(() => {
    console.log(mappedOptions.length)
  }, [mappedOptions])

  const previousState = React.useRef<{ isLoading: boolean; data: TFilterOption[] }>()

  React.useEffect(() => {
    if (!onFetchStateChange) return

    const currentState = {
      isLoading,
      data: options
    }

    if (
      !previousState.current ||
      previousState.current.isLoading !== currentState.isLoading ||
      previousState.current.data !== currentState.data
    ) {
      previousState.current = currentState
      onFetchStateChange(currentState)
    }
  }, [isLoading, options, onFetchStateChange])

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant='outline'
          size='sm'
          className='h-8 border-dashed'
          disabled={externalLoading}
        >
          {externalLoading ? (
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
                  mappedOptions
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
          <CommandInput 
            placeholder={title} 
            disabled={externalLoading || isLoading} 
            value={search}
            onValueChange={handleSearch}
          />
          <CommandList className="max-h-[200px] overflow-y-auto" onScroll={handleScroll}>
            <CommandEmpty>No results found.</CommandEmpty>
            <CommandGroup>
              {isLoading && !isInitialized ? (
                <div className='flex items-center justify-center py-6'>
                  <div className='h-4 w-4 animate-spin rounded-full border-2 border-primary border-t-transparent' />
                </div>
              ) : options.length === 0 && !isLoading ? (
                <div className="flex flex-col items-center justify-center py-6 text-sm text-muted-foreground">
                  <span>No options available</span>
                </div>
              ) : (
                <>
                  {mappedOptions.map((option) => {
                    const isSelected = isValueSelected(option.value)
                    return (
                      <CommandItem
                        key={option.value}
                        onSelect={() => {
                          const currentValues = new Set((column?.getFilterValue() as (string | number)[] || []).map(String))
                          if (isSelected) {
                            currentValues.delete(option.value)
                          } else {
                            if (!multiSelect) {
                              currentValues.clear()
                            }
                            currentValues.add(option.value)
                          }
                          const filterValues = Array.from(currentValues)
                          column?.setFilterValue(
                            filterValues.length ? filterValues : undefined
                          )
                        }}
                        className="flex items-center"
                      >
                        <div className="flex-shrink-0 flex items-center">
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
                        </div>
                        <span className="truncate flex-1 min-w-0">{option.label}</span>
                        {facets?.get(option.value) && showFacets && (
                          <span className='ml-auto flex-shrink-0 h-4 w-4 items-center justify-center font-mono text-xs'>
                            {facets.get(option.value)}
                          </span>
                        )}
                      </CommandItem>
                    )
                  })}
                  {isLoading && currentPage > 1 && (
                    <div className='flex items-center justify-center py-2'>
                      <div className='h-4 w-4 animate-spin rounded-full border-2 border-primary border-t-transparent' />
                    </div>
                  )}
                </>
              )}
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
        </Command>
      </PopoverContent>
    </Popover>
  )
}