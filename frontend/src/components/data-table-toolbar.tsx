import { Table } from '@tanstack/react-table'

import { Input } from '@/components/ui/input'
import { DataTableViewOptions } from './data-table-view-options'

interface DataTableToolbarProps<TData> {
  table: Table<TData>
  onSearch?: (query: string) => void
  searchPlaceholder?: string
  header?: React.ReactNode
}

export function DataTableToolbar<TData>({
  table,
  onSearch,
  searchPlaceholder,
  header,
}: DataTableToolbarProps<TData>) {
  return (
    <div className='flex items-center justify-between'>
      <div className='flex flex-1 flex-col-reverse items-start gap-y-2 sm:flex-row sm:items-center sm:space-x-2'>
        {onSearch && (
          <Input
            placeholder={searchPlaceholder ?? 'Search crypto...'}
            value={(table.getColumn('name')?.getFilterValue() as string) ?? ''}
            onChange={(event) => {
              table.getColumn('name')?.setFilterValue(event.target.value)
              if (onSearch) {
                onSearch(event.target.value)
              }
            }}
            className='h-8 w-[150px] lg:w-[250px]'
          />
        )}
      </div>
      {header}
      <DataTableViewOptions table={table}  />
    </div>
  )
}
