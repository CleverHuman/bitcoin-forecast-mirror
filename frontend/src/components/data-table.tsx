import * as React from 'react'
import {
  ColumnDef,
  ColumnFiltersState,
  SortingState,
  Table,
  VisibilityState,
  getCoreRowModel,
  getFacetedRowModel,
  getFacetedUniqueValues,
  getFilteredRowModel,
  getPaginationRowModel,
  getSortedRowModel,
  useReactTable,
} from '@tanstack/react-table'

import { DataTablePagination } from '../components/data-table-pagination'
import { DataTableToolbar } from '../components/data-table-toolbar'
import { TableView } from './table-view'
import { useEffect, useRef } from 'react'

interface DataTableProps<TData, TValue> {
  columns: ColumnDef<TData, TValue>[]
  data: TData[]
  hideToolbar?: boolean
  onItemClick?: (item: TData) => void
  onSearch?: (query: string) => void
  searchPlacheholder?: string
  enablePageSize?: boolean
  toolbar?: (table: Table<TData>) => React.ReactNode
  onTableReady?: (table: Table<TData>) => void
  onColumnFiltersChange?: (filters: ColumnFiltersState) => void
  isLoading?: boolean
  emptyContent?: React.ReactNode
  content?: React.ReactNode
}

export function DataTable<TData, TValue>({
  columns,
  data,
  onItemClick,
  hideToolbar,
  onSearch,
  searchPlacheholder,
  enablePageSize,
  toolbar,
  onTableReady,
  onColumnFiltersChange,
  content,
}: DataTableProps<TData, TValue>) {
  const [rowSelection, setRowSelection] = React.useState({})
  const [columnVisibility, setColumnVisibility] =
    React.useState<VisibilityState>({})
  const [columnFilters, setColumnFilters] = React.useState<ColumnFiltersState>(
    []
  )
  const [sorting, setSorting] = React.useState<SortingState>([])
  const isInitialMount = useRef(true)
  const prevFiltersRef = useRef<ColumnFiltersState>([])

  const table = useReactTable({
    data,
    columns,
    state: {
      sorting,
      columnVisibility,
      rowSelection,
      columnFilters,
    },
    enableRowSelection: true,
    onRowSelectionChange: setRowSelection,
    onSortingChange: setSorting,
    onColumnFiltersChange: setColumnFilters,
    onColumnVisibilityChange: setColumnVisibility,
    getCoreRowModel: getCoreRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFacetedRowModel: getFacetedRowModel(),
    getFacetedUniqueValues: getFacetedUniqueValues(),
  })

  useEffect(() => {
    const currentFilters = table.getState().columnFilters
    const prevFilters = prevFiltersRef.current

    if (isInitialMount.current) {
      prevFiltersRef.current = currentFilters
      return
    }

    const hasChanged =
      JSON.stringify(currentFilters) !== JSON.stringify(prevFilters)

    if (hasChanged && onColumnFiltersChange) {
      onColumnFiltersChange(currentFilters)
      prevFiltersRef.current = currentFilters
    }
  }, [table.getState().columnFilters, onColumnFiltersChange])

  useEffect(() => {
    if (isInitialMount.current && onTableReady) {
      isInitialMount.current = false
      onTableReady(table)
    }
  }, [table, onTableReady])

  return (
    <div className='space-y-4'>
      {toolbar
        ? toolbar(table)
        : !hideToolbar && (
            <DataTableToolbar
              table={table}
              onSearch={onSearch}
              searchPlaceholder={searchPlacheholder}
            />
          )}
      <div className=''>
        {content ? content : <TableView table={table} columns={columns} onItemClick={onItemClick} /> }
      </div>
      <DataTablePagination table={table} enablePageSize={enablePageSize} />
    </div>
  )
}
