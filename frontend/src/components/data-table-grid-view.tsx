import * as React from 'react'
import {
  ColumnDef,
  ColumnFiltersState,
  Row,
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

import { DataTablePagination } from './data-table-pagination'
import { DataTableToolbar } from './data-table-toolbar'
import { TableView } from './table-view'
import { useEffect, useRef } from 'react'

interface DataTableGridViewProps<TData, TValue> {
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
  renderItem?: (item: Row<TData>) => React.ReactNode
  viewType?: 'grid' | 'list'
  pageCount?: number
  onPageChange?: (page: number) => void
  
}

export function DataTableGridView<TData, TValue>({
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
  isLoading,
  emptyContent,
  renderItem,
  viewType = 'grid',
  pageCount,
  onPageChange,
}: DataTableGridViewProps<TData, TValue>) {
  const [rowSelection, setRowSelection] = React.useState({})
  const [columnVisibility, setColumnVisibility] =
    React.useState<VisibilityState>({})
  const [columnFilters, setColumnFilters] = React.useState<ColumnFiltersState>(
    []
  )
  const [sorting, setSorting] = React.useState<SortingState>([])
  const isInitialMount = useRef(true)
  const prevFiltersRef = useRef<ColumnFiltersState>([])
  const [pagination, setPagination] = React.useState({
    pageIndex: 0,
    pageSize: 10,
  });
  
  // Add a ref to store the previous page index
  const prevPageIndexRef = useRef<number>(0);

  const table = useReactTable({
    data,
    columns,
    state: {
      sorting,
      columnVisibility,
      rowSelection,
      columnFilters,
      pagination,
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
    onPaginationChange: setPagination,
    manualPagination: Boolean(pageCount),
    pageCount: pageCount ?? Math.ceil(data.length / pagination.pageSize),
  })

  useEffect(() => {
    if (onPageChange && pagination.pageIndex !== prevPageIndexRef.current) {
      onPageChange(pagination.pageIndex);
      prevPageIndexRef.current = pagination.pageIndex;
    }
  }, [pagination.pageIndex, onPageChange]);

  useEffect(() => {
    console.log(columnFilters)
  }, [columnFilters])

  const columnFiltersState = table.getState().columnFilters

  useEffect(() => {
    const prevFilters = prevFiltersRef.current

    const hasChanged =
      JSON.stringify(columnFilters) !== JSON.stringify(prevFilters)

    if (hasChanged && onColumnFiltersChange) {
      onColumnFiltersChange(columnFilters)
      prevFiltersRef.current = columnFilters
    }
  }, [columnFiltersState, onColumnFiltersChange, columnFilters])

  useEffect(() => {
    if (isInitialMount.current && onTableReady) {
      isInitialMount.current = false
      onTableReady(table)
    }
  }, [table, onTableReady])


  const defaultRenderItem = (item: Row<TData>) => {
    return <div>{item.getValue('name') ?? item.id}</div>
  }

  const renderSkeleton = () => {
    if (viewType === 'grid') {
      return (
        <div className="grid grid-cols-2 gap-4">
          {[...Array(6)].map((_, index) => (
            <div 
              key={index} 
              className="h-32 bg-gray-100 rounded-lg animate-pulse"
            />
          ))}
        </div>
      )
    }
    
    return (
      <div className="space-y-3">
        {[...Array(5)].map((_, index) => (
          <div key={index} className="flex space-x-4">
            <div className="h-12 bg-gray-100 rounded w-full animate-pulse" />
          </div>
        ))}
      </div>
    )
  }

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
      {isLoading ? (
        renderSkeleton()
      ) : viewType === 'grid' ? (
        <div className={`${viewType === 'grid' ? 'grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4 4xl:grid-cols-5 gap-4' : ''}`}>
          {table.getRowModel().rows.map((row) => (
            <div key={row.id}>{renderItem ? renderItem(row) : defaultRenderItem(row)}</div>
          ))}
        </div>
      ) : (
        <div className=''>
           <TableView table={table} columns={columns} onItemClick={onItemClick} showEmptyContent={false} /> 
        </div>
      )}
      {emptyContent && table.getRowModel().rows.length === 0 && emptyContent}
      <DataTablePagination table={table} enablePageSize={enablePageSize} />
    </div>
  )
}
