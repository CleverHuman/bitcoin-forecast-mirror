import * as React from 'react'
import {
  ColumnDef,
  ColumnFiltersState,
  SortingState,
  VisibilityState,
  getCoreRowModel,
  getFacetedRowModel,
  getFacetedUniqueValues,
  getFilteredRowModel,
  getSortedRowModel,
  useReactTable,
  OnChangeFn,
  Table,
  Row,
} from '@tanstack/react-table'

import { DataTableToolbar } from './data-table-toolbar'
import { TableView } from './table-view'
import { useEffect, useRef } from 'react'

interface DataTableProps<TData, TValue> {
  columns: ColumnDef<TData, TValue>[]
  data: TData[]
  hideToolbar?: boolean
  onItemClick?: (item: TData) => void
  onSearch?: (query: string) => void
  searchPlaceholder?: string
  onLoadMore?: () => Promise<void>
  hasMore: boolean
  isLoading: boolean
  onSort?: (key: string, desc: boolean) => void
  columnVisibility?: VisibilityState
  onColumnVisibilityChange?: OnChangeFn<VisibilityState>
  header?: React.ReactNode
  toolbar?: (table: Table<TData>) => React.ReactNode
  onTableReady?: (table: Table<TData>) => void
  onColumnFiltersChange?: (filters: ColumnFiltersState) => void
  emptyContent?: React.ReactNode
  viewType?: 'grid' | 'list'
  gridClassName?: string
  renderItem?: (item: Row<TData>) => React.ReactNode
}

export function DataEndlessTable<TData, TValue>({
                                                  columns,
                                                  data,
                                                  onItemClick,
                                                  hideToolbar,
                                                  onSearch,
                                                  searchPlaceholder,
                                                  onLoadMore,
                                                  hasMore,
                                                  isLoading,
                                                  onSort,
                                                  columnVisibility,
                                                  onColumnVisibilityChange,
                                                  header,
                                                  toolbar,
                                                  onTableReady,
                                                  onColumnFiltersChange,
                                                  emptyContent,
                                                  viewType = 'list',
                                                  renderItem,
                                                  gridClassName
                                                }: DataTableProps<TData, TValue>) {
  const [rowSelection, setRowSelection] = React.useState({})
  const [columnFilters, setColumnFilters] = React.useState<ColumnFiltersState>([])
  const [sorting, setSorting] = React.useState<SortingState>([])
  const containerRef = useRef<HTMLDivElement>(null)
  const isInitialMount = useRef(true)
  const prevFiltersRef = useRef<ColumnFiltersState>([])
  const loadingRef = useRef<HTMLDivElement>(null)

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
    onSortingChange: (updater) => {
      const newSorting = typeof updater === 'function' ? updater(sorting) : updater;
      if (sorting.length === 0 && newSorting.length > 0) {
        const descendingFirst = [{ ...newSorting[0], desc: true }];
        setSorting(descendingFirst);
        if (onSort) {
          onSort(descendingFirst[0].id, true);
        }
      } else {
        setSorting(newSorting);
        if (newSorting.length > 0 && onSort) {
          onSort(newSorting[0].id, newSorting[0].desc);
        }
      }
    },
    onColumnFiltersChange: setColumnFilters,
    onColumnVisibilityChange,
    getCoreRowModel: getCoreRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFacetedRowModel: getFacetedRowModel(),
    getFacetedUniqueValues: getFacetedUniqueValues(),
    manualFiltering: true
  })
  const currentFilters = table.getState().columnFilters
  useEffect(() => {
    const prevFilters = prevFiltersRef.current

    const hasChanged = JSON.stringify(currentFilters) !== JSON.stringify(prevFilters)

    if (hasChanged && onColumnFiltersChange) {
      onColumnFiltersChange(columnFilters)
      prevFiltersRef.current = currentFilters
    }
  }, [currentFilters, onColumnFiltersChange, columnFilters])

  useEffect(() => {
    if (isInitialMount.current && onTableReady) {
      isInitialMount.current = false
      onTableReady(table)
    }
  }, [table, onTableReady])

  useEffect(() => {
    if (!onLoadMore || !hasMore || isLoading || table.getRowCount() === 0) return;

    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting && hasMore && !isLoading) {
          onLoadMore();
        }
      },
      { threshold: 0.1 }
    );

    if (loadingRef.current) {
      observer.observe(loadingRef.current);
    }

    return () => {
      if (loadingRef.current) {
        observer.unobserve(loadingRef.current);
      }
    };
  }, [onLoadMore, hasMore, isLoading]);

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

  const renderContent = () => {
    if (data.length === 0) {
      return !isLoading && emptyContent;
    }

    if (viewType === 'grid') {
      return (
        <>
          <div className={`grid ${gridClassName ?? "grid-cols-1 md:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4 4xl:grid-cols-5 gap-4"}`}>
            {table.getRowModel().rows.map((row) => (
              <div key={row.id}>{renderItem ? renderItem(row) : defaultRenderItem(row)}</div>
            ))}
          </div>
          <div ref={loadingRef} className="h-10 flex items-center justify-center mt-4">
            {isLoading && <div className="text-sm text-muted-foreground">Loading more data...</div>}
          </div>
        </>
      );
    }

    return (
      <>
        <TableView table={table} columns={columns} onItemClick={onItemClick} />
        <div ref={loadingRef} className="h-10 flex items-center justify-center">
          {isLoading && <div className="text-sm text-muted-foreground">Loading more data...</div>}
        </div>
      </>
    );
  };

  return (
    <div className='space-y-4 w-full'>
      {toolbar ? toolbar(table) : !hideToolbar && <DataTableToolbar table={table} onSearch={onSearch} searchPlaceholder={searchPlaceholder} header={header} />}
      <div ref={containerRef} className='overflow-auto'>
        {isLoading && data.length === 0 ? renderSkeleton() : renderContent()}
      </div>
    </div>
  )
}