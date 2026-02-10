import { ColumnDef, flexRender } from "@tanstack/react-table";
import { Table as TableData } from '@tanstack/react-table'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "./ui/table";

interface TableViewProps<TData, TValue> {
    table: TableData<TData>
    columns: ColumnDef<TData, TValue>[]
    onItemClick?: (item: TData) => void
    showEmptyContent?: boolean
}
export function TableView<TData, TValue> ({table, columns, onItemClick, showEmptyContent = true}: TableViewProps<TData, TValue>) {
    return (
        <Table>
          <TableHeader>
            {table.getHeaderGroups().map((headerGroup) => (
              <TableRow key={headerGroup.id}>
                {headerGroup.headers.map((header) => {
                  return (
                    <TableHead key={header.id} colSpan={header.colSpan}>
                      {header.isPlaceholder
                        ? null
                        : flexRender(
                            header.column.columnDef.header,
                            header.getContext()
                          )}
                    </TableHead>
                  )
                })}
              </TableRow>
            ))}
          </TableHeader>
          <TableBody>
            {table.getRowModel().rows?.length ? (
              table.getRowModel().rows.map((row) => (
                <TableRow
                  key={row.id}
                  data-state={row.getIsSelected() && 'selected'}
                >
                  {row.getVisibleCells().map((cell) => (
                    <TableCell
                      key={cell.id}
                      className="cursor-pointer"
                      onClick={
                        onItemClick
                          ? () => onItemClick(row.original)
                          : undefined
                      }
                    >
                      {flexRender(
                        cell.column.columnDef.cell,
                        cell.getContext()
                      )}
                    </TableCell>
                  ))}
                </TableRow>
              ))
            ) : (
              showEmptyContent && (
                <TableRow>
                  <TableCell colSpan={columns.length} className="h-24 text-center">No results.</TableCell>
                </TableRow>
              )
            )}
          </TableBody>
        </Table>
    );
}