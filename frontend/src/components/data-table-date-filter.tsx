import { CalendarIcon, Cross2Icon } from '@radix-ui/react-icons'
import { Column } from '@tanstack/react-table'
import { format } from 'date-fns'

import { Button } from '@/components/custom/button'
import { Calendar } from '@/components/ui/calendar'
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover'
import { Separator } from '@/components/ui/separator'

interface DataTableDateFilterProps<TData, TValue> {
  column?: Column<TData, TValue>
  title?: string
  loading?: boolean
}

export function DataTableDateFilter<TData, TValue>({
  column,
  title = 'Pick date',
  loading = false,
}: DataTableDateFilterProps<TData, TValue>) {
  const selectedDate = column?.getFilterValue() as string | undefined

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
            <div className="mr-2 h-4 w-4 animate-spin rounded-full border-2 border-primary border-t-transparent" />
          ) : (
            <CalendarIcon className='mr-2 h-4 w-4' />
          )}
          {selectedDate ? (
            format(new Date(selectedDate), 'PP')
          ) : (
            title
          )}
          {selectedDate && (
            <>
              <Separator orientation='vertical' className='mx-2 h-4' />
              <Cross2Icon
                className='h-4 w-4'
                onClick={(e) => {
                  e.stopPropagation()
                  column?.setFilterValue(undefined)
                }}
              />
            </>
          )}
        </Button>
      </PopoverTrigger>
      <PopoverContent className='w-auto p-0' align='start'>
        {loading ? (
          <div className="flex items-center justify-center p-6">
            <div className="h-4 w-4 animate-spin rounded-full border-2 border-primary border-t-transparent" />
          </div>
        ) : (
          <Calendar
            mode='single'
            selected={selectedDate ? new Date(selectedDate) : undefined}
            onSelect={(date) => {
                column?.setFilterValue(date?.toISOString().split('T')[0])
            }}
            disabled={(date) => date > new Date()}
            initialFocus
          />
        )}
      </PopoverContent>
    </Popover>
  )
} 