import React from 'react'
import { cn } from '@/lib/utils.ts'

interface LoadingProps extends React.HTMLAttributes<HTMLDivElement> {}

const Loading = React.forwardRef<HTMLDivElement, LoadingProps>(
  ({ className, ...props }, ref) => {
    return (
      <div className={cn('animate-spin rounded-full border-2 border-t-primary h-5 w-5', className)} ref={ref} {...props} />
    )
  }
)
Loading.displayName = 'Loading'

export { Loading }