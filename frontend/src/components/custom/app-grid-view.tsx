import { cn } from '@/lib/utils'

interface AppGridViewProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode
}

export function AppGridView({ children, className, ...props }: AppGridViewProps) {
  return (
    <div
      className={cn(
        'grid gap-4 md:grid-cols-2 lg:grid-cols-3 2xl:grid-cols-4 3xl:grid-cols-5',
        className
      )}
      {...props}
    >
      {children}
    </div>
  )
}
