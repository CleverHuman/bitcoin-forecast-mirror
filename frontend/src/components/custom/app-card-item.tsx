import { Card } from '@/components/ui/card'
import { cn } from '@/lib/utils'
import { forwardRef } from 'react'

interface HeaderProps extends React.HTMLAttributes<HTMLDivElement> {}
interface ContentProps extends React.HTMLAttributes<HTMLDivElement> {}
interface OptionsProps extends React.HTMLAttributes<HTMLDivElement> {}

interface AppCardItemProps extends React.ComponentPropsWithoutRef<typeof Card> {}

const Header = forwardRef<HTMLDivElement, HeaderProps>(
  ({ className, children, ...props }, ref) => (
    <div 
      ref={ref}
      className={cn('relative h-3/5', className)} 
      {...props}
    >
      {children}
    </div>
  )
)
Header.displayName = 'AppCardItem.Header'

const Content = forwardRef<HTMLDivElement, ContentProps>(
  ({ className, children, ...props }, ref) => (
    <div 
      ref={ref}
      className={cn('flex flex-col justify-between flex-1 space-y-1 py-2 px-4', className)} 
      {...props}
    >
      {children}
    </div>
  )
)
Content.displayName = 'AppCardItem.Content'

const Options = forwardRef<HTMLDivElement, OptionsProps>(
  ({ className, children, ...props }, ref) => (
    <div 
      ref={ref}
      className={cn('absolute right-2 top-2 flex flex-row gap-2', className)} 
      {...props}
    >
      {children}
    </div>
  )
)
Options.displayName = 'AppCardItem.Options'

const AppCardItemRoot = forwardRef<HTMLDivElement, AppCardItemProps>(
  ({ className, children, ...props }, ref) => (
    <Card
      ref={ref}
      className={cn('h-[200px] 2xl:h-[220px] cursor-pointer hover:bg-accent group relative', className)}
      {...props}
    >
      <div className='flex flex-col h-full'>
        {children}
      </div>
    </Card>
  )
)
AppCardItemRoot.displayName = 'AppCardItem'

export const AppCardItem = Object.assign(AppCardItemRoot, {
  Header,
  Content,
  Options,
})