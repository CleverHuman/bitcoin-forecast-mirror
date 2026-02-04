import { Link } from 'react-router-dom'
import { ChevronRight } from 'lucide-react'
import { cn } from '@/lib/utils'

interface BreadcrumbItem {
  label: string
  href: string
  current: boolean
}

interface BreadcrumbsProps {
  items: BreadcrumbItem[]
  className?: string
}

export function Breadcrumbs({ items, className }: BreadcrumbsProps) {
  return (
    <nav className={cn('flex', className)} aria-label="Breadcrumb">
      <ol className="flex items-center space-x-2">
        {items.map((item, index) => (
          <li key={item.href} className="flex items-center">
            {index > 0 && <ChevronRight className="h-4 w-4 text-gray-500" />}
            <Link
              to={item.href}
              className={cn(
                'ml-2',
                item.current
                  ? 'font-semibold text-gray-900'
                  : 'text-gray-500 hover:text-gray-700'
              )}
              aria-current={item.current ? 'page' : undefined}
            >
              {item.label.replace(/-/g, ' ').split(' ').map((word) => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}
            </Link>
          </li>
        ))}
      </ol>
    </nav>
  )
} 