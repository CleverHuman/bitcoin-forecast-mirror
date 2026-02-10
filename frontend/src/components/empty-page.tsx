import { Button } from '@/components/ui/button'
import { useNavigate } from 'react-router-dom'

interface EmptyPageProps {
  title?: string
  description?: string
  hideButton?: boolean | false
  buttonText?: string
}

export default function EmptyPage({ title, description, hideButton }: EmptyPageProps) {
  const navigate = useNavigate()
  return (
    <div className="flex flex-col items-center justify-center py-12 px-4">
      <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-2">
        {title || 'No results found'}
      </h3>
      <p className="text-sm text-gray-500 dark:text-gray-400 text-center mb-4">
        {description || 'Try adjusting your search or filters to find what you\'re looking for.'}
      </p>
      {!hideButton && (
      <Button
        variant="outline"
        onClick={() => navigate(0)}
        >
          Refresh Page
        </Button>
      )}
    </div>
  )
} 