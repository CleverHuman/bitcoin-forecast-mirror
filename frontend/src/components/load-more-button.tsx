import { Loader2 } from 'lucide-react'
import React from 'react'
import { Button } from './ui/button'


interface LoadMoreButtonProps {
  onClick: () => void
  isLoading: boolean
}

const LoadMoreButton: React.FC<LoadMoreButtonProps> = ({ onClick, isLoading }) => {
  return (
    <div className="flex justify-center mt-4">
      <Button
        onClick={onClick}
        disabled={isLoading}
      >
        {isLoading ? (
          <div className="flex items-center gap-2">
            <Loader2 className="h-4 w-4 animate-spin" />
            Loading more...
          </div>
        ) : (
          'Load More'
        )}
      </Button>
    </div>
  )
}

export default LoadMoreButton 