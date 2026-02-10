import { useCallback, useEffect, useRef, useState } from 'react'

interface UseInfiniteScrollOptions {
  threshold?: number
  isLoading?: boolean
  hasMore?: boolean
  onLoadMore: () => Promise<void>
  containerRef?: React.RefObject<HTMLElement>
}

export function useInfiniteScroll({
  threshold = 0.8,
  isLoading = false,
  hasMore = false,
  onLoadMore,
  containerRef,
}: UseInfiniteScrollOptions) {
  const [isScrolling, setIsScrolling] = useState(false)
  const scrollTimeout = useRef<NodeJS.Timeout>()

  const handleScroll = useCallback(() => {
    if (scrollTimeout.current) {
      clearTimeout(scrollTimeout.current)
    }

    scrollTimeout.current = setTimeout(() => {
      if (isScrolling || isLoading || !hasMore) return

      let thresholdMet = false

      if (containerRef?.current) {
        const container = containerRef.current
        const scrollPosition = container.scrollTop + container.clientHeight
        const containerHeight = container.scrollHeight
        const thresholdPosition = containerHeight * threshold

        thresholdMet = scrollPosition >= thresholdPosition
      } else {
        const scrollPosition = window.scrollY + window.innerHeight
        const documentHeight = document.documentElement.scrollHeight
        const thresholdPosition = documentHeight * threshold

        thresholdMet = scrollPosition >= thresholdPosition
      }

      if (thresholdMet) {
        setIsScrolling(true)
        onLoadMore().finally(() => {
          setIsScrolling(false)
        })
      }
    }, 100)
  }, [threshold, isScrolling, isLoading, hasMore, onLoadMore, containerRef])

  useEffect(() => {
    const element = containerRef?.current || window
    const throttledScroll = () => {
      requestAnimationFrame(() => handleScroll())
    }

    element.addEventListener('scroll', throttledScroll, { passive: true })
    return () => {
      if (scrollTimeout.current) {
        clearTimeout(scrollTimeout.current)
      }
      element.removeEventListener('scroll', throttledScroll)
    }
  }, [handleScroll, containerRef])

  return {
    isScrolling,
  }
} 