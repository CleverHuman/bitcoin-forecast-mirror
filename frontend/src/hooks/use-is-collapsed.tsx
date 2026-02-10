import { useEffect } from 'react'
import useLocalStorage from './use-local-storage'

type UseIsCollapsedProps = {  
  defaultValue?: boolean | undefined
  key?: string | undefined
}

export default function useIsCollapsed({defaultValue, key}: UseIsCollapsedProps = {
  key: 'collapsed-sidebar',
  defaultValue: false,
}) {
  const [isCollapsed, setIsCollapsed] = useLocalStorage({
    key: key || 'collapsed-sidebar',
    defaultValue: defaultValue || false,
  })

  useEffect(() => {
    const handleResize = () => {
      // Update isCollapsed based on window.innerWidth
      setIsCollapsed(window.innerWidth < 768 ? false : isCollapsed)
    }

    // Initial setup
    handleResize()

    // Add event listener for window resize
    window.addEventListener('resize', handleResize)

    // Cleanup event listener on component unmount
    return () => {
      window.removeEventListener('resize', handleResize)
    }
  }, [isCollapsed, setIsCollapsed])

  return [isCollapsed, setIsCollapsed] as const
}
