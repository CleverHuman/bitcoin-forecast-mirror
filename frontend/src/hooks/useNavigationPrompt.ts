import { useEffect, useCallback } from 'react'
import { useBlocker } from 'react-router-dom'

export function useNavigationPrompt(
  shouldBlock: boolean,
  onBeforeNavigate: () => Promise<boolean> | boolean
) {
  const blocker = useBlocker(shouldBlock)

  const handleBeforeUnload = useCallback(
    (event: BeforeUnloadEvent) => {
      if (shouldBlock) {
        event.preventDefault()
        event.returnValue = ''
      }
    },
    [shouldBlock]
  )

  useEffect(() => {
    if (blocker.state === 'blocked') {
      const proceed = async () => {
        const canProceed = await onBeforeNavigate()
        if (canProceed) {
          blocker.proceed()
        } else {
          blocker.reset()
        }
      }
      proceed()
    }
  }, [blocker, onBeforeNavigate])

  useEffect(() => {
    window.addEventListener('beforeunload', handleBeforeUnload)
    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload)
    }
  }, [handleBeforeUnload])

  return blocker
} 