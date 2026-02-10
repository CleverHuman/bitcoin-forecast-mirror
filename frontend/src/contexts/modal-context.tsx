import { createContext, useContext, ReactNode } from 'react'
import { useModal } from '@/hooks/use-modal'

type ModalContextType = ReturnType<typeof useModal>

const ModalContext = createContext<ModalContextType | undefined>(undefined)

export function ModalProvider({ children }: { children: ReactNode }) {
  const modal = useModal()

  return (
    <ModalContext.Provider value={modal}>
      {children}
      {modal.renderModal()}
    </ModalContext.Provider>
  )
}

export function useModalContext() {
  const context = useContext(ModalContext)
  if (!context) {
    throw new Error('useModalContext must be used within a ModalProvider')
  }
  return context
}