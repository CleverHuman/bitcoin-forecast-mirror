import { createElement, useCallback, useState } from 'react'
import { AppDialog } from '@/components/app-dialog'

type ModalComponent<T = any> = (props: T) => JSX.Element
type BaseModalProps = Omit<React.ComponentProps<typeof AppDialog>, 'children'>
type ModalContentProps<T = any> = T

export const withDialog = <T extends object>(Component: ModalComponent<T>) => {
  const WrappedComponent = (props: T) => Component(props)
  WrappedComponent.isDialog = true
  return WrappedComponent
}

export const useModal = () => {
  const [isOpen, setIsOpen] = useState(false)
  const [modalProps, setModalProps] = useState<any>({})
  const [ModalComponent, setModalComponent] = useState<ModalComponent | null>(null)

  const showModal = useCallback(<T extends BaseModalProps>(
    Component: ModalComponent<ModalContentProps<T>>,
    contentProps?: T,
  ) => {
    setModalComponent(() => Component)
    setModalProps(contentProps || {})
    setIsOpen(true)
  }, [])

  const hideModal = useCallback(() => {
    setIsOpen(false)
    setModalComponent(null)
    setModalProps({})
  }, [])

  const renderModal = useCallback(() => {
    if (!ModalComponent) return null

    const isAppDialogComponent = (ModalComponent as any).isDialog === true

    if (isAppDialogComponent) {
      return createElement(ModalComponent, {
        ...modalProps,
        isOpen,
        onOpenChange: setIsOpen,
        onClose: hideModal,
      })
    }

    return (
      <AppDialog
        isOpen={isOpen}
        onOpenChange={setIsOpen}
        {...modalProps}
      >
        {createElement(ModalComponent, {
          ...modalProps,
          onClose: hideModal,
        })}
      </AppDialog>
    )
  }, [ModalComponent, modalProps, isOpen, hideModal])

  return {
    showModal,
    hideModal,
    renderModal,
  }
}