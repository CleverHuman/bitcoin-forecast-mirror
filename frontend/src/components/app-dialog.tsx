import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { X } from 'lucide-react'
import React, { useCallback, useEffect } from 'react'

export interface AppDialogProps {
  isOpen?: boolean
  onOpenChange?: (open: boolean) => void
  onConfirm?: () => void
  onCancel?: () => void
  title?: string | React.ReactNode
  description?: string
  confirmText?: string
  cancelText?: string
  children?: React.ReactNode
  variant?: 'default' | 'destructive'
  showCloseButton?: boolean
  onClose?: () => void
  hideFooter?: boolean
  options?: React.ReactNode
  size?: 'sm' | 'md' | 'lg' | 'xl' | '2xl' | 'full'
}

export function AppDialog({
                            isOpen,
                            onOpenChange,
                            onConfirm,
                            onCancel,
                            title,
                            description,
                            confirmText = 'Confirm',
                            cancelText = 'Cancel',
                            children,
                            variant = 'default',
                            showCloseButton = true,
                            onClose,
                            hideFooter = false,
                            options,
                            size = 'sm',
                          }: AppDialogProps) {
  const handleClose = () => {
    if (onClose) {
      onClose()
    } else {
      onOpenChange && onOpenChange(false)
    }
  }

  const handleCancel = () => {
    if (onCancel) {
      onCancel()
    } else {
      onOpenChange && onOpenChange(false)
    }
  }

  const handleConfirm = useCallback(() => {
    onConfirm && onConfirm()
    onOpenChange && onOpenChange(false)
  }, [onConfirm, onOpenChange])

  const sizeClasses = {
    sm: 'sm:max-w-[425px]',
    md: 'sm:max-w-[640px]',
    lg: 'sm:max-w-[768px]',
    xl: 'sm:max-w-[1024px]',
    '2xl': 'sm:max-w-[1280px]',
    full: 'sm:max-w-[calc(100vw-64px)]',
  }

  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.key === 'Enter') {
        e.preventDefault()
        handleConfirm()
      }
    }

    document.addEventListener('keydown', down)
    return () => document.removeEventListener('keydown', down)
  }, [handleConfirm])

  return (
    <Dialog open={isOpen} onOpenChange={onOpenChange}>
      <DialogContent
        className={`flex max-h-[calc(100vh-64px)] flex-col gap-0 overflow-clip p-0 ${sizeClasses[size]}`}
        withoutCloseButton
      >
        <div className='sticky top-0 z-50 bg-background px-6 py-4'>
          <div className='flex items-center justify-between'>
            <DialogHeader className='p-0'>
              <div className='flex flex-col'>
                {typeof title === 'string' ? (
                  <DialogTitle>{title}</DialogTitle>
                ) : (
                  title
                )}
                {description && (
                  <DialogDescription>{description}</DialogDescription>
                )}
              </div>
            </DialogHeader>
            {showCloseButton && (
              <div className='flex items-center gap-2'>
                {options}
                <Button variant='ghost' size='icon' onClick={handleClose}>
                  <X className='h-4 w-4' />
                </Button>
              </div>
            )}
          </div>
        </div>

        {children && (
          <div
            className={`min-h-0 flex-1 overflow-y-auto px-6 py-4 ${hideFooter ? 'border-t' : 'border-y'}`}
          >
            {children}
          </div>
        )}

        {!hideFooter && (
          <div className='sticky bottom-0 bg-background px-6 py-4'>
            <DialogFooter className='gap-2 sm:gap-0'>
              <Button variant='outline' onClick={handleCancel}>
                {cancelText}
              </Button>
              <Button variant={variant} onClick={handleConfirm}>
                {confirmText}
              </Button>
            </DialogFooter>
          </div>
        )}
      </DialogContent>
    </Dialog>
  )
}
