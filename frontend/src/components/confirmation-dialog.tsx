import { AppDialog, AppDialogProps } from '@/components/app-dialog'

interface ConfirmationDialogProps extends Omit<AppDialogProps, 'onConfirm'> {
  onClose?: () => void
  onConfirm: () => void
  title: string
  description: string
  confirmText?: string
  cancelText?: string
  variant?: 'default' | 'destructive'
}

export function ConfirmationDialog({
  isOpen,
  onClose,
  onConfirm,
  title,
  description,
  confirmText = "Continue",
  cancelText = "Cancel",
  variant,
  ...props
}: ConfirmationDialogProps) {
  return (
    <AppDialog
      isOpen={isOpen}
      onOpenChange={() => {
        onClose?.()
        props.onOpenChange && props.onOpenChange(false)
      }}
      onConfirm={onConfirm}
      title={title}
      variant={variant}
      cancelText={cancelText}
      {...props}
    >
      <div>
        <p>{description}</p>
      </div>
    </AppDialog>
  )
}