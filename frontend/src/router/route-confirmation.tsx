import { useNavigate } from 'react-router-dom'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Button } from '@/components/ui/button'
import { useState } from 'react'

interface RouteConfirmationProps {
  title: string
  description: string
  children: React.ReactNode
}

export default function RouteConfirmation({ title, description, children }: RouteConfirmationProps) {
  const [showConfirmation, setShowConfirmation] = useState(true)
  const navigate = useNavigate()

  if (!showConfirmation) {
    return <>{children}</>
  }

  return (
    <Dialog open={true} onOpenChange={() => navigate(-1)}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
          <DialogDescription>{description}</DialogDescription>
        </DialogHeader>
        <div className="flex justify-end gap-3 mt-6">
          <Button variant="outline" onClick={() => navigate(-1)}>
            Cancel
          </Button>
          <Button onClick={() => setShowConfirmation(false)}>
            Continue
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  )
}