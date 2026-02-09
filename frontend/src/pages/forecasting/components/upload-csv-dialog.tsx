import React, { useRef, useState } from 'react'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
  DialogFooter,
} from '@/components/ui/dialog'
import { Upload } from 'lucide-react'
import Papa from 'papaparse'
import { useDispatch } from 'react-redux'
import { fetchForecastData } from '@/stores/slices/coingecko/coins-market'
import { Alert, AlertDescription } from '@/components/ui/alert'

interface UploadCSVDialogProps {
  onUploadSuccess?: () => void
}

const UploadCSVDialog: React.FC<UploadCSVDialogProps> = ({ onUploadSuccess }) => {
  const [open, setOpen] = useState(false)
  const [file, setFile] = useState<File | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isUploading, setIsUploading] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const dispatch = useDispatch()

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setError(null)
    const selectedFile = e.target.files?.[0]
    
    if (!selectedFile) {
      return
    }
    
    if (selectedFile.type !== 'text/csv' && !selectedFile.name.endsWith('.csv')) {
      setError('Please select a valid CSV file')
      return
    }
    
    setFile(selectedFile)
  }

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a file first')
      return
    }

    setIsUploading(true)
    setError(null)

    try {
      // Parse the CSV to validate format
      const reader = new FileReader()
      
      reader.onload = async (e) => {
        const text = e.target?.result as string
        
        Papa.parse(text, {
          header: true,
          skipEmptyLines: true,
          complete: async (results) => {
            // Check if the CSV has the required columns
            const headers = results.meta.fields || []
            const requiredColumns = ['ds', 'yhat_ensemble', 'yhat_ensemble_lower', 'yhat_ensemble_upper']

            const missingColumns = requiredColumns.filter(col => !headers.includes(col))
            
            if (missingColumns.length > 0) {
              setError(`CSV is missing required columns: ${missingColumns.join(', ')}`)
              setIsUploading(false)
              return
            }
            
            // Create a temporary URL for the file
            const blob = new Blob([text], { type: 'text/csv' })
            const url = URL.createObjectURL(blob)
            
            try {
              // Dispatch the action with the temporary URL
              await dispatch<any>(fetchForecastData(url))
              
              // Clean up
              URL.revokeObjectURL(url)
              setFile(null)
              setOpen(false)
              if (fileInputRef.current) fileInputRef.current.value = ''
              if (onUploadSuccess) onUploadSuccess()
            } catch (error) {
              setError('Failed to process the forecast data')
              URL.revokeObjectURL(url)
            }
            
            setIsUploading(false)
          },
          error: (error: any) => {
            setError(`Failed to parse CSV: ${error.message}`)
            setIsUploading(false)
          }
        })
      }
      
      reader.readAsText(file)
    } catch (error) {
      setError('An error occurred while processing the file')
      setIsUploading(false)
    }
  }

  const handleOpenChange = (newOpen: boolean) => {
    setOpen(newOpen)
    if (!newOpen) {
      setFile(null)
      setError(null)
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogTrigger asChild>
        <Button className="w-full sm:w-auto" variant="outline">
          <Upload className="mr-2 h-4 w-4" /> Upload CSV
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Upload Forecast CSV</DialogTitle>
        </DialogHeader>
        <div className="flex flex-col gap-4 py-4">
          {error && (
            <Alert variant="destructive">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
          
          <div className="grid w-full max-w-sm items-center gap-1.5">
            <label htmlFor="csv-file" className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
              Select CSV File
            </label>
            <input
              id="csv-file"
              type="file"
              ref={fileInputRef}
              accept=".csv"
              onChange={handleFileChange}
              className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
            />
            <p className="text-xs text-muted-foreground">
              The CSV file must include columns: ds, yhat_ensemble, yhat_ensemble_lower, yhat_ensemble_upper
            </p>
          </div>
        </div>
        <DialogFooter className="sm:justify-end">
          <Button
            type="button"
            variant="secondary"
            onClick={() => setOpen(false)}
            disabled={isUploading}
          >
            Cancel
          </Button>
          <Button
            type="button"
            onClick={handleUpload}
            disabled={!file || isUploading}
          >
            {isUploading ? 'Uploading...' : 'Upload and Run'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

export default UploadCSVDialog 