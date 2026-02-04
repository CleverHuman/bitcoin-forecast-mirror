'use client'

import { forwardRef, useMemo, useState } from 'react'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Input } from '@/components/ui/input'
import { Palette } from 'lucide-react'

const colorPalette = {
  grays: ['#000000', '#262626', '#434343', '#595959', '#737373', '#8C8C8C', '#A6A6A6', '#BFBFBF', '#D9D9D9', '#F0F0F0'],
  colors: [
    ['#F5222D', '#FA8C16', '#FADB14', '#52C41A', '#13C2C2', '#1890FF', '#722ED1', '#EB2F96', '#A389D3'],
    ['#FF4D4F', '#FFA940', '#FFEC3D', '#73D13D', '#36CFC9', '#40A9FF', '#9254DE', '#F759AB', '#B37FEB'],
    ['#FF7875', '#FFC069', '#FFF566', '#95DE64', '#5CDBD3', '#69C0FF', '#B37FEB', '#FF85C0', '#C5A3FF'],
    ['#FFA39E', '#FFD591', '#FFF899', '#B7EB8F', '#87E8DE', '#91D5FF', '#D3ADF7', '#FFA7D3', '#D3ADF7'],
    ['#FFC1B6', '#FFE3BA', '#FFFBC1', '#D9F7BE', '#B5F5EC', '#BAE7FF', '#EFE6FF', '#FFC3E5', '#EFE6FF'],
    ['#FFE0D9', '#FFF1E1', '#FFFDE8', '#F4FFB8', '#E6FFFB', '#E6F7FF', '#F9F0FF', '#FFE5F3', '#F9F0FF'],
  ]
}

interface ColorPickerProps {
  value: string
  onChange: (value: string) => void
  onClear?: () => void
}

export const ColorPicker = forwardRef<HTMLInputElement, ColorPickerProps>(
  ({ value, onChange, onClear }, ref) => {
    const [open, setOpen] = useState(false)

    const parsedValue = useMemo(() => {
      return value || '#000000'
    }, [value])

    return (
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <Button
            variant="outline"
            size="icon"
            className="w-[40px] p-0"
            title='Choose color'
            style={{
              backgroundColor: parsedValue
            }}
          >
            <Palette className="h-4 w-4" />
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-[320px] p-4">
          <div className="space-y-4">
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium">Color</span>
              <Input
                ref={ref}
                type="text"
                value={parsedValue}
                className="h-8 w-24"
                readOnly
              />
              <div
                className="w-8 h-8 rounded-md border"
                style={{ backgroundColor: parsedValue }}
              />
            </div>

            <div className="space-y-2">
              <div className="grid grid-cols-10 gap-1">
                {colorPalette.grays.map((color) => (
                  <button
                    key={color}
                    className={cn(
                      "h-6 w-6 rounded-md border",
                      "hover:scale-110 transition-transform",
                      "ring-offset-background",
                      "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                    )}
                    style={{ backgroundColor: color }}
                    onClick={() => onChange(color)}
                  />
                ))}
              </div>

              <div className="grid grid-cols-9 gap-1">
                {colorPalette.colors[0].map((_, colIndex) => (
                  <div key={colIndex} className="space-y-1">
                    {colorPalette.colors.map((row, rowIndex) => (
                      <button
                        key={`${rowIndex}-${colIndex}`}
                        className={cn(
                          "h-6 w-6 rounded-md border",
                          "hover:scale-110 transition-transform",
                          "ring-offset-background",
                          "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                        )}
                        style={{ backgroundColor: row[colIndex] }}
                        onClick={() => onChange(row[colIndex])}
                      />
                    ))}
                  </div>
                ))}
              </div>
            </div>

            {onClear && (
              <Button
                variant="outline"
                size="sm"
                className="w-full"
                onClick={onClear}
              >
                Clear color
              </Button>
            )}
          </div>
        </PopoverContent>
      </Popover>
    )
  }
)

ColorPicker.displayName = 'ColorPicker' 