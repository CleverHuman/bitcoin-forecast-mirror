import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
  DialogTrigger,
} from '@/components/ui/dialog'
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from '@/components/ui/form'
import { Input } from '@/components/ui/input'
import { Switch } from '@/components/ui/switch'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { zodResolver } from '@hookform/resolvers/zod'
import { useForm } from 'react-hook-form'
import * as z from 'zod'
import { SymbolSelect } from './symbol-select'
import { PlusIcon } from 'lucide-react'
import { ForecastHistoryItem } from '../data/dummy'
import { useEffect } from 'react'

const formSchema = z.object({
  start_date: z.string().min(1, 'Start date is required'),
  end_date: z.string().min(1, 'End date is required'),
  trading_symbol: z.string(),
  side: z.string().default('Sell'),
  dates: z.array(z.string()).default(['2024-04-19']),
  events: z.array(z.string()).default(['4th Halving']),
  interval_width: z.number().default(0.95),
  growth: z.enum(['linear', 'logistic']).default('linear'),
  seasonality_mode: z
    .enum(['additive', 'multiplicative'])
    .default('multiplicative'),
  changepoint_prior_scale: z.number().default(0.1),
  changepoint_range: z.number().default(0.8),
  n_changepoints: z.number().default(300),
  seasonality_prior_scale: z.number().default(10),
  daily_seasonality: z.boolean().default(false),
  weekly_seasonality: z.boolean().default(true),
  yearly_seasonality: z.boolean().default(true),
  holidays: z.any().optional(),
  cap: z.number().optional(),
  floor: z.number().optional(),
  trading_volume: z.any().optional(),
})

export type ForecastFormData = z.infer<typeof formSchema>

interface FormForecastDialogProps {
  forecast?: ForecastHistoryItem | null
  open: boolean
  onOpenChange: (open: boolean) => void
  onSubmit: (data: ForecastFormData) => void
}

export const FormForecastDialog: React.FC<FormForecastDialogProps> = ({
  open,
  onOpenChange,
  onSubmit,
  forecast,
}) => {
  const form = useForm<ForecastFormData>({
    resolver: zodResolver(formSchema),
  })

  useEffect(() => {
    if (open && !forecast) {
      form.reset({
        start_date: new Date().toISOString().split('T')[0],
        end_date: new Date(new Date().setDate(new Date().getDate() + 1))
          .toISOString()
          .split('T')[0],
        side: 'Sell',
        trading_symbol: '',
        dates: ['2024-04-19'],
        events: ['4th Halving'],
        interval_width: 0.95,
        growth: 'linear',
        seasonality_mode: 'multiplicative',
        changepoint_prior_scale: 0.1,
        changepoint_range: 0.8,
        n_changepoints: 300,
        seasonality_prior_scale: 10,
        daily_seasonality: false,
        weekly_seasonality: true,
        yearly_seasonality: true,
        cap: undefined,
        floor: undefined,
        trading_volume: undefined,
      })
    }
  }, [open, form, forecast])

  useEffect(() => {
    if (forecast) {
      const formattedForecast = {
        ...forecast,
        side:
          forecast.side?.charAt(0).toUpperCase() +
            forecast.side?.slice(1).toLowerCase() || 'Sell',
        start_date:
          forecast.start_date || new Date().toISOString().split('T')[0],
        end_date:
          forecast.end_date ||
          new Date(new Date().setDate(new Date().getDate() + 1))
            .toISOString()
            .split('T')[0],
        trading_symbol: forecast.trading_symbol || '',
        dates: forecast.dates || ['2024-04-19'],
        events: forecast.events || ['4th Halving'],
        interval_width: forecast.interval_width || 0.95,
        growth: forecast.growth || 'linear',
        seasonality_mode: forecast.seasonality_mode || 'multiplicative',
        changepoint_prior_scale: forecast.changepoint_prior_scale || 0.1,
        changepoint_range: forecast.changepoint_range || 0.8,
        n_changepoints: forecast.n_changepoints || 300,
        seasonality_prior_scale: forecast.seasonality_prior_scale || 10,
        daily_seasonality: forecast.daily_seasonality || false,
        weekly_seasonality: forecast.weekly_seasonality || true,
        yearly_seasonality: forecast.yearly_seasonality || true,
      }
      form.reset(formattedForecast)
    }
  }, [forecast, form])

  const handleSubmit = (values: ForecastFormData) => {
    onSubmit(values)
    onOpenChange(false)
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogTrigger asChild>
        <Button className='w-full sm:w-auto' variant='outline' onClick={() => onOpenChange(true)}>
          <PlusIcon className='mr-2 h-4 w-4' />
          Create Forecast
        </Button>
      </DialogTrigger>
      <DialogContent className='max-w-lg '>
        <DialogHeader>
          <DialogTitle>Create New Forecast</DialogTitle>
        </DialogHeader>
        <div className='max-h-[80vh] overflow-y-auto pl-1 pr-4'>
          <Form {...form}>
            <form
              onSubmit={form.handleSubmit(handleSubmit)}
              className='space-y-2'
            >
              <div className='grid grid-cols-2 gap-4'>
                <FormField
                  control={form.control}
                  name='start_date'
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Start Date</FormLabel>
                      <FormControl>
                        <Input type='date' {...field} />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <FormField
                  control={form.control}
                  name='end_date'
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>End Date</FormLabel>
                      <FormControl>
                        <Input type='date' {...field} />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
              </div>
              <FormField
                control={form.control}
                name='trading_symbol'
                render={({ field }) => <SymbolSelect field={field} />}
              />
              <FormField
                control={form.control}
                name='side'
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Side</FormLabel>
                    <Select onValueChange={field.onChange} value={field.value}>
                      <FormControl>
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                      </FormControl>
                      <SelectContent>
                        <SelectItem value='Buy'>Buy</SelectItem>
                        <SelectItem value='Sell'>Sell</SelectItem>
                      </SelectContent>
                    </Select>
                    <FormMessage />
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name='growth'
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Growth Model</FormLabel>
                    <Select onValueChange={field.onChange} value={field.value}>
                      <FormControl>
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                      </FormControl>
                      <SelectContent>
                        <SelectItem value='linear'>Linear</SelectItem>
                        <SelectItem value='logistic'>Logistic</SelectItem>
                      </SelectContent>
                    </Select>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <FormField
                control={form.control}
                name='seasonality_mode'
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Seasonality Mode</FormLabel>
                    <Select onValueChange={field.onChange} value={field.value}>
                      <FormControl>
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                      </FormControl>
                      <SelectContent>
                        <SelectItem value='additive'>Additive</SelectItem>
                        <SelectItem value='multiplicative'>
                          Multiplicative
                        </SelectItem>
                      </SelectContent>
                    </Select>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <FormField
                control={form.control}
                name='interval_width'
                render={({ field: { value, onChange, ...field } }) => (
                  <FormItem>
                    <FormLabel>Confidence Interval</FormLabel>
                    <FormControl>
                      <Input
                        type='number'
                        step='0.01'
                        min='0'
                        max='1'
                        {...field}
                        value={value}
                        onChange={(event) =>
                          onChange(parseFloat(event.target.value))
                        }
                      />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name='daily_seasonality'
                render={({ field }) => (
                  <FormItem className='flex items-center justify-between space-x-2'>
                    <FormLabel>Daily Seasonality</FormLabel>
                    <FormControl>
                      <Switch
                        checked={field.value}
                        onCheckedChange={field.onChange}
                      />
                    </FormControl>
                  </FormItem>
                )}
              />
              <FormField
                control={form.control}
                name='weekly_seasonality'
                render={({ field }) => (
                  <FormItem className='flex items-center justify-between space-x-2'>
                    <FormLabel>Weekly Seasonality</FormLabel>
                    <FormControl>
                      <Switch
                        checked={field.value}
                        onCheckedChange={field.onChange}
                      />
                    </FormControl>
                  </FormItem>
                )}
              />
              <FormField
                control={form.control}
                name='yearly_seasonality'
                render={({ field }) => (
                  <FormItem className='flex items-center justify-between space-x-2'>
                    <FormLabel>Yearly Seasonality</FormLabel>
                    <FormControl>
                      <Switch
                        checked={field.value}
                        onCheckedChange={field.onChange}
                      />
                    </FormControl>
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name='seasonality_prior_scale'
                render={({ field: { value, onChange, ...field } }) => (
                  <FormItem>
                    <FormLabel>Seasonality Prior Scale</FormLabel>
                    <FormControl>
                      <Input
                        type='number'
                        step='0.1'
                        min='0'
                        {...field}
                        value={value}
                        onChange={(event) =>
                          onChange(parseFloat(event.target.value))
                        }
                      />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <FormField
                control={form.control}
                name='changepoint_prior_scale'
                render={({ field: { value, onChange, ...field } }) => (
                  <FormItem>
                    <FormLabel>Changepoint Prior Scale</FormLabel>
                    <FormControl>
                      <Input
                        type='number'
                        step='0.01'
                        min='0'
                        {...field}
                        value={value}
                        onChange={(event) =>
                          onChange(parseFloat(event.target.value))
                        }
                      />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <FormField
                control={form.control}
                name='changepoint_range'
                render={({ field: { value, onChange, ...field } }) => (
                  <FormItem>
                    <FormLabel>Changepoint Range</FormLabel>
                    <FormControl>
                      <Input
                        type='number'
                        step='0.1'
                        min='0'
                        max='1'
                        {...field}
                        value={value}
                        onChange={(event) =>
                          onChange(parseFloat(event.target.value))
                        }
                      />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <FormField
                control={form.control}
                name='n_changepoints'
                render={({ field: { value, onChange, ...field } }) => (
                  <FormItem>
                    <FormLabel>Number of Changepoints</FormLabel>
                    <FormControl>
                      <Input
                        type='number'
                        min='0'
                        {...field}
                        value={value}
                        onChange={(event) =>
                          onChange(parseInt(event.target.value))
                        }
                      />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />

              <div className='grid grid-cols-2 gap-4'>
                <FormField
                  control={form.control}
                  name='cap'
                  render={({ field: { value, onChange, ...field } }) => (
                    <FormItem>
                      <FormLabel>Cap (Optional)</FormLabel>
                      <FormControl>
                        <Input
                          type='number'
                          {...field}
                          value={value ?? ''}
                          onChange={(event) =>
                            onChange(
                              event.target.value
                                ? parseFloat(event.target.value)
                                : undefined
                            )
                          }
                        />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <FormField
                  control={form.control}
                  name='floor'
                  render={({ field: { value, onChange, ...field } }) => (
                    <FormItem>
                      <FormLabel>Floor (Optional)</FormLabel>
                      <FormControl>
                        <Input
                          type='number'
                          {...field}
                          value={value ?? ''}
                          onChange={(event) =>
                            onChange(
                              event.target.value
                                ? parseFloat(event.target.value)
                                : undefined
                            )
                          }
                        />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
              </div>
              <DialogFooter>
                <Button type='submit' className='mt-4'>
                  {forecast ? 'Update Forecast' : 'Create Forecast'}
                </Button>
              </DialogFooter>
            </form>
          </Form>
        </div>
      </DialogContent>
    </Dialog>
  )
}
