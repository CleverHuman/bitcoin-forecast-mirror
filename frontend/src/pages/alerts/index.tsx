import React from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import BasePageView from '@/components/base-page-view'
import { useAuth } from '@/contexts/auth-context'
import { useNavigate } from 'react-router-dom'

const AlertsPage: React.FC = () => {
  const { isAuthenticated } = useAuth()
  const navigate = useNavigate()
  const alerts = [
    { id: 1, asset: 'Bitcoin', condition: 'Price above', value: '$30,000' },
    { id: 2, asset: 'Ethereum', condition: 'Price below', value: '$1,800' },
    { id: 3, asset: 'Cardano', condition: '24h change above', value: '5%' },
  ]

  const handleNavigation = (path: string) => {
    navigate(path)
  }

  return (
    <BasePageView
      title='Alerts'
      description='Manage your cryptocurrency alerts'
    >
      {isAuthenticated ? (
        <>
          <Card className='mb-4'>
            <CardHeader>
              <CardTitle>Create New Alert</CardTitle>
            </CardHeader>
            <CardContent>
              <div className='grid grid-cols-1 md:grid-cols-4 gap-4'>
                <Input placeholder='Asset name' />
                <Select>
                  <SelectTrigger>
                    <SelectValue placeholder='Condition' />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value='above'>Price above</SelectItem>
                    <SelectItem value='below'>Price below</SelectItem>
                    <SelectItem value='change-above'>
                      24h change above
                    </SelectItem>
                    <SelectItem value='change-below'>
                      24h change below
                    </SelectItem>
                  </SelectContent>
                </Select>
                <Input placeholder='Value' />
                <Button>Create Alert</Button>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle>Your Alerts</CardTitle>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Asset</TableHead>
                    <TableHead>Condition</TableHead>
                    <TableHead>Value</TableHead>
                    <TableHead>Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {alerts.map((alert) => (
                    <TableRow key={alert.id}>
                      <TableCell>{alert.asset}</TableCell>
                      <TableCell>{alert.condition}</TableCell>
                      <TableCell>{alert.value}</TableCell>
                      <TableCell>
                        <Button variant='destructive' size='sm'>
                          Delete
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </>
      ) : (
        <div className='flex items-center justify-center h-40'>
          <p className='text-center mx-auto'>
            Please{' '}
            <p
              onClick={() => handleNavigation('/sign-in')}
              className='ml-auto inline-block text-sm underline'
            >
              log in
            </p>{' '}
            to view your alerts.
          </p>
        </div>
      )}
    </BasePageView>
  )
}

export default AlertsPage
