import React from 'react'
import { Layout } from '@/components/custom/layout'
import BlockchainSwitcher from '@/components/blockchain-switcher'
import { Button } from './ui/button'
import { ArrowLeft } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import HeaderPage from './header-page'

interface BasePageViewProps {
  children: React.ReactNode
  title: string | React.ReactNode
  description?: string | React.ReactNode  
  hideBlockchainSwitcher?: boolean
  hideDownloadButton?: boolean
  fixedLayout?: boolean
  actions?: React.ReactNode
  className?: string
  withBackButton?: boolean
  hideHeader?: boolean
}

const BasePageViewContent: React.FC<BasePageViewProps> = ({
  children,
  title,
  description,
  hideBlockchainSwitcher = true,
  hideDownloadButton = true,
  fixedLayout = false,
  actions,
  className,
  withBackButton = false,
  hideHeader = false,
}) => {


  const navigate = useNavigate()
  return (
    <Layout fixed={fixedLayout} className={className}>
      {!hideHeader && <HeaderPage />}
      <Layout.Header className='flex w-full flex-col items-start gap-0 px-4 sm:px-6 lg:px-8'>
        <div className='w-full'>
          {withBackButton && (
            <Button
              variant='ghost'
              onClick={() => navigate(-1)}
              className='mb-4'
            >
              <ArrowLeft className='mr-2 h-4 w-4' />
              Back
            </Button>
          )}
        </div>
        <div className=' flex w-full flex-col gap-4 lg:flex-row lg:items-center lg:justify-between'>
          <div className='flex flex-col space-y-2'>
            {typeof title === 'string' ? (
              <h2 className='text-xl font-bold tracking-tight sm:text-2xl lg:text-3xl'>
                {title}
              </h2>
            ) : (
              title
            )}
            {typeof description === 'string' ? (
              <p className='text-sm text-muted-foreground sm:text-base'>
                {description}
              </p>
            ) : (
              description
            )}
          </div>
          <div className='flex w-full flex-col gap-2 lg:w-auto lg:flex-row lg:items-center'>
            {!hideBlockchainSwitcher && (
              <div className='w-full lg:w-auto'>
                <BlockchainSwitcher />
              </div>
            )}
            {!hideDownloadButton && (
              <Button className='w-full lg:w-auto'>Download</Button>
            )}
            <div className='w-full lg:w-auto'>{actions}</div>
          </div>
        </div>
      </Layout.Header>
      <Layout.Body className='flex flex-1 flex-col px-2 py-0 sm:px-3 lg:px-6'>
        {children}
      </Layout.Body>
    </Layout>
  )
}

const BasePageView: React.FC<BasePageViewProps> = (props) => (
  <BasePageViewContent {...props} />
)

export default BasePageView
