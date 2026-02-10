import { Linkedin, Facebook, Share2 } from 'lucide-react'

import { Button } from '@/components/ui/button'
import {
  DropdownMenuItem,
  DropdownMenuContent,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { Twitter } from 'lucide-react'
import { Download } from 'lucide-react'

import { DropdownMenu } from '@/components/ui/dropdown-menu'

interface ShareForecastDropdownProps {
  handleShare: (platform: 'twitter' | 'facebook' | 'linkedin' | 'download') => Promise<void>
}

export default function ShareForecastDropdown({ handleShare }: ShareForecastDropdownProps) {


  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant='ghost' size='icon'>
          <Share2 className='h-4 w-4' />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align='end'>
        <DropdownMenuItem onClick={() => handleShare('download')}>
          <Download className='mr-2 h-4 w-4' />
          Download Image
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => handleShare('twitter')}>
          <Twitter className='mr-2 h-4 w-4' />
          Share on Twitter
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => handleShare('facebook')}>
          <Facebook className='mr-2 h-4 w-4' />
          Share on Facebook
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => handleShare('linkedin')}>
          <Linkedin className='mr-2 h-4 w-4' />
          Share on LinkedIn
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
