import React from 'react'
import { useTheme } from './theme-provider'

interface LogoOpieShortProps {
  className?: string
}

const LogoOpieShort: React.FC<LogoOpieShortProps> = ({ className}) => {
  const { theme } = useTheme()
  return (
    <img
      src={theme === 'dark' ? '/images/logo-opie-short-dark.png' : '/images/logo-opie-short-light.png'}
      alt='opie logo'
      className={className}
    />
  )
}

export default LogoOpieShort
