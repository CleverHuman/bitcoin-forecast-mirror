import React from 'react'
import { useTheme } from './theme-provider'

interface LogoOpieProps {
  className?: string
}

const LogoOpie: React.FC<LogoOpieProps> = ({ className}) => {
  const { theme } = useTheme()
  return (
    <img
      src={theme === 'dark' ? '/images/logo-opie-dark.png' : '/images/logo-opie-light.png'}
      alt='opie logo'
      className={className}
    />
  )
}

export default LogoOpie
