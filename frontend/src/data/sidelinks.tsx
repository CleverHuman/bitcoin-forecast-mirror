import { RouterPath } from '@/router/router-path'
import {
  IconChartLine} from '@tabler/icons-react'

export interface NavLink {
  title: string
  label?: string
  href: string
  icon: JSX.Element,
  modalBeforeNavigation?: boolean
}

export interface SideLink extends NavLink {
  sub?: NavLink[]
}

export const sidelinks: SideLink[] = [
  {
    title: 'Forecasting',
    label: '',
    href: RouterPath.FORECASTING,
    icon: <IconChartLine size={18} />,
  },
]
